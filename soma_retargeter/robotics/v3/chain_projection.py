"""Offline bounded chain-only projection references."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine_jacobian import engine_relative_jacobian
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .numerical_jacobian import numerical_relative_jacobian
from .projection_solver import (
    ContinuationSolverConfig,
    SO3_LOG_JACOBIAN_CONVENTION,
    ResidualJacobianEvaluation,
    deterministic_seed_candidates,
    rank_zero_kkt_evidence,
    so3_log_residual_jacobian,
    solve_bounded_continuation,
)
from .spatial import so3_exp, so3_log

GLOBAL_SEED_RESIDUAL_TOLERANCE = 1e-7
GLOBAL_TASK_SPACE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ProjectionResult:
    desired: np.ndarray
    projected: np.ndarray
    chain_q: np.ndarray
    residual: float
    normalized_residual: float
    normalization_scale: float
    converged: bool
    status: str
    iterations: int = 0
    active_coordinates: list[int] = field(default_factory=list)
    prior_residual_norm: float = 0.0
    solver_message: str = ""
    demand_residual: float | None = None
    unreachable_demand: bool | None = None
    rank_zero_reason: str | None = None
    normalization_reference: str = "neutral_chain_length"
    residual_parameterization: str = "position"
    residual_jacobian_source: str | None = None
    residual_jacobian_scalar_dtype: str | None = None
    deterministic_start_count: int = 0
    seed_results: list[dict] = field(default_factory=list)
    selected_seed_index: int | None = None
    seed_consensus_passed: bool = True
    continuation_history: list[dict] = field(default_factory=list)
    deterministic_adaptive_subdivision: dict = field(default_factory=dict)
    active_limit_kkt: dict = field(default_factory=dict)
    task_gradient: np.ndarray | None = None
    task_gradient_inf_norm: float = 0.0
    prior_gradient: np.ndarray | None = None
    prior_gradient_inf_norm: float = 0.0
    capability_certificate: dict = field(default_factory=dict)
    so3_jacobian_convention: str | None = None

    def to_json(self) -> dict:
        active_limit_kkt = self.active_limit_kkt
        if not active_limit_kkt and self.status in {"rank_zero", "unreachable/rank_zero"}:
            demand_residual = self.residual if self.demand_residual is None else self.demand_residual
            active_limit_kkt = rank_zero_kkt_evidence(demand_residual)
        payload = {
            "desired": self.desired.tolist(),
            "projected": self.projected.tolist(),
            "chain_q": self.chain_q.tolist(),
            "residual": self.residual,
            "normalized_residual": self.normalized_residual,
            "normalization_scale": self.normalization_scale,
            "converged": self.converged,
            "status": self.status,
            "iterations": self.iterations,
            "active_coordinates": self.active_coordinates,
            "prior_residual_norm": self.prior_residual_norm,
            "solver_message": self.solver_message,
            "normalization_reference": self.normalization_reference,
            "residual_parameterization": self.residual_parameterization,
            "residual_jacobian_source": self.residual_jacobian_source,
            "residual_jacobian_scalar_dtype": self.residual_jacobian_scalar_dtype,
            "deterministic_start_count": self.deterministic_start_count,
            "seed_results": self.seed_results,
            "selected_seed_index": self.selected_seed_index,
            "seed_consensus_passed": self.seed_consensus_passed,
            "continuation_history": [step.to_json() if hasattr(step, "to_json") else dict(step) for step in self.continuation_history],
            "deterministic_adaptive_subdivision": self.deterministic_adaptive_subdivision
            or _empty_subdivision_summary(),
            "active_limit_kkt": active_limit_kkt,
            "task_gradient": [] if self.task_gradient is None else np.asarray(self.task_gradient, dtype=float).tolist(),
            "task_gradient_inf_norm": self.task_gradient_inf_norm,
            "prior_gradient": [] if self.prior_gradient is None else np.asarray(self.prior_gradient, dtype=float).tolist(),
            "prior_gradient_inf_norm": self.prior_gradient_inf_norm,
            "capability_certificate": self.capability_certificate,
        }
        if self.so3_jacobian_convention is not None:
            payload["so3_jacobian_convention"] = self.so3_jacobian_convention
        if self.status in {"rank_zero", "unreachable/rank_zero"}:
            demand_residual = self.residual if self.demand_residual is None else self.demand_residual
            payload["demand_residual"] = demand_residual
            payload["unreachable_demand"] = (
                demand_residual > 1e-10 if self.unreachable_demand is None else self.unreachable_demand
            )
            payload["rank_zero_reason"] = self.rank_zero_reason or _rank_zero_reason(demand_residual)
        else:
            if self.demand_residual is not None:
                payload["demand_residual"] = self.demand_residual
            if self.unreachable_demand is not None:
                payload["unreachable_demand"] = self.unreachable_demand
            if self.rank_zero_reason is not None:
                payload["rank_zero_reason"] = self.rank_zero_reason
        return payload


def project_endpoint_position(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_reference_position: np.ndarray,
    *,
    neutral_q: np.ndarray | None = None,
    previous_q: np.ndarray | None = None,
    neutral_prior_weight: float = 1e-12,
    continuity_prior_weight: float = 1e-8,
    exact_threshold: float | None = None,
) -> ProjectionResult:
    scale = _position_normalization_scale(adapter, q_seed, reference, target, active_coordinates)
    desired_reference_position = np.asarray(desired_reference_position, dtype=float).copy()
    current_seed_position = adapter.relative_transform(adapter.forward_kinematics(q_seed), reference, target)[:3, 3]
    seed_residual_abs = float(np.linalg.norm(current_seed_position - desired_reference_position))
    if not active_coordinates:
        return _rank_zero_result(
            desired_reference_position,
            current_seed_position,
            q_seed,
            seed_residual_abs,
            scale,
            residual_parameterization="position",
        )
    active_values, lower, upper = _initial_and_bounds(adapter, q_seed, active_coordinates)
    if seed_residual_abs <= 1e-10 and _seed_within_actual_bounds(adapter, q_seed, active_coordinates):
        return _exact_seed_result(
            desired_reference_position,
            current_seed_position,
            q_seed,
            seed_residual_abs,
            scale,
            active_coordinates,
            active_values=active_values,
            lower_bounds=lower,
            upper_bounds=upper,
            residual_parameterization="position",
        )

    x0, lo, hi = active_values, lower, upper
    prior = _prior_context(
        adapter,
        active_coordinates,
        neutral_q=neutral_q,
        previous_q=previous_q,
        neutral_prior_weight=neutral_prior_weight,
        continuity_prior_weight=continuity_prior_weight,
    )
    seeds = _solver_seeds(x0, lo, hi, prior)

    def make_evaluator(seed_x: np.ndarray):
        seed_q = adapter.set_velocity_coordinates(q_seed, active_coordinates, seed_x)
        seed_position = adapter.relative_transform(adapter.forward_kinematics(seed_q), reference, target)[:3, 3]
        demand = desired_reference_position - seed_position
        prior_jacobian = _prior_jacobian(prior)

        def evaluate(x: np.ndarray, alpha: float) -> ResidualJacobianEvaluation:
            q = adapter.set_velocity_coordinates(q_seed, active_coordinates, x)
            relative = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)
            target_position = seed_position + float(alpha) * demand
            task_residual = (relative[:3, 3] - target_position) / scale
            jacobian = _relative_jacobian(adapter, q, reference, target, active_coordinates)
            return ResidualJacobianEvaluation(
                task_residual=task_residual,
                task_jacobian=jacobian["translation"] / scale,
                prior_residual=_prior_residual(x, prior),
                prior_jacobian=prior_jacobian,
                jacobian_source=str(jacobian["source"]),
                scalar_dtype=str(jacobian["scalar_dtype"]),
                residual_parameterization="position",
            )

        return evaluate

    solve = solve_bounded_continuation(
        make_evaluator,
        seeds,
        lo,
        hi,
        active_coordinates=active_coordinates,
        config=_continuation_config(exact_threshold),
    )
    q = adapter.set_velocity_coordinates(q_seed, active_coordinates, solve.x)
    projected = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, 3]
    residual_abs = float(np.linalg.norm(projected - desired_reference_position))
    seed_results = _decorate_seed_results(
        adapter,
        q_seed,
        reference,
        target,
        active_coordinates,
        solve.seed_results,
        desired_reference_position,
        scale,
        task_block="translation",
    )
    return _projection_result_from_solve(
        desired_reference_position,
        projected,
        q,
        residual_abs,
        scale,
        active_coordinates,
        solve,
        seed_results=seed_results,
        residual_parameterization="position",
    )


def project_torso_orientation(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_relative_rotation: np.ndarray,
    *,
    neutral_q: np.ndarray | None = None,
    previous_q: np.ndarray | None = None,
    neutral_prior_weight: float = 1e-12,
    continuity_prior_weight: float = 1e-8,
    exact_threshold: float | None = None,
) -> ProjectionResult:
    scale = np.pi
    desired_relative_rotation = np.asarray(desired_relative_rotation, dtype=float).copy()
    current_seed_rotation = adapter.relative_transform(adapter.forward_kinematics(q_seed), reference, target)[:3, :3]
    seed_error = so3_log(current_seed_rotation.T @ desired_relative_rotation)
    seed_residual_abs = float(np.linalg.norm(seed_error))
    if not active_coordinates:
        return _rank_zero_result(
            so3_log(desired_relative_rotation),
            so3_log(current_seed_rotation),
            q_seed,
            seed_residual_abs,
            scale,
            residual_parameterization="so3_log",
            so3_jacobian_convention=SO3_LOG_JACOBIAN_CONVENTION,
        )
    active_values, lower, upper = _initial_and_bounds(adapter, q_seed, active_coordinates)
    if seed_residual_abs <= 1e-10 and _seed_within_actual_bounds(adapter, q_seed, active_coordinates):
        return _exact_seed_result(
            so3_log(desired_relative_rotation),
            so3_log(current_seed_rotation),
            q_seed,
            seed_residual_abs,
            scale,
            active_coordinates,
            active_values=active_values,
            lower_bounds=lower,
            upper_bounds=upper,
            residual_parameterization="so3_log",
            so3_jacobian_convention=SO3_LOG_JACOBIAN_CONVENTION,
        )

    x0, lo, hi = active_values, lower, upper
    prior = _prior_context(
        adapter,
        active_coordinates,
        neutral_q=neutral_q,
        previous_q=previous_q,
        neutral_prior_weight=neutral_prior_weight,
        continuity_prior_weight=continuity_prior_weight,
    )
    seeds = _solver_seeds(x0, lo, hi, prior)

    def make_evaluator(seed_x: np.ndarray):
        seed_q = adapter.set_velocity_coordinates(q_seed, active_coordinates, seed_x)
        seed_rotation = adapter.relative_transform(adapter.forward_kinematics(seed_q), reference, target)[:3, :3]
        demand_log = so3_log(seed_rotation.T @ desired_relative_rotation)
        prior_jacobian = _prior_jacobian(prior)

        def evaluate(x: np.ndarray, alpha: float) -> ResidualJacobianEvaluation:
            q = adapter.set_velocity_coordinates(q_seed, active_coordinates, x)
            relative = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)
            current_rotation = relative[:3, :3]
            target_rotation = seed_rotation @ so3_exp(float(alpha) * demand_log)
            error = so3_log(current_rotation.T @ target_rotation)
            jacobian = _relative_jacobian(adapter, q, reference, target, active_coordinates)
            task_jacobian = so3_log_residual_jacobian(current_rotation, error, jacobian["rotation"]) / scale
            return ResidualJacobianEvaluation(
                task_residual=error / scale,
                task_jacobian=task_jacobian,
                prior_residual=_prior_residual(x, prior),
                prior_jacobian=prior_jacobian,
                jacobian_source=str(jacobian["source"]),
                scalar_dtype=str(jacobian["scalar_dtype"]),
                residual_parameterization="so3_log",
                so3_jacobian_convention=SO3_LOG_JACOBIAN_CONVENTION,
            )

        return evaluate

    solve = solve_bounded_continuation(
        make_evaluator,
        seeds,
        lo,
        hi,
        active_coordinates=active_coordinates,
        config=_continuation_config(exact_threshold),
    )
    q = adapter.set_velocity_coordinates(q_seed, active_coordinates, solve.x)
    rot = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, :3]
    err = so3_log(rot.T @ desired_relative_rotation)
    residual_abs = float(np.linalg.norm(err))
    desired_log = so3_log(desired_relative_rotation)
    seed_results = _decorate_seed_results(
        adapter,
        q_seed,
        reference,
        target,
        active_coordinates,
        solve.seed_results,
        desired_log,
        scale,
        task_block="rotation",
    )
    return _projection_result_from_solve(
        desired_log,
        so3_log(rot),
        q,
        residual_abs,
        scale,
        active_coordinates,
        solve,
        seed_results=seed_results,
        residual_parameterization="so3_log",
        so3_jacobian_convention=SO3_LOG_JACOBIAN_CONVENTION,
    )


def _projection_result_from_solve(
    desired: np.ndarray,
    projected: np.ndarray,
    q: np.ndarray,
    residual_abs: float,
    scale: float,
    active_coordinates: list[int],
    solve,
    *,
    seed_results: list[dict] | None = None,
    residual_parameterization: str,
    so3_jacobian_convention: str | None = None,
) -> ProjectionResult:
    rows = list(solve.seed_results if seed_results is None else seed_results)
    return ProjectionResult(
        np.asarray(desired, dtype=float).copy(),
        np.asarray(projected, dtype=float).copy(),
        np.asarray(q, dtype=float).copy(),
        residual_abs,
        float(residual_abs / scale),
        scale,
        bool(solve.success),
        _solver_status(bool(solve.success), residual_abs),
        iterations=int(solve.nfev),
        active_coordinates=list(active_coordinates),
        prior_residual_norm=float(np.linalg.norm(solve.prior_residual)),
        solver_message=(
            f"{solve.message}; deterministic_start_count={solve.deterministic_start_count}; "
            f"selected_seed_index={solve.selected_seed_index}; "
            f"jacobian_callback={solve.residual_jacobian_source}"
        ),
        residual_parameterization=residual_parameterization,
        residual_jacobian_source=solve.residual_jacobian_source,
        residual_jacobian_scalar_dtype=solve.residual_jacobian_scalar_dtype,
        deterministic_start_count=solve.deterministic_start_count,
        seed_results=rows,
        selected_seed_index=solve.selected_seed_index,
        seed_consensus_passed=_seed_consensus_passed(rows, solve.selected_seed_index, task_scale=scale),
        continuation_history=solve.continuation_history,
        deterministic_adaptive_subdivision=solve.deterministic_adaptive_subdivision,
        active_limit_kkt=solve.active_limit_kkt,
        task_gradient=solve.task_gradient,
        task_gradient_inf_norm=solve.task_gradient_inf_norm,
        prior_gradient=solve.prior_gradient,
        prior_gradient_inf_norm=solve.prior_gradient_inf_norm,
        so3_jacobian_convention=so3_jacobian_convention,
    )


def _decorate_seed_results(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    seed_results: list[dict],
    desired: np.ndarray,
    scale: float,
    *,
    task_block: str,
) -> list[dict]:
    rows: list[dict] = []
    desired = np.asarray(desired, dtype=float).reshape(3)
    for row in seed_results:
        payload = dict(row)
        final_x = np.asarray(payload.get("final_q_active", payload.get("final_x", [])), dtype=float)
        final_task_vector: np.ndarray | None = None
        if final_x.shape == (len(active_coordinates),):
            q = adapter.set_velocity_coordinates(q_seed, active_coordinates, final_x)
            relative = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)
            if task_block == "rotation":
                final_task_vector = so3_log(relative[:3, :3])
            else:
                final_task_vector = relative[:3, 3].copy()
        if final_task_vector is not None:
            payload["final_task_vector"] = final_task_vector.tolist()
            payload["task_space_endpoint"] = payload["final_task_vector"]
            payload["task_space_error_norm"] = float(np.linalg.norm(final_task_vector - desired))
        normalized = float(payload.get("normalized_residual", payload.get("task_residual_norm", 0.0)) or 0.0)
        payload["normalized_residual"] = normalized
        payload["certificate_class"] = _seed_certificate_class(payload, normalized, scale)
        rows.append(payload)
    return rows


def _seed_certificate_class(seed_result: dict, normalized_residual: float, scale: float) -> str:
    if not seed_result.get("accepted", False):
        return "solver_failed"
    residual_tolerance = 1e-7 * max(1.0, float(scale))
    if normalized_residual <= residual_tolerance:
        return "exact_reachable"
    active_limits = seed_result.get("active_limits", {})
    if isinstance(active_limits, dict) and int(active_limits.get("count", 0) or 0) > 0:
        return "capability_limited_joint_limits"
    return "capability_limited_rank"


def _rank_zero_result(
    desired: np.ndarray,
    projected: np.ndarray,
    q_seed: np.ndarray,
    residual_abs: float,
    scale: float,
    *,
    residual_parameterization: str,
    so3_jacobian_convention: str | None = None,
) -> ProjectionResult:
    return ProjectionResult(
        np.asarray(desired, dtype=float).copy(),
        np.asarray(projected, dtype=float).copy(),
        np.asarray(q_seed, dtype=float).copy(),
        residual_abs,
        float(residual_abs / scale),
        scale,
        residual_abs <= 1e-10,
        _rank_zero_status(residual_abs),
        active_coordinates=[],
        demand_residual=residual_abs,
        unreachable_demand=residual_abs > 1e-10,
        rank_zero_reason=_rank_zero_reason(residual_abs),
        residual_parameterization=residual_parameterization,
        residual_jacobian_source="rank_zero_no_active_coordinates",
        residual_jacobian_scalar_dtype="float64",
        deterministic_start_count=0,
        selected_seed_index=None,
        continuation_history=[],
        deterministic_adaptive_subdivision=_empty_subdivision_summary(),
        active_limit_kkt=rank_zero_kkt_evidence(residual_abs),
        task_gradient=np.zeros(0),
        task_gradient_inf_norm=0.0,
        prior_gradient=np.zeros(0),
        prior_gradient_inf_norm=0.0,
        so3_jacobian_convention=so3_jacobian_convention,
    )


def _exact_seed_result(
    desired: np.ndarray,
    projected: np.ndarray,
    q_seed: np.ndarray,
    residual_abs: float,
    scale: float,
    active_coordinates: list[int],
    *,
    active_values: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    residual_parameterization: str,
    so3_jacobian_convention: str | None = None,
) -> ProjectionResult:
    final_active = [] if active_values is None else np.asarray(active_values, dtype=float).tolist()
    active_limits = _exact_active_limit_summary(active_values, lower_bounds, upper_bounds, active_coordinates)
    free_coordinates = [
        int(coord)
        for coord in active_coordinates
        if coord not in set(active_limits["lower"]) | set(active_limits["upper"]) | set(active_limits["fixed"])
    ]
    return ProjectionResult(
        np.asarray(desired, dtype=float).copy(),
        np.asarray(projected, dtype=float).copy(),
        np.asarray(q_seed, dtype=float).copy(),
        residual_abs,
        float(residual_abs / scale),
        scale,
        True,
        "converged",
        iterations=0,
        active_coordinates=list(active_coordinates),
        solver_message="exact at deterministic seed; nonlinear solve skipped",
        residual_parameterization=residual_parameterization,
        residual_jacobian_source="exact_seed_no_nonlinear_solve",
        residual_jacobian_scalar_dtype="float64",
        deterministic_start_count=1,
        seed_results=[
            {
                "seed_index": 0,
                "accepted": True,
                "initial_x": final_active,
                "final_x": final_active,
                "final_q_active": final_active,
                "final_task_vector": np.asarray(projected, dtype=float).reshape(3).tolist(),
                "task_space_endpoint": np.asarray(projected, dtype=float).reshape(3).tolist(),
                "task_residual_norm": float(residual_abs / scale),
                "normalized_residual": float(residual_abs / scale),
                "prior_residual_norm": 0.0,
                "total_residual_norm": float(residual_abs / scale),
                "active_limits": active_limits,
                "certificate_class": "exact_reachable",
                "accepted_step_count": 0,
                "rejected_step_count": 0,
                "nfev": 1,
                "njev": 0,
            }
        ],
        selected_seed_index=0,
        seed_consensus_passed=True,
        continuation_history=[],
        deterministic_adaptive_subdivision=_empty_subdivision_summary(),
        active_limit_kkt={
            "rank_zero": False,
            "objective": "task_only_least_squares",
            "gradient_source": "exact_seed_zero_residual",
            "scalar_dtype": "float64",
            "active_bound_tolerance": 1e-9,
            "stationarity_tolerance": 1e-7,
            "primal_violation_inf_norm": 0.0,
            "primal_feasible": True,
            "dual_feasibility_inf_norm": 0.0,
            "dual_feasible": True,
            "complementarity_inf_norm": 0.0,
            "complementarity_passed": True,
            "stationarity_inf_norm": 0.0,
            "satisfied": True,
            "active_lower": active_limits["lower"],
            "active_upper": active_limits["upper"],
            "fixed_coordinates": active_limits["fixed"],
            "free_coordinates": free_coordinates,
            "multipliers": [],
            "projected_gradient": [0.0 for _ in active_coordinates],
            "projected_gradient_norm": 0.0,
            "task_gradient": [0.0 for _ in active_coordinates],
            "task_gradient_inf_norm": 0.0,
        },
        task_gradient=np.zeros(len(active_coordinates), dtype=float),
        task_gradient_inf_norm=0.0,
        prior_gradient=np.zeros(len(active_coordinates), dtype=float),
        prior_gradient_inf_norm=0.0,
        so3_jacobian_convention=so3_jacobian_convention,
    )


def _exact_active_limit_summary(
    values: np.ndarray | None,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    active_coordinates: list[int],
) -> dict:
    if values is None or lower is None or upper is None:
        return {"lower": [], "upper": [], "fixed": [], "count": 0, "boundary_tolerance": 1e-9}
    active_tol = 1e-9
    lower_hits: list[int] = []
    upper_hits: list[int] = []
    fixed: list[int] = []
    for idx, coord in enumerate(active_coordinates):
        finite_lower = bool(np.isfinite(lower[idx]))
        finite_upper = bool(np.isfinite(upper[idx]))
        if finite_lower and finite_upper and abs(float(upper[idx] - lower[idx])) <= active_tol:
            fixed.append(int(coord))
            continue
        if finite_lower and values[idx] <= lower[idx] + active_tol:
            lower_hits.append(int(coord))
        if finite_upper and values[idx] >= upper[idx] - active_tol:
            upper_hits.append(int(coord))
    return {
        "lower": lower_hits,
        "upper": upper_hits,
        "fixed": fixed,
        "count": len(lower_hits) + len(upper_hits) + len(fixed),
        "boundary_tolerance": active_tol,
    }


def _initial_and_bounds(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    active_coordinates: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = []
    lo = []
    hi = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type in {"revolute", "prismatic"}:
            value = float(q_seed[info.qpos_adr])
        else:
            value = 0.0
        lower = info.lower if np.isfinite(info.lower) else -np.pi
        upper = info.upper if np.isfinite(info.upper) else np.pi
        x0.append(value)
        lo.append(float(lower))
        hi.append(float(upper))
    x0_arr = np.asarray(x0, dtype=float)
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    x0_arr = np.clip(x0_arr, lo_arr, hi_arr)
    return x0_arr, lo_arr, hi_arr


def _seed_within_actual_bounds(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    active_coordinates: list[int],
) -> bool:
    tol = 1e-9
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type not in {"revolute", "prismatic"}:
            continue
        value = float(q_seed[info.qpos_adr])
        if np.isfinite(info.lower) and value < float(info.lower) - tol:
            return False
        if np.isfinite(info.upper) and value > float(info.upper) + tol:
            return False
    return True


def _position_normalization_scale(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> float:
    del q_seed
    state = adapter.forward_kinematics(adapter.neutral_q())
    body_path = adapter.body_path(reference.body_name, target.body_name)
    points = [adapter.site_transform(state, reference)[:3, 3]]
    for body_name in body_path:
        points.append(adapter.body_transform(state, body_name)[:3, 3])
    points.append(adapter.site_transform(state, target)[:3, 3])
    length = 0.0
    for a, b in zip(points, points[1:]):
        length += float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
    direct = float(np.linalg.norm(points[-1] - points[0]))
    return max(length, direct, _degenerate_prismatic_span(adapter, active_coordinates), 1e-6)


def _degenerate_prismatic_span(adapter: MuJoCoRuntimeModelAdapter, active_coordinates: list[int]) -> float:
    span = 0.0
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type != "prismatic" or not (np.isfinite(info.lower) and np.isfinite(info.upper)):
            continue
        span += abs(float(info.upper) - float(info.lower))
    return span


def _rank_zero_status(residual_abs: float) -> str:
    if residual_abs <= 1e-10:
        return "rank_zero"
    return "unreachable/rank_zero"


def _rank_zero_reason(demand_residual: float) -> str:
    if demand_residual <= 1e-10:
        return "no_active_coordinates_zero_demand"
    return "no_active_coordinates_nonzero_demand"


def _solver_status(success: bool, residual_abs: float) -> str:
    if not success:
        return "failed"
    if residual_abs <= 1e-8:
        return "converged"
    return "converged/with_residual"


def _continuation_config(exact_threshold: float | None) -> ContinuationSolverConfig | None:
    if exact_threshold is None:
        return None
    return ContinuationSolverConfig(final_numeric_polish_residual_threshold=float(exact_threshold))


def _prior_context(
    adapter: MuJoCoRuntimeModelAdapter,
    active_coordinates: list[int],
    *,
    neutral_q: np.ndarray | None,
    previous_q: np.ndarray | None,
    neutral_prior_weight: float,
    continuity_prior_weight: float,
) -> dict[str, np.ndarray | float | None]:
    neutral = adapter.neutral_q() if neutral_q is None else np.asarray(neutral_q, dtype=float)
    previous = None if previous_q is None else np.asarray(previous_q, dtype=float)
    return {
        "neutral_values": _coordinate_values(adapter, neutral, active_coordinates),
        "previous_values": None if previous is None else _coordinate_values(adapter, previous, active_coordinates),
        "scales": _coordinate_scales(adapter, active_coordinates),
        "neutral_weight": float(max(0.0, neutral_prior_weight)),
        "continuity_weight": float(max(0.0, continuity_prior_weight if previous is not None else 0.0)),
    }


def _prior_residual(x: np.ndarray, prior: dict[str, np.ndarray | float | None]) -> np.ndarray:
    scales = prior["scales"]
    assert isinstance(scales, np.ndarray)
    residuals: list[np.ndarray] = []
    neutral_weight = float(prior["neutral_weight"])
    if neutral_weight > 0.0:
        neutral_values = prior["neutral_values"]
        assert isinstance(neutral_values, np.ndarray)
        residuals.append(np.sqrt(neutral_weight) * (x - neutral_values) / scales)
    continuity_weight = float(prior["continuity_weight"])
    previous_values = prior["previous_values"]
    if continuity_weight > 0.0 and isinstance(previous_values, np.ndarray):
        residuals.append(np.sqrt(continuity_weight) * (x - previous_values) / scales)
    if not residuals:
        return np.zeros(0)
    return np.concatenate(residuals)


def _prior_jacobian(prior: dict[str, np.ndarray | float | None]) -> np.ndarray:
    scales = prior["scales"]
    assert isinstance(scales, np.ndarray)
    rows: list[np.ndarray] = []
    neutral_weight = float(prior["neutral_weight"])
    if neutral_weight > 0.0:
        rows.append(np.diag(np.sqrt(neutral_weight) / scales))
    continuity_weight = float(prior["continuity_weight"])
    previous_values = prior["previous_values"]
    if continuity_weight > 0.0 and isinstance(previous_values, np.ndarray):
        rows.append(np.diag(np.sqrt(continuity_weight) / scales))
    if not rows:
        return np.zeros((0, scales.size))
    return np.vstack(rows)


def _solver_seeds(
    x0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    prior: dict[str, np.ndarray | float | None],
) -> list[np.ndarray]:
    neutral_values = prior["neutral_values"] if isinstance(prior["neutral_values"], np.ndarray) else None
    previous_values = prior["previous_values"] if isinstance(prior["previous_values"], np.ndarray) else None
    return deterministic_seed_candidates(x0, lo, hi, neutral_values=neutral_values, previous_values=previous_values)


def _relative_jacobian(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> dict[str, np.ndarray | str]:
    try:
        engine = engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
        return {
            "translation": engine.translation,
            "rotation": engine.rotation,
            "source": "engine_relative_jacobian",
            "scalar_dtype": engine.scalar_dtype,
        }
    except Exception:
        fd = numerical_relative_jacobian(
            adapter,
            q,
            reference,
            target,
            active_coordinates,
            engine_validation=False,
        )
        return {
            "translation": fd.translation,
            "rotation": fd.rotation,
            "source": "finite_difference_fallback",
            "scalar_dtype": "float64",
        }


def _coordinate_values(adapter: MuJoCoRuntimeModelAdapter, q: np.ndarray, active_coordinates: list[int]) -> np.ndarray:
    values = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type in {"revolute", "prismatic"}:
            values.append(float(q[info.qpos_adr]))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=float)


def _coordinate_scales(adapter: MuJoCoRuntimeModelAdapter, active_coordinates: list[int]) -> np.ndarray:
    scales = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.limited and np.isfinite(info.lower) and np.isfinite(info.upper):
            scale = abs(info.upper - info.lower)
        elif info.joint_type == "prismatic":
            scale = 1.0
        elif info.joint_type == "revolute":
            scale = 2.0 * np.pi
        else:
            scale = np.pi
        scales.append(max(float(scale), 1e-6))
    return np.asarray(scales, dtype=float)


def _seed_consensus_passed(seed_results: list[dict], selected_seed_index: int | None, *, task_scale: float) -> bool:
    if not seed_results or selected_seed_index is None:
        return True
    selected = next((seed for seed in seed_results if seed["seed_index"] == selected_seed_index), None)
    if selected is None:
        return False
    accepted = [seed for seed in seed_results if seed.get("accepted", False)]
    if not accepted:
        return False
    best_norm = min(float(seed.get("normalized_residual", seed.get("task_residual_norm", 0.0))) for seed in accepted)
    residual_tolerance = GLOBAL_SEED_RESIDUAL_TOLERANCE
    competitive = [
        seed
        for seed in accepted
        if float(seed.get("normalized_residual", seed.get("task_residual_norm", 0.0))) <= best_norm + residual_tolerance
    ]
    if len(competitive) < 2:
        return True
    vectors = []
    for seed in competitive:
        vector = seed.get("final_task_vector")
        if vector is None:
            return False
        vectors.append(np.asarray(vector, dtype=float).reshape(3))
    max_delta = 0.0
    for i, lhs in enumerate(vectors):
        for rhs in vectors[i + 1 :]:
            max_delta = max(max_delta, float(np.linalg.norm(lhs - rhs)))
    normalized_spread = max_delta / max(float(task_scale), np.finfo(float).tiny)
    classes = {str(seed.get("certificate_class")) for seed in competitive if seed.get("certificate_class") is not None}
    return bool(normalized_spread <= GLOBAL_TASK_SPACE_TOLERANCE and len(classes) <= 1)


def _empty_subdivision_summary() -> dict:
    return {
        "strategy": "deterministic_residual_space_adaptive_subdivision",
        "configured_max_task_step_norm": 0.25,
        "max_task_step_norm": 0.0,
        "accepted_step_count": 0,
        "all_seed_accepted_step_count": 0,
        "all_seed_rejected_step_count": 0,
        "max_subdivision_depth": 0,
    }
