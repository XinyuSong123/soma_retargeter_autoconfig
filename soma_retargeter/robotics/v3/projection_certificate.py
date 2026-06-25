"""Capability projection certificate decisions and deterministic evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any

import numpy as np

from .reachability_metrics import rank_with_uncertainty, retained_left_singular_vectors


CERTIFICATE_CLASSES = (
    "exact_reachable",
    "capability_limited_rank",
    "capability_limited_joint_limits",
    "capability_limited_mixed",
    "unsupported_rank_zero",
    "solver_failed",
    "numerical_invalid",
    "invalid_target_geometry",
)
GLOBAL_SEED_RESIDUAL_TOLERANCE = 1e-7
GLOBAL_TASK_SPACE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class ProjectionCertificateEvidence:
    task_block: str
    desired: np.ndarray
    projected: np.ndarray
    seed: np.ndarray
    jacobian: np.ndarray
    demand: np.ndarray | None = None
    residual: np.ndarray | None = None
    scale: float = 1.0
    noise_norm: float = 0.0
    rank_abs_floor: float = 1e-9
    rank_rel_tol: float = 1e-5
    rank_uncertainty_multiplier: float = 10.0
    converged: bool = True
    solver_status: str = ""
    active_coordinates: list[int] = field(default_factory=list)
    coordinate_values: np.ndarray | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    seed_consensus: dict[str, Any] | None = None
    seed_consensus_passed: bool | None = None
    jacobian_source: str = "unknown"
    solver_message: str = ""
    residual_tolerance: float | None = None
    component_tolerance: float | None = None
    exact_threshold: float | None = None
    normalized_residual: float | None = None
    task_residual: float | None = None
    active_lower_bounds: list[int] = field(default_factory=list)
    active_upper_bounds: list[int] = field(default_factory=list)
    kkt: dict[str, Any] | None = None
    task_gradient: np.ndarray | None = None
    prior_gradient: np.ndarray | None = None
    seed_results: list[dict[str, Any]] = field(default_factory=list)
    continuation_history: list[dict[str, Any]] = field(default_factory=list)
    scalar_dtype: str | None = None
    decomposition: dict[str, Any] | None = None
    continuation_passed: bool | None = None
    joint_limits_passed: bool | None = None
    numerical_gate_passed: bool | None = None
    residual_parameterization: str | None = None
    residual_jacobian_source: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    kkt_abs_tolerance: float = 1e-7
    kkt_rel_tolerance: float = 1e-5
    leakage_abs_tolerance: float = 1e-7
    boundary_abs_tolerance: float = 1e-9
    boundary_rel_tolerance: float = 1e-8


@dataclass(frozen=True)
class ProjectionCertificate:
    certificate_class: str
    task_block: str
    decomposition: dict[str, Any]
    gates: dict[str, bool | None]
    active_limits: dict[str, Any]
    kkt: dict[str, Any]
    seed_consensus: dict[str, Any]
    rank_evidence: dict[str, Any]
    reasons: list[str]
    deterministic_digest: str = ""
    schema_version: int = 2
    passed: bool = False
    motion_class: str = "ordinary"
    exact_threshold: float | None = None
    exact_threshold_passed: bool | None = None
    continuation: dict[str, Any] = field(default_factory=dict)
    joint_limits: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    audit_evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "certificate_class": self.certificate_class,
            "passed": bool(self.passed),
            "motion_class": self.motion_class,
            "task_block": self.task_block,
            "exact_threshold": self.exact_threshold,
            "exact_threshold_passed": self.exact_threshold_passed,
            "gates": self.gates.copy(),
            "decomposition": _json_clean(self.decomposition),
            "kkt": _json_clean(self.kkt),
            "seed_consensus": _json_clean(self.seed_consensus),
            "continuation": _json_clean(self.continuation),
            "joint_limits": _json_clean(self.joint_limits),
            "numerical": _json_clean(self.numerical),
            "active_limits": _json_clean(self.active_limits),
            "rank_evidence": _json_clean(self.rank_evidence),
            "audit_evidence": _json_clean(self.audit_evidence),
            "reasons": list(self.reasons),
            "deterministic_digest": self.deterministic_digest,
        }


def build_projection_certificate(evidence: ProjectionCertificateEvidence) -> ProjectionCertificate:
    shape_error = _shape_error(evidence)
    if shape_error is not None:
        return _finalize(
            _minimal_certificate(
                "invalid_target_geometry",
                evidence.task_block,
                [shape_error],
                seed_consensus=evidence.seed_consensus,
            )
        )
    finite_error = _finite_error(evidence)
    if finite_error is not None:
        return _finalize(
            _minimal_certificate(
                "numerical_invalid",
                evidence.task_block,
                [finite_error],
                seed_consensus=evidence.seed_consensus,
            )
        )
    desired = np.asarray(evidence.desired, dtype=float).reshape(3)
    projected = np.asarray(evidence.projected, dtype=float).reshape(3)
    seed = np.asarray(evidence.seed, dtype=float).reshape(3)
    jacobian = np.asarray(evidence.jacobian, dtype=float)
    scale = max(float(evidence.scale), np.finfo(float).tiny)
    residual = (
        np.asarray(evidence.residual, dtype=float).reshape(3)
        if evidence.residual is not None
        else projected - desired
    )
    demand = (
        np.asarray(evidence.demand, dtype=float).reshape(3)
        if evidence.demand is not None
        else desired - seed
    )
    retained = projected - seed
    residual_norm = float(np.linalg.norm(residual))
    demand_norm = float(np.linalg.norm(demand))
    retained_norm = float(np.linalg.norm(retained))
    residual_tolerance = _residual_tolerance(evidence, desired, projected, scale)
    component_tolerance = _component_tolerance(evidence, desired, projected, scale)
    exact_threshold = (
        float(evidence.exact_threshold)
        if evidence.exact_threshold is not None
        else float(residual_tolerance / scale)
    )
    exact_threshold_passed = bool(residual_norm <= residual_tolerance)
    rank, singular_values, rank_threshold = rank_with_uncertainty(
        jacobian,
        noise_norm=float(evidence.noise_norm),
        abs_floor=float(evidence.rank_abs_floor),
        rel_tol=float(evidence.rank_rel_tol),
        k_uncertainty=float(evidence.rank_uncertainty_multiplier),
    )
    tangent_basis = retained_left_singular_vectors(jacobian, rank)
    projector = tangent_basis @ tangent_basis.T if tangent_basis.size else np.zeros((3, 3))
    normal_projector = np.eye(3) - projector
    compatible_demand = projector @ demand
    compatible_retained = projector @ retained
    compatible_retention_error = compatible_retained - compatible_demand
    tangent_residual = projector @ residual
    rank_incompatible_residual = normal_projector @ residual
    orthogonal_leakage = normal_projector @ retained
    gradient = (jacobian.T @ residual) / (scale * scale) if jacobian.shape[1] else np.zeros(0)
    active_limits = _active_limits(evidence)
    scalar_dtype = _scalar_dtype(evidence)
    prior_gradient = _prior_gradient(evidence, gradient)
    kkt = _kkt_evidence(evidence, gradient, prior_gradient, active_limits, scalar_dtype)
    kkt_passed = bool(kkt.get("satisfied", False))
    tangent_residual_norm = float(np.linalg.norm(tangent_residual))
    rank_incompatible_residual_norm = float(np.linalg.norm(rank_incompatible_residual))
    active_limit_residual_norm = (
        tangent_residual_norm
        if active_limits["count"] > 0 and kkt_passed and tangent_residual_norm > component_tolerance
        else 0.0
    )
    explained_norm = float(np.hypot(rank_incompatible_residual_norm, active_limit_residual_norm))
    residual_explained = residual_norm <= residual_tolerance or abs(explained_norm - residual_norm) <= max(
        component_tolerance,
        1e-5 * max(1.0, residual_norm),
    )
    no_orthogonal_leakage = float(np.linalg.norm(orthogonal_leakage)) <= max(
        float(evidence.leakage_abs_tolerance),
        component_tolerance,
    )
    compatible_demand_retained = float(np.linalg.norm(compatible_retention_error)) <= component_tolerance
    if rank == 0 and residual_norm > residual_tolerance:
        compatible_demand_retained = False
    seed_consensus = _seed_consensus(
        evidence.seed_consensus,
        evidence.seed_consensus_passed,
        evidence.seed_results,
        scale,
    )
    seed_consensus_passed = bool(seed_consensus.get("passed", False))
    continuation = _continuation_evidence(evidence, residual_norm, residual_tolerance)
    continuation_passed = bool(continuation.get("passed", False))
    joint_limits = _joint_limits_evidence(active_limits, kkt, evidence.joint_limits_passed)
    joint_limits_passed = bool(joint_limits.get("passed", False))
    numerical = _numerical_evidence(evidence, scalar_dtype)
    numerical_passed = bool(numerical.get("passed", False))
    gates = {
        "exact_threshold_passed": exact_threshold_passed,
        "compatible_demand_retained": bool(compatible_demand_retained),
        "residual_explained": bool(residual_explained),
        "projected_gradient_kkt": kkt_passed,
        "seed_consensus": seed_consensus_passed,
        "continuation": continuation_passed,
        "joint_limits": joint_limits_passed,
        "numerical": numerical_passed,
        "no_orthogonal_leakage": bool(no_orthogonal_leakage),
    }
    certificate_class, reasons = _decide_certificate_class(
        evidence,
        residual_norm=residual_norm,
        residual_tolerance=residual_tolerance,
        component_tolerance=component_tolerance,
        rank=rank,
        rank_incompatible_residual_norm=rank_incompatible_residual_norm,
        active_limit_residual_norm=active_limit_residual_norm,
        kkt_passed=kkt_passed,
    )
    decomposition = {
        "rank": int(rank),
        "residual_norm": residual_norm,
        "demand_norm": demand_norm,
        "retained_norm": retained_norm,
        "compatible_demand_norm": float(np.linalg.norm(compatible_demand)),
        "compatible_retained_norm": float(np.linalg.norm(compatible_retained)),
        "compatible_retention_error_norm": float(np.linalg.norm(compatible_retention_error)),
        "tangent_residual_norm": tangent_residual_norm,
        "rank_incompatible_residual_norm": rank_incompatible_residual_norm,
        "reachable_residual_norm": tangent_residual_norm,
        "orthogonal_residual_norm": rank_incompatible_residual_norm,
        "active_limit_residual_norm": active_limit_residual_norm,
        "orthogonal_leakage_norm": float(np.linalg.norm(orthogonal_leakage)),
        "rank_incompatible_fraction": _safe_fraction(rank_incompatible_residual_norm, residual_norm),
        "reachable_residual_fraction": _safe_fraction(tangent_residual_norm, residual_norm),
        "orthogonal_residual_fraction": _safe_fraction(rank_incompatible_residual_norm, residual_norm),
        "active_limit_fraction": _safe_fraction(active_limit_residual_norm, residual_norm),
        "residual_tolerance": residual_tolerance,
        "component_tolerance": component_tolerance,
        "rank_threshold": rank_threshold,
        "singular_values": singular_values.tolist(),
    }
    rank_evidence = {
        "singular_values": singular_values.tolist(),
        "rank_threshold": rank_threshold,
        "rank_abs_floor": float(evidence.rank_abs_floor),
        "rank_rel_tol": float(evidence.rank_rel_tol),
        "rank_uncertainty_multiplier": float(evidence.rank_uncertainty_multiplier),
        "noise_norm": float(evidence.noise_norm),
        "jacobian_source": evidence.residual_jacobian_source or evidence.jacobian_source,
        "residual_parameterization": evidence.residual_parameterization,
        "residual_jacobian_source": evidence.residual_jacobian_source,
        "normalized_residual": evidence.normalized_residual,
        "task_residual": evidence.task_residual,
        "decomposition_input": evidence.decomposition or {},
        "evidence": evidence.evidence,
    }
    audit_evidence = _audit_evidence(
        evidence,
        desired=desired,
        projected=projected,
        seed=seed,
        residual=residual,
        demand=demand,
        jacobian=jacobian,
        gradient=gradient,
        prior_gradient=prior_gradient,
        scalar_dtype=scalar_dtype,
        scale=scale,
    )
    passed = _certificate_passed(certificate_class, gates)
    motion_class = str(evidence.evidence.get("motion_class", "ordinary")) if isinstance(evidence.evidence, dict) else "ordinary"
    return _finalize(
        ProjectionCertificate(
            certificate_class=certificate_class,
            task_block=evidence.task_block,
            decomposition=decomposition,
            gates=gates,
            active_limits=active_limits,
            kkt=kkt,
            seed_consensus=seed_consensus,
            rank_evidence=rank_evidence,
            reasons=reasons,
            passed=passed,
            motion_class=motion_class,
            exact_threshold=exact_threshold,
            exact_threshold_passed=exact_threshold_passed,
            continuation=continuation,
            joint_limits=joint_limits,
            numerical=numerical,
            audit_evidence=audit_evidence,
        )
    )


def _decide_certificate_class(
    evidence: ProjectionCertificateEvidence,
    *,
    residual_norm: float,
    residual_tolerance: float,
    component_tolerance: float,
    rank: int,
    rank_incompatible_residual_norm: float,
    active_limit_residual_norm: float,
    kkt_passed: bool,
) -> tuple[str, list[str]]:
    if residual_norm <= residual_tolerance and kkt_passed:
        return "exact_reachable", ["projection residual is within exact-reachable tolerance"]
    if not evidence.converged and evidence.active_coordinates:
        return "solver_failed", [f"solver did not converge: {evidence.solver_status or 'unknown_status'}"]
    if rank <= 0:
        return "unsupported_rank_zero", ["rank-zero task has nonzero demand"]
    if not kkt_passed:
        return "solver_failed", ["projected-gradient/KKT gate failed"]
    rank_limited = rank_incompatible_residual_norm > component_tolerance
    limit_limited = active_limit_residual_norm > component_tolerance
    if rank_limited and limit_limited:
        return "capability_limited_mixed", ["residual has rank-incompatible and active-limit components"]
    if limit_limited:
        return "capability_limited_joint_limits", ["compatible demand is blocked by active joint limits"]
    if rank_limited:
        return "capability_limited_rank", ["residual is outside the local tangent subspace"]
    return "solver_failed", ["residual remains but is not explained by rank or active limits"]


def _shape_error(evidence: ProjectionCertificateEvidence) -> str | None:
    for name in ("desired", "projected", "seed"):
        value = np.asarray(getattr(evidence, name), dtype=float)
        if value.shape != (3,):
            return f"{name} must be shaped (3,), got {value.shape}"
    for name in ("demand", "residual"):
        raw = getattr(evidence, name)
        if raw is None:
            continue
        value = np.asarray(raw, dtype=float)
        if value.shape != (3,):
            return f"{name} must be shaped (3,), got {value.shape}"
    jacobian = np.asarray(evidence.jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 3:
        return f"jacobian must be shaped (3, n), got {jacobian.shape}"
    return None


def _finite_error(evidence: ProjectionCertificateEvidence) -> str | None:
    arrays = {
        "desired": evidence.desired,
        "projected": evidence.projected,
        "seed": evidence.seed,
        "jacobian": evidence.jacobian,
    }
    if evidence.demand is not None:
        arrays["demand"] = evidence.demand
    if evidence.residual is not None:
        arrays["residual"] = evidence.residual
    for name, value in arrays.items():
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            return f"{name} contains non-finite values"
    for name in ("scale", "noise_norm", "rank_abs_floor", "rank_rel_tol"):
        if not np.isfinite(float(getattr(evidence, name))):
            return f"{name} is non-finite"
    return None


def _active_limits(evidence: ProjectionCertificateEvidence) -> dict[str, Any]:
    if evidence.active_lower_bounds or evidence.active_upper_bounds:
        boundary = None
        if isinstance(evidence.kkt, dict):
            boundary = evidence.kkt.get("active_bound_tolerance")
        return {
            "lower": [int(x) for x in evidence.active_lower_bounds],
            "upper": [int(x) for x in evidence.active_upper_bounds],
            "count": len(evidence.active_lower_bounds) + len(evidence.active_upper_bounds),
            "boundary_tolerance": float(boundary) if boundary is not None else float(evidence.boundary_abs_tolerance),
        }
    values = _optional_array(evidence.coordinate_values)
    lower = _optional_bound_array(evidence.lower_bounds)
    upper = _optional_bound_array(evidence.upper_bounds)
    active_coordinates = list(evidence.active_coordinates)
    if values is None or lower is None or upper is None or not active_coordinates:
        return {"lower": [], "upper": [], "count": 0, "boundary_tolerance": float(evidence.boundary_abs_tolerance)}
    lower_hits: list[int] = []
    upper_hits: list[int] = []
    tolerances: list[float] = []
    for idx, (coord, value, lo, hi) in enumerate(zip(active_coordinates, values, lower, upper)):
        finite_lo = np.isfinite(lo)
        finite_hi = np.isfinite(hi)
        span = abs(float(hi) - float(lo)) if finite_lo and finite_hi else max(1.0, abs(float(value)))
        tol = max(float(evidence.boundary_abs_tolerance), float(evidence.boundary_rel_tolerance) * max(1.0, span))
        tolerances.append(tol)
        if finite_lo and float(value) <= float(lo) + tol:
            lower_hits.append(int(coord))
        if finite_hi and float(value) >= float(hi) - tol:
            upper_hits.append(int(coord))
    return {
        "lower": lower_hits,
        "upper": upper_hits,
        "count": len(lower_hits) + len(upper_hits),
        "boundary_tolerance": max(tolerances, default=float(evidence.boundary_abs_tolerance)),
    }


def _kkt_evidence(
    evidence: ProjectionCertificateEvidence,
    gradient: np.ndarray,
    prior_gradient: np.ndarray,
    active_limits: dict[str, Any],
    scalar_dtype: str,
) -> dict[str, Any]:
    del active_limits
    values = _optional_array(evidence.coordinate_values)
    lower = _optional_bound_array(evidence.lower_bounds)
    upper = _optional_bound_array(evidence.upper_bounds)
    active_coordinates = list(evidence.active_coordinates)
    raw_kkt = evidence.kkt if isinstance(evidence.kkt, dict) else {}
    stationarity_tol = _stationarity_tolerance(evidence, scalar_dtype, raw_kkt)
    active_tol = _active_bound_tolerance(evidence, scalar_dtype, raw_kkt)
    projected_values: list[float] = []
    stationarity_violations: list[float] = []
    dual_violations: list[float] = []
    complementarity_violations: list[float] = []
    primal_violations: list[float] = []
    active_lower: list[int] = []
    active_upper: list[int] = []
    fixed: list[int] = []
    free: list[int] = []
    multipliers: list[dict[str, float | int | str]] = []
    if gradient.size == 0:
        projected = np.zeros(0)
    elif values is None or lower is None or upper is None or len(active_coordinates) != gradient.size:
        projected = np.asarray(gradient, dtype=float)
        projected_values = projected.tolist()
        stationarity_violations = [abs(float(g)) for g in gradient]
    else:
        for idx, g_raw in enumerate(gradient):
            coordinate = int(active_coordinates[idx])
            value = float(values[idx])
            lo = float(lower[idx])
            hi = float(upper[idx])
            g = float(g_raw)
            finite_lower = bool(np.isfinite(lo))
            finite_upper = bool(np.isfinite(hi))
            lower_gap = value - lo if finite_lower else float("inf")
            upper_gap = hi - value if finite_upper else float("inf")
            primal_violations.append(max(-lower_gap if finite_lower else 0.0, -upper_gap if finite_upper else 0.0, 0.0))
            is_fixed = bool(finite_lower and finite_upper and abs(hi - lo) <= active_tol)
            at_lower = bool(finite_lower and value <= lo + active_tol)
            at_upper = bool(finite_upper and value >= hi - active_tol)
            if is_fixed:
                fixed.append(coordinate)
                projected_values.append(0.0)
                stationarity_violations.append(0.0)
                dual_violations.append(0.0)
                complementarity_violations.append(0.0)
            elif at_lower:
                active_lower.append(coordinate)
                multiplier = max(g, 0.0)
                multipliers.append({"coordinate": coordinate, "side": "lower", "value": multiplier})
                projected_values.append(min(0.0, g))
                stationarity_violations.append(max(-g, 0.0))
                dual_violations.append(max(-g, 0.0))
                complementarity_violations.append(multiplier * abs(lower_gap))
            elif at_upper:
                active_upper.append(coordinate)
                multiplier = max(-g, 0.0)
                multipliers.append({"coordinate": coordinate, "side": "upper", "value": multiplier})
                projected_values.append(max(0.0, g))
                stationarity_violations.append(max(g, 0.0))
                dual_violations.append(max(g, 0.0))
                complementarity_violations.append(multiplier * abs(upper_gap))
            else:
                free.append(coordinate)
                projected_values.append(g)
                stationarity_violations.append(abs(g))
                dual_violations.append(0.0)
                complementarity_violations.append(0.0)
        projected = np.asarray(projected_values, dtype=float)
    if not primal_violations and values is not None and lower is not None and upper is not None:
        for value, lo, hi in zip(values, lower, upper):
            finite_lower = bool(np.isfinite(lo))
            finite_upper = bool(np.isfinite(hi))
            primal_violations.append(
                max(
                    float(lo - value) if finite_lower else 0.0,
                    float(value - hi) if finite_upper else 0.0,
                    0.0,
                )
            )
    stationarity_inf = float(max(stationarity_violations, default=0.0))
    primal_inf = float(max(primal_violations, default=0.0))
    dual_inf = float(max(dual_violations, default=0.0))
    complementarity_inf = float(max(complementarity_violations, default=0.0))
    grad_norm = float(np.linalg.norm(gradient))
    grad_inf = _inf_norm(gradient)
    prior_inf = _inf_norm(prior_gradient)
    provided_task_consistent = _provided_task_gradient_consistent(raw_kkt, gradient, stationarity_tol)
    provided_prior_consistent = _provided_prior_gradient_consistent(raw_kkt, prior_gradient, stationarity_tol)
    primal_feasible = bool(primal_inf <= active_tol)
    dual_feasible = bool(dual_inf <= stationarity_tol)
    complementarity_passed = bool(complementarity_inf <= stationarity_tol)
    satisfied = bool(
        primal_feasible
        and dual_feasible
        and complementarity_passed
        and stationarity_inf <= stationarity_tol
        and provided_task_consistent
        and provided_prior_consistent
    )
    if grad_inf <= stationarity_tol and prior_inf <= stationarity_tol:
        prior_cancellation_ratio = 0.0
    else:
        prior_cancellation_ratio = float(prior_inf / max(grad_inf, stationarity_tol))
    return {
        "gradient": gradient.tolist(),
        "task_gradient": gradient.tolist(),
        "prior_gradient": prior_gradient.tolist(),
        "projected_gradient": projected.tolist(),
        "gradient_norm": grad_norm,
        "task_gradient_inf_norm": grad_inf,
        "prior_gradient_inf_norm": prior_inf,
        "prior_cancellation_ratio": prior_cancellation_ratio,
        "projected_gradient_norm": stationarity_inf,
        "projected_gradient_inf_norm": stationarity_inf,
        "stationarity_inf_norm": stationarity_inf,
        "tolerance": stationarity_tol,
        "stationarity_tolerance": stationarity_tol,
        "active_bound_tolerance": active_tol,
        "primal_violation_inf_norm": primal_inf,
        "primal_feasible": primal_feasible,
        "dual_feasibility_inf_norm": dual_inf,
        "dual_feasible": dual_feasible,
        "complementarity_inf_norm": complementarity_inf,
        "complementarity_passed": complementarity_passed,
        "active_lower": active_lower,
        "active_upper": active_upper,
        "fixed_coordinates": fixed,
        "free_coordinates": free,
        "multipliers": multipliers,
        "provided_task_gradient_consistent": provided_task_consistent,
        "provided_prior_gradient_consistent": provided_prior_consistent,
        "satisfied": satisfied,
        "source": "independent_recompute",
        "raw": raw_kkt,
    }


def _seed_consensus(
    seed_consensus: dict[str, Any] | None,
    passed: bool | None = None,
    seed_results: list[dict[str, Any]] | None = None,
    task_scale: float = 1.0,
) -> dict[str, Any]:
    if seed_consensus is None:
        payload = {
            "checked": passed is not None,
            "passed": bool(passed) if passed is not None else False,
            "start_count": 0,
            "max_projected_delta": None,
            "max_residual_delta": None,
        }
    else:
        payload = dict(seed_consensus)
    if passed is not None:
        payload["passed"] = bool(passed)
        payload.setdefault("checked", True)
    rows = list(payload.get("seed_results") or seed_results or [])
    if rows:
        payload["seed_results"] = rows
        payload.setdefault("checked", True)
        payload.setdefault("start_count", len(rows))
        payload.setdefault("task_scale", float(task_scale))
        _recompute_seed_task_space_consensus(payload, rows, float(payload.get("task_scale", task_scale) or task_scale))
    return payload


def _recompute_seed_task_space_consensus(payload: dict[str, Any], rows: list[dict[str, Any]], task_scale: float) -> None:
    accepted = [row for row in rows if row.get("accepted", False)]
    if not accepted:
        payload["competitive_seed_count"] = 0
        payload["max_task_space_delta"] = None
        payload["task_space_spread"] = None
        payload["certificate_class_consensus"] = False
        payload["passed"] = False
        return
    residuals = [
        float(row.get("normalized_residual", row.get("task_residual_norm", 0.0)) or 0.0)
        for row in accepted
    ]
    best = min(residuals)
    residual_tolerance = float(payload.get("global_seed_residual_tolerance", GLOBAL_SEED_RESIDUAL_TOLERANCE))
    task_space_tolerance = max(
        float(payload.get("task_space_tolerance", payload.get("tolerance", GLOBAL_SEED_RESIDUAL_TOLERANCE)) or 0.0),
        GLOBAL_TASK_SPACE_TOLERANCE,
    )
    competitive = [
        row
        for row in accepted
        if float(row.get("normalized_residual", row.get("task_residual_norm", 0.0)) or 0.0) <= best + residual_tolerance
    ]
    vectors: list[np.ndarray] = []
    missing_vectors = False
    for row in competitive:
        vector = row.get("final_task_vector")
        if vector is None:
            missing_vectors = True
            continue
        vectors.append(np.asarray(vector, dtype=float).reshape(3))
    max_delta = 0.0
    if missing_vectors and len(competitive) >= 2:
        max_delta = float("inf")
    else:
        for i, lhs in enumerate(vectors):
            for rhs in vectors[i + 1 :]:
                max_delta = max(max_delta, float(np.linalg.norm(lhs - rhs)))
    normalized_spread = max_delta / max(task_scale, np.finfo(float).tiny)
    exact_threshold = _optional_float(payload.get("exact_threshold"))
    classes = {
        _seed_consensus_certificate_class(row, exact_threshold=exact_threshold)
        for row in competitive
        if row.get("certificate_class") is not None
    }
    class_consensus = len(classes) <= 1
    task_space_passed = bool(normalized_spread <= task_space_tolerance)
    payload["competitive_seed_count"] = len(competitive)
    payload["max_task_space_delta"] = max_delta
    payload["task_space_spread"] = normalized_spread
    payload["task_space_tolerance"] = task_space_tolerance
    payload["global_seed_residual_tolerance"] = residual_tolerance
    payload["certificate_class_consensus"] = class_consensus
    payload["certificate_classes"] = sorted(classes)
    payload["passed"] = bool(competitive and task_space_passed and class_consensus)


def _seed_consensus_certificate_class(row: dict[str, Any], *, exact_threshold: float | None) -> str:
    normalized = float(row.get("normalized_residual", row.get("task_residual_norm", 0.0)) or 0.0)
    if exact_threshold is not None and normalized <= exact_threshold:
        return "exact_reachable"
    return str(row.get("certificate_class"))


def _continuation_evidence(
    evidence: ProjectionCertificateEvidence,
    residual_norm: float,
    residual_tolerance: float,
) -> dict[str, Any]:
    history = [dict(step) for step in evidence.continuation_history]
    if not history:
        if evidence.continuation_passed is not None:
            passed = bool(evidence.continuation_passed)
            source = "explicit_gate_no_history"
        elif not evidence.active_coordinates:
            passed = True
            source = "trivial_rank_zero"
        elif evidence.converged and residual_norm <= residual_tolerance:
            passed = True
            source = "exact_seed_no_continuation"
        else:
            passed = False
            source = "missing_continuation_history"
        return {
            "checked": True,
            "passed": passed,
            "source": source,
            "history_length": 0,
            "starts_at_zero": passed,
            "accepted_alpha_strictly_increasing": passed,
            "reached_alpha_one": passed,
            "final_alpha": 1.0 if passed else None,
            "all_steps_finite": True,
            "all_steps_within_bounds": True if passed else None,
        }
    starts_at_zero = abs(float(history[0].get("alpha_start", float("nan")))) <= 1e-12
    accepted = all(bool(step.get("accepted", False)) for step in history)
    finite = all(_step_finite(step) for step in history)
    strictly_increasing = True
    previous_alpha = None
    for step in history:
        start = float(step.get("alpha_start", float("nan")))
        end = float(step.get("alpha_end", float("nan")))
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            strictly_increasing = False
        if previous_alpha is not None and start < previous_alpha - 1e-12:
            strictly_increasing = False
        previous_alpha = end
    final_alpha = float(history[-1].get("alpha_end", float("nan")))
    reached_alpha_one = bool(abs(final_alpha - 1.0) <= 1e-12)
    within_bounds = _continuation_steps_within_bounds(evidence, history)
    explicit_passed = True if evidence.continuation_passed is None else bool(evidence.continuation_passed)
    passed = bool(
        starts_at_zero
        and accepted
        and finite
        and strictly_increasing
        and reached_alpha_one
        and within_bounds
        and explicit_passed
    )
    return {
        "checked": True,
        "passed": passed,
        "source": "continuation_history",
        "history_length": len(history),
        "starts_at_zero": bool(starts_at_zero),
        "accepted_alpha_strictly_increasing": bool(strictly_increasing),
        "all_steps_accepted": bool(accepted),
        "reached_alpha_one": reached_alpha_one,
        "final_alpha": final_alpha if np.isfinite(final_alpha) else None,
        "all_steps_finite": bool(finite),
        "all_steps_within_bounds": bool(within_bounds),
    }


def _step_finite(step: dict[str, Any]) -> bool:
    for key in ("alpha_start", "alpha_end", "task_residual_norm", "prior_residual_norm"):
        if key in step and not np.isfinite(float(step[key])):
            return False
    if "q_active" in step:
        q = np.asarray(step["q_active"], dtype=float)
        return bool(np.all(np.isfinite(q)))
    return True


def _continuation_steps_within_bounds(evidence: ProjectionCertificateEvidence, history: list[dict[str, Any]]) -> bool:
    lower = _optional_bound_array(evidence.lower_bounds)
    upper = _optional_bound_array(evidence.upper_bounds)
    if lower is None or upper is None:
        return all(bool(step.get("within_bounds", True)) for step in history)
    active_tol = _active_bound_tolerance(evidence, _scalar_dtype(evidence), evidence.kkt if isinstance(evidence.kkt, dict) else {})
    for step in history:
        if not bool(step.get("within_bounds", True)):
            return False
        if "q_active" not in step:
            continue
        q = np.asarray(step["q_active"], dtype=float)
        if q.size != lower.size:
            continue
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if np.any(q[finite_lower] < lower[finite_lower] - active_tol):
            return False
        if np.any(q[finite_upper] > upper[finite_upper] + active_tol):
            return False
    return True


def _joint_limits_evidence(
    active_limits: dict[str, Any],
    kkt: dict[str, Any],
    explicit_passed: bool | None,
) -> dict[str, Any]:
    passed = bool(kkt.get("primal_feasible", False))
    if explicit_passed is not None:
        passed = bool(passed and explicit_passed)
    return {
        "checked": True,
        "passed": passed,
        "active_bound_count": int(active_limits.get("count", 0) or 0),
        "active_lower": list(active_limits.get("lower", [])),
        "active_upper": list(active_limits.get("upper", [])),
        "fixed_coordinates": list(kkt.get("fixed_coordinates", [])),
        "primal_feasible": bool(kkt.get("primal_feasible", False)),
        "primal_violation_inf_norm": kkt.get("primal_violation_inf_norm"),
        "boundary_tolerance": active_limits.get("boundary_tolerance"),
    }


def _numerical_evidence(evidence: ProjectionCertificateEvidence, scalar_dtype: str) -> dict[str, Any]:
    arrays = [
        evidence.desired,
        evidence.projected,
        evidence.seed,
        evidence.jacobian,
        [] if evidence.task_gradient is None else evidence.task_gradient,
        [] if evidence.prior_gradient is None else evidence.prior_gradient,
    ]
    nonfinite_count = 0
    for value in arrays:
        arr = np.asarray(value, dtype=float)
        nonfinite_count += int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    explicit_passed = True if evidence.numerical_gate_passed is None else bool(evidence.numerical_gate_passed)
    passed = bool(nonfinite_count == 0 and explicit_passed)
    return {
        "checked": True,
        "passed": passed,
        "jacobian_source": evidence.residual_jacobian_source or evidence.jacobian_source,
        "scalar_dtype": scalar_dtype,
        "rank_stability_gate": "not_evaluated",
        "engine_fd_validation_status": "not_evaluated",
        "nonfinite_count": nonfinite_count,
    }


def _audit_evidence(
    evidence: ProjectionCertificateEvidence,
    *,
    desired: np.ndarray,
    projected: np.ndarray,
    seed: np.ndarray,
    residual: np.ndarray,
    demand: np.ndarray,
    jacobian: np.ndarray,
    gradient: np.ndarray,
    prior_gradient: np.ndarray,
    scalar_dtype: str,
    scale: float,
) -> dict[str, Any]:
    normalized_residual = residual / max(scale, np.finfo(float).tiny)
    return {
        "desired_vector": desired.tolist(),
        "projected_vector": projected.tolist(),
        "seed_vector": seed.tolist(),
        "normalized_residual_vector": normalized_residual.tolist(),
        "demand_vector": demand.tolist(),
        "relevant_task_jacobian": jacobian.tolist(),
        "active_coordinates": [int(coord) for coord in evidence.active_coordinates],
        "q_active": _optional_json_array(evidence.coordinate_values),
        "lower_bounds": _optional_json_array(evidence.lower_bounds),
        "upper_bounds": _optional_json_array(evidence.upper_bounds),
        "task_gradient": gradient.tolist(),
        "provided_task_gradient": _optional_json_array(evidence.task_gradient),
        "prior_gradient": prior_gradient.tolist(),
        "seed_results": [dict(row) for row in evidence.seed_results],
        "continuation_history": [dict(step) for step in evidence.continuation_history],
        "scalar_dtype": scalar_dtype,
        "normalization_scale": float(scale),
        "residual_parameterization": evidence.residual_parameterization,
        "jacobian_source": evidence.residual_jacobian_source or evidence.jacobian_source,
    }


def _certificate_passed(certificate_class: str, gates: dict[str, bool | None]) -> bool:
    if certificate_class in {"solver_failed", "numerical_invalid", "invalid_target_geometry"}:
        return False
    required = ["projected_gradient_kkt", "seed_consensus", "continuation", "joint_limits", "numerical"]
    if certificate_class != "exact_reachable":
        required.append("residual_explained")
    return all(bool(gates.get(name, False)) for name in required)


def _residual_tolerance(
    evidence: ProjectionCertificateEvidence,
    desired: np.ndarray,
    projected: np.ndarray,
    scale: float,
) -> float:
    if evidence.residual_tolerance is not None:
        return float(evidence.residual_tolerance)
    if evidence.exact_threshold is not None:
        return float(evidence.exact_threshold) * float(scale)
    return 1e-7 * max(1.0, scale, float(np.linalg.norm(desired)), float(np.linalg.norm(projected)))


def _component_tolerance(
    evidence: ProjectionCertificateEvidence,
    desired: np.ndarray,
    projected: np.ndarray,
    scale: float,
) -> float:
    if evidence.component_tolerance is not None:
        return float(evidence.component_tolerance)
    return 1e-7 * max(1.0, scale, float(np.linalg.norm(desired)), float(np.linalg.norm(projected)))


def _optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _optional_bound_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if np.any(np.isnan(arr)):
        return None
    return arr


def _optional_json_array(value: np.ndarray | None) -> list[Any]:
    if value is None:
        return []
    return np.asarray(value, dtype=float).tolist()


def _scalar_dtype(evidence: ProjectionCertificateEvidence) -> str:
    if evidence.scalar_dtype:
        dtype = str(evidence.scalar_dtype).lower()
    elif isinstance(evidence.kkt, dict) and evidence.kkt.get("scalar_dtype"):
        dtype = str(evidence.kkt.get("scalar_dtype")).lower()
    else:
        dtype = "float64"
    if "float32" in dtype or dtype in {"single", "fp32"}:
        return "float32"
    return "float64"


def _stationarity_tolerance(
    evidence: ProjectionCertificateEvidence,
    scalar_dtype: str,
    raw_kkt: dict[str, Any],
) -> float:
    del raw_kkt
    if scalar_dtype == "float32":
        return 5e-5
    return max(float(evidence.kkt_abs_tolerance), 1e-7)


def _active_bound_tolerance(
    evidence: ProjectionCertificateEvidence,
    scalar_dtype: str,
    raw_kkt: dict[str, Any],
) -> float:
    del raw_kkt
    if scalar_dtype == "float32":
        return 1e-5
    return max(float(evidence.boundary_abs_tolerance), 1e-9)


def _prior_gradient(evidence: ProjectionCertificateEvidence, gradient: np.ndarray) -> np.ndarray:
    if evidence.prior_gradient is None:
        return np.zeros_like(gradient, dtype=float)
    arr = np.asarray(evidence.prior_gradient, dtype=float).reshape(-1)
    if arr.size == gradient.size:
        return arr
    if arr.size == 0:
        return np.zeros_like(gradient, dtype=float)
    return arr


def _inf_norm(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _provided_task_gradient_consistent(raw_kkt: dict[str, Any], gradient: np.ndarray, tolerance: float) -> bool:
    if not raw_kkt or "task_gradient" not in raw_kkt:
        return True
    provided = np.asarray(raw_kkt.get("task_gradient", []), dtype=float).reshape(-1)
    expected = np.asarray(gradient, dtype=float).reshape(-1)
    if provided.size != expected.size:
        return False
    return bool(_inf_norm(provided - expected) <= max(float(tolerance), 1e-9 * max(1.0, _inf_norm(expected))))


def _provided_prior_gradient_consistent(raw_kkt: dict[str, Any], prior_gradient: np.ndarray, tolerance: float) -> bool:
    if not raw_kkt:
        return True
    expected_inf = _inf_norm(prior_gradient)
    if "prior_gradient" in raw_kkt:
        provided = np.asarray(raw_kkt.get("prior_gradient", []), dtype=float).reshape(-1)
        expected = np.asarray(prior_gradient, dtype=float).reshape(-1)
        if provided.size != expected.size:
            return False
        return bool(_inf_norm(provided - expected) <= max(float(tolerance), 1e-9 * max(1.0, _inf_norm(expected))))
    if "prior_gradient_inf_norm" in raw_kkt:
        provided_inf = float(raw_kkt.get("prior_gradient_inf_norm") or 0.0)
        return bool(abs(provided_inf - expected_inf) <= max(float(tolerance), 1e-9 * max(1.0, expected_inf)))
    return True


def _safe_fraction(part: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return float(part / total)


def _minimal_certificate(
    certificate_class: str,
    task_block: str,
    reasons: list[str],
    *,
    seed_consensus: dict[str, Any] | None,
) -> ProjectionCertificate:
    return ProjectionCertificate(
        certificate_class=certificate_class,
        task_block=task_block,
        decomposition={
            "rank": 0,
            "residual_norm": None,
            "demand_norm": None,
            "retained_norm": None,
            "compatible_demand_norm": None,
            "compatible_retained_norm": None,
            "compatible_retention_error_norm": None,
            "tangent_residual_norm": None,
            "rank_incompatible_residual_norm": None,
            "active_limit_residual_norm": None,
            "orthogonal_leakage_norm": None,
            "rank_incompatible_fraction": None,
            "active_limit_fraction": None,
            "residual_tolerance": None,
            "component_tolerance": None,
        },
        gates={
            "compatible_demand_retained": None,
            "residual_explained": None,
            "projected_gradient_kkt": None,
            "seed_consensus": None,
            "no_orthogonal_leakage": None,
        },
        active_limits={"lower": [], "upper": [], "count": 0, "boundary_tolerance": None},
        kkt={
            "gradient": [],
            "projected_gradient": [],
            "gradient_norm": None,
            "projected_gradient_norm": None,
            "tolerance": None,
        },
        seed_consensus=_seed_consensus(seed_consensus),
        rank_evidence={
            "singular_values": [],
            "rank_threshold": None,
            "rank_abs_floor": None,
            "rank_rel_tol": None,
            "rank_uncertainty_multiplier": None,
            "noise_norm": None,
            "jacobian_source": "unavailable",
        },
        reasons=reasons,
    )


def _finalize(certificate: ProjectionCertificate) -> ProjectionCertificate:
    payload = certificate.to_json()
    payload["deterministic_digest"] = ""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return replace(certificate, deterministic_digest=digest)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
