"""Deterministic bounded continuation solver for chain projection tasks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .spatial import skew


SO3_LOG_JACOBIAN_CONVENTION = "left_perturbation_log_current_transpose_target"
BOUND_SNAP_SEARCH_MULTIPLIER = 10.0
FINAL_POLISH_EXACT_COMPLETION_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ResidualJacobianEvaluation:
    task_residual: np.ndarray
    task_jacobian: np.ndarray
    prior_residual: np.ndarray
    prior_jacobian: np.ndarray
    jacobian_source: str
    scalar_dtype: str = "float64"
    residual_parameterization: str = "position"
    so3_jacobian_convention: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def residual(self) -> np.ndarray:
        if self.prior_residual.size == 0:
            return self.task_residual
        return np.concatenate([self.task_residual, self.prior_residual])

    @property
    def jacobian(self) -> np.ndarray:
        if self.prior_jacobian.size == 0:
            return self.task_jacobian
        return np.vstack([self.task_jacobian, self.prior_jacobian])


@dataclass(frozen=True)
class ContinuationSolverConfig:
    max_task_step_norm: float = 0.25
    min_alpha_step: float = 1e-6
    max_subdivision_depth: int = 10
    max_accepted_steps: int = 64
    xtol: float = 1e-14
    ftol: float = 1e-14
    gtol: float = 1e-12
    max_nfev_per_step: int = 60
    cost_tie_tolerance: float = 1e-15
    successful_seed_tie_residual_tolerance: float = 1e-7
    final_numeric_polish_residual_threshold: float | None = None
    final_numeric_polish_max_nfev: int = 60


@dataclass(frozen=True)
class ContinuationSolveResult:
    x: np.ndarray
    success: bool
    message: str
    nfev: int
    njev: int
    task_residual: np.ndarray
    prior_residual: np.ndarray
    task_jacobian: np.ndarray
    residual_jacobian_source: str
    residual_jacobian_scalar_dtype: str
    residual_parameterization: str
    so3_jacobian_convention: str | None
    continuation_history: list[dict]
    deterministic_start_count: int
    selected_seed_index: int
    seed_results: list[dict]
    deterministic_adaptive_subdivision: dict
    active_limit_kkt: dict
    task_gradient: np.ndarray
    task_gradient_inf_norm: float
    prior_gradient: np.ndarray
    prior_gradient_inf_norm: float


def kkt_tolerances_for_scalar_dtype(scalar_dtype: str) -> dict[str, float | str]:
    dtype = str(scalar_dtype).lower()
    if "float32" in dtype or dtype in {"single", "fp32"}:
        return {
            "scalar_dtype": "float32",
            "stationarity_tolerance": 5e-5,
            "active_bound_tolerance": 1e-5,
        }
    return {
        "scalar_dtype": "float64",
        "stationarity_tolerance": 1e-7,
        "active_bound_tolerance": 1e-9,
    }


def deterministic_seed_candidates(
    x0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    neutral_values: np.ndarray | None = None,
    previous_values: np.ndarray | None = None,
    max_corner_dim: int = 4,
    max_axis_starts: int = 4,
) -> list[np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), x0)
    candidates: list[np.ndarray] = [x0]
    if neutral_values is not None:
        candidates.append(np.asarray(neutral_values, dtype=float))
    if previous_values is not None:
        candidates.append(np.asarray(previous_values, dtype=float))
    candidates.append(mid)
    axes = list(range(x0.size))
    if x0.size > max_corner_dim and x0.size > max_axis_starts:
        axes = np.linspace(0, x0.size - 1, max_axis_starts, dtype=int).tolist()
    for axis in axes:
        if np.isfinite(lo[axis]):
            point = mid.copy()
            point[axis] = lo[axis]
            candidates.append(point)
        if np.isfinite(hi[axis]):
            point = mid.copy()
            point[axis] = hi[axis]
            candidates.append(point)
    if 0 < x0.size <= max_corner_dim and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
        for bits in product((0, 1), repeat=x0.size):
            candidates.append(np.asarray([hi[i] if bit else lo[i] for i, bit in enumerate(bits)], dtype=float))
    unique: list[np.ndarray] = []
    for candidate in candidates:
        clipped = np.clip(np.asarray(candidate, dtype=float), lo, hi)
        if not any(np.allclose(clipped, seen, rtol=0.0, atol=1e-12) for seen in unique):
            unique.append(clipped.copy())
    return unique


def solve_bounded_continuation(
    make_evaluator: Callable[[np.ndarray], Callable[[np.ndarray, float], ResidualJacobianEvaluation]],
    seeds: list[np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    active_coordinates: list[int] | None = None,
    config: ContinuationSolverConfig | None = None,
) -> ContinuationSolveResult:
    cfg = config or ContinuationSolverConfig()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("lower and upper bounds must have the same shape")
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds")
    clipped_seeds = [np.clip(np.asarray(seed, dtype=float), lo, hi) for seed in seeds]
    seed_runs = [
        _solve_one_seed(make_evaluator(seed), seed, seed_index, lo, hi, cfg)
        for seed_index, seed in enumerate(clipped_seeds)
    ]
    best = _select_best_seed(seed_runs, cfg)
    final_eval = best["final_eval"]
    assert isinstance(final_eval, ResidualJacobianEvaluation)
    task_gradient = final_eval.task_jacobian.T @ final_eval.task_residual
    task_gradient_inf = float(np.max(np.abs(task_gradient))) if task_gradient.size else 0.0
    prior_gradient = final_eval.prior_jacobian.T @ final_eval.prior_residual
    prior_gradient_inf = float(np.max(np.abs(prior_gradient))) if prior_gradient.size else 0.0
    coordinate_ids = list(active_coordinates) if active_coordinates is not None else list(range(best["x"].size))
    kkt = active_limit_kkt_evidence(
        best["x"],
        lo,
        hi,
        final_eval.task_residual,
        final_eval.task_jacobian,
        prior_residual=final_eval.prior_residual,
        prior_jacobian=final_eval.prior_jacobian,
        coordinate_ids=coordinate_ids,
        scalar_dtype=final_eval.scalar_dtype,
    )
    seed_results = [_seed_summary(run, lo, hi, coordinate_ids) for run in seed_runs]
    continuation_history = [dict(step) for step in best["history"]]
    subdivision = _subdivision_summary(seed_runs, continuation_history, cfg.max_task_step_norm)
    return ContinuationSolveResult(
        x=np.asarray(best["x"], dtype=float).copy(),
        success=bool(best["success"]),
        message=str(best["message"]),
        nfev=int(best["nfev"]),
        njev=int(best["njev"]),
        task_residual=np.asarray(final_eval.task_residual, dtype=float).copy(),
        prior_residual=np.asarray(final_eval.prior_residual, dtype=float).copy(),
        task_jacobian=np.asarray(final_eval.task_jacobian, dtype=float).copy(),
        residual_jacobian_source=str(final_eval.jacobian_source),
        residual_jacobian_scalar_dtype=str(final_eval.scalar_dtype),
        residual_parameterization=str(final_eval.residual_parameterization),
        so3_jacobian_convention=final_eval.so3_jacobian_convention,
        continuation_history=continuation_history,
        deterministic_start_count=len(clipped_seeds),
        selected_seed_index=int(best["seed_index"]),
        seed_results=seed_results,
        deterministic_adaptive_subdivision=subdivision,
        active_limit_kkt=kkt,
        task_gradient=task_gradient,
        task_gradient_inf_norm=task_gradient_inf,
        prior_gradient=prior_gradient,
        prior_gradient_inf_norm=prior_gradient_inf,
    )


def active_limit_kkt_evidence(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    task_residual: np.ndarray,
    task_jacobian: np.ndarray,
    *,
    prior_residual: np.ndarray | None = None,
    prior_jacobian: np.ndarray | None = None,
    coordinate_ids: list[int] | None = None,
    scalar_dtype: str = "float64",
) -> dict:
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    task_residual = np.asarray(task_residual, dtype=float)
    task_jacobian = np.asarray(task_jacobian, dtype=float)
    prior_residual_arr = np.asarray([] if prior_residual is None else prior_residual, dtype=float).reshape(-1)
    if prior_jacobian is None:
        prior_jacobian_arr = np.zeros((0, x.size), dtype=float)
    else:
        prior_jacobian_arr = np.asarray(prior_jacobian, dtype=float)
        if prior_jacobian_arr.size == 0:
            prior_jacobian_arr = np.zeros((0, x.size), dtype=float)
    gradient = task_jacobian.T @ task_residual if x.size else np.zeros(0)
    prior_gradient = prior_jacobian_arr.T @ prior_residual_arr if x.size and prior_jacobian_arr.size else np.zeros_like(gradient)
    tolerances = kkt_tolerances_for_scalar_dtype(scalar_dtype)
    active_tol = float(tolerances["active_bound_tolerance"])
    stationarity_tol = float(tolerances["stationarity_tolerance"])
    ids = coordinate_ids if coordinate_ids is not None else list(range(x.size))
    active_lower: list[int] = []
    active_upper: list[int] = []
    fixed: list[int] = []
    free: list[int] = []
    multipliers: list[dict[str, float | int | str]] = []
    projected_gradient: list[float] = []
    stationarity_violations: list[float] = []
    dual_violations: list[float] = []
    complementarity_violations: list[float] = []
    primal_violations: list[float] = []
    for local_idx, coordinate in enumerate(ids):
        finite_lower = bool(np.isfinite(lo[local_idx]))
        finite_upper = bool(np.isfinite(hi[local_idx]))
        lower_gap = float(x[local_idx] - lo[local_idx]) if finite_lower else float("inf")
        upper_gap = float(hi[local_idx] - x[local_idx]) if finite_upper else float("inf")
        fixed_coordinate = bool(finite_lower and finite_upper and abs(float(hi[local_idx] - lo[local_idx])) <= active_tol)
        lower_active = bool(finite_lower and x[local_idx] <= lo[local_idx] + active_tol)
        upper_active = bool(finite_upper and x[local_idx] >= hi[local_idx] - active_tol)
        g = float(gradient[local_idx])
        primal_violations.append(max(-lower_gap if finite_lower else 0.0, -upper_gap if finite_upper else 0.0, 0.0))
        if fixed_coordinate:
            fixed.append(int(coordinate))
            projected_gradient.append(0.0)
            stationarity_violations.append(0.0)
            dual_violations.append(0.0)
            complementarity_violations.append(0.0)
        elif lower_active:
            active_lower.append(int(coordinate))
            multipliers.append({"coordinate": int(coordinate), "side": "lower", "value": max(g, 0.0)})
            projected_gradient.append(min(0.0, g))
            stationarity_violations.append(max(-g, 0.0))
            dual_violations.append(max(-g, 0.0))
            complementarity_violations.append(max(g, 0.0) * abs(lower_gap))
        elif upper_active:
            active_upper.append(int(coordinate))
            multipliers.append({"coordinate": int(coordinate), "side": "upper", "value": max(-g, 0.0)})
            projected_gradient.append(max(0.0, g))
            stationarity_violations.append(max(g, 0.0))
            dual_violations.append(max(g, 0.0))
            complementarity_violations.append(max(-g, 0.0) * abs(upper_gap))
        else:
            free.append(int(coordinate))
            projected_gradient.append(g)
            stationarity_violations.append(abs(g))
            dual_violations.append(0.0)
            complementarity_violations.append(0.0)
    stationarity_inf = float(max(stationarity_violations, default=0.0))
    dual_inf = float(max(dual_violations, default=0.0))
    complementarity_inf = float(max(complementarity_violations, default=0.0))
    primal_inf = float(max(primal_violations, default=0.0))
    primal_feasible = bool(primal_inf <= active_tol)
    dual_feasible = bool(dual_inf <= stationarity_tol)
    complementarity_passed = bool(complementarity_inf <= stationarity_tol)
    satisfied = bool(
        primal_feasible
        and dual_feasible
        and complementarity_passed
        and stationarity_inf <= stationarity_tol
    )
    return {
        "rank_zero": False,
        "objective": "task_only_least_squares",
        "gradient_source": "task_only",
        "scalar_dtype": str(tolerances["scalar_dtype"]),
        "active_bound_tolerance": active_tol,
        "stationarity_tolerance": stationarity_tol,
        "primal_violation_inf_norm": primal_inf,
        "primal_feasible": primal_feasible,
        "dual_feasibility_inf_norm": dual_inf,
        "dual_feasible": dual_feasible,
        "complementarity_inf_norm": complementarity_inf,
        "complementarity_passed": complementarity_passed,
        "stationarity_inf_norm": stationarity_inf,
        "satisfied": satisfied,
        "active_lower": active_lower,
        "active_upper": active_upper,
        "fixed_coordinates": fixed,
        "free_coordinates": free,
        "multipliers": multipliers,
        "projected_gradient": projected_gradient,
        "projected_gradient_norm": stationarity_inf,
        "task_gradient": gradient.tolist(),
        "task_gradient_inf_norm": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
        "prior_gradient": prior_gradient.tolist(),
        "prior_gradient_inf_norm": float(np.max(np.abs(prior_gradient))) if prior_gradient.size else 0.0,
        "raw_evidence": {
            "q": x.tolist(),
            "lower_bounds": lo.tolist(),
            "upper_bounds": hi.tolist(),
            "task_jacobian": task_jacobian.tolist(),
            "task_residual": task_residual.tolist(),
            "prior_jacobian": prior_jacobian_arr.tolist(),
            "prior_residual": prior_residual_arr.tolist(),
        },
    }


def rank_zero_kkt_evidence(task_residual_norm: float, *, scalar_dtype: str = "float64") -> dict:
    tolerances = kkt_tolerances_for_scalar_dtype(scalar_dtype)
    return {
        "rank_zero": True,
        "objective": "task_only_least_squares",
        "gradient_source": "task_only",
        "scalar_dtype": str(tolerances["scalar_dtype"]),
        "active_bound_tolerance": float(tolerances["active_bound_tolerance"]),
        "stationarity_tolerance": float(tolerances["stationarity_tolerance"]),
        "stationarity_inf_norm": 0.0,
        "primal_violation_inf_norm": 0.0,
        "primal_feasible": True,
        "dual_feasibility_inf_norm": 0.0,
        "dual_feasible": True,
        "complementarity_inf_norm": 0.0,
        "complementarity_passed": True,
        "satisfied": True,
        "active_lower": [],
        "active_upper": [],
        "fixed_coordinates": [],
        "free_coordinates": [],
        "multipliers": [],
        "projected_gradient": [],
        "projected_gradient_norm": 0.0,
        "task_gradient": [],
        "task_gradient_inf_norm": 0.0,
        "task_residual_norm": float(task_residual_norm),
    }


def so3_left_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    k = skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * k + (k @ k) / 12.0
    half_theta = 0.5 * theta
    denom = 2.0 * theta * np.sin(theta)
    if abs(denom) < 1e-12:
        coefficient = 1.0 / (theta * theta)
    else:
        coefficient = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / denom
    del half_theta
    return np.eye(3) - 0.5 * k + coefficient * (k @ k)


def so3_log_residual_jacobian(
    current_rotation: np.ndarray,
    log_residual: np.ndarray,
    relative_rotation_jacobian: np.ndarray,
) -> np.ndarray:
    current_rotation = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    log_residual = np.asarray(log_residual, dtype=float).reshape(3)
    relative_rotation_jacobian = np.asarray(relative_rotation_jacobian, dtype=float)
    return -so3_left_jacobian_inverse(log_residual) @ current_rotation.T @ relative_rotation_jacobian


def _solve_one_seed(
    evaluate: Callable[[np.ndarray, float], ResidualJacobianEvaluation],
    seed: np.ndarray,
    seed_index: int,
    lo: np.ndarray,
    hi: np.ndarray,
    cfg: ContinuationSolverConfig,
) -> dict:
    x = np.clip(np.asarray(seed, dtype=float), lo, hi)
    alpha = 0.0
    nfev = 0
    njev = 0
    rejected_steps = 0
    history: list[dict] = []
    success = True
    messages: list[str] = []
    while alpha < 1.0 - 1e-12 and len(history) < cfg.max_accepted_steps:
        eval_cache: dict[tuple[float, tuple[int, ...], bytes], ResidualJacobianEvaluation] = {}

        def cached_eval(values: np.ndarray, trial_alpha: float) -> ResidualJacobianEvaluation:
            arr = np.asarray(values, dtype=float)
            contiguous = np.ascontiguousarray(arr)
            key = (float(trial_alpha), contiguous.shape, contiguous.tobytes())
            cached = eval_cache.get(key)
            if cached is None:
                cached = evaluate(contiguous, trial_alpha)
                eval_cache[key] = cached
            return cached

        base_eval = cached_eval(x, alpha)
        target_eval = cached_eval(x, 1.0)
        task_delta = np.asarray(target_eval.task_residual - base_eval.task_residual, dtype=float)
        task_delta_norm = float(np.linalg.norm(task_delta))
        if task_delta_norm <= cfg.max_task_step_norm or cfg.max_task_step_norm <= 0.0:
            proposed_alpha = 1.0
        else:
            proposed_alpha = alpha + (1.0 - alpha) * cfg.max_task_step_norm / task_delta_norm
        proposed_alpha = min(1.0, max(proposed_alpha, alpha + cfg.min_alpha_step))
        accepted = False
        last_res = None
        last_eval = None
        trial_alpha = proposed_alpha
        subdivision_depth = 0
        while not accepted:
            trial_eval = cached_eval(x, trial_alpha)
            task_step_norm = float(np.linalg.norm(trial_eval.task_residual - base_eval.task_residual))

            def residual_fun(values: np.ndarray) -> np.ndarray:
                return cached_eval(values, trial_alpha).residual

            def jacobian_fun(values: np.ndarray) -> np.ndarray:
                return cached_eval(values, trial_alpha).jacobian

            try:
                res = least_squares(
                    residual_fun,
                    x,
                    jac=jacobian_fun,
                    bounds=(lo, hi),
                    xtol=cfg.xtol,
                    ftol=cfg.ftol,
                    gtol=cfg.gtol,
                    max_nfev=cfg.max_nfev_per_step,
                )
                res_eval = cached_eval(res.x, trial_alpha)
                finite = bool(
                    np.all(np.isfinite(res.x))
                    and np.all(np.isfinite(res.fun))
                    and np.all(np.isfinite(res_eval.task_jacobian))
                )
                accepted = bool(res.success and finite)
                last_res = res
                last_eval = res_eval
            except Exception as exc:  # pragma: no cover - defensive against optimizer backend failures
                res = None
                res_eval = None
                finite = False
                accepted = False
                last_res = None
                last_eval = None
                messages.append(f"alpha={trial_alpha:.12g}: {type(exc).__name__}: {exc}")
            if accepted:
                break
            rejected_steps += 1
            if subdivision_depth >= cfg.max_subdivision_depth or (trial_alpha - alpha) <= cfg.min_alpha_step:
                success = False
                messages.append(f"failed at alpha={trial_alpha:.12g}")
                break
            subdivision_depth += 1
            trial_alpha = alpha + 0.5 * (trial_alpha - alpha)
        if not accepted or last_res is None or last_eval is None:
            break
        nfev += int(last_res.nfev)
        njev += int(getattr(last_res, "njev", 0) or 0)
        x = np.asarray(last_res.x, dtype=float)
        history.append(
            {
                "seed_index": int(seed_index),
                "step_index": len(history),
                "alpha_start": float(alpha),
                "alpha_end": float(trial_alpha),
                "accepted": True,
                "subdivision_depth": int(subdivision_depth),
                "task_step_norm": task_step_norm,
                "task_residual_norm": float(np.linalg.norm(last_eval.task_residual)),
                "prior_residual_norm": float(np.linalg.norm(last_eval.prior_residual)),
                "q_active": x.tolist(),
                "within_bounds": bool(np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)),
                "nfev": int(last_res.nfev),
                "njev": int(getattr(last_res, "njev", 0) or 0),
                "status": int(last_res.status),
            }
        )
        messages.append(str(last_res.message))
        alpha = float(trial_alpha)
    if len(history) >= cfg.max_accepted_steps and alpha < 1.0 - 1e-12:
        success = False
        messages.append("maximum accepted continuation steps reached")
    final_eval = evaluate(x, 1.0)
    final_polish = _final_task_polish(evaluate, x, lo, hi, cfg)
    if final_polish["accepted"]:
        x = np.asarray(final_polish["x"], dtype=float)
        polished_eval = final_polish["final_eval"]
        assert isinstance(polished_eval, ResidualJacobianEvaluation)
        final_eval = polished_eval
        nfev += int(final_polish["nfev"])
        njev += int(final_polish["njev"])
    if _final_polish_completed_exact_task(final_polish, final_eval, cfg):
        alpha_start = float(history[-1].get("alpha_end", 0.0)) if history else 0.0
        if alpha_start < 1.0 - 1e-12:
            history.append(
                {
                    "seed_index": int(seed_index),
                    "step_index": len(history),
                    "alpha_start": alpha_start,
                    "alpha_end": 1.0,
                    "accepted": True,
                    "subdivision_depth": 0,
                    "task_step_norm": float(np.linalg.norm(final_eval.task_residual)),
                    "task_residual_norm": float(np.linalg.norm(final_eval.task_residual)),
                    "prior_residual_norm": float(np.linalg.norm(final_eval.prior_residual)),
                    "q_active": x.tolist(),
                    "within_bounds": bool(np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)),
                    "nfev": int(final_polish["nfev"]),
                    "njev": int(final_polish["njev"]),
                    "status": int(final_polish["status"]),
                    "source": "final_task_polish_exact_completion",
                }
            )
        success = True
        messages.append("final task-only polish completed exact alpha=1 task")
    messages.append(str(final_polish["message"]))
    return {
        "seed_index": int(seed_index),
        "initial_x": np.asarray(seed, dtype=float).copy(),
        "x": x.copy(),
        "success": bool(success),
        "message": "; ".join(messages),
        "nfev": int(nfev),
        "njev": int(njev),
        "history": history,
        "rejected_steps": int(rejected_steps),
        "final_task_polish": _final_polish_summary(final_polish),
        "final_eval": final_eval,
        "task_cost": float(np.dot(final_eval.task_residual, final_eval.task_residual)),
        "total_cost": float(np.dot(final_eval.residual, final_eval.residual)),
    }


def _final_polish_completed_exact_task(
    final_polish: dict,
    final_eval: ResidualJacobianEvaluation,
    cfg: ContinuationSolverConfig,
) -> bool:
    threshold = cfg.final_numeric_polish_residual_threshold
    if threshold is None or not bool(final_polish.get("accepted", False)):
        return False
    kkt_after = final_polish.get("kkt_after", {})
    if not isinstance(kkt_after, dict) or not bool(kkt_after.get("satisfied", False)):
        return False
    residual_norm = float(np.linalg.norm(final_eval.task_residual))
    exact_completion_threshold = min(float(threshold), FINAL_POLISH_EXACT_COMPLETION_TOLERANCE)
    return bool(residual_norm <= exact_completion_threshold)


def _select_best_seed(seed_runs: list[dict], cfg: ContinuationSolverConfig) -> dict:
    best = seed_runs[0]
    for run in seed_runs[1:]:
        if _run_is_better(run, best, cfg):
            best = run
    return best


def _final_task_polish(
    evaluate: Callable[[np.ndarray, float], ResidualJacobianEvaluation],
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    cfg: ContinuationSolverConfig,
) -> dict:
    eval_cache: dict[tuple[tuple[int, ...], bytes], ResidualJacobianEvaluation] = {}

    def cached_eval(values: np.ndarray) -> ResidualJacobianEvaluation:
        arr = np.asarray(values, dtype=float)
        contiguous = np.ascontiguousarray(arr)
        key = (contiguous.shape, contiguous.tobytes())
        cached = eval_cache.get(key)
        if cached is None:
            cached = evaluate(contiguous, 1.0)
            eval_cache[key] = cached
        return cached

    x = np.clip(np.asarray(x, dtype=float), lo, hi)
    base_eval = cached_eval(x)
    base_task_cost = _task_cost(base_eval)
    before_kkt = active_limit_kkt_evidence(
        x,
        lo,
        hi,
        base_eval.task_residual,
        base_eval.task_jacobian,
        scalar_dtype=base_eval.scalar_dtype,
    )
    if before_kkt["satisfied"]:
        return {
            "accepted": False,
            "skipped": True,
            "x": x.copy(),
            "x_delta_norm": 0.0,
            "final_eval": base_eval,
            "nfev": 0,
            "njev": 0,
            "status": 0,
            "success": True,
            "message": "final task-only polish skipped: task-only KKT already satisfied",
            "task_cost_before": base_task_cost,
            "task_cost_after": base_task_cost,
            "kkt_before": _compact_kkt_summary(before_kkt, x, lo, hi),
            "kkt_after": _compact_kkt_summary(before_kkt, x, lo, hi),
            "bound_snap": _empty_bound_snap_summary(),
            "numeric_fallback": _empty_numeric_fallback_summary(),
        }

    def residual_fun(values: np.ndarray) -> np.ndarray:
        return cached_eval(values).task_residual

    def jacobian_fun(values: np.ndarray) -> np.ndarray:
        return cached_eval(values).task_jacobian

    try:
        res = least_squares(
            residual_fun,
            x,
            jac=jacobian_fun,
            bounds=(lo, hi),
            xtol=cfg.xtol,
            ftol=cfg.ftol,
            gtol=cfg.gtol,
            max_nfev=cfg.max_nfev_per_step,
        )
        final_eval = cached_eval(res.x)
        finite = bool(
            np.all(np.isfinite(res.x))
            and np.all(np.isfinite(res.fun))
            and np.all(np.isfinite(final_eval.task_jacobian))
        )
        final_task_cost = _task_cost(final_eval)
        after_kkt = active_limit_kkt_evidence(
            res.x,
            lo,
            hi,
            final_eval.task_residual,
            final_eval.task_jacobian,
            scalar_dtype=final_eval.scalar_dtype,
        )
        ls_accepted = bool(
            finite
            and final_task_cost <= base_task_cost + cfg.cost_tie_tolerance
            and float(after_kkt["stationarity_inf_norm"])
            <= float(before_kkt["stationarity_inf_norm"]) + cfg.cost_tie_tolerance
        )
        use_ls_start = bool(finite and final_task_cost <= base_task_cost + cfg.cost_tie_tolerance)
        snap_start_x = np.asarray(res.x, dtype=float) if use_ls_start else x
        snap_start_eval = final_eval if use_ls_start else base_eval
        snap_start_kkt = after_kkt if use_ls_start else before_kkt
        snap = _bound_snap_candidate(
            cached_eval,
            snap_start_x,
            snap_start_eval,
            lo,
            hi,
            base_task_cost=base_task_cost,
            before_kkt=before_kkt,
            start_kkt=snap_start_kkt,
            tie_tolerance=cfg.cost_tie_tolerance,
        )
        if snap["accepted"]:
            final_x = np.asarray(snap["x"], dtype=float)
            final_eval = snap["final_eval"]
            assert isinstance(final_eval, ResidualJacobianEvaluation)
            final_task_cost = float(snap["task_cost_after"])
            after_kkt = snap["kkt_after"]
            accepted = True
        else:
            final_x = np.asarray(res.x, dtype=float)
            accepted = ls_accepted
        numeric_start_x = final_x if accepted or (finite and final_task_cost <= base_task_cost + cfg.cost_tie_tolerance) else x
        numeric_start_eval = final_eval if numeric_start_x is final_x else base_eval
        numeric_start_kkt = after_kkt if numeric_start_x is final_x else before_kkt
        numeric = _numeric_task_polish_candidate(
            cached_eval,
            numeric_start_x,
            numeric_start_eval,
            lo,
            hi,
            base_task_cost=base_task_cost,
            before_kkt=before_kkt,
            start_kkt=numeric_start_kkt,
            cfg=cfg,
        )
        if numeric["accepted"]:
            final_x = np.asarray(numeric["x"], dtype=float)
            final_eval = numeric["final_eval"]
            assert isinstance(final_eval, ResidualJacobianEvaluation)
            final_task_cost = float(numeric["task_cost_after"])
            after_kkt = numeric["kkt_after"]
            accepted = True
        message = f"final task-only polish: {res.message}"
        return {
            "accepted": accepted,
            "skipped": False,
            "x": final_x.copy(),
            "x_delta_norm": float(np.linalg.norm(final_x - x)),
            "final_eval": final_eval,
            "nfev": int(res.nfev),
            "njev": int(getattr(res, "njev", 0) or 0),
            "status": int(res.status),
            "success": bool(res.success),
            "message": message,
            "task_cost_before": base_task_cost,
            "task_cost_after": final_task_cost,
            "kkt_before": _compact_kkt_summary(before_kkt, x, lo, hi),
            "kkt_after": _compact_kkt_summary(after_kkt, final_x, lo, hi),
            "bound_snap": _bound_snap_summary(snap),
            "numeric_fallback": _numeric_fallback_summary(numeric),
        }
    except Exception as exc:  # pragma: no cover - defensive against optimizer backend failures
        return {
            "accepted": False,
            "skipped": False,
            "x": x.copy(),
            "x_delta_norm": 0.0,
            "final_eval": base_eval,
            "nfev": 0,
            "njev": 0,
            "status": 0,
            "success": False,
            "message": f"final task-only polish: {type(exc).__name__}: {exc}",
            "task_cost_before": base_task_cost,
            "task_cost_after": base_task_cost,
            "kkt_before": _compact_kkt_summary(before_kkt, x, lo, hi),
            "kkt_after": _compact_kkt_summary(before_kkt, x, lo, hi),
            "bound_snap": _empty_bound_snap_summary(),
            "numeric_fallback": _empty_numeric_fallback_summary(),
        }


def _final_polish_summary(polish: dict) -> dict:
    return {
        "objective": "task_only_least_squares",
        "accepted": bool(polish["accepted"]),
        "skipped": bool(polish.get("skipped", False)),
        "success": bool(polish["success"]),
        "status": int(polish["status"]),
        "nfev": int(polish["nfev"]),
        "njev": int(polish["njev"]),
        "x_delta_norm": float(polish.get("x_delta_norm", 0.0)),
        "task_cost_before": float(polish["task_cost_before"]),
        "task_cost_after": float(polish["task_cost_after"]),
        "kkt_before": dict(polish["kkt_before"]),
        "kkt_after": dict(polish["kkt_after"]),
        "bound_snap": dict(polish["bound_snap"]),
        "numeric_fallback": dict(polish["numeric_fallback"]),
        "message": str(polish["message"]),
    }


def _bound_snap_candidate(
    evaluate: Callable[[np.ndarray], ResidualJacobianEvaluation],
    x: np.ndarray,
    evaluation: ResidualJacobianEvaluation,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    base_task_cost: float,
    before_kkt: dict,
    start_kkt: dict,
    tie_tolerance: float,
) -> dict:
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    gradient = np.asarray(evaluation.task_jacobian, dtype=float).T @ np.asarray(evaluation.task_residual, dtype=float)
    active_tol = float(start_kkt["active_bound_tolerance"])
    stationarity_tol = float(start_kkt["stationarity_tolerance"])
    search_tol = BOUND_SNAP_SEARCH_MULTIPLIER * active_tol
    snapped_x = x.copy()
    snapped: list[dict[str, float | int | str]] = []
    for index, g in enumerate(gradient):
        lower_gap = float(x[index] - lo[index]) if np.isfinite(lo[index]) else float("inf")
        upper_gap = float(hi[index] - x[index]) if np.isfinite(hi[index]) else float("inf")
        if active_tol < lower_gap <= search_tol and float(g) > stationarity_tol:
            snapped_x[index] = lo[index]
            snapped.append(
                {
                    "coordinate": int(index),
                    "side": "lower",
                    "gap": lower_gap,
                    "gradient": float(g),
                }
            )
        elif active_tol < upper_gap <= search_tol and float(g) < -stationarity_tol:
            snapped_x[index] = hi[index]
            snapped.append(
                {
                    "coordinate": int(index),
                    "side": "upper",
                    "gap": upper_gap,
                    "gradient": float(g),
                }
            )
    if not snapped:
        return {
            "accepted": False,
            "attempted": False,
            "snapped_coordinates": [],
            "x": x.copy(),
            "final_eval": evaluation,
            "task_cost_after": _task_cost(evaluation),
            "kkt_after": start_kkt,
            "message": "no near-bound descent coordinates",
        }
    snapped_eval = evaluate(snapped_x)
    snapped_cost = _task_cost(snapped_eval)
    snapped_kkt = active_limit_kkt_evidence(
        snapped_x,
        lo,
        hi,
        snapped_eval.task_residual,
        snapped_eval.task_jacobian,
        scalar_dtype=snapped_eval.scalar_dtype,
    )
    finite = bool(
        np.all(np.isfinite(snapped_x))
        and np.all(np.isfinite(snapped_eval.task_residual))
        and np.all(np.isfinite(snapped_eval.task_jacobian))
    )
    accepted = bool(
        finite
        and snapped_cost <= base_task_cost + tie_tolerance
        and float(snapped_kkt["stationarity_inf_norm"])
        <= float(before_kkt["stationarity_inf_norm"]) + tie_tolerance
    )
    return {
        "accepted": accepted,
        "attempted": True,
        "snapped_coordinates": snapped,
        "x": snapped_x.copy(),
        "final_eval": snapped_eval,
        "task_cost_after": snapped_cost,
        "kkt_after": snapped_kkt,
        "message": "accepted near-bound descent snap" if accepted else "rejected near-bound descent snap",
    }


def _numeric_task_polish_candidate(
    evaluate: Callable[[np.ndarray], ResidualJacobianEvaluation],
    x: np.ndarray,
    evaluation: ResidualJacobianEvaluation,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    base_task_cost: float,
    before_kkt: dict,
    start_kkt: dict,
    cfg: ContinuationSolverConfig,
) -> dict:
    threshold = cfg.final_numeric_polish_residual_threshold
    if threshold is None:
        return _numeric_fallback_result(
            accepted=False,
            attempted=False,
            x=x,
            evaluation=evaluation,
            task_cost=_task_cost(evaluation),
            kkt=start_kkt,
            message="not configured",
        )
    residual_norm = float(np.linalg.norm(evaluation.task_residual))
    if start_kkt["satisfied"]:
        return _numeric_fallback_result(
            accepted=False,
            attempted=False,
            x=x,
            evaluation=evaluation,
            task_cost=_task_cost(evaluation),
            kkt=start_kkt,
            message="task-only KKT already satisfied",
        )
    if residual_norm > float(threshold):
        return _numeric_fallback_result(
            accepted=False,
            attempted=False,
            x=x,
            evaluation=evaluation,
            task_cost=_task_cost(evaluation),
            kkt=start_kkt,
            message="residual exceeds exact-threshold fallback gate",
        )

    def residual_fun(values: np.ndarray) -> np.ndarray:
        return evaluate(values).task_residual

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\\.optimize\\._lsq\\..*")
            res = least_squares(
                residual_fun,
                np.asarray(x, dtype=float),
                jac="2-point",
                bounds=(lo, hi),
                method="trf",
                xtol=None,
                ftol=None,
                gtol=cfg.gtol,
                max_nfev=cfg.final_numeric_polish_max_nfev,
            )
        final_x = np.asarray(res.x, dtype=float)
        final_eval = evaluate(final_x)
        final_task_cost = _task_cost(final_eval)
        final_kkt = active_limit_kkt_evidence(
            final_x,
            lo,
            hi,
            final_eval.task_residual,
            final_eval.task_jacobian,
            scalar_dtype=final_eval.scalar_dtype,
        )
        finite = bool(
            np.all(np.isfinite(final_x))
            and np.all(np.isfinite(final_eval.task_residual))
            and np.all(np.isfinite(final_eval.task_jacobian))
        )
        accepted = bool(
            finite
            and final_task_cost <= base_task_cost + cfg.cost_tie_tolerance
            and float(final_kkt["stationarity_inf_norm"])
            <= float(before_kkt["stationarity_inf_norm"]) + cfg.cost_tie_tolerance
            and final_kkt["satisfied"]
        )
        return {
            "accepted": accepted,
            "attempted": True,
            "x": final_x.copy(),
            "final_eval": final_eval,
            "task_cost_after": final_task_cost,
            "kkt_after": final_kkt,
            "nfev": int(res.nfev),
            "njev": int(getattr(res, "njev", 0) or 0),
            "status": int(res.status),
            "success": bool(res.success),
            "message": str(res.message),
            "residual_threshold": float(threshold),
            "residual_norm_before": residual_norm,
        }
    except Exception as exc:  # pragma: no cover - defensive against optimizer backend failures
        return _numeric_fallback_result(
            accepted=False,
            attempted=True,
            x=x,
            evaluation=evaluation,
            task_cost=_task_cost(evaluation),
            kkt=start_kkt,
            message=f"{type(exc).__name__}: {exc}",
            residual_threshold=float(threshold),
            residual_norm_before=residual_norm,
        )


def _numeric_fallback_result(
    *,
    accepted: bool,
    attempted: bool,
    x: np.ndarray,
    evaluation: ResidualJacobianEvaluation,
    task_cost: float,
    kkt: dict,
    message: str,
    residual_threshold: float | None = None,
    residual_norm_before: float | None = None,
) -> dict:
    return {
        "accepted": bool(accepted),
        "attempted": bool(attempted),
        "x": np.asarray(x, dtype=float).copy(),
        "final_eval": evaluation,
        "task_cost_after": float(task_cost),
        "kkt_after": kkt,
        "nfev": 0,
        "njev": 0,
        "status": 0,
        "success": bool(not attempted),
        "message": message,
        "residual_threshold": residual_threshold,
        "residual_norm_before": residual_norm_before,
    }


def _numeric_fallback_summary(numeric: dict) -> dict:
    return {
        "attempted": bool(numeric["attempted"]),
        "accepted": bool(numeric["accepted"]),
        "residual_threshold": numeric["residual_threshold"],
        "residual_norm_before": numeric["residual_norm_before"],
        "nfev": int(numeric["nfev"]),
        "njev": int(numeric["njev"]),
        "status": int(numeric["status"]),
        "success": bool(numeric["success"]),
        "task_cost_after": float(numeric["task_cost_after"]),
        "stationarity_after": float(numeric["kkt_after"]["stationarity_inf_norm"]),
        "kkt_satisfied_after": bool(numeric["kkt_after"]["satisfied"]),
        "message": str(numeric["message"]),
    }


def _empty_numeric_fallback_summary() -> dict:
    return {
        "attempted": False,
        "accepted": False,
        "residual_threshold": None,
        "residual_norm_before": None,
        "nfev": 0,
        "njev": 0,
        "status": 0,
        "success": True,
        "task_cost_after": None,
        "stationarity_after": None,
        "kkt_satisfied_after": None,
        "message": "not attempted",
    }


def _bound_snap_summary(snap: dict) -> dict:
    return {
        "attempted": bool(snap["attempted"]),
        "accepted": bool(snap["accepted"]),
        "search_multiplier": float(BOUND_SNAP_SEARCH_MULTIPLIER),
        "snapped_coordinates": list(snap["snapped_coordinates"]),
        "task_cost_after": float(snap["task_cost_after"]),
        "stationarity_after": float(snap["kkt_after"]["stationarity_inf_norm"]),
        "message": str(snap["message"]),
    }


def _empty_bound_snap_summary() -> dict:
    return {
        "attempted": False,
        "accepted": False,
        "search_multiplier": float(BOUND_SNAP_SEARCH_MULTIPLIER),
        "snapped_coordinates": [],
        "task_cost_after": None,
        "stationarity_after": None,
        "message": "not attempted",
    }


def _compact_kkt_summary(kkt: dict, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    return {
        "satisfied": bool(kkt["satisfied"]),
        "stationarity_inf_norm": float(kkt["stationarity_inf_norm"]),
        "stationarity_tolerance": float(kkt["stationarity_tolerance"]),
        "active_bound_tolerance": float(kkt["active_bound_tolerance"]),
        "primal_feasible": bool(kkt["primal_feasible"]),
        "dual_feasible": bool(kkt["dual_feasible"]),
        "complementarity_passed": bool(kkt["complementarity_passed"]),
        "active_lower": list(kkt["active_lower"]),
        "active_upper": list(kkt["active_upper"]),
        "free_coordinates": list(kkt["free_coordinates"]),
        "minimum_lower_gap": _minimum_bound_gap(x, lo, lower=True),
        "minimum_upper_gap": _minimum_bound_gap(x, hi, lower=False),
    }


def _minimum_bound_gap(x: np.ndarray, bound: np.ndarray, *, lower: bool) -> float | None:
    x = np.asarray(x, dtype=float)
    bound = np.asarray(bound, dtype=float)
    finite = np.isfinite(bound)
    if not np.any(finite):
        return None
    gaps = x[finite] - bound[finite] if lower else bound[finite] - x[finite]
    return float(np.min(gaps))


def _task_cost(evaluation: ResidualJacobianEvaluation) -> float:
    task_residual = np.asarray(evaluation.task_residual, dtype=float)
    return float(np.dot(task_residual, task_residual))


def _run_is_better(candidate: dict, incumbent: dict, cfg: ContinuationSolverConfig) -> bool:
    cand_task = float(candidate["task_cost"])
    inc_task = float(incumbent["task_cost"])
    if _success_tie_should_choose_candidate(candidate, incumbent, cfg):
        return True
    if _success_tie_should_choose_candidate(incumbent, candidate, cfg):
        return False
    tie_tolerance = cfg.cost_tie_tolerance
    if cand_task < inc_task - tie_tolerance:
        return True
    if cand_task > inc_task + tie_tolerance:
        return False
    cand_total = float(candidate["total_cost"])
    inc_total = float(incumbent["total_cost"])
    if cand_total < inc_total - tie_tolerance:
        return True
    if cand_total > inc_total + tie_tolerance:
        return False
    return int(candidate["seed_index"]) < int(incumbent["seed_index"])


def _success_tie_should_choose_candidate(
    candidate: dict,
    incumbent: dict,
    cfg: ContinuationSolverConfig,
) -> bool:
    threshold = cfg.final_numeric_polish_residual_threshold
    if threshold is None or bool(candidate["success"]) == bool(incumbent["success"]):
        return False
    if not bool(candidate["success"]):
        return False
    cand_residual = float(np.sqrt(max(float(candidate["task_cost"]), 0.0)))
    inc_residual = float(np.sqrt(max(float(incumbent["task_cost"]), 0.0)))
    if cand_residual > float(threshold) or inc_residual > float(threshold):
        return False
    return abs(cand_residual - inc_residual) <= cfg.successful_seed_tie_residual_tolerance


def _seed_summary(run: dict, lo: np.ndarray, hi: np.ndarray, coordinate_ids: list[int]) -> dict:
    final_eval = run["final_eval"]
    assert isinstance(final_eval, ResidualJacobianEvaluation)
    task_residual_norm = float(np.linalg.norm(final_eval.task_residual))
    return {
        "seed_index": int(run["seed_index"]),
        "accepted": bool(run["success"]),
        "initial_x": np.asarray(run["initial_x"], dtype=float).tolist(),
        "final_x": np.asarray(run["x"], dtype=float).tolist(),
        "final_q_active": np.asarray(run["x"], dtype=float).tolist(),
        "task_residual_norm": task_residual_norm,
        "normalized_residual": task_residual_norm,
        "prior_residual_norm": float(np.linalg.norm(final_eval.prior_residual)),
        "total_residual_norm": float(np.linalg.norm(final_eval.residual)),
        "active_limits": _seed_active_limit_summary(run["x"], lo, hi, coordinate_ids, final_eval.scalar_dtype),
        "accepted_step_count": len(run["history"]),
        "rejected_step_count": int(run["rejected_steps"]),
        "final_task_polish": dict(run.get("final_task_polish", {})),
        "nfev": int(run["nfev"]),
        "njev": int(run["njev"]),
    }


def _seed_active_limit_summary(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    coordinate_ids: list[int],
    scalar_dtype: str,
) -> dict:
    tolerances = kkt_tolerances_for_scalar_dtype(scalar_dtype)
    active_tol = float(tolerances["active_bound_tolerance"])
    x = np.asarray(x, dtype=float)
    lower: list[int] = []
    upper: list[int] = []
    fixed: list[int] = []
    for local_idx, coordinate in enumerate(coordinate_ids):
        finite_lower = bool(np.isfinite(lo[local_idx]))
        finite_upper = bool(np.isfinite(hi[local_idx]))
        is_fixed = bool(finite_lower and finite_upper and abs(float(hi[local_idx] - lo[local_idx])) <= active_tol)
        if is_fixed:
            fixed.append(int(coordinate))
            continue
        if finite_lower and x[local_idx] <= lo[local_idx] + active_tol:
            lower.append(int(coordinate))
        if finite_upper and x[local_idx] >= hi[local_idx] - active_tol:
            upper.append(int(coordinate))
    return {
        "lower": lower,
        "upper": upper,
        "fixed": fixed,
        "count": len(lower) + len(upper) + len(fixed),
        "boundary_tolerance": active_tol,
    }


def _subdivision_summary(seed_runs: list[dict], selected_history: list[dict], configured_step_norm: float) -> dict:
    all_steps = [step for run in seed_runs for step in run["history"]]
    return {
        "strategy": "deterministic_residual_space_adaptive_subdivision",
        "configured_max_task_step_norm": float(configured_step_norm),
        "max_task_step_norm": float(max((step["task_step_norm"] for step in selected_history), default=0.0)),
        "accepted_step_count": len(selected_history),
        "all_seed_accepted_step_count": len(all_steps),
        "all_seed_rejected_step_count": int(sum(int(run["rejected_steps"]) for run in seed_runs)),
        "max_subdivision_depth": int(max((step["subdivision_depth"] for step in all_steps), default=0)),
    }


@dataclass(frozen=True)
class ContinuationStep:
    alpha_start: float
    alpha_end: float
    accepted: bool
    task_residual_norm: float
    normalized_residual: float
    iterations: int
    max_task_step_norm: float

    def to_json(self) -> dict:
        return {
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "accepted": self.accepted,
            "task_residual_norm": self.task_residual_norm,
            "normalized_residual": self.normalized_residual,
            "iterations": self.iterations,
            "max_task_step_norm": self.max_task_step_norm,
        }


@dataclass(frozen=True)
class SeedConsensusReport:
    seed_results: list[dict]
    selected_seed_index: int
    spread: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ProjectionSolveResult:
    x: np.ndarray
    success: bool
    iterations: int
    message: str
    task_residual_norm: float
    total_cost: float
    deterministic_start_count: int
    seed_consensus: SeedConsensusReport


def solve_bounded_multistart(
    residual: Callable[[np.ndarray], np.ndarray],
    task_residual: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    neutral_values: np.ndarray | None = None,
    previous_values: np.ndarray | None = None,
    max_nfev: int = 240,
    consensus_tolerance: float = 1e-3,
) -> ProjectionSolveResult:
    starts = deterministic_seed_candidates(
        x0,
        lower,
        upper,
        neutral_values=neutral_values,
        previous_values=previous_values,
    )
    rows: list[dict] = []
    best = None
    best_key: tuple[float, int] | None = None
    best_task_norm = float("inf")
    for seed_index, start in enumerate(starts):
        res = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=max_nfev,
        )
        task_vec = np.asarray(task_residual(res.x), dtype=float)
        task_norm = float(np.linalg.norm(task_vec))
        total_norm = float(np.linalg.norm(res.fun))
        rows.append(
            {
                "seed_index": int(seed_index),
                "success": bool(res.success),
                "task_residual_norm": task_norm,
                "total_residual_norm": total_norm,
                "iterations": int(res.nfev),
                "x": np.asarray(res.x, dtype=float).tolist(),
                "message": str(res.message),
            }
        )
        key_task = 0.0 if task_norm < 1e-8 else task_norm
        key = (key_task, int(seed_index))
        if best is None or best_key is None or key < best_key:
            best = res
            best_key = key
            best_task_norm = task_norm
    assert best is not None and best_key is not None
    selected = int(best_key[1])
    competitive = [row for row in rows if row["success"] and row["task_residual_norm"] <= best_task_norm + consensus_tolerance]
    if len(competitive) >= 2:
        xs = np.asarray([row["x"] for row in competitive], dtype=float)
        spread = float(np.max(np.linalg.norm(xs - xs[0], axis=1)))
    else:
        spread = 0.0
    return ProjectionSolveResult(
        x=np.asarray(best.x, dtype=float),
        success=bool(best.success),
        iterations=int(best.nfev),
        message=str(best.message),
        task_residual_norm=float(best_task_norm),
        total_cost=float(np.dot(best.fun, best.fun)),
        deterministic_start_count=len(starts),
        seed_consensus=SeedConsensusReport(
            seed_results=rows,
            selected_seed_index=selected,
            spread=spread,
            tolerance=consensus_tolerance,
            passed=bool(spread <= consensus_tolerance),
        ),
    )


def projected_gradient_report(
    jacobian: np.ndarray,
    residual: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float | None = None,
) -> dict:
    report = active_limit_kkt_evidence(
        x,
        lower,
        upper,
        residual,
        jacobian,
        coordinate_ids=list(range(np.asarray(x).size)),
    )
    if tolerance is not None:
        report["stationarity_tolerance"] = float(tolerance)
        report["satisfied"] = bool(float(report["stationarity_inf_norm"]) <= float(tolerance))
    report["projected_gradient_norm"] = float(report["stationarity_inf_norm"])
    report["complementarity_passed"] = bool(report["satisfied"])
    report["task_gradient_inf_norm"] = float(np.max(np.abs(report.get("task_gradient", []) or [0.0])))
    return report
