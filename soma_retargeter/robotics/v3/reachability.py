"""Deterministic multi-pose local-rank summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine_jacobian import engine_relative_jacobian
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .numerical_jacobian import matrix_rank_and_singular_values, numerical_relative_jacobian
from .reachability_metrics import (
    percentile,
    principal_angles,
    projector_distance,
    rank_with_uncertainty,
    retained_left_singular_vectors,
)


@dataclass(frozen=True)
class ReachabilityReport:
    regular_rank_translation: int
    nominal_rank_translation: int
    regular_rank_rotation: int
    nominal_rank_rotation: int
    singularity_fraction_translation: float
    singularity_fraction_rotation: float
    epsilon_unstable_fraction: float
    epsilon_unstable_sample_fraction: float
    epsilon_stability_gate_passed: bool
    epsilon_unstable_columns: list[int]
    conditioning_percentiles_translation: dict[str, float | None]
    conditioning_percentiles_rotation: dict[str, float | None]
    samples: int
    selected_singular_values_translation: list[list[float]]
    selected_singular_values_rotation: list[list[float]]
    sample_diagnostics: list[dict]
    numerical_stability_gate_passed: bool = True
    stable_sample_fraction: float = 1.0
    task_block: str = "translation+rotation"
    engine_rank_translation: int = 0
    engine_rank_rotation: int = 0
    fd_rank_translation: int = 0
    fd_rank_rotation: int = 0
    rank_agreement_rate_translation: float = 1.0
    rank_agreement_rate_rotation: float = 1.0
    relevant_rank_agreement_rate: float = 1.0
    projector_distance_p95: float | None = None
    principal_angle_p95: float | None = None
    engine_fd_normalized_error_p95: float | None = None
    rank_method: str = "per-sample-local-rank-regular-fraction"
    regular_rank_fraction_threshold: float = 0.20
    stable_sample_fraction_threshold: float = 0.95

    def to_json(self) -> dict:
        return self.__dict__.copy()


def deterministic_chain_samples(
    adapter: MuJoCoRuntimeModelAdapter,
    active_coordinates: list[int],
    *,
    low_discrepancy_count: int = 32,
) -> list[np.ndarray]:
    neutral = adapter.neutral_q()
    samples = [neutral]
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        lower, upper, center = _sample_interval(adapter, neutral, dof)
        span = upper - lower
        if span <= 0.0:
            continue
        for sign in (-1.0, 1.0):
            value = np.clip(center + sign * 0.1 * span, lower, upper)
            samples.append(adapter.set_velocity_coordinates(neutral, [dof], np.asarray([value])))
    if active_coordinates:
        halton = _halton(low_discrepancy_count, len(active_coordinates))
        for row in halton:
            values = []
            for u, dof in zip(row, active_coordinates):
                lower, upper, _ = _sample_interval(adapter, neutral, dof)
                lo = lower + 0.2 * (upper - lower)
                hi = upper - 0.2 * (upper - lower)
                values.append(lo + u * (hi - lo))
            samples.append(adapter.set_velocity_coordinates(neutral, active_coordinates, np.asarray(values)))
    return samples


def analyze_reachability(
    adapter: MuJoCoRuntimeModelAdapter,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    *,
    low_discrepancy_count: int = 32,
    raise_on_unstable: bool = False,
    task_name: str | None = None,
) -> ReachabilityReport:
    ranks_p: list[int] = []
    ranks_r: list[int] = []
    cond_p: list[float] = []
    cond_r: list[float] = []
    unstable = 0
    unstable_samples = 0
    unstable_columns_seen: set[int] = set()
    sv_p_selected: list[list[float]] = []
    sv_r_selected: list[list[float]] = []
    sample_diagnostics: list[dict] = []
    fd_ranks_p: list[int] = []
    fd_ranks_r: list[int] = []
    rank_agree_p = 0
    rank_agree_r = 0
    relevant_rank_agree = 0
    engine_available_count = 0
    projector_distances: list[float] = []
    principal_angle_maxima: list[float] = []
    normalized_errors: list[float] = []
    samples = deterministic_chain_samples(adapter, active_coordinates, low_discrepancy_count=low_discrepancy_count)
    task_block = _task_block(task_name, reference, target)
    for sample_idx, q in enumerate(samples):
        jac = numerical_relative_jacobian(
            adapter,
            q,
            reference,
            target,
            active_coordinates,
            raise_on_unstable=raise_on_unstable,
            engine_validation=False,
        )
        rp, sp = matrix_rank_and_singular_values(jac.translation)
        rr, sr = matrix_rank_and_singular_values(jac.rotation)
        cp = _conditioning(sp, rp)
        cr = _conditioning(sr, rr)
        ranks_p.append(rp)
        ranks_r.append(rr)
        if cp is not None:
            cond_p.append(cp)
        if cr is not None:
            cond_r.append(cr)
        unstable += len(jac.unstable_columns)
        relevant_stable = _relevant_block_stable(jac, task_block)
        if not relevant_stable:
            unstable_samples += 1
            unstable_columns_seen.update(jac.unstable_columns)
        sample_diagnostics.append(
            {
                "sample_index": sample_idx,
                "translation": {
                    "local_rank": rp,
                    "singular_values": sp.tolist(),
                    "conditioning": cp,
                },
                "rotation": {
                    "local_rank": rr,
                    "singular_values": sr.tolist(),
                    "conditioning": cr,
                },
                "unstable_columns": list(jac.unstable_columns),
                "epsilon_discrepancies": jac.epsilon_discrepancies,
                "epsilon_stability_gate_passed": jac.stability_gate_passed,
                "numerical_stability_gate_passed": relevant_stable,
                "task_block": task_block,
                "column_classifications": jac.column_classifications or [],
                "jacobian_noise_norm": jac.jacobian_noise_norm,
            }
        )
        try:
            engine = engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
            engine_available = True
            engine_available_count += 1
            engine_note = None
            engine_translation = engine.translation
            engine_rotation = engine.rotation
        except Exception as exc:
            engine_available = False
            engine_note = f"{type(exc).__name__}: {exc}"
            engine_translation = jac.translation
            engine_rotation = jac.rotation

        rp_engine, sp_engine, tau_p = rank_with_uncertainty(
            engine_translation,
            noise_norm=jac.jacobian_noise_norm,
            abs_floor=_backend_abs_floor(adapter),
        )
        rr_engine, sr_engine, tau_r = rank_with_uncertainty(
            engine_rotation,
            noise_norm=jac.jacobian_noise_norm,
            abs_floor=_backend_abs_floor(adapter),
        )
        rp_fd, sp_fd, tau_p_fd = rank_with_uncertainty(
            jac.translation,
            noise_norm=jac.jacobian_noise_norm,
            abs_floor=_backend_abs_floor(adapter),
        )
        rr_fd, sr_fd, tau_r_fd = rank_with_uncertainty(
            jac.rotation,
            noise_norm=jac.jacobian_noise_norm,
            abs_floor=_backend_abs_floor(adapter),
        )
        ranks_p[-1] = rp_engine
        ranks_r[-1] = rr_engine
        fd_ranks_p.append(rp_fd)
        fd_ranks_r.append(rr_fd)
        rank_agree_p += int(rp_engine == rp_fd)
        rank_agree_r += int(rr_engine == rr_fd)
        relevant_engine = _select_block(engine_translation, engine_rotation, task_block)
        relevant_fd = _select_block(jac.translation, jac.rotation, task_block)
        relevant_rank_engine, _, tau_relevant = rank_with_uncertainty(
            relevant_engine,
            noise_norm=jac.jacobian_noise_norm,
            abs_floor=_backend_abs_floor(adapter),
        )
        relevant_rank_fd, _, _ = rank_with_uncertainty(
            relevant_fd,
            noise_norm=jac.jacobian_noise_norm,
            abs_floor=_backend_abs_floor(adapter),
        )
        relevant_rank_agree += int(relevant_rank_engine == relevant_rank_fd)
        u_engine = retained_left_singular_vectors(relevant_engine, relevant_rank_engine)
        u_fd = retained_left_singular_vectors(relevant_fd, relevant_rank_fd)
        distance = projector_distance(u_engine, u_fd)
        if distance is not None:
            projector_distances.append(distance)
        angles = principal_angles(u_engine, u_fd)
        if angles:
            principal_angle_maxima.append(float(max(angles)))
        norm_error = float(np.linalg.norm(relevant_engine - relevant_fd) / max(np.linalg.norm(relevant_engine), jac.jacobian_noise_norm, 1e-12))
        normalized_errors.append(norm_error)
        sample_diagnostics[-1]["engine_jacobian"] = {
            "available": engine_available,
            "note": engine_note,
            "finite": bool(np.all(np.isfinite(engine_translation)) and np.all(np.isfinite(engine_rotation))),
            "rank_translation": rp_engine,
            "rank_rotation": rr_engine,
            "rank_threshold_translation": tau_p,
            "rank_threshold_rotation": tau_r,
            "source": getattr(engine, "source", "finite_difference_fallback") if engine_available else "unavailable",
            "backend": getattr(engine, "backend", adapter.__class__.__name__) if engine_available else adapter.__class__.__name__,
        }
        sample_diagnostics[-1]["fd_rank"] = {
            "rank_translation": rp_fd,
            "rank_rotation": rr_fd,
            "rank_threshold_translation": tau_p_fd,
            "rank_threshold_rotation": tau_r_fd,
        }
        sample_diagnostics[-1]["subspace"] = {
            "task_block": task_block,
            "relevant_rank_engine": relevant_rank_engine,
            "relevant_rank_fd": relevant_rank_fd,
            "rank_agreement": relevant_rank_engine == relevant_rank_fd,
            "rank_threshold": tau_relevant,
            "projector_distance": distance,
            "principal_angles": angles,
            "engine_fd_normalized_error": norm_error,
        }
        if sample_idx < 3:
            sv_p_selected[-1:] = [sp_engine.tolist()]
            sv_r_selected[-1:] = [sr_engine.tolist()]
    regular_p = _regular_rank(ranks_p)
    regular_r = _regular_rank(ranks_r)
    stable_sample_fraction = 1.0 - float(unstable_samples / max(1, len(samples)))
    relevant_rank_agreement_rate = float(relevant_rank_agree / max(1, len(samples)))
    projector_distance_p95 = percentile(projector_distances, 95)
    principal_angle_p95 = percentile(principal_angle_maxima, 95)
    engine_fd_normalized_error_p95 = percentile(normalized_errors, 95)
    engine_available_rate = float(engine_available_count / max(1, len(samples)))
    subspace_gate_passed = (
        relevant_rank_agreement_rate >= 0.95
        and (projector_distance_p95 is None or projector_distance_p95 <= 0.05)
        and (principal_angle_p95 is None or principal_angle_p95 <= 1e-5)
    )
    engine_fd_error_gate_passed = (
        engine_fd_normalized_error_p95 is None
        or engine_fd_normalized_error_p95 <= 0.02
        or subspace_gate_passed
    )
    numerical_gate_passed = (
        stable_sample_fraction >= 0.95
        and engine_available_rate >= 1.0
        and subspace_gate_passed
        and engine_fd_error_gate_passed
    )
    return ReachabilityReport(
        regular_rank_translation=regular_p,
        nominal_rank_translation=int(np.median(ranks_p)) if ranks_p else 0,
        regular_rank_rotation=regular_r,
        nominal_rank_rotation=int(np.median(ranks_r)) if ranks_r else 0,
        singularity_fraction_translation=_fraction_below(ranks_p, regular_p),
        singularity_fraction_rotation=_fraction_below(ranks_r, regular_r),
        epsilon_unstable_fraction=float(unstable / max(1, len(samples) * max(1, len(active_coordinates)))),
        epsilon_unstable_sample_fraction=float(unstable_samples / max(1, len(samples))),
        epsilon_stability_gate_passed=stable_sample_fraction >= 0.95,
        epsilon_unstable_columns=sorted(unstable_columns_seen),
        conditioning_percentiles_translation=_percentiles(cond_p),
        conditioning_percentiles_rotation=_percentiles(cond_r),
        samples=len(samples),
        selected_singular_values_translation=sv_p_selected,
        selected_singular_values_rotation=sv_r_selected,
        sample_diagnostics=sample_diagnostics,
        numerical_stability_gate_passed=numerical_gate_passed,
        stable_sample_fraction=stable_sample_fraction,
        task_block=task_block,
        engine_rank_translation=regular_p,
        engine_rank_rotation=regular_r,
        fd_rank_translation=_regular_rank(fd_ranks_p),
        fd_rank_rotation=_regular_rank(fd_ranks_r),
        rank_agreement_rate_translation=float(rank_agree_p / max(1, len(samples))),
        rank_agreement_rate_rotation=float(rank_agree_r / max(1, len(samples))),
        relevant_rank_agreement_rate=relevant_rank_agreement_rate,
        projector_distance_p95=projector_distance_p95,
        principal_angle_p95=principal_angle_p95,
        engine_fd_normalized_error_p95=engine_fd_normalized_error_p95,
    )


def _task_block(task_name: str | None, reference: SemanticSite, target: SemanticSite) -> str:
    label = (task_name or f"{reference.semantic_name}:{target.semantic_name}").lower()
    if "torso" in label or "chest" in label:
        return "rotation"
    if "hand" in label or "foot" in label:
        return "translation"
    return "translation+rotation"


def _relevant_block_stable(jac, task_block: str) -> bool:
    if jac.stability_gate_passed:
        return True
    classifications = jac.column_classifications or []
    severe = {"nonfinite", "unstable_nonsmooth", "engine_fd_mismatch"}
    if any(item.get("class") in severe for item in classifications):
        return False
    if task_block == "translation":
        return all(d.get("translation_max_abs", 0.0) <= d.get("translation_tolerance", 0.0) or d.get("classification") in {"numerically_zero", "unstable_roundoff"} for d in jac.epsilon_discrepancies)
    if task_block == "rotation":
        return all(d.get("rotation_max_abs", 0.0) <= d.get("rotation_tolerance", 0.0) or d.get("classification") in {"numerically_zero", "unstable_roundoff"} for d in jac.epsilon_discrepancies)
    return False


def _select_block(translation: np.ndarray, rotation: np.ndarray, task_block: str) -> np.ndarray:
    if task_block == "translation":
        return translation
    if task_block == "rotation":
        return rotation
    return np.vstack([translation, rotation])


def _backend_abs_floor(adapter: MuJoCoRuntimeModelAdapter) -> float:
    backend = adapter.__class__.__name__.lower()
    if "newton" in backend:
        return 1e-5
    return 1e-9


def _sample_interval(adapter: MuJoCoRuntimeModelAdapter, neutral: np.ndarray, dof: int) -> tuple[float, float, float]:
    info = adapter.coordinate(dof)
    center = float(neutral[info.qpos_adr]) if info.joint_type in {"revolute", "prismatic"} else 0.0
    if info.limited and np.isfinite(info.lower) and np.isfinite(info.upper):
        return float(info.lower), float(info.upper), float(np.clip(center, info.lower, info.upper))
    if info.joint_type == "revolute":
        radius = np.pi
    elif info.joint_type == "prismatic":
        radius = 0.5
    else:
        radius = 0.5
    return center - radius, center + radius, center


def _regular_rank(ranks: list[int]) -> int:
    if not ranks:
        return 0
    for r in range(max(ranks), -1, -1):
        if sum(x >= r for x in ranks) / len(ranks) >= 0.20:
            return r
    return 0


def _fraction_below(ranks: list[int], rank: int) -> float:
    if not ranks:
        return 0.0
    return float(sum(x < rank for x in ranks) / len(ranks))


def _conditioning(singular_values: np.ndarray, rank: int) -> float | None:
    if rank <= 0 or singular_values.size < rank or singular_values[rank - 1] <= 0:
        return None
    return float(singular_values[0] / singular_values[rank - 1])


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p99": None}
    arr = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99))}


def _halton(n: int, dim: int) -> np.ndarray:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if dim > len(primes):
        raise ValueError("halton dimension too high for built-in bases")
    return np.asarray([[_radical_inverse(i + 1, primes[d]) for d in range(dim)] for i in range(n)], dtype=float)


def _radical_inverse(i: int, base: int) -> float:
    f = 1.0 / base
    r = 0.0
    while i > 0:
        r += f * (i % base)
        i //= base
        f /= base
    return r
