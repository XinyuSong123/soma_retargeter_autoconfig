"""Deterministic multi-pose local-rank summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .numerical_jacobian import matrix_rank_and_singular_values, numerical_relative_jacobian


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
    rank_method: str = "per-sample-local-rank-regular-fraction"
    regular_rank_fraction_threshold: float = 0.20

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
    samples = deterministic_chain_samples(adapter, active_coordinates, low_discrepancy_count=low_discrepancy_count)
    for sample_idx, q in enumerate(samples):
        jac = numerical_relative_jacobian(
            adapter,
            q,
            reference,
            target,
            active_coordinates,
            raise_on_unstable=raise_on_unstable,
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
        if not jac.stability_gate_passed:
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
            }
        )
        if sample_idx < 3:
            sv_p_selected.append(sp.tolist())
            sv_r_selected.append(sr.tolist())
    regular_p = _regular_rank(ranks_p)
    regular_r = _regular_rank(ranks_r)
    return ReachabilityReport(
        regular_rank_translation=regular_p,
        nominal_rank_translation=int(np.median(ranks_p)) if ranks_p else 0,
        regular_rank_rotation=regular_r,
        nominal_rank_rotation=int(np.median(ranks_r)) if ranks_r else 0,
        singularity_fraction_translation=_fraction_below(ranks_p, regular_p),
        singularity_fraction_rotation=_fraction_below(ranks_r, regular_r),
        epsilon_unstable_fraction=float(unstable / max(1, len(samples) * max(1, len(active_coordinates)))),
        epsilon_unstable_sample_fraction=float(unstable_samples / max(1, len(samples))),
        epsilon_stability_gate_passed=unstable == 0,
        epsilon_unstable_columns=sorted(unstable_columns_seen),
        conditioning_percentiles_translation=_percentiles(cond_p),
        conditioning_percentiles_rotation=_percentiles(cond_r),
        samples=len(samples),
        selected_singular_values_translation=sv_p_selected,
        selected_singular_values_rotation=sv_r_selected,
        sample_diagnostics=sample_diagnostics,
    )


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
