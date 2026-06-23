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

    def to_json(self) -> dict[str, Any]:
        return {
            "certificate_class": self.certificate_class,
            "task_block": self.task_block,
            "decomposition": _json_clean(self.decomposition),
            "gates": self.gates.copy(),
            "active_limits": _json_clean(self.active_limits),
            "kkt": _json_clean(self.kkt),
            "seed_consensus": _json_clean(self.seed_consensus),
            "rank_evidence": _json_clean(self.rank_evidence),
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
    kkt = _kkt_evidence(evidence, gradient, active_limits)
    kkt_passed = bool(kkt["projected_gradient_norm"] <= kkt["tolerance"])
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
    seed_consensus = _seed_consensus(evidence.seed_consensus, evidence.seed_consensus_passed)
    seed_consensus_passed = bool(seed_consensus.get("passed", False))
    gates = {
        "compatible_demand_retained": bool(compatible_demand_retained),
        "residual_explained": bool(residual_explained),
        "projected_gradient_kkt": kkt_passed,
        "seed_consensus": seed_consensus_passed,
        "no_orthogonal_leakage": bool(no_orthogonal_leakage),
    }
    if evidence.continuation_passed is not None:
        gates["continuation"] = bool(evidence.continuation_passed)
    if evidence.joint_limits_passed is not None:
        gates["joint_limits"] = bool(evidence.joint_limits_passed)
    if evidence.numerical_gate_passed is not None:
        gates["numerical"] = bool(evidence.numerical_gate_passed)
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
        "active_limit_residual_norm": active_limit_residual_norm,
        "orthogonal_leakage_norm": float(np.linalg.norm(orthogonal_leakage)),
        "rank_incompatible_fraction": _safe_fraction(rank_incompatible_residual_norm, residual_norm),
        "active_limit_fraction": _safe_fraction(active_limit_residual_norm, residual_norm),
        "residual_tolerance": residual_tolerance,
        "component_tolerance": component_tolerance,
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
    if not evidence.converged and evidence.active_coordinates:
        return "solver_failed", [f"solver did not converge: {evidence.solver_status or 'unknown_status'}"]
    if residual_norm <= residual_tolerance and kkt_passed:
        return "exact_reachable", ["projection residual is within exact-reachable tolerance"]
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
    lower = _optional_array(evidence.lower_bounds)
    upper = _optional_array(evidence.upper_bounds)
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
    active_limits: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(evidence.kkt, dict) and evidence.kkt:
        projected_norm = evidence.kkt.get("stationarity_inf_norm", evidence.kkt.get("projected_gradient_norm", 0.0))
        tolerance = evidence.kkt.get("stationarity_tolerance", evidence.kkt.get("tolerance", evidence.kkt.get("active_bound_tolerance", 1e-7)))
        task_gradient = evidence.kkt.get("task_gradient", gradient.tolist())
        return {
            "gradient": task_gradient,
            "projected_gradient": evidence.kkt.get("projected_gradient", []),
            "gradient_norm": float(np.linalg.norm(np.asarray(task_gradient, dtype=float))) if task_gradient is not None else None,
            "projected_gradient_norm": float(projected_norm),
            "tolerance": float(tolerance),
            "source": "provided_kkt",
            "raw": evidence.kkt,
        }
    values = _optional_array(evidence.coordinate_values)
    lower = _optional_array(evidence.lower_bounds)
    upper = _optional_array(evidence.upper_bounds)
    active_coordinates = list(evidence.active_coordinates)
    if gradient.size == 0:
        projected = np.zeros(0)
    elif values is None or lower is None or upper is None or not active_coordinates:
        projected = gradient
    else:
        projected_values = []
        for idx, g in enumerate(gradient):
            value = float(values[idx])
            lo = float(lower[idx])
            hi = float(upper[idx])
            boundary_tol = float(active_limits["boundary_tolerance"])
            at_lower = np.isfinite(lo) and value <= lo + boundary_tol
            at_upper = np.isfinite(hi) and value >= hi - boundary_tol
            if at_lower:
                projected_values.append(min(0.0, float(g)))
            elif at_upper:
                projected_values.append(max(0.0, float(g)))
            else:
                projected_values.append(float(g))
        projected = np.asarray(projected_values, dtype=float)
    norm = float(np.linalg.norm(projected))
    grad_norm = float(np.linalg.norm(gradient))
    tolerance = max(float(evidence.kkt_abs_tolerance), float(evidence.kkt_rel_tolerance) * max(1.0, grad_norm))
    return {
        "gradient": gradient.tolist(),
        "projected_gradient": projected.tolist(),
        "gradient_norm": grad_norm,
        "projected_gradient_norm": norm,
        "tolerance": tolerance,
    }


def _seed_consensus(seed_consensus: dict[str, Any] | None, passed: bool | None = None) -> dict[str, Any]:
    if seed_consensus is None:
        return {
            "checked": passed is not None,
            "passed": bool(passed) if passed is not None else False,
            "start_count": 0,
            "max_projected_delta": None,
            "max_residual_delta": None,
        }
    payload = dict(seed_consensus)
    if passed is not None:
        payload["passed"] = bool(passed)
        payload.setdefault("checked", True)
    return payload


def _residual_tolerance(
    evidence: ProjectionCertificateEvidence,
    desired: np.ndarray,
    projected: np.ndarray,
    scale: float,
) -> float:
    if evidence.residual_tolerance is not None:
        return float(evidence.residual_tolerance)
    if evidence.exact_threshold is not None:
        return float(evidence.exact_threshold)
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
