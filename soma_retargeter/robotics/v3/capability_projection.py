"""Capability-aware projection wrappers with certificate evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .engine_jacobian import engine_relative_jacobian
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .numerical_jacobian import numerical_relative_jacobian
from .projection_certificate import ProjectionCertificate, ProjectionCertificateEvidence, build_projection_certificate
from .reachability_metrics import backend_rank_abs_floor, rank_with_uncertainty, retained_left_singular_vectors
from .spatial import so3_log


@dataclass(frozen=True)
class TaskResidualDecomposition:
    rank: int
    residual_norm: float
    demand_norm: float
    compatible_demand_norm: float
    tangent_residual_norm: float
    rank_incompatible_residual_norm: float
    active_limit_residual_norm: float
    rank_threshold: float
    singular_values: list[float]
    active_limit: bool = False

    def to_json(self) -> dict:
        return {
            "rank": self.rank,
            "residual_norm": self.residual_norm,
            "demand_norm": self.demand_norm,
            "compatible_demand_norm": self.compatible_demand_norm,
            "tangent_residual_norm": self.tangent_residual_norm,
            "rank_incompatible_residual_norm": self.rank_incompatible_residual_norm,
            "active_limit_residual_norm": self.active_limit_residual_norm,
            "rank_threshold": self.rank_threshold,
            "singular_values": self.singular_values,
            "active_limit": self.active_limit,
        }


def decompose_task_residual(
    jacobian: np.ndarray,
    demand: np.ndarray,
    residual: np.ndarray,
    *,
    active_limit: bool = False,
    noise_norm: float = 0.0,
    rank_abs_floor: float = 1e-9,
    rank_rel_tol: float = 1e-5,
) -> TaskResidualDecomposition:
    jac = np.asarray(jacobian, dtype=float)
    demand_vec = np.asarray(demand, dtype=float).reshape(3)
    residual_vec = np.asarray(residual, dtype=float).reshape(3)
    rank, singular_values, threshold = rank_with_uncertainty(
        jac,
        noise_norm=noise_norm,
        abs_floor=rank_abs_floor,
        rel_tol=rank_rel_tol,
    )
    basis = retained_left_singular_vectors(jac, rank)
    projector = basis @ basis.T if basis.size else np.zeros((3, 3))
    tangent_residual = projector @ residual_vec
    rank_incompatible_residual = (np.eye(3) - projector) @ residual_vec
    tangent_norm = float(np.linalg.norm(tangent_residual))
    return TaskResidualDecomposition(
        rank=int(rank),
        residual_norm=float(np.linalg.norm(residual_vec)),
        demand_norm=float(np.linalg.norm(demand_vec)),
        compatible_demand_norm=float(np.linalg.norm(projector @ demand_vec)),
        tangent_residual_norm=tangent_norm,
        rank_incompatible_residual_norm=float(np.linalg.norm(rank_incompatible_residual)),
        active_limit_residual_norm=tangent_norm if active_limit else 0.0,
        rank_threshold=float(threshold),
        singular_values=singular_values.tolist(),
        active_limit=bool(active_limit),
    )


@dataclass(frozen=True)
class CapabilityProjectionResult:
    projection: Any
    certificate: ProjectionCertificate

    @property
    def status(self) -> str:
        return self.projection.status

    @property
    def chain_q(self) -> np.ndarray:
        return self.projection.chain_q

    def to_json(self) -> dict:
        payload = self.projection.to_json()
        payload["capability_certificate"] = self.certificate.to_json()
        return payload


def project_endpoint_position_with_certificate(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_reference_position: np.ndarray,
    *,
    neutral_q: np.ndarray | None = None,
    previous_q: np.ndarray | None = None,
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-8,
    seed_consensus_atol: float = 1e-7,
) -> CapabilityProjectionResult:
    from .chain_projection import project_endpoint_position

    projection = project_endpoint_position(
        adapter,
        q_seed,
        reference,
        target,
        active_coordinates,
        desired_reference_position,
        neutral_q=neutral_q,
        previous_q=previous_q,
        neutral_prior_weight=neutral_prior_weight,
        continuity_prior_weight=continuity_prior_weight,
    )
    consensus = _projection_seed_consensus(projection, active_coordinates, atol=seed_consensus_atol)
    certificate = certify_projection_result(
        adapter,
        q_seed,
        reference,
        target,
        active_coordinates,
        projection,
        task_block="translation",
        seed_consensus=consensus,
    )
    return CapabilityProjectionResult(projection, certificate)


def project_torso_orientation_with_certificate(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_relative_rotation: np.ndarray,
    *,
    neutral_q: np.ndarray | None = None,
    previous_q: np.ndarray | None = None,
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-8,
    seed_consensus_atol: float = 1e-7,
) -> CapabilityProjectionResult:
    from .chain_projection import project_torso_orientation

    projection = project_torso_orientation(
        adapter,
        q_seed,
        reference,
        target,
        active_coordinates,
        desired_relative_rotation,
        neutral_q=neutral_q,
        previous_q=previous_q,
        neutral_prior_weight=neutral_prior_weight,
        continuity_prior_weight=continuity_prior_weight,
    )
    consensus = _projection_seed_consensus(projection, active_coordinates, atol=seed_consensus_atol)
    certificate = certify_projection_result(
        adapter,
        q_seed,
        reference,
        target,
        active_coordinates,
        projection,
        task_block="rotation",
        seed_consensus=consensus,
    )
    return CapabilityProjectionResult(projection, certificate)


def certify_projection_result(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    projection: Any,
    *,
    task_block: str,
    seed_consensus: dict | None = None,
) -> ProjectionCertificate:
    try:
        seed = _task_vector(adapter, q_seed, reference, target, task_block)
        jacobian, noise_norm, jacobian_source = _task_jacobian(
            adapter,
            projection.chain_q,
            reference,
            target,
            active_coordinates,
            task_block,
        )
        values, lower, upper = _coordinate_values_and_bounds(adapter, projection.chain_q, active_coordinates)
        evidence = ProjectionCertificateEvidence(
            task_block=task_block,
            desired=np.asarray(projection.desired, dtype=float),
            projected=np.asarray(projection.projected, dtype=float),
            seed=seed,
            jacobian=jacobian,
            scale=float(projection.normalization_scale),
            noise_norm=noise_norm,
            rank_abs_floor=backend_rank_abs_floor(adapter),
            converged=bool(projection.converged),
            solver_status=projection.status,
            active_coordinates=list(active_coordinates),
            coordinate_values=values,
            lower_bounds=lower,
            upper_bounds=upper,
            seed_consensus=seed_consensus,
            jacobian_source=jacobian_source,
            solver_message=projection.solver_message,
            normalized_residual=getattr(projection, "normalized_residual", None),
            task_residual=getattr(projection, "residual", None),
            kkt=getattr(projection, "active_limit_kkt", None),
            residual_parameterization=getattr(projection, "residual_parameterization", None),
            residual_jacobian_source=getattr(projection, "residual_jacobian_source", None),
        )
    except Exception:
        evidence = ProjectionCertificateEvidence(
            task_block=task_block,
            desired=np.full(3, np.nan),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.zeros((3, 0)),
            converged=False,
            solver_status="certificate_evidence_failed",
            seed_consensus=seed_consensus,
        )
    return build_projection_certificate(evidence)


def _task_vector(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    task_block: str,
) -> np.ndarray:
    relative = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)
    if task_block == "rotation":
        return so3_log(relative[:3, :3])
    return relative[:3, 3].copy()


def _task_jacobian(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    task_block: str,
) -> tuple[np.ndarray, float, str]:
    if not active_coordinates:
        return np.zeros((3, 0)), 0.0, "rank_zero_no_active_coordinates"
    try:
        fd = numerical_relative_jacobian(
            adapter,
            q,
            reference,
            target,
            active_coordinates,
            engine_validation=False,
        )
        noise_norm = float(fd.jacobian_noise_norm)
    except Exception:
        fd = None
        noise_norm = 0.0
    try:
        engine = engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
        matrix = engine.rotation if task_block == "rotation" else engine.translation
        return matrix, noise_norm, "engine_relative_jacobian"
    except Exception:
        if fd is None:
            raise
        matrix = fd.rotation if task_block == "rotation" else fd.translation
        return matrix, noise_norm, "finite_difference_fallback"


def _coordinate_values_and_bounds(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    active_coordinates: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type in {"revolute", "prismatic"}:
            values.append(float(q[info.qpos_adr]))
        else:
            values.append(0.0)
        lower.append(float(info.lower) if np.isfinite(info.lower) else -np.inf)
        upper.append(float(info.upper) if np.isfinite(info.upper) else np.inf)
    return np.asarray(values, dtype=float), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _seed_consensus(
    projector: Callable[[np.ndarray], Any],
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    projection: Any,
    active_coordinates: list[int],
    *,
    neutral_q: np.ndarray | None,
    previous_q: np.ndarray | None,
    atol: float,
) -> dict:
    starts = _seed_candidates(adapter, q_seed, projection, active_coordinates, neutral_q=neutral_q, previous_q=previous_q)
    projected_deltas: list[float] = []
    residual_deltas: list[float] = []
    statuses: list[str] = []
    for seed in starts:
        candidate = projector(seed)
        projected_deltas.append(float(np.linalg.norm(candidate.projected - projection.projected)))
        residual_deltas.append(abs(float(candidate.residual) - float(projection.residual)))
        statuses.append(candidate.status)
    max_projected_delta = max(projected_deltas, default=0.0)
    max_residual_delta = max(residual_deltas, default=0.0)
    return {
        "checked": True,
        "passed": bool(max_projected_delta <= atol and max_residual_delta <= atol),
        "start_count": len(starts),
        "max_projected_delta": max_projected_delta,
        "max_residual_delta": max_residual_delta,
        "tolerance": float(atol),
        "statuses": statuses,
    }


def _seed_candidates(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    projection: Any,
    active_coordinates: list[int],
    *,
    neutral_q: np.ndarray | None,
    previous_q: np.ndarray | None,
) -> list[np.ndarray]:
    candidates = [
        np.asarray(q_seed, dtype=float),
        adapter.neutral_q() if neutral_q is None else np.asarray(neutral_q, dtype=float),
        np.asarray(projection.chain_q, dtype=float),
    ]
    if previous_q is not None:
        candidates.append(np.asarray(previous_q, dtype=float))
    midpoint = _midpoint_seed(adapter, q_seed, active_coordinates)
    if midpoint is not None:
        candidates.append(midpoint)
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(candidate.shape == seen.shape and np.allclose(candidate, seen, rtol=0.0, atol=1e-12) for seen in unique):
            unique.append(candidate.copy())
    return unique


def _midpoint_seed(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    active_coordinates: list[int],
) -> np.ndarray | None:
    if not active_coordinates:
        return None
    midpoints = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if not (np.isfinite(info.lower) and np.isfinite(info.upper)):
            return None
        midpoints.append(0.5 * (float(info.lower) + float(info.upper)))
    return adapter.set_velocity_coordinates(q_seed, active_coordinates, np.asarray(midpoints, dtype=float))


def _projection_seed_consensus(projection: Any, active_coordinates: list[int], *, atol: float) -> dict:
    rows = list(getattr(projection, "seed_results", []) or [])
    selected_seed_index = getattr(projection, "selected_seed_index", None)
    if not rows:
        trivial = not active_coordinates
        return {
            "checked": True,
            "passed": bool(trivial or getattr(projection, "seed_consensus_passed", False)),
            "start_count": 1 if trivial else int(getattr(projection, "deterministic_start_count", 0) or 0),
            "selected_seed_index": selected_seed_index,
            "max_projected_delta": 0.0 if trivial else None,
            "max_residual_delta": 0.0 if trivial else None,
            "tolerance": float(atol),
            "source": "projection_result_seed_results",
            "statuses": ["trivial_rank_zero"] if trivial else [],
        }
    selected = next((row for row in rows if row.get("seed_index") == selected_seed_index), None)
    selected_norm = float((selected or rows[0]).get("task_residual_norm", 0.0))
    accepted = [row for row in rows if row.get("accepted", False)]
    residual_deltas = [abs(float(row.get("task_residual_norm", 0.0)) - selected_norm) for row in accepted]
    return {
        "checked": True,
        "passed": bool(getattr(projection, "seed_consensus_passed", False)),
        "start_count": int(getattr(projection, "deterministic_start_count", len(rows)) or len(rows)),
        "selected_seed_index": selected_seed_index,
        "max_projected_delta": None,
        "max_residual_delta": max(residual_deltas, default=0.0),
        "tolerance": float(atol),
        "source": "projection_result_seed_results",
        "statuses": ["accepted" if row.get("accepted", False) else "rejected" for row in rows],
        "seed_results": rows,
    }
