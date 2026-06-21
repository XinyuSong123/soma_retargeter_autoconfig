"""Offline bounded chain-only projection references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import so3_log


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
    iterations: int
    active_coordinates: list[int]
    prior_residual_norm: float = 0.0
    solver_message: str = ""

    def to_json(self) -> dict:
        return {
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
        }


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
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-8,
) -> ProjectionResult:
    scale = _position_normalization_scale(adapter, q_seed, reference, target, active_coordinates)
    desired_reference_position = np.asarray(desired_reference_position, dtype=float)
    if not active_coordinates:
        state = adapter.forward_kinematics(q_seed)
        current = adapter.relative_transform(state, reference, target)[:3, 3]
        residual_abs = float(np.linalg.norm(current - desired_reference_position))
        return ProjectionResult(
            desired_reference_position,
            current,
            q_seed.copy(),
            residual_abs,
            float(residual_abs / scale),
            scale,
            residual_abs <= 1e-10,
            _rank_zero_status(residual_abs),
            0,
            [],
        )
    x0, lo, hi = _initial_and_bounds(adapter, q_seed, active_coordinates)
    prior = _prior_context(
        adapter,
        active_coordinates,
        neutral_q=neutral_q,
        previous_q=previous_q,
        neutral_prior_weight=neutral_prior_weight,
        continuity_prior_weight=continuity_prior_weight,
    )

    def residual(x: np.ndarray) -> np.ndarray:
        q = adapter.set_velocity_coordinates(q_seed, active_coordinates, x)
        pos = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, 3]
        return np.concatenate([(pos - desired_reference_position) / scale, _prior_residual(x, prior)])

    res = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    q = adapter.set_velocity_coordinates(q_seed, active_coordinates, res.x)
    projected = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, 3]
    residual_abs = float(np.linalg.norm(projected - desired_reference_position))
    return ProjectionResult(
        desired_reference_position,
        projected,
        q,
        residual_abs,
        float(residual_abs / scale),
        scale,
        bool(res.success),
        "converged" if res.success else "failed",
        int(res.nfev),
        list(active_coordinates),
        float(np.linalg.norm(_prior_residual(res.x, prior))),
        str(res.message),
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
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-8,
) -> ProjectionResult:
    scale = np.pi
    desired_relative_rotation = np.asarray(desired_relative_rotation, dtype=float)
    if not active_coordinates:
        current = adapter.relative_transform(adapter.forward_kinematics(q_seed), reference, target)[:3, :3]
        residual_abs = float(np.linalg.norm(so3_log(current.T @ desired_relative_rotation)))
        return ProjectionResult(
            so3_log(desired_relative_rotation),
            so3_log(current),
            q_seed.copy(),
            residual_abs,
            float(residual_abs / scale),
            scale,
            residual_abs <= 1e-10,
            _rank_zero_status(residual_abs),
            0,
            [],
        )
    x0, lo, hi = _initial_and_bounds(adapter, q_seed, active_coordinates)
    prior = _prior_context(
        adapter,
        active_coordinates,
        neutral_q=neutral_q,
        previous_q=previous_q,
        neutral_prior_weight=neutral_prior_weight,
        continuity_prior_weight=continuity_prior_weight,
    )

    def residual(x: np.ndarray) -> np.ndarray:
        q = adapter.set_velocity_coordinates(q_seed, active_coordinates, x)
        rot = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, :3]
        return np.concatenate([so3_log(rot.T @ desired_relative_rotation) / scale, _prior_residual(x, prior)])

    res = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    q = adapter.set_velocity_coordinates(q_seed, active_coordinates, res.x)
    rot = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, :3]
    err = so3_log(rot.T @ desired_relative_rotation)
    residual_abs = float(np.linalg.norm(err))
    return ProjectionResult(
        so3_log(desired_relative_rotation),
        so3_log(rot),
        q,
        residual_abs,
        float(residual_abs / scale),
        scale,
        bool(res.success),
        "converged" if res.success else "failed",
        int(res.nfev),
        list(active_coordinates),
        float(np.linalg.norm(_prior_residual(res.x, prior))),
        str(res.message),
    )


def _initial_and_bounds(adapter: MuJoCoRuntimeModelAdapter, q_seed: np.ndarray, active_coordinates: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = []
    lo = []
    hi = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type in {"revolute", "prismatic"}:
            x0.append(float(q_seed[info.qpos_adr]))
        else:
            x0.append(0.0)
        lo.append(info.lower if np.isfinite(info.lower) else -np.pi)
        hi.append(info.upper if np.isfinite(info.upper) else np.pi)
    x0_arr = np.asarray(x0)
    lo_arr = np.asarray(lo)
    hi_arr = np.asarray(hi)
    x0_arr = np.clip(x0_arr, lo_arr, hi_arr)
    return x0_arr, lo_arr, hi_arr


def _position_normalization_scale(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> float:
    state = adapter.forward_kinematics(q_seed)
    seed_position = adapter.relative_transform(state, reference, target)[:3, 3]
    positions = [seed_position]
    if active_coordinates:
        x0, lo, hi = _initial_and_bounds(adapter, q_seed, active_coordinates)
        for values in (lo, hi):
            q = adapter.set_velocity_coordinates(q_seed, active_coordinates, values)
            positions.append(adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, 3])
        for idx in range(len(active_coordinates)):
            for bound in (lo[idx], hi[idx]):
                values = x0.copy()
                values[idx] = bound
                q = adapter.set_velocity_coordinates(q_seed, active_coordinates, values)
                positions.append(adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, 3])
    offsets = [float(np.linalg.norm(pos - seed_position)) for pos in positions]
    return max(max(offsets, default=0.0), 1e-6)


def _rank_zero_status(residual_abs: float) -> str:
    if residual_abs <= 1e-10:
        return "rank_zero"
    return "unreachable/rank_zero"


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
