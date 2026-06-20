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
        }


def project_endpoint_position(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_reference_position: np.ndarray,
) -> ProjectionResult:
    scale = _position_normalization_scale(adapter, q_seed, reference, target, active_coordinates)
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
            True,
            "rank_zero",
        )
    x0, lo, hi = _initial_and_bounds(adapter, q_seed, active_coordinates)

    def residual(x: np.ndarray) -> np.ndarray:
        q = adapter.set_velocity_coordinates(q_seed, active_coordinates, x)
        pos = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, 3]
        return (pos - desired_reference_position) / scale

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
        str(res.message),
    )


def project_torso_orientation(
    adapter: MuJoCoRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_relative_rotation: np.ndarray,
) -> ProjectionResult:
    scale = np.pi
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
            True,
            "rank_zero",
        )
    x0, lo, hi = _initial_and_bounds(adapter, q_seed, active_coordinates)

    def residual(x: np.ndarray) -> np.ndarray:
        q = adapter.set_velocity_coordinates(q_seed, active_coordinates, x)
        rot = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)[:3, :3]
        return so3_log(rot.T @ desired_relative_rotation) / scale

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
