"""Objective runtime quality metrics for Step 3.1 fleet evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from soma_retargeter.robotics.v3.spatial import rotation_error


@dataclass(frozen=True)
class NumericSummary:
    mean: float
    p95: float
    max: float

    def to_json(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": self.mean,
            f"{prefix}_p95": self.p95,
            f"{prefix}_max": self.max,
        }


def summarize(values: np.ndarray | list[float]) -> NumericSummary:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return NumericSummary(mean=0.0, p95=0.0, max=0.0)
    return NumericSummary(
        mean=_stable(float(np.mean(arr))),
        p95=_stable(float(np.percentile(arr, 95))),
        max=_stable(float(np.max(arr))),
    )


def target_stream_metrics(transforms: Mapping[str, np.ndarray], *, capability_status: Mapping[str, str] | None = None) -> dict[str, Any]:
    capability_status = capability_status or {}
    per_semantic: dict[str, dict[str, Any]] = {}
    all_translation_velocities: list[float] = []
    all_rotation_velocities: list[float] = []
    finite_count = 0
    nan_count = 0
    target_jump_count = 0
    orthogonality_error_max = 0.0
    for semantic, stack_value in sorted(transforms.items()):
        stack = np.asarray(stack_value, dtype=np.float64)
        if stack.ndim != 3 or stack.shape[1:] != (4, 4):
            raise ValueError(f"target stream for {semantic} must have shape [F,4,4], got {stack.shape}")
        finite_mask = np.isfinite(stack).all(axis=(1, 2))
        finite_count += int(np.count_nonzero(finite_mask))
        nan_count += int(stack.shape[0] - np.count_nonzero(finite_mask))
        rotations = stack[:, :3, :3]
        orth = np.linalg.norm(np.matmul(np.swapaxes(rotations, 1, 2), rotations) - np.eye(3), axis=(1, 2))
        orthogonality_error_max = max(orthogonality_error_max, float(np.max(orth)) if orth.size else 0.0)
        translation_velocity = np.linalg.norm(np.diff(stack[:, :3, 3], axis=0), axis=1) if stack.shape[0] > 1 else np.zeros(0)
        rotation_velocity = _rotation_velocity(stack)
        all_translation_velocities.extend(float(v) for v in translation_velocity)
        all_rotation_velocities.extend(float(v) for v in rotation_velocity)
        jumps = int(np.count_nonzero(translation_velocity > 0.5)) + int(np.count_nonzero(rotation_velocity > math.radians(60.0)))
        target_jump_count += jumps
        per_semantic[semantic] = {
            "finite_count": int(np.count_nonzero(finite_mask)),
            "nan_count": int(stack.shape[0] - np.count_nonzero(finite_mask)),
            "se3_orthogonality_error_max": _stable(float(np.max(orth)) if orth.size else 0.0),
            "frame_to_frame_translation_velocity_p95": summarize(translation_velocity).p95,
            "frame_to_frame_translation_velocity_max": summarize(translation_velocity).max,
            "frame_to_frame_rotation_velocity_p95": summarize(rotation_velocity).p95,
            "frame_to_frame_rotation_velocity_max": summarize(rotation_velocity).max,
            "target_jump_count": jumps,
            "capability_status": capability_status.get(semantic, "unknown"),
        }
    translation_velocity_summary = summarize(all_translation_velocities)
    rotation_velocity_summary = summarize(all_rotation_velocities)
    return {
        "finite_count": finite_count,
        "nan_count": nan_count,
        "se3_orthogonality_error_max": _stable(orthogonality_error_max),
        "frame_to_frame_translation_velocity_p95": translation_velocity_summary.p95,
        "frame_to_frame_translation_velocity_max": translation_velocity_summary.max,
        "frame_to_frame_rotation_velocity_p95": rotation_velocity_summary.p95,
        "frame_to_frame_rotation_velocity_max": rotation_velocity_summary.max,
        "target_jump_count": target_jump_count,
        "per_semantic": per_semantic,
    }


def smoke_output_metrics(
    *,
    q_sequence: np.ndarray,
    coordinate_info: list[Any],
    residuals: np.ndarray,
    runtime_seconds: float,
    solver_iterations: np.ndarray | None = None,
) -> dict[str, Any]:
    q = np.asarray(q_sequence, dtype=np.float64)
    residual_arr = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if q.ndim == 1:
        q = q[None, :]
    finite = np.isfinite(q)
    nan_count = int(np.isnan(q).sum())
    inf_count = int(np.isinf(q).sum())
    joint_abs = summarize(np.abs(q))
    velocity = np.diff(q, axis=0) if q.shape[0] > 1 else np.zeros((0, q.shape[1]))
    acceleration = np.diff(velocity, axis=0) if velocity.shape[0] > 1 else np.zeros((0, q.shape[1]))
    joint_limits = joint_limit_metrics(q, coordinate_info)
    residual_summary = summarize(residual_arr)
    normalized = residual_arr / max(1.0, float(np.nanmax(residual_arr)) if residual_arr.size else 1.0)
    normalized_summary = summarize(normalized)
    iterations = np.asarray(solver_iterations if solver_iterations is not None else np.zeros(q.shape[0]), dtype=np.float64)
    iter_summary = summarize(iterations)
    return {
        "output_frame_count": int(q.shape[0]),
        "joint_coord_count": int(q.shape[1]) if q.ndim == 2 else 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "output_finite": bool(finite.all()),
        "joint_limit_violation_count": joint_limits["joint_limit_violation_count"],
        "max_joint_limit_violation": joint_limits["max_joint_limit_violation"],
        "joint_position_abs_p95": joint_abs.p95,
        "joint_position_abs_max": joint_abs.max,
        "joint_velocity_p95": summarize(np.abs(velocity)).p95,
        "joint_velocity_max": summarize(np.abs(velocity)).max,
        "joint_acceleration_p95": summarize(np.abs(acceleration)).p95,
        "joint_acceleration_max": summarize(np.abs(acceleration)).max,
        "task_residual_mean": residual_summary.mean,
        "task_residual_p95": residual_summary.p95,
        "task_residual_max": residual_summary.max,
        "normalized_task_residual_mean": normalized_summary.mean,
        "normalized_task_residual_p95": normalized_summary.p95,
        "normalized_task_residual_max": normalized_summary.max,
        "solver_success_fraction": 1.0 if nan_count == 0 and inf_count == 0 else 0.0,
        "solver_iteration_mean": iter_summary.mean,
        "solver_iteration_p95": iter_summary.p95,
        "solver_iteration_max": iter_summary.max,
        "runtime_seconds": _stable(float(runtime_seconds)),
    }


def joint_limit_metrics(q_sequence: np.ndarray, coordinate_info: list[Any]) -> dict[str, Any]:
    q = np.asarray(q_sequence, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    count = 0
    max_violation = 0.0
    for coord in coordinate_info:
        index = int(getattr(coord, "index", coord.get("index") if isinstance(coord, dict) else -1))
        limited = bool(getattr(coord, "limited", coord.get("limited") if isinstance(coord, dict) else False))
        if index < 0 or index >= q.shape[1] or not limited:
            continue
        lower = float(getattr(coord, "lower", coord.get("lower") if isinstance(coord, dict) else -math.inf))
        upper = float(getattr(coord, "upper", coord.get("upper") if isinstance(coord, dict) else math.inf))
        values = q[:, index]
        violation = np.maximum(lower - values, values - upper)
        violation = np.maximum(violation, 0.0)
        count += int(np.count_nonzero(violation > 0.0))
        max_violation = max(max_violation, float(np.max(violation)) if violation.size else 0.0)
    return {
        "joint_limit_violation_count": count,
        "max_joint_limit_violation": _stable(max_violation),
    }


def contact_diagnostics(transforms: Mapping[str, np.ndarray]) -> dict[str, Any]:
    foot_heights: list[float] = []
    for semantic in ("LeftFoot", "RightFoot"):
        if semantic in transforms:
            arr = np.asarray(transforms[semantic], dtype=np.float64)
            foot_heights.extend(float(v) for v in arr[:, 2, 3])
    heights = np.asarray(foot_heights, dtype=np.float64)
    if heights.size == 0:
        return {
            "support_height_min": None,
            "support_height_max": None,
            "foot_height_below_ground_count": 0,
            "foot_sliding_proxy_if_contact_scores_available": None,
            "stance_width_p95": None,
            "stance_width_max": None,
        }
    stance = np.zeros(0)
    if "LeftFoot" in transforms and "RightFoot" in transforms:
        left = np.asarray(transforms["LeftFoot"], dtype=np.float64)[:, :3, 3]
        right = np.asarray(transforms["RightFoot"], dtype=np.float64)[:, :3, 3]
        frame_count = min(left.shape[0], right.shape[0])
        stance = np.linalg.norm(left[:frame_count, :2] - right[:frame_count, :2], axis=1)
    return {
        "support_height_min": _stable(float(np.min(heights))),
        "support_height_max": _stable(float(np.max(heights))),
        "foot_height_below_ground_count": int(np.count_nonzero(heights < -1e-6)),
        "foot_sliding_proxy_if_contact_scores_available": None,
        "stance_width_p95": summarize(stance).p95 if stance.size else None,
        "stance_width_max": summarize(stance).max if stance.size else None,
    }


def _rotation_velocity(stack: np.ndarray) -> np.ndarray:
    if stack.shape[0] <= 1:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [rotation_error(stack[index - 1, :3, :3], stack[index, :3, :3]) for index in range(1, stack.shape[0])],
        dtype=np.float64,
    )


def _stable(value: float) -> float:
    if not math.isfinite(value):
        return value
    if abs(value) < 1e-15:
        return 0.0
    return round(float(value), 12)
