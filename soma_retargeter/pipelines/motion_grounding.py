# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

import soma_retargeter.robot_registry_parser as robot_registry_parser
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.foot_anchors import (
    generate_virtual_sole_anchors,
    load_capability_profile,
    resolve_capability_profile,
)


SUPPORT_CONTACT_ANCHORS = ("toe", "heel", "inner_edge", "outer_edge")


@dataclass
class MotionGroundingStats:
    applied: bool
    frames: int = 0
    lifted_frames: int = 0
    max_lift_m: float = 0.0
    min_support_z_before_m: float = 0.0
    min_support_z_after_m: float = 0.0
    reason: str = ""


def _quat_rotate_xyzw(quat_xyzw, point) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat = quat / norm
    q_vec = quat[0:3]
    point = np.asarray(point, dtype=np.float64)
    t = 2.0 * np.cross(q_vec, point)
    return point + quat[3] * t + np.cross(q_vec, t)


def _transform_point_xyzw(transform_values, local_point) -> np.ndarray:
    transform_values = np.asarray(transform_values, dtype=np.float64)
    return transform_values[0:3] + _quat_rotate_xyzw(transform_values[3:7], local_point)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values
    window = min(int(window), len(values))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _support_anchor_heights(
    model,
    state,
    robot_builder,
    resolved_capability: dict[str, Any],
    anchors: dict[str, Any],
    q_values: np.ndarray,
) -> np.ndarray:
    wp.copy(model.joint_q, wp.array(np.asarray(q_values, dtype=np.float32), dtype=wp.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    body_q = state.body_q.numpy()
    body_map = {
        newton_utils.get_name_from_label(label): index
        for index, label in enumerate(robot_builder.body_label)
    }
    links = resolved_capability.get("links", {})
    heights: list[float] = []
    for side in ("left", "right"):
        link_name = links.get(f"{side}_foot")
        if not link_name or link_name not in body_map or not isinstance(anchors.get(side), dict):
            continue
        foot_tx = body_q[body_map[link_name]]
        for anchor_name in SUPPORT_CONTACT_ANCHORS:
            local_point = anchors[side].get(anchor_name)
            if local_point is None:
                continue
            heights.append(float(_transform_point_xyzw(foot_tx, local_point)[2]))
    return np.asarray(heights, dtype=np.float64)


def apply_virtual_foot_grounding_to_frames(
    frames: np.ndarray,
    *,
    model,
    robot_builder,
    robot_name: str,
    ground_height_m: float = 0.0,
    smooth_window: int = 5,
) -> tuple[np.ndarray, MotionGroundingStats]:
    """Lift retargeted root frames so virtual foot support anchors do not penetrate the ground."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    if frames.size == 0:
        return frames, MotionGroundingStats(applied=False, reason="empty motion")

    try:
        profile, _ = load_capability_profile(robot_name)
        resolution = resolve_capability_profile(robot_name, profile=profile)
        resolution.raise_for_errors()
        anchors = generate_virtual_sole_anchors(robot_name, resolution.payload, profile=profile)
    except Exception as exc:
        return frames, MotionGroundingStats(applied=False, reason=f"capability load failed: {exc}")

    if not anchors.get("enabled"):
        return frames, MotionGroundingStats(applied=False, reason="virtual sole anchors disabled")

    grounded = np.asarray(frames, dtype=np.float32).copy()
    state = model.state()
    min_heights = np.zeros(len(grounded), dtype=np.float64)
    for frame_idx, q_values in enumerate(grounded):
        heights = _support_anchor_heights(
            model,
            state,
            robot_builder,
            resolution.payload,
            anchors,
            q_values,
        )
        if len(heights) == 0:
            return frames, MotionGroundingStats(applied=False, reason="no support anchors resolved")
        min_heights[frame_idx] = float(np.min(heights))

    required_lifts = np.maximum(0.0, ground_height_m - min_heights)
    smoothed_lifts = _moving_average(required_lifts, smooth_window)
    lifts = np.maximum(required_lifts, smoothed_lifts)
    grounded[:, 2] += lifts.astype(np.float32)
    min_after = min_heights + lifts

    lifted_frames = int(np.count_nonzero(lifts > 1e-6))
    return grounded, MotionGroundingStats(
        applied=lifted_frames > 0,
        frames=len(grounded),
        lifted_frames=lifted_frames,
        max_lift_m=float(np.max(lifts)) if len(lifts) else 0.0,
        min_support_z_before_m=float(np.min(min_heights)) if len(min_heights) else 0.0,
        min_support_z_after_m=float(np.min(min_after)) if len(min_after) else 0.0,
        reason="ok",
    )
