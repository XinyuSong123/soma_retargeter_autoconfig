# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp

import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.foot_anchors import (
    generate_virtual_sole_anchors,
    load_capability_profile,
    resolve_capability_profile,
)
from soma_retargeter.tools.pose_io import apply_robot_pose_to_model, create_joint_name_to_q_index_map


SUPPORT_CONTACT_ANCHORS = ("toe", "heel", "inner_edge", "outer_edge")


@dataclass
class FootGroundingOptions:
    contact_threshold_m: float = 0.025
    max_joint_delta_rad: float = 0.22
    enable_joint_tuning: bool = True
    joint_delta_penalty: float = 0.01
    descent_steps_rad: tuple[float, ...] = (0.08, 0.04, 0.02, 0.01, 0.005)


@dataclass
class FootGroundingResult:
    payload: dict[str, Any]
    changed: bool
    contact_success: bool
    max_abs_support_anchor_z_m: float
    rms_support_anchor_z_m: float
    tuned_joint_deltas_rad: dict[str, float] = field(default_factory=dict)
    candidate_joints: list[str] = field(default_factory=list)
    used_hip_fallback: bool = False
    support_anchor_z_m: dict[str, dict[str, float]] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


def _round_float(value: float, digits: int = 8) -> float:
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 10 ** -digits else rounded


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


def _set_model_q(model, state, q_values: np.ndarray) -> None:
    wp.copy(model.joint_q, wp.array(np.asarray(q_values, dtype=np.float32), dtype=wp.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)


def _body_name_to_index(robot_builder) -> dict[str, int]:
    return {
        newton_utils.get_name_from_label(label): index
        for index, label in enumerate(robot_builder.body_label)
    }


def _support_world_points(
    model,
    state,
    robot_builder,
    resolved_capability: dict[str, Any],
    anchors: dict[str, Any],
    q_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    _set_model_q(model, state, q_values)
    body_q = state.body_q.numpy()
    body_map = _body_name_to_index(robot_builder)
    links = resolved_capability.get("links", {})
    world_points: dict[str, dict[str, np.ndarray]] = {}
    z_values: list[float] = []

    for side in ("left", "right"):
        link_name = links.get(f"{side}_foot")
        if not link_name or link_name not in body_map or not isinstance(anchors.get(side), dict):
            continue
        foot_tx = body_q[body_map[link_name]]
        side_points: dict[str, np.ndarray] = {}
        for anchor_name in SUPPORT_CONTACT_ANCHORS:
            local_point = anchors[side].get(anchor_name)
            if local_point is None:
                continue
            point = _transform_point_xyzw(foot_tx, local_point)
            side_points[anchor_name] = point
            z_values.append(float(point[2]))
        if side_points:
            world_points[side] = side_points

    return np.asarray(z_values, dtype=np.float64), world_points


def _normalize_root_height(
    model,
    state,
    robot_builder,
    resolved_capability: dict[str, Any],
    anchors: dict[str, Any],
    q_values: np.ndarray,
) -> np.ndarray:
    q_values = np.array(q_values, dtype=np.float32, copy=True)
    z_values, _ = _support_world_points(model, state, robot_builder, resolved_capability, anchors, q_values)
    if len(z_values) == 0:
        return q_values
    q_values[2] -= 0.5 * (float(np.min(z_values)) + float(np.max(z_values)))
    return q_values


def _single_coord_q_index(model, robot_builder, joint_name: str) -> int | None:
    joint_map = create_joint_name_to_q_index_map(model, robot_builder)
    if joint_name not in joint_map:
        return None
    joint_index = joint_map[joint_name]
    joint_q_start = model.joint_q_start.numpy()
    start = int(joint_q_start[joint_index])
    end = int(joint_q_start[joint_index + 1]) if joint_index + 1 < len(joint_q_start) else int(model.joint_coord_count)
    if end - start != 1:
        return None
    return start


def _joint_limit(model, q_index: int) -> tuple[float, float]:
    lower = model.joint_limit_lower.numpy()
    upper = model.joint_limit_upper.numpy()
    limit_offset = int(model.joint_coord_count) - len(lower)
    limit_index = q_index - limit_offset
    if 0 <= limit_index < len(lower):
        return float(lower[limit_index]), float(upper[limit_index])
    return -float("inf"), float("inf")


def _side_hip_fallback_names(side: str, joint_names: list[str]) -> list[str]:
    priority = [
        f"{side}_thigh_pitch_joint",
        f"{side}_thigh_roll_joint",
        f"{side}_thigh_yaw_joint",
        f"{side}_hip_pitch_joint",
        f"{side}_hip_roll_joint",
        f"{side}_hip_yaw_joint",
    ]
    found = [name for name in priority if name in joint_names]
    for name in sorted(joint_names):
        lowered = name.lower()
        if name in found:
            continue
        if side not in lowered:
            continue
        if not any(token in lowered for token in ("hip", "thigh")):
            continue
        if any(axis in lowered for axis in ("pitch", "roll", "yaw")):
            found.append(name)
    return found


def _select_tune_candidates(
    model,
    robot_builder,
    resolved_capability: dict[str, Any],
) -> tuple[list[str], bool]:
    joint_map = create_joint_name_to_q_index_map(model, robot_builder)
    joint_names = list(joint_map.keys())
    ankle = resolved_capability.get("ankle", {})
    candidate_names: list[str] = []
    used_hip_fallback = False

    def add_candidate(name: str | None) -> None:
        if not name or name in candidate_names:
            return
        if _single_coord_q_index(model, robot_builder, name) is None:
            return
        candidate_names.append(name)

    for side in ("left", "right"):
        side_ankles = []
        for field_name in (f"{side}_pitch", f"{side}_roll"):
            joint_name = ankle.get(field_name)
            if joint_name and _single_coord_q_index(model, robot_builder, joint_name) is not None:
                side_ankles.append(joint_name)
                add_candidate(joint_name)

        if len(side_ankles) < 2:
            used_hip_fallback = True
            for joint_name in _side_hip_fallback_names(side, joint_names):
                add_candidate(joint_name)

    return candidate_names, used_hip_fallback


def _evaluate_grounding(
    model,
    state,
    robot_builder,
    resolved_capability: dict[str, Any],
    anchors: dict[str, Any],
    q_values: np.ndarray,
) -> tuple[float, float, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    z_values, world_points = _support_world_points(
        model,
        state,
        robot_builder,
        resolved_capability,
        anchors,
        q_values,
    )
    if len(z_values) == 0:
        return float("inf"), float("inf"), z_values, world_points
    max_abs = float(np.max(np.abs(z_values)))
    rms = float(np.sqrt(np.mean(z_values * z_values)))
    return max_abs, rms, z_values, world_points


def _optimize_contact_joints(
    model,
    state,
    robot_builder,
    resolved_capability: dict[str, Any],
    anchors: dict[str, Any],
    q_start: np.ndarray,
    candidates: list[str],
    options: FootGroundingOptions,
) -> np.ndarray:
    if not options.enable_joint_tuning or not candidates:
        return q_start

    candidate_q_indices = {
        name: _single_coord_q_index(model, robot_builder, name)
        for name in candidates
    }
    candidate_q_indices = {
        name: q_index
        for name, q_index in candidate_q_indices.items()
        if q_index is not None
    }
    if not candidate_q_indices:
        return q_start

    reference_q = np.array(q_start, dtype=np.float32, copy=True)

    def objective(q_values: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        q_grounded = _normalize_root_height(
            model,
            state,
            robot_builder,
            resolved_capability,
            anchors,
            q_values,
        )
        max_abs, rms, _, _ = _evaluate_grounding(
            model,
            state,
            robot_builder,
            resolved_capability,
            anchors,
            q_grounded,
        )
        joint_delta = sum(
            float((q_grounded[q_index] - reference_q[q_index]) ** 2)
            for q_index in candidate_q_indices.values()
        )
        loss = max_abs + 0.25 * rms + options.joint_delta_penalty * joint_delta
        return loss, q_grounded, max_abs, rms

    best_loss, best_q, _, _ = objective(q_start)
    for step in options.descent_steps_rad:
        improved = True
        while improved:
            improved = False
            for joint_name, q_index in candidate_q_indices.items():
                limit_lower, limit_upper = _joint_limit(model, q_index)
                lower = max(limit_lower, float(reference_q[q_index] - options.max_joint_delta_rad))
                upper = min(limit_upper, float(reference_q[q_index] + options.max_joint_delta_rad))
                for sign in (-1.0, 1.0):
                    candidate = np.array(best_q, dtype=np.float32, copy=True)
                    candidate[q_index] = np.clip(candidate[q_index] + sign * step, lower, upper)
                    loss, grounded, _, _ = objective(candidate)
                    if loss + 1e-8 < best_loss:
                        best_loss = loss
                        best_q = grounded
                        improved = True
    return best_q


def _support_z_payload(world_points: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    return {
        side: {
            anchor_name: _round_float(float(point[2]))
            for anchor_name, point in side_points.items()
        }
        for side, side_points in world_points.items()
    }


def _payload_needs_update(
    original_payload: dict[str, Any],
    q_values: np.ndarray,
    tuned_joint_deltas: dict[str, float],
) -> bool:
    root_payload = original_payload.get("root_transform")
    if not isinstance(root_payload, dict):
        return True

    old_translation = np.asarray(root_payload.get("translation_m", []), dtype=np.float64)
    old_rotation = np.asarray(root_payload.get("rotation_xyzw", []), dtype=np.float64)
    if old_translation.shape != (3,) or old_rotation.shape != (4,):
        return True
    if float(np.max(np.abs(old_translation - q_values[0:3]))) > 1e-5:
        return True
    if float(np.max(np.abs(old_rotation - q_values[3:7]))) > 1e-5:
        return True
    if tuned_joint_deltas:
        return True
    return not isinstance(original_payload.get("auto_grounding"), dict)


def _apply_grounding_to_payload(
    original_payload: dict[str, Any],
    q_values: np.ndarray,
    model,
    robot_builder,
    candidate_joints: list[str],
    reference_q: np.ndarray,
    *,
    max_abs: float,
    rms: float,
    support_z: dict[str, dict[str, float]],
    options: FootGroundingOptions,
    used_hip_fallback: bool,
) -> tuple[dict[str, Any], bool, dict[str, float]]:
    joint_map = create_joint_name_to_q_index_map(model, robot_builder)
    joint_q_start = model.joint_q_start.numpy()
    tuned_joint_deltas: dict[str, float] = {}
    for joint_name in candidate_joints:
        if joint_name not in joint_map:
            continue
        q_index = int(joint_q_start[joint_map[joint_name]])
        delta = float(q_values[q_index] - reference_q[q_index])
        if abs(delta) > 1e-6:
            tuned_joint_deltas[joint_name] = _round_float(delta)

    changed = _payload_needs_update(original_payload, q_values, tuned_joint_deltas)
    auto_grounding = original_payload.get("auto_grounding", {})
    if not changed and isinstance(auto_grounding, dict):
        changed = (
            auto_grounding.get("support_anchor_z_m") != support_z
            or auto_grounding.get("candidate_joints") != candidate_joints
            or bool(auto_grounding.get("used_hip_fallback")) != bool(used_hip_fallback)
            or auto_grounding.get("max_joint_delta_rad") != _round_float(options.max_joint_delta_rad)
        )
    if not changed:
        return original_payload, False, tuned_joint_deltas

    payload = copy.deepcopy(original_payload)
    payload["root_transform"] = {
        "translation_m": [_round_float(value) for value in q_values[0:3]],
        "rotation_xyzw": [_round_float(value) for value in q_values[3:7]],
    }
    joint_positions = payload.setdefault("joint_positions_rad", {})
    for joint_name in tuned_joint_deltas:
        q_index = int(joint_q_start[joint_map[joint_name]])
        joint_positions[joint_name] = _round_float(float(q_values[q_index]))

    payload["auto_grounding"] = {
        "method": "support_anchor_midrange_height_plus_limited_joint_coordinate_descent",
        "support_anchors": list(SUPPORT_CONTACT_ANCHORS),
        "candidate_joints": candidate_joints,
        "tuned_joints": list(tuned_joint_deltas.keys()),
        "used_hip_fallback": used_hip_fallback,
        "max_abs_support_anchor_z_m": _round_float(max_abs),
        "rms_support_anchor_z_m": _round_float(rms),
        "max_joint_delta_rad": _round_float(options.max_joint_delta_rad),
        "contact_threshold_m": _round_float(options.contact_threshold_m),
        "contact_success": max_abs <= options.contact_threshold_m,
        "support_anchor_z_m": support_z,
    }
    return payload, True, tuned_joint_deltas


def ground_robot_pose_payload(
    robot_name: str,
    payload: dict[str, Any],
    *,
    options: FootGroundingOptions | None = None,
    profile: dict[str, Any] | None = None,
) -> FootGroundingResult:
    """Return a robot pose payload with root height and optional leg joints adjusted for foot contact."""

    options = options or FootGroundingOptions()
    if profile is None:
        profile, _ = load_capability_profile(robot_name)
    resolution = resolve_capability_profile(robot_name, profile=profile)
    resolution.raise_for_errors()
    anchors = generate_virtual_sole_anchors(robot_name, resolution.payload, profile=profile)
    if not anchors.get("enabled"):
        return FootGroundingResult(
            payload=payload,
            changed=False,
            contact_success=False,
            max_abs_support_anchor_z_m=float("inf"),
            rms_support_anchor_z_m=float("inf"),
            messages=["Virtual sole anchors are disabled; foot grounding was skipped."],
        )

    robot_builder = newton.ModelBuilder()
    robot_builder.add_mjcf(str(pipeline_utils.get_robot_mjcf_path(robot_name)))
    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder, wp.transform_identity())
    builder.add_ground_plane()
    model = builder.finalize()
    state = model.state()

    q_initial, _ = apply_robot_pose_to_model(model, robot_builder, payload)
    q_initial = np.asarray(q_initial, dtype=np.float32)
    q_grounded = _normalize_root_height(
        model,
        state,
        robot_builder,
        resolution.payload,
        anchors,
        q_initial,
    )
    max_abs, rms, _, world_points = _evaluate_grounding(
        model,
        state,
        robot_builder,
        resolution.payload,
        anchors,
        q_grounded,
    )
    candidates, used_hip_fallback = _select_tune_candidates(model, robot_builder, resolution.payload)

    q_final = q_grounded
    if max_abs > options.contact_threshold_m:
        q_final = _optimize_contact_joints(
            model,
            state,
            robot_builder,
            resolution.payload,
            anchors,
            q_grounded,
            candidates,
            options,
        )
        max_abs, rms, _, world_points = _evaluate_grounding(
            model,
            state,
            robot_builder,
            resolution.payload,
            anchors,
            q_final,
        )

    support_z = _support_z_payload(world_points)
    grounded_payload, changed, tuned_joint_deltas = _apply_grounding_to_payload(
        payload,
        q_final,
        model,
        robot_builder,
        candidates,
        q_initial,
        max_abs=max_abs,
        rms=rms,
        support_z=support_z,
        options=options,
        used_hip_fallback=used_hip_fallback,
    )

    messages = []
    if used_hip_fallback:
        messages.append("One or both feet have fewer than two ankle DoFs; hip/thigh joints were enabled as fallback tuners.")
    if max_abs > options.contact_threshold_m:
        messages.append(
            f"Foot grounding did not fully reach threshold: max_abs_z={max_abs:.5f} m, "
            f"threshold={options.contact_threshold_m:.5f} m."
        )

    return FootGroundingResult(
        payload=grounded_payload,
        changed=changed,
        contact_success=max_abs <= options.contact_threshold_m,
        max_abs_support_anchor_z_m=max_abs,
        rms_support_anchor_z_m=rms,
        tuned_joint_deltas_rad=tuned_joint_deltas,
        candidate_joints=candidates,
        used_hip_fallback=used_hip_fallback,
        support_anchor_z_m=support_z,
        messages=messages,
    )
