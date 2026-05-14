# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import soma_retargeter.robot_registry_parser as robot_registry_parser
from soma_retargeter.teacher_refinement.capability_loader import load_capability_profile
from soma_retargeter.teacher_refinement.capability_schema import (
    ANKLE_FIELDS,
    AUTO,
    LINK_FIELDS,
    TOE_FIELDS,
    WAIST_FIELDS,
    CapabilityResolution,
    is_auto,
    is_false,
)
from soma_retargeter.teacher_refinement.model_introspection import (
    RobotModelInfo,
    axis_matches,
    candidate_by_name,
    load_robot_model_info,
    sorted_names,
)


def _section(profile: dict[str, Any], name: str) -> dict[str, Any]:
    value = profile.get(name, {})
    return value if isinstance(value, dict) else {}


def _normalize_declared_value(value: Any) -> Any:
    if value is None:
        return AUTO
    if isinstance(value, str):
        value = value.strip()
        return value if value else AUTO
    return value


def _find_joint_candidate(info: RobotModelInfo, key: str) -> str | None:
    names = sorted_names(info.joints.keys())
    joint_items = list(info.joints.values())

    if key == "waist.yaw":
        by_name = candidate_by_name(names, [("waist", "yaw"), ("torso", "yaw"), ("torso",), ("waist",)])
        if by_name:
            return by_name
        for joint in joint_items:
            if ("waist" in joint.name.lower() or "torso" in joint.name.lower()) and axis_matches(joint.axis, "z"):
                return joint.name
    if key == "waist.pitch":
        by_name = candidate_by_name(names, [("waist", "pitch"), ("torso", "pitch")])
        if by_name:
            return by_name
    if key == "waist.roll":
        by_name = candidate_by_name(names, [("waist", "roll"), ("torso", "roll")])
        if by_name:
            return by_name

    side_tokens = {
        "ankle.left_pitch": ("left", "ankle", "pitch"),
        "ankle.left_roll": ("left", "ankle", "roll"),
        "ankle.right_pitch": ("right", "ankle", "pitch"),
        "ankle.right_roll": ("right", "ankle", "roll"),
        "toe.left": ("left", "toe"),
        "toe.right": ("right", "toe"),
    }
    if key in side_tokens:
        return candidate_by_name(names, [side_tokens[key]])
    return None


def _find_link_candidate(info: RobotModelInfo, key: str) -> str | None:
    names = sorted_names(info.links)
    predicates = {
        "pelvis": [("pelvis",), ("base_link",), ("base",), ("root",)],
        "chest": [("chest",), ("torso",), ("trunk",)],
        "left_foot": [("left", "foot"), ("left", "sole"), ("left", "ankle", "roll"), ("left", "toe")],
        "right_foot": [("right", "foot"), ("right", "sole"), ("right", "ankle", "roll"), ("right", "toe")],
        "left_hand": [("left", "hand"), ("left", "wrist"), ("left", "gripper"), ("left", "elbow", "yaw")],
        "right_hand": [("right", "hand"), ("right", "wrist"), ("right", "gripper"), ("right", "elbow", "yaw")],
    }
    return candidate_by_name(names, predicates.get(key, []))


def _resolve_named_value(
    *,
    declared: Any,
    allowed_names: set[str],
    auto_candidate: str | None,
    label: str,
    errors: list[str],
    warnings: list[str],
) -> str | None:
    declared = _normalize_declared_value(declared)
    if is_false(declared):
        return None
    if is_auto(declared):
        if auto_candidate is None:
            warnings.append(f"{label} could not be inferred; it will be treated as unavailable.")
        return auto_candidate
    if not isinstance(declared, str):
        errors.append(f"{label} must be a string, false, or auto.")
        return None
    if declared not in allowed_names:
        errors.append(f"{label} declares {declared!r}, but that name does not exist in the robot model.")
        return None
    return declared


def _dof_class(dof_count: int) -> str:
    if dof_count >= 28:
        return "full_29dof_like"
    if dof_count >= 20:
        return "mid_dof"
    return "low_dof"


def _waist_type(waist: dict[str, str | None]) -> str:
    yaw = bool(waist.get("yaw"))
    pitch = bool(waist.get("pitch"))
    roll = bool(waist.get("roll"))
    if yaw and pitch and roll:
        return "full"
    if yaw and not pitch and not roll:
        return "yaw_only"
    if not yaw and not pitch and not roll:
        return "none"
    return "partial"


def _ankle_type(ankle: dict[str, str | None]) -> str:
    has_pitch = bool(ankle.get("left_pitch") and ankle.get("right_pitch"))
    has_roll = bool(ankle.get("left_roll") and ankle.get("right_roll"))
    if has_pitch and has_roll:
        return "pitch_roll"
    if has_pitch:
        return "pitch_only"
    if has_roll:
        return "roll_only"
    return "none"


def _toe_type(toe: dict[str, str | None]) -> str:
    has_left = bool(toe.get("left"))
    has_right = bool(toe.get("right"))
    if has_left and has_right:
        return "toe_joints"
    if has_left or has_right:
        return "partial"
    return "none"


def _select_template(
    *,
    dof_class: str,
    waist_type: str,
    ankle_type: str,
    requested_mode: Any,
) -> str:
    if isinstance(requested_mode, str) and requested_mode.strip().lower() not in {"", AUTO}:
        mode = requested_mode.strip()
        mode_to_template = {
            "full_body": "normal_29dof_template",
            "low_dof_safe": "low_dof_yaw_only_waist_template",
            "foot_priority": "foot_priority_template",
            "normal_29dof_template": "normal_29dof_template",
            "low_dof_yaw_only_waist_template": "low_dof_yaw_only_waist_template",
            "no_waist_template": "no_waist_template",
            "simplified_ankle_template": "simplified_ankle_template",
            "foot_priority_template": "foot_priority_template",
        }
        return mode_to_template.get(mode, mode)

    if waist_type == "none":
        return "no_waist_template"
    if waist_type == "yaw_only":
        return "low_dof_yaw_only_waist_template"
    if ankle_type != "pitch_roll":
        return "simplified_ankle_template"
    if dof_class == "full_29dof_like":
        return "normal_29dof_template"
    return "foot_priority_template"


def resolve_capability_profile(
    robot_name: str,
    *,
    profile: dict[str, Any] | None = None,
    profile_path: str | None = None,
) -> CapabilityResolution:
    """Resolve user declarations and auto fields into an internal profile."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    loaded_path = None
    if profile is None:
        profile, loaded_path = load_capability_profile(robot_name, profile_path)
    elif profile_path is not None:
        loaded_path = profile_path

    info = load_robot_model_info(robot_name)
    warnings: list[str] = []
    errors: list[str] = []

    if not info.links:
        warnings.append("No robot links were discovered; capability inference will be limited.")
    if not info.joints:
        warnings.append("No robot joints were discovered; DoF and capability inference will be limited.")

    waist_raw = _section(profile, "waist")
    ankle_raw = _section(profile, "ankle")
    toe_raw = _section(profile, "toe")
    links_raw = _section(profile, "links")

    waist = {
        field_name: _resolve_named_value(
            declared=waist_raw.get(field_name, AUTO),
            allowed_names=set(info.joints.keys()),
            auto_candidate=_find_joint_candidate(info, f"waist.{field_name}"),
            label=f"waist.{field_name}",
            errors=errors,
            warnings=warnings,
        )
        for field_name in WAIST_FIELDS
    }
    ankle = {
        field_name: _resolve_named_value(
            declared=ankle_raw.get(field_name, AUTO),
            allowed_names=set(info.joints.keys()),
            auto_candidate=_find_joint_candidate(info, f"ankle.{field_name}"),
            label=f"ankle.{field_name}",
            errors=errors,
            warnings=warnings,
        )
        for field_name in ANKLE_FIELDS
    }
    toe = {
        field_name: _resolve_named_value(
            declared=toe_raw.get(field_name, AUTO),
            allowed_names=set(info.joints.keys()),
            auto_candidate=_find_joint_candidate(info, f"toe.{field_name}"),
            label=f"toe.{field_name}",
            errors=errors,
            warnings=warnings,
        )
        for field_name in TOE_FIELDS
    }
    links = {
        field_name: _resolve_named_value(
            declared=links_raw.get(field_name, AUTO),
            allowed_names=set(info.links),
            auto_candidate=_find_link_candidate(info, field_name),
            label=f"links.{field_name}",
            errors=errors,
            warnings=warnings,
        )
        for field_name in LINK_FIELDS
    }

    dof_count = info.actuated_dof_count
    dof_class = _dof_class(dof_count)
    waist_kind = _waist_type(waist)
    ankle_kind = _ankle_type(ankle)
    toe_kind = _toe_type(toe)
    tracking_policy = _section(profile, "tracking_policy")
    priority_template = _select_template(
        dof_class=dof_class,
        waist_type=waist_kind,
        ankle_type=ankle_kind,
        requested_mode=tracking_policy.get("mode", AUTO),
    )

    foot_raw = _section(profile, "foot")
    anchor_mode = foot_raw.get("sole_anchor_mode", AUTO)
    if is_auto(anchor_mode):
        anchor_mode = "bbox" if links.get("left_foot") or links.get("right_foot") else "disabled"

    payload = {
        "schema_version": 1,
        "robot": robot_name,
        "profile_source": str(loaded_path) if loaded_path is not None else "auto",
        "dof_count": dof_count,
        "dof_class": dof_class,
        "waist": waist,
        "waist_type": waist_kind,
        "ankle": ankle,
        "ankle_type": ankle_kind,
        "toe": toe,
        "toe_type": toe_kind,
        "links": links,
        "has_chest_link": bool(links.get("chest")),
        "has_hand_links": bool(links.get("left_hand") and links.get("right_hand")),
        "foot_anchor_mode": str(anchor_mode),
        "torso_tracking_mode": "yaw_only_or_weak" if waist_kind in {"yaw_only", "none"} else "full_or_medium",
        "priority_template": priority_template,
        "model_sources": info.source_paths,
        "warnings": warnings,
    }
    return CapabilityResolution(payload=payload, warnings=warnings, errors=errors)
