# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any


_BASE_WEIGHTS = {
    "Hips": (10.0, 2.0),
    "Chest": (0.5, 0.5),
    "LeftArm": (0.5, 0.1),
    "RightArm": (0.5, 0.1),
    "LeftForeArm": (1.0, 0.5),
    "RightForeArm": (1.0, 0.5),
    "LeftHand": (1.0, 0.5),
    "RightHand": (1.0, 0.5),
    "LeftLeg": (0.5, 0.2),
    "RightLeg": (0.5, 0.2),
    "LeftShin": (1.0, 0.5),
    "RightShin": (1.0, 0.5),
    "LeftFoot": (2.0, 1.0),
    "RightFoot": (2.0, 1.0),
}


_TEMPLATES: dict[str, dict[str, Any]] = {
    "normal_29dof_template": {
        "ik_weights": {
            **_BASE_WEIGHTS,
            "Chest": (0.75, 0.65),
            "LeftHand": (1.1, 0.45),
            "RightHand": (1.1, 0.45),
            "LeftFoot": (3.0, 1.35),
            "RightFoot": (3.0, 1.35),
        },
        "runtime": {
            "ik_iterations": 28,
            "joint_limit_weight": 11.0,
            "smooth_joint_filter_weight": 6.0,
            "enable_post_processing": False,
        },
    },
    "low_dof_yaw_only_waist_template": {
        "ik_weights": {
            **_BASE_WEIGHTS,
            "Hips": (11.0, 1.8),
            "Chest": (0.45, 0.12),
            "LeftArm": (0.45, 0.05),
            "RightArm": (0.45, 0.05),
            "LeftForeArm": (0.9, 0.18),
            "RightForeArm": (0.9, 0.18),
            "LeftHand": (0.95, 0.12),
            "RightHand": (0.95, 0.12),
            "LeftFoot": (3.6, 1.15),
            "RightFoot": (3.6, 1.15),
        },
        "runtime": {
            "ik_iterations": 30,
            "joint_limit_weight": 14.0,
            "smooth_joint_filter_weight": 7.0,
            "enable_post_processing": False,
        },
    },
    "no_waist_template": {
        "ik_weights": {
            **_BASE_WEIGHTS,
            "Hips": (11.0, 1.5),
            "Chest": (0.25, 0.05),
            "LeftArm": (0.4, 0.03),
            "RightArm": (0.4, 0.03),
            "LeftForeArm": (0.8, 0.12),
            "RightForeArm": (0.8, 0.12),
            "LeftHand": (0.85, 0.08),
            "RightHand": (0.85, 0.08),
            "LeftFoot": (3.8, 0.95),
            "RightFoot": (3.8, 0.95),
        },
        "runtime": {
            "ik_iterations": 32,
            "joint_limit_weight": 15.0,
            "smooth_joint_filter_weight": 7.5,
            "enable_post_processing": False,
        },
    },
    "simplified_ankle_template": {
        "ik_weights": {
            **_BASE_WEIGHTS,
            "Hips": (10.5, 1.6),
            "Chest": (0.45, 0.22),
            "LeftFoot": (3.7, 0.45),
            "RightFoot": (3.7, 0.45),
            "LeftShin": (1.2, 0.45),
            "RightShin": (1.2, 0.45),
        },
        "runtime": {
            "ik_iterations": 30,
            "joint_limit_weight": 15.0,
            "smooth_joint_filter_weight": 7.5,
            "enable_post_processing": False,
        },
    },
    "foot_priority_template": {
        "ik_weights": {
            **_BASE_WEIGHTS,
            "Hips": (10.8, 1.8),
            "Chest": (0.45, 0.25),
            "LeftHand": (0.9, 0.18),
            "RightHand": (0.9, 0.18),
            "LeftFoot": (3.9, 1.05),
            "RightFoot": (3.9, 1.05),
        },
        "runtime": {
            "ik_iterations": 30,
            "joint_limit_weight": 13.5,
            "smooth_joint_filter_weight": 7.0,
            "enable_post_processing": False,
        },
    },
}


def get_tracking_template(name: str) -> dict[str, Any]:
    if name not in _TEMPLATES:
        raise ValueError(f"Unknown tracking template: {name}")
    return deepcopy(_TEMPLATES[name])


def apply_tracking_template(
    raw_config: dict[str, Any],
    template_name: str,
    *,
    resolved_capability: dict[str, Any] | None = None,
    virtual_sole_anchors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a refined raw retargeter config with template IK weights."""

    template = get_tracking_template(template_name)
    ik_weights = template["ik_weights"]
    refined = deepcopy(raw_config)
    raw_ik_map = refined.get("ik_map", {})
    if not isinstance(raw_ik_map, dict):
        raw_ik_map = {}

    new_ik_map = {}
    for joint_name, entry in raw_ik_map.items():
        t_weight, r_weight = ik_weights.get(str(joint_name), (1.0, 0.5))
        if isinstance(entry, str):
            body = entry.strip()
            if not body:
                continue
            new_ik_map[str(joint_name)] = {
                "t_body": body,
                "r_body": body,
                "t_weight": t_weight,
                "r_weight": r_weight,
            }
        elif isinstance(entry, dict):
            t_body = entry.get("t_body") or entry.get("body") or entry.get("link")
            r_body = entry.get("r_body") or t_body
            if not t_body or not r_body:
                continue
            new_ik_map[str(joint_name)] = {
                **entry,
                "t_body": str(t_body),
                "r_body": str(r_body),
                "t_weight": float(t_weight),
                "r_weight": float(r_weight),
            }
    refined["ik_map"] = new_ik_map

    for key, value in template["runtime"].items():
        refined[key] = value

    if resolved_capability is not None:
        refined["resolved_capability_summary"] = {
            "dof_class": resolved_capability.get("dof_class"),
            "waist_type": resolved_capability.get("waist_type"),
            "ankle_type": resolved_capability.get("ankle_type"),
            "toe_type": resolved_capability.get("toe_type"),
            "priority_template": template_name,
        }
    if virtual_sole_anchors is not None:
        refined["virtual_sole_anchors"] = virtual_sole_anchors
    return refined
