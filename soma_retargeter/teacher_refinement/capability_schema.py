# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AUTO = "auto"

WAIST_FIELDS = ("yaw", "pitch", "roll")
ANKLE_FIELDS = ("left_pitch", "left_roll", "right_pitch", "right_roll")
TOE_FIELDS = ("left", "right")
LINK_FIELDS = ("pelvis", "chest", "left_foot", "right_foot", "left_hand", "right_hand")
ANCHOR_NAMES = ("sole_center", "toe", "heel", "inner_edge", "outer_edge")

TRACKING_TEMPLATES = (
    "normal_29dof_template",
    "low_dof_yaw_only_waist_template",
    "no_waist_template",
    "simplified_ankle_template",
    "foot_priority_template",
)


@dataclass
class CapabilityResolution:
    """Resolved robot capability profile plus validation messages."""

    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Invalid robot capability profile:\n- " + "\n- ".join(self.errors))


def default_capability_profile(robot_name: str) -> dict[str, Any]:
    """Return an all-auto profile used when no robot_capability file exists."""

    return {
        "schema_version": 1,
        "robot": robot_name,
        "capability_mode": "auto_inferred",
        "waist": {field_name: AUTO for field_name in WAIST_FIELDS},
        "ankle": {field_name: AUTO for field_name in ANKLE_FIELDS},
        "toe": {field_name: AUTO for field_name in TOE_FIELDS},
        "links": {field_name: AUTO for field_name in LINK_FIELDS},
        "foot": {
            "sole_anchor_mode": AUTO,
            "forward_axis": AUTO,
            "up_axis": AUTO,
            "manual_size": {
                "left": {"length": AUTO, "width": AUTO, "sole_z": AUTO},
                "right": {"length": AUTO, "width": AUTO, "sole_z": AUTO},
            },
        },
        "tracking_policy": {"mode": AUTO},
    }


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO


def is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() == "false")


def scalar_to_float(value: Any) -> float | None:
    if is_auto(value) or is_false(value) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
