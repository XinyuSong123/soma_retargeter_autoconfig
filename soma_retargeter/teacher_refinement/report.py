# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from soma_retargeter.utils import io_utils


def debug_dir_for_config(config_path: str | Path) -> Path:
    return Path(config_path).resolve().parent / "debug"


def save_debug_json(config_path: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    return io_utils.save_json(debug_dir_for_config(config_path) / filename, payload, indent=4, ensure_ascii=False)


def build_refine_report(
    *,
    robot_name: str,
    teacher: str,
    template_name: str,
    decision: dict[str, Any],
    base_metrics: dict[str, Any],
    refined_metrics: dict[str, Any],
    resolved_capability: dict[str, Any],
    generated_sole_anchors: dict[str, Any],
) -> dict[str, Any]:
    base_score = float(base_metrics["total_score"])
    refined_score = float(refined_metrics["total_score"])
    improvement = (refined_score - base_score) / base_score if base_score > 0 else 0.0
    return {
        "robot": robot_name,
        "teacher": teacher,
        "template": template_name,
        "decision": decision["decision"],
        "accepted": decision["accepted"],
        "reason": decision["reason"],
        "base_score": base_score,
        "refined_score": refined_score,
        "improvement": improvement,
        "evaluation_mode": refined_metrics.get("evaluation_mode"),
        "acceptance_checks": decision.get("checks", {}),
        "metrics": {
            "foot_slip_m": {
                "base": base_metrics["foot_slip"],
                "refined": refined_metrics["foot_slip"],
            },
            "foot_penetration_m": {
                "base": base_metrics["foot_penetration"],
                "refined": refined_metrics["foot_penetration"],
            },
            "joint_limit_violation": {
                "base": base_metrics["joint_limit_violation"],
                "refined": refined_metrics["joint_limit_violation"],
            },
            "foot_score": {
                "base": base_metrics["foot_score"],
                "refined": refined_metrics["foot_score"],
            },
            "smoothness": {
                "base": base_metrics["smoothness_score"],
                "refined": refined_metrics["smoothness_score"],
            },
            "body_teacher_score": {
                "base": base_metrics["body_teacher_score"],
                "refined": refined_metrics["body_teacher_score"],
            },
        },
        "resolved_capability": resolved_capability,
        "generated_sole_anchors": generated_sole_anchors,
        "notes": [
            "The MVP uses capability-aware templates and a config-level validation proxy when teacher motion caches are absent.",
            "G1 joint angles are not copied into the target robot config.",
        ],
    }
