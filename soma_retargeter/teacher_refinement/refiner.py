# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import soma_retargeter.robot_registry_parser as robot_registry_parser
from soma_retargeter.foot_anchors import generate_virtual_sole_anchors
from soma_retargeter.teacher_refinement.accept_reject import decide_acceptance
from soma_retargeter.teacher_refinement.capability_loader import load_capability_profile
from soma_retargeter.teacher_refinement.capability_resolver import resolve_capability_profile
from soma_retargeter.teacher_refinement.evaluator import evaluate_base_and_refined
from soma_retargeter.teacher_refinement.report import build_refine_report, save_debug_json
from soma_retargeter.teacher_refinement.tracking_templates import apply_tracking_template
from soma_retargeter.utils import io_utils


def _raw_retargeter_path(robot_name: str, retargeter_config_path: str | Path | None) -> Path:
    if retargeter_config_path is not None:
        return io_utils.resolve_path(retargeter_config_path)
    path = robot_registry_parser.get_profile_path(robot_name, "retargeter_config")
    if path is None:
        raise FileNotFoundError(f"Retargeter config is not registered for robot: {robot_name}")
    return Path(path)


def refine_registered_robot_config(
    robot_name: str,
    *,
    teacher: str = "unitree_g1",
    retargeter_config_path: str | Path | None = None,
    capability_profile_path: str | Path | None = None,
    output_path: str | Path | None = None,
    force_accept: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Run the deprecated teacher-guided refinement flow for a registered robot."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    teacher = robot_registry_parser.resolve_robot_name(teacher)
    config_path = _raw_retargeter_path(robot_name, retargeter_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Retargeter config not found: {config_path}")

    base_config = io_utils.load_json(config_path)
    final_path = io_utils.resolve_path(output_path) if output_path is not None else config_path

    if robot_name == teacher and not force_accept:
        report = {
            "robot": robot_name,
            "teacher": teacher,
            "template": "teacher_identity",
            "decision": "keep_base",
            "accepted": False,
            "reason": "target robot is the teacher robot; no teacher-guided refinement is needed",
        }
        if write and output_path is not None:
            io_utils.save_json(final_path, base_config, indent=4, ensure_ascii=False)
        return {
            "robot": robot_name,
            "teacher": teacher,
            "retargeter_config": str(config_path),
            "output_path": str(final_path),
            "accepted": False,
            "decision": "keep_base",
            "reason": report["reason"],
            "template": "teacher_identity",
            "resolved_capability": {},
            "generated_sole_anchors": {},
            "base_metrics": {},
            "refined_metrics": {},
            "report": report,
        }

    capability_profile, loaded_profile_path = load_capability_profile(robot_name, capability_profile_path)
    resolution = resolve_capability_profile(
        robot_name,
        profile=capability_profile,
        profile_path=str(loaded_profile_path) if loaded_profile_path else None,
    )
    resolution.raise_for_errors()
    resolved = resolution.payload

    generated_anchors = generate_virtual_sole_anchors(
        robot_name,
        resolved,
        profile=capability_profile,
    )
    template_name = str(resolved["priority_template"])
    refined_config = apply_tracking_template(
        base_config,
        template_name,
        resolved_capability=resolved,
        virtual_sole_anchors=generated_anchors,
    )
    refined_config["teacher_refinement"] = {
        "schema_version": 1,
        "teacher": teacher,
        "teacher_role": "canonical_motion_quality_teacher_not_joint_angle_source",
        "capability_profile": str(loaded_profile_path) if loaded_profile_path else "auto",
        "priority_template": template_name,
        "validation_mode": "config_static_proxy",
    }

    base_metrics, refined_metrics = evaluate_base_and_refined(base_config, refined_config, resolved)
    decision = decide_acceptance(base_metrics, refined_metrics, force_accept=force_accept)

    if write:
        save_debug_json(config_path, "base_autoconfig.json", base_config)
        save_debug_json(config_path, "resolved_capability.json", resolved)
        save_debug_json(config_path, "generated_sole_anchors.json", generated_anchors)
        save_debug_json(config_path, "base_eval_report.json", base_metrics.to_dict())
        save_debug_json(config_path, "refined_eval_report.json", refined_metrics.to_dict())
        save_debug_json(config_path, "g1_teacher_refined.json", refined_config)
        report = build_refine_report(
            robot_name=robot_name,
            teacher=teacher,
            template_name=template_name,
            decision=decision,
            base_metrics=base_metrics.to_dict(),
            refined_metrics=refined_metrics.to_dict(),
            resolved_capability=resolved,
            generated_sole_anchors=generated_anchors,
        )
        save_debug_json(config_path, "refine_report.json", report)

        if decision["accepted"]:
            io_utils.save_json(final_path, refined_config, indent=4, ensure_ascii=False)
        elif output_path is not None:
            io_utils.save_json(final_path, base_config, indent=4, ensure_ascii=False)
    else:
        report = build_refine_report(
            robot_name=robot_name,
            teacher=teacher,
            template_name=template_name,
            decision=decision,
            base_metrics=base_metrics.to_dict(),
            refined_metrics=refined_metrics.to_dict(),
            resolved_capability=resolved,
            generated_sole_anchors=generated_anchors,
        )

    return {
        "robot": robot_name,
        "teacher": teacher,
        "retargeter_config": str(config_path),
        "output_path": str(final_path),
        "accepted": bool(decision["accepted"]),
        "decision": decision["decision"],
        "reason": decision["reason"],
        "template": template_name,
        "resolved_capability": resolved,
        "generated_sole_anchors": generated_anchors,
        "base_metrics": base_metrics.to_dict(),
        "refined_metrics": refined_metrics.to_dict(),
        "report": report,
    }
