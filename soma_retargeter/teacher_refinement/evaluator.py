# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from soma_retargeter.teacher_refinement.metrics import RefinementMetrics, clamp01, weighted_total


def _runtime_ik_entries(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    entries = {}
    for joint_name, entry in config.get("ik_map", {}).items():
        if isinstance(entry, str):
            default = {
                "Hips": (10.0, 2.0),
                "Chest": (0.5, 0.5),
                "LeftFoot": (2.0, 1.0),
                "RightFoot": (2.0, 1.0),
            }.get(str(joint_name), (1.0, 0.5))
            entries[str(joint_name)] = {"t_weight": default[0], "r_weight": default[1]}
        elif isinstance(entry, dict):
            entries[str(joint_name)] = {
                "t_weight": float(entry.get("t_weight", 1.0)),
                "r_weight": float(entry.get("r_weight", 0.5)),
            }
    return entries


def _mean_weight(entries: dict[str, dict[str, float]], joints: tuple[str, ...], key: str, default: float) -> float:
    values = [entries[joint][key] for joint in joints if joint in entries]
    return sum(values) / len(values) if values else default


def evaluate_config_static_proxy(
    config: dict[str, Any],
    resolved_capability: dict[str, Any],
) -> RefinementMetrics:
    """Score a config without motion rollout when teacher validation assets are unavailable.

    The result is intentionally labeled as a proxy in reports. It checks whether the
    config matches the capability-aware tracking policy: feet and pelvis should be
    strong, torso orientation should be weak for low-DoF waists, and joint-limit
    regularization should be present.
    """

    entries = _runtime_ik_entries(config)
    foot_t = _mean_weight(entries, ("LeftFoot", "RightFoot"), "t_weight", 2.0)
    foot_r = _mean_weight(entries, ("LeftFoot", "RightFoot"), "r_weight", 1.0)
    hip_t = _mean_weight(entries, ("Hips",), "t_weight", 10.0)
    chest_r = _mean_weight(entries, ("Chest",), "r_weight", 0.5)
    hand_r = _mean_weight(entries, ("LeftHand", "RightHand"), "r_weight", 0.5)
    smooth_weight = float(config.get("smooth_joint_filter_weight", 5.5))
    joint_limit_weight = float(config.get("joint_limit_weight", 10.0))
    waist_type = resolved_capability.get("waist_type")
    ankle_type = resolved_capability.get("ankle_type")

    foot_pos_score = clamp01(foot_t / 3.6)
    if ankle_type == "pitch_roll":
        foot_rot_score = clamp01(foot_r / 1.15)
        foot_overconstraint = max(0.0, foot_r - 1.5) * 0.08
    else:
        foot_rot_score = clamp01(1.0 - abs(foot_r - 0.45) / 1.0)
        foot_overconstraint = max(0.0, foot_r - 0.75) * 0.18

    foot_score = clamp01(0.72 * foot_pos_score + 0.28 * foot_rot_score - foot_overconstraint)
    anchor_bonus = 0.08 if config.get("virtual_sole_anchors", {}).get("enabled") else 0.0
    contact_score = clamp01(0.86 * foot_pos_score + anchor_bonus)

    if waist_type in {"yaw_only", "none"}:
        torso_score = clamp01(1.0 - max(0.0, chest_r - 0.18) * (1.4 if waist_type == "yaw_only" else 2.0))
    else:
        torso_score = clamp01(0.6 + min(chest_r, 0.65) * 0.5)
    pelvis_score = clamp01(hip_t / 11.0)
    hand_score = clamp01(1.0 - max(0.0, hand_r - (0.15 if waist_type in {"yaw_only", "none"} else 0.5)))
    body_teacher_score = clamp01(0.5 * pelvis_score + 0.35 * torso_score + 0.15 * hand_score)

    smoothness_score = clamp01(1.0 - abs(smooth_weight - 7.0) / 9.0)
    joint_limit_score = clamp01(1.0 - abs(joint_limit_weight - 14.0) / 18.0)
    feasibility_penalty = 0.0
    if waist_type in {"yaw_only", "none"}:
        feasibility_penalty += max(0.0, chest_r - 0.25) * 0.35
        feasibility_penalty += max(0.0, hand_r - 0.25) * 0.12
    feasibility_penalty += foot_overconstraint
    robot_feasibility_score = clamp01(0.92 + joint_limit_score * 0.08 - feasibility_penalty)

    total_score = weighted_total(
        foot_score=foot_score,
        contact_score=contact_score,
        body_teacher_score=body_teacher_score,
        smoothness_score=smoothness_score,
        joint_limit_score=joint_limit_score,
        robot_feasibility_score=robot_feasibility_score,
    )

    foot_slip = max(0.004, 0.080 * (1.0 - contact_score + 0.15))
    foot_penetration = max(0.001, 0.025 * (1.0 - foot_score + 0.10))
    joint_limit_violation = max(0.001, 0.040 * (1.0 - joint_limit_score + feasibility_penalty + 0.05))
    hard_gate_passed = bool(entries) and all(
        value >= 0.0
        for entry in entries.values()
        for value in (entry["t_weight"], entry["r_weight"])
    )

    return RefinementMetrics(
        evaluation_mode="config_static_proxy",
        hard_gate_passed=hard_gate_passed,
        total_score=float(total_score),
        foot_score=float(foot_score),
        contact_score=float(contact_score),
        body_teacher_score=float(body_teacher_score),
        smoothness_score=float(smoothness_score),
        joint_limit_score=float(joint_limit_score),
        robot_feasibility_score=float(robot_feasibility_score),
        foot_slip=float(foot_slip),
        foot_penetration=float(foot_penetration),
        joint_limit_violation=float(joint_limit_violation),
    )


def evaluate_base_and_refined(
    base_config: dict[str, Any],
    refined_config: dict[str, Any],
    resolved_capability: dict[str, Any],
) -> tuple[RefinementMetrics, RefinementMetrics]:
    return (
        evaluate_config_static_proxy(base_config, resolved_capability),
        evaluate_config_static_proxy(refined_config, resolved_capability),
    )
