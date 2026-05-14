# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RefinementMetrics:
    evaluation_mode: str
    hard_gate_passed: bool
    total_score: float
    foot_score: float
    contact_score: float
    body_teacher_score: float
    smoothness_score: float
    joint_limit_score: float
    robot_feasibility_score: float
    foot_slip: float
    foot_penetration: float
    joint_limit_violation: float
    ik_fail_frames: int = 0
    severe_jitter: float = 0.0
    root_pelvis_drift: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def weighted_total(
    *,
    foot_score: float,
    contact_score: float,
    body_teacher_score: float,
    smoothness_score: float,
    joint_limit_score: float,
    robot_feasibility_score: float,
) -> float:
    return (
        0.35 * foot_score
        + 0.20 * contact_score
        + 0.15 * body_teacher_score
        + 0.10 * smoothness_score
        + 0.10 * joint_limit_score
        + 0.10 * robot_feasibility_score
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
