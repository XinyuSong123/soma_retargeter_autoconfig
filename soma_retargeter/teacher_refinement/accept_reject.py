# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from soma_retargeter.teacher_refinement.metrics import RefinementMetrics


def decide_acceptance(
    base: RefinementMetrics,
    refined: RefinementMetrics,
    *,
    force_accept: bool = False,
) -> dict:
    if force_accept:
        return {
            "accepted": True,
            "decision": "use_teacher_refined",
            "reason": "forced by caller",
        }

    checks = {
        "hard_gate_passed": refined.hard_gate_passed,
        "total_score_improved_5pct": refined.total_score > base.total_score * 1.05,
        "foot_score_improved_8pct": refined.foot_score > base.foot_score * 1.08,
        "foot_slip_not_worse": refined.foot_slip <= base.foot_slip * 1.02,
        "joint_limits_not_worse": refined.joint_limit_violation <= base.joint_limit_violation * 1.02,
    }
    accepted = all(checks.values())
    if accepted:
        reason = (
            f"refined score improved from {base.total_score:.3f} to {refined.total_score:.3f} "
            f"and foot score improved from {base.foot_score:.3f} to {refined.foot_score:.3f}"
        )
    else:
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        reason = f"kept base config because acceptance checks failed: {failed}"
    return {
        "accepted": accepted,
        "decision": "use_teacher_refined" if accepted else "keep_base",
        "reason": reason,
        "checks": checks,
    }
