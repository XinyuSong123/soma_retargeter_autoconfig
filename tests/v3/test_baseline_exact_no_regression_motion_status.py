from __future__ import annotations

from copy import deepcopy

from soma_retargeter.robotics.v3.capability_status import PROFILE_PASSED, evaluate_profile_status
from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES


def _exact_task() -> dict:
    return {
        "status": "converged",
        "converged": True,
        "normalized_residual": 0.0,
        "residual": 0.0,
        "active_coordinates": [0, 1],
        "capability_certificate": {
            "certificate_class": "exact_reachable",
            "gates": {
                "exact_threshold_passed": True,
                "primal_feasible": True,
                "projected_gradient_kkt": True,
                "seed_consensus": True,
                "continuation": True,
                "joint_limits": True,
                "numerical": True,
                "residual_explained": True,
            },
            "decomposition": {
                "rank": 3,
                "residual_norm": 0.0,
                "demand_norm": 0.0,
                "rank_incompatible_fraction": 0.0,
                "active_limit_fraction": 0.0,
            },
            "active_limits": {"lower": [], "upper": [], "count": 0},
            "seed_consensus": {"checked": True, "passed": True, "start_count": 4},
        },
    }


def _stress_limited_task() -> dict:
    task = _exact_task()
    task["status"] = "converged/with_residual"
    task["normalized_residual"] = 0.9
    task["residual"] = 0.9
    task["capability_certificate"] = {
        "certificate_class": "capability_limited_rank",
        "gates": {
            "exact_threshold_passed": False,
            "primal_feasible": True,
            "projected_gradient_kkt": True,
            "seed_consensus": True,
            "continuation": True,
            "joint_limits": True,
            "numerical": True,
            "residual_explained": True,
        },
        "decomposition": {
            "rank": 2,
            "residual_norm": 0.9,
            "demand_norm": 0.9,
            "rank_incompatible_fraction": 0.99,
            "active_limit_fraction": 0.0,
            "rank_incompatible_residual_norm": 0.891,
            "active_limit_residual_norm": 0.0,
            "component_tolerance": 1e-7,
        },
        "active_limits": {"lower": [], "upper": [], "count": 0},
        "seed_consensus": {"checked": True, "passed": True, "start_count": 4},
    }
    return task


def test_exact_required_motion_matrix_is_not_downgraded_by_stress_diagnostics() -> None:
    required_tasks = ("torso", "left_hand", "right_hand", "left_foot", "right_foot")
    exact_tasks = {task: _exact_task() for task in required_tasks}
    motions = {
        motion: {"tasks": deepcopy(exact_tasks)}
        for motion in CANONICAL_MOTION_NAMES
    }
    motions["extreme_but_valid_joint_limit_stress"]["tasks"]["left_hand"] = _stress_limited_task()

    result = evaluate_profile_status(
        {"motion_order": list(CANONICAL_MOTION_NAMES), "motions": motions},
        required_tasks=required_tasks,
    )

    assert result.status == PROFILE_PASSED
    assert result.failures == ()
