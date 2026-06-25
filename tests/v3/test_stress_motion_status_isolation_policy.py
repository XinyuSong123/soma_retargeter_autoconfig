from __future__ import annotations

from copy import deepcopy

from soma_retargeter.robotics.v3.capability_status import PROFILE_PASSED, evaluate_profile_status
from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES


REQUIRED_TASKS = ("torso", "left_hand")


def _certificate(certificate_class: str) -> dict:
    return {
        "certificate_class": certificate_class,
        "gates": {
            "exact_threshold_passed": certificate_class == "exact_reachable",
            "primal_feasible": True,
            "projected_gradient_kkt": True,
            "seed_consensus": True,
            "continuation": True,
            "joint_limits": True,
            "numerical": True,
            "residual_explained": True,
        },
        "decomposition": {
            "rank": 2 if certificate_class != "unsupported_rank_zero" else 0,
            "residual_norm": 0.2,
            "demand_norm": 0.2,
            "rank_incompatible_fraction": 0.98,
            "active_limit_fraction": 0.0,
            "rank_incompatible_residual_norm": 0.196,
            "active_limit_residual_norm": 0.0,
            "component_tolerance": 1e-7,
        },
        "active_limits": {"lower": [], "upper": [], "count": 0},
        "seed_consensus": {"checked": True, "passed": True, "start_count": 4},
    }


def _task(normalized_residual: float, certificate_class: str = "exact_reachable") -> dict:
    return {
        "status": "converged" if certificate_class == "exact_reachable" else "converged/with_residual",
        "converged": certificate_class != "solver_failed",
        "normalized_residual": normalized_residual,
        "residual": normalized_residual,
        "active_coordinates": [0, 1],
        "capability_certificate": _certificate(certificate_class),
    }


def _projection(stress_task: dict) -> dict:
    exact_tasks = {task: _task(0.0) for task in REQUIRED_TASKS}
    motions = {
        motion: {"tasks": deepcopy(exact_tasks)}
        for motion in CANONICAL_MOTION_NAMES
    }
    motions["extreme_but_valid_joint_limit_stress"]["tasks"]["left_hand"] = stress_task
    return {
        "motion_order": list(CANONICAL_MOTION_NAMES),
        "motions": motions,
    }


def test_stress_only_limited_certificate_does_not_downgrade_exact_profile() -> None:
    result = evaluate_profile_status(
        _projection(_task(0.9, "capability_limited_rank")),
        required_tasks=REQUIRED_TASKS,
    )

    assert result.status == PROFILE_PASSED
    assert result.failures == ()


def test_stress_only_failed_certificate_does_not_downgrade_exact_profile() -> None:
    result = evaluate_profile_status(
        _projection(_task(0.9, "solver_failed")),
        required_tasks=REQUIRED_TASKS,
    )

    assert result.status == PROFILE_PASSED
    assert result.failures == ()
