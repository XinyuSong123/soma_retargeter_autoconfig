from __future__ import annotations

from copy import deepcopy

from soma_retargeter.robotics.v3.capability_status import (
    PROFILE_ALGORITHM_FAILED,
    PROFILE_CAPABILITY_LIMITED_PASSED,
    PROFILE_PASSED,
    evaluate_profile_status,
    profile_status_from_semantic_readiness,
)
from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES


REQUIRED_TASKS = ("torso", "left_hand")


def _gates(**overrides: bool) -> dict[str, bool]:
    gates = {
        "exact_threshold_passed": True,
        "primal_feasible": True,
        "projected_gradient_kkt": True,
        "seed_consensus": True,
        "continuation": True,
        "joint_limits": True,
        "numerical": True,
        "residual_explained": True,
    }
    gates.update(overrides)
    return gates


def _certificate(certificate_class: str, *, gates: dict[str, bool] | None = None) -> dict:
    decomposition = {
        "rank": 3,
        "residual_norm": 0.0,
        "demand_norm": 0.0,
        "rank_incompatible_fraction": 0.0,
        "active_limit_fraction": 0.0,
        "tangent_residual_norm": 0.0,
        "rank_incompatible_residual_norm": 0.0,
        "active_limit_residual_norm": 0.0,
        "component_tolerance": 1e-7,
    }
    active_limits = {"lower": [], "upper": [], "count": 0}
    if certificate_class == "capability_limited_rank":
        decomposition.update(
            {
                "rank": 2,
                "residual_norm": 0.2,
                "demand_norm": 0.2,
                "rank_incompatible_fraction": 0.98,
                "rank_incompatible_residual_norm": 0.196,
                "tangent_residual_norm": 0.004,
            }
        )
    elif certificate_class == "capability_limited_joint_limits":
        decomposition.update(
            {
                "rank": 3,
                "residual_norm": 0.2,
                "demand_norm": 0.2,
                "active_limit_fraction": 1.0,
                "active_limit_residual_norm": 0.2,
            }
        )
        active_limits = {"lower": [0], "upper": [], "count": 1}
    elif certificate_class == "capability_limited_mixed":
        decomposition.update(
            {
                "rank": 2,
                "residual_norm": 0.2,
                "demand_norm": 0.2,
                "rank_incompatible_fraction": 0.96,
                "active_limit_fraction": 0.6,
                "rank_incompatible_residual_norm": 0.192,
                "active_limit_residual_norm": 0.12,
            }
        )
        active_limits = {"lower": [], "upper": [1], "count": 1}
    elif certificate_class == "unsupported_rank_zero":
        decomposition.update(
            {
                "rank": 0,
                "residual_norm": 0.2,
                "demand_norm": 0.2,
                "rank_incompatible_fraction": 1.0,
                "rank_incompatible_residual_norm": 0.2,
            }
        )
    default_gates = _gates(exact_threshold_passed=certificate_class == "exact_reachable")
    return {
        "certificate_class": certificate_class,
        "gates": default_gates if gates is None else gates,
        "decomposition": decomposition,
        "active_limits": active_limits,
        "kkt": {"projected_gradient_norm": 1e-9, "tolerance": 1e-6},
        "seed_consensus": {"checked": True, "passed": True, "start_count": 4},
    }


def _task(normalized_residual: float, certificate_class: str = "exact_reachable", **overrides) -> dict:
    payload = {
        "status": "converged" if normalized_residual <= 1e-9 else "converged/with_residual",
        "converged": True,
        "normalized_residual": normalized_residual,
        "residual": normalized_residual,
        "active_coordinates": [0, 1, 2],
        "capability_certificate": _certificate(certificate_class),
    }
    payload.update(overrides)
    return payload


def _projection(overrides: dict[tuple[str, str], dict] | None = None) -> dict:
    tasks = {task: _task(0.0) for task in REQUIRED_TASKS}
    motions = {
        motion: {"tasks": deepcopy(tasks)}
        for motion in CANONICAL_MOTION_NAMES
    }
    for (motion, task), payload in (overrides or {}).items():
        motions[motion]["tasks"][task] = payload
    return {
        "motion_order": list(CANONICAL_MOTION_NAMES),
        "motions": motions,
    }


def test_all_required_non_stress_tasks_exact_passes() -> None:
    result = evaluate_profile_status(_projection(), required_tasks=REQUIRED_TASKS)

    assert result.status == PROFILE_PASSED
    assert result.failures == ()


def test_ordinary_or_extended_limited_certificate_sets_capability_limited_status() -> None:
    projection = _projection(
        {
            ("arms_forward", "left_hand"): _task(0.2, "capability_limited_rank"),
            ("crossed_body_reach", "torso"): _task(0.2, "capability_limited_joint_limits"),
        }
    )

    result = evaluate_profile_status(projection, required_tasks=REQUIRED_TASKS)

    assert result.status == PROFILE_CAPABILITY_LIMITED_PASSED
    assert result.failures == ()


def test_invariance_motion_is_exact_only_even_with_limited_certificate() -> None:
    result = evaluate_profile_status(
        _projection({("neutral", "left_hand"): _task(0.2, "capability_limited_rank")}),
        required_tasks=REQUIRED_TASKS,
    )

    assert result.status == PROFILE_ALGORITHM_FAILED
    assert any("neutral.left_hand" in failure and "invariance" in failure for failure in result.failures)


def test_missing_or_failed_certificate_fails_required_motion() -> None:
    missing = _task(0.2, "capability_limited_rank")
    missing.pop("capability_certificate")
    failed = _task(0.2, "solver_failed")

    result = evaluate_profile_status(
        _projection(
            {
                ("arms_forward", "left_hand"): missing,
                ("overhead_reach", "torso"): failed,
            }
        ),
        required_tasks=REQUIRED_TASKS,
    )

    assert result.status == PROFILE_ALGORITHM_FAILED
    assert any("arms_forward.left_hand" in failure and "capability_certificate missing" in failure for failure in result.failures)
    assert any("overhead_reach.torso" in failure and "solver_failed" in failure for failure in result.failures)


def test_limited_certificate_false_gate_fails_with_specific_gate_name() -> None:
    bad_certificate = _certificate(
        "capability_limited_rank",
        gates=_gates(exact_threshold_passed=False, seed_consensus=False),
    )
    result = evaluate_profile_status(
        _projection(
            {
                ("asymmetric_arm_reach", "left_hand"): _task(
                    0.2,
                    "capability_limited_rank",
                    capability_certificate=bad_certificate,
                )
            }
        ),
        required_tasks=REQUIRED_TASKS,
    )

    assert result.status == PROFILE_ALGORITHM_FAILED
    assert any("asymmetric_arm_reach.left_hand" in failure and "seed_consensus" in failure for failure in result.failures)


def test_partial_and_negative_profile_statuses_are_not_promoted_to_capability_limited() -> None:
    assert (
        profile_status_from_semantic_readiness("partial_humanoid", PROFILE_CAPABILITY_LIMITED_PASSED)
        == "partial_humanoid"
    )
    assert (
        profile_status_from_semantic_readiness("negative_control_passed", PROFILE_CAPABILITY_LIMITED_PASSED)
        == "negative_control_passed"
    )
    assert profile_status_from_semantic_readiness("full_humanoid_ready", PROFILE_PASSED) == "full_humanoid_ready"
    assert (
        profile_status_from_semantic_readiness("full_humanoid_ready", PROFILE_CAPABILITY_LIMITED_PASSED)
        == PROFILE_CAPABILITY_LIMITED_PASSED
    )
