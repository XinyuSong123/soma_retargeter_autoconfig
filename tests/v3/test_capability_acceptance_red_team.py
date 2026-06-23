from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_capability import (
    EXPECTED_MODEL_COUNT,
    EXACT_PROJECTION_THRESHOLDS,
    audit_deterministic_rerun,
    audit_negative_controls,
    audit_projection_reports,
    audit_threshold_calibration,
)


def _task(
    *,
    normalized_residual: float,
    task: str = "left_hand",
    motion: str = "crossed_body_reach",
    status: str = "converged/with_residual",
    kkt_certificate: dict | None = None,
) -> dict:
    payload = {
        "active_coordinates": [0, 1],
        "converged": True,
        "desired": [0.2, 0.0, 0.0],
        "projected": [0.0, 0.0, 0.0],
        "residual": normalized_residual,
        "normalized_residual": normalized_residual,
        "normalization_scale": 1.0,
        "status": status,
        "reference": "Chest",
        "target": "LeftHand" if "hand" in task else "Chest",
        "desired_source": "canonical_targets.transforms",
    }
    if kkt_certificate is not None:
        payload["kkt_certificate"] = kkt_certificate
    payload["motion"] = motion
    payload["task"] = task
    return payload


def _report(*, task_payload: dict, status: str = "algorithm_failed", expected: str = "positive") -> dict:
    motion = task_payload.pop("motion")
    task = task_payload.pop("task")
    return {
        "status": status,
        "manifest_entry": {
            "id": "fixture_humanoid",
            "expected_capability": expected,
            "robot_class": "humanoid",
            "required": True,
        },
        "failures": [],
        "canonical_target_validation": {"failures": []},
        "canonical_projection_reports": {
            "motion_order": ["neutral", motion],
            "motions": {
                motion: {
                    "tasks": {
                        task: task_payload,
                    }
                }
            },
        },
    }


def _valid_kkt_certificate() -> dict:
    return {
        "certified": True,
        "stationarity_inf_norm": 1e-9,
        "stationarity_tolerance": 1e-6,
        "complementarity_inf_norm": 1e-9,
        "complementarity_tolerance": 1e-6,
        "primal_feasible": True,
        "dual_feasible": True,
        "task_gradient_inf_norm": 0.25,
        "prior_gradient_inf_norm": 1e-9,
        "prior_cancellation_ratio": 1e-8,
        "seed_consistency": {
            "checked": True,
            "status": "consistent",
            "max_task_cost_delta": 0.0,
        },
    }


def test_threshold_audit_locks_exact_global_values() -> None:
    payload = {
        "projection_quality": {
            "neutral_position_abs_m": EXACT_PROJECTION_THRESHOLDS["neutral"],
            "foot_rho_p": EXACT_PROJECTION_THRESHOLDS["foot"],
            "hand_rho_p": 0.13,
            "torso_rho_r": EXACT_PROJECTION_THRESHOLDS["torso"],
        }
    }

    failures = audit_threshold_calibration(payload, check_live_function=False)

    assert any("hand_rho_p" in failure and "0.12" in failure for failure in failures)


def test_over_threshold_residual_requires_kkt_certificate() -> None:
    report = _report(task_payload=_task(normalized_residual=0.2))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("missing kkt_certificate" in failure for failure in failures)


def test_kkt_certificate_rejects_prior_gradient_cancellation_and_seed_inconsistency() -> None:
    certificate = _valid_kkt_certificate()
    certificate["prior_cancellation_ratio"] = 0.75
    certificate["seed_consistency"] = {"checked": True, "status": "inconsistent"}
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("prior_cancellation_ratio" in failure for failure in failures)
    assert any("seed_consistency" in failure for failure in failures)


def test_valid_over_threshold_kkt_certificate_is_accepted_for_algorithm_failure() -> None:
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=_valid_kkt_certificate()))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert failures == []


def test_rank_zero_nonzero_demand_must_be_preserved_as_unreachable() -> None:
    task = _task(normalized_residual=0.0, status="rank_zero")
    task["desired"] = [0.3, 0.0, 0.0]
    task["projected"] = [0.0, 0.0, 0.0]
    report = _report(task_payload=task)

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("rank-zero nonzero demand" in failure for failure in failures)


def test_target_geometry_failure_cannot_be_packaged_as_capability_pass() -> None:
    report = _report(task_payload=_task(normalized_residual=0.0), status="passed")
    report["canonical_target_validation"]["failures"] = ["source torso geometry is inconsistent"]

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("target geometry failure packaged as pass" in failure for failure in failures)


def test_stress_motion_does_not_mask_ordinary_failure() -> None:
    report = _report(task_payload=_task(normalized_residual=0.0), status="passed")
    report["canonical_projection_reports"]["motions"]["extreme_but_valid_joint_limit_stress"] = {
        "tasks": {"left_hand": _task(normalized_residual=0.9, motion="stress", task="left_hand")}
    }
    report["canonical_projection_reports"]["motions"]["crossed_body_reach"] = {
        "tasks": {"left_hand": _task(normalized_residual=0.21, motion="ordinary", task="left_hand")}
    }

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("ordinary residual over threshold packaged as pass" in failure for failure in failures)
    assert not any("extreme_but_valid_joint_limit_stress.left_hand missing kkt" in failure for failure in failures)


def test_negative_controls_are_not_humanoid_profile_passes() -> None:
    summary = {"negative_control_passed": 1, "profile_passed": 1}
    reports = {
        "bolt_urdf": {
            "status": "passed",
            "manifest_entry": {
                "id": "bolt_urdf",
                "expected_capability": "negative_control",
                "robot_class": "biped_no_arms",
            },
            "capability_status": "full_humanoid_ready",
            "canonical_projection_reports": {"motions": {}},
        }
    }

    failures = audit_negative_controls(summary, reports)

    assert any("negative control status" in failure for failure in failures)
    assert any("humanoid capability payload" in failure for failure in failures)
    assert any("profile_passed" in failure for failure in failures)


def test_deterministic_rerun_must_compare_all_44_models() -> None:
    deterministic = {
        "status": "passed",
        "totals": {
            "model_count": EXPECTED_MODEL_COUNT,
            "compared_count": EXPECTED_MODEL_COUNT - 1,
            "matched_count": EXPECTED_MODEL_COUNT - 1,
            "mismatch_count": 0,
            "skipped_non_pass_count": 1,
        },
        "models": {
            f"model_{index}": {"compared": index < EXPECTED_MODEL_COUNT - 1}
            for index in range(EXPECTED_MODEL_COUNT)
        },
    }

    failures = audit_deterministic_rerun(deterministic, expected_model_count=EXPECTED_MODEL_COUNT)

    assert any("compared_count" in failure for failure in failures)
    assert any("uncompared deterministic models" in failure for failure in failures)


def test_projection_audit_accepts_rank_zero_zero_demand_evidence() -> None:
    task = _task(normalized_residual=0.0, status="rank_zero")
    task["desired"] = [0.0, 0.0, 0.0]
    task["projected"] = [0.0, 0.0, 0.0]
    task["demand_residual"] = 0.0
    task["unreachable_demand"] = False
    task["rank_zero_reason"] = "no_active_coordinates_zero_demand"
    report = _report(task_payload=task)

    assert audit_projection_reports({"fixture_humanoid": report}) == []


def test_live_assets44_deterministic_fixture_shape_is_documented() -> None:
    path = Path("artifacts/retargeting_v3_step2_assets44/deterministic_rerun.json")
    deterministic = json.loads(path.read_text())

    assert deterministic["totals"]["model_count"] == EXPECTED_MODEL_COUNT
