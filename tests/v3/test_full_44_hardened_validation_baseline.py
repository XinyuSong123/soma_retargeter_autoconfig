from __future__ import annotations

from collections import Counter

import pytest

import soma_retargeter.robotics.v3.validation as validation_module


BASELINE_COMMIT = "5ad5a001c445c525d4c8bbaf6339dec5c5c2c719"
BASELINE_ROOT = "artifacts/retargeting_v3_step2_assets44"
EXPECTED_BASELINE_COUNTS = {
    "algorithm_failed": 11,
    "negative_control_passed": 9,
    "partial_passed": 3,
    "passed": 21,
}


def _legal_current_reports(baseline) -> dict[str, dict]:
    old_failures = sorted(
        model_id
        for model_id, status in baseline.status_by_model.items()
        if status == "algorithm_failed"
    )
    limited_ids = set(old_failures[::2])
    reports = {}
    for model_id, before_status in sorted(baseline.status_by_model.items()):
        if before_status == "algorithm_failed":
            after_status = "capability_limited_passed" if model_id in limited_ids else "passed"
        else:
            after_status = before_status
        reports[model_id] = {
            "status": after_status,
            "status_reason": f"synthetic current status for {model_id}",
            "task_certificate_summary": {"status": "available", "task_count": 1},
        }
    return reports


def test_fixed_assets44_baseline_loads_from_git_object_with_exact_counts() -> None:
    baseline = validation_module._load_immutable_assets44_baseline()

    assert baseline.commit == BASELINE_COMMIT
    assert baseline.artifact_root == BASELINE_ROOT
    assert baseline.status_counts == EXPECTED_BASELINE_COUNTS
    assert len(baseline.status_by_model) == 44
    assert baseline.source_command == f"git show {BASELINE_COMMIT}:{BASELINE_ROOT}/summary.json"


def test_before_after_uses_true_baseline_and_records_11_old_failure_transitions() -> None:
    baseline = validation_module._load_immutable_assets44_baseline()
    reports = _legal_current_reports(baseline)

    matrix = validation_module._before_after_matrix(reports, baseline=baseline)

    assert matrix["basis"] == "immutable_assets44_git_baseline"
    assert matrix["baseline_commit"] == BASELINE_COMMIT
    assert matrix["baseline_artifact_root"] == BASELINE_ROOT
    assert matrix["baseline_status_counts"] == EXPECTED_BASELINE_COUNTS
    assert matrix["row_count"] == 44

    transition_counts = matrix["status_transition_counts"]
    assert transition_counts["passed->passed"] == 21
    assert transition_counts["partial_passed->partial_passed"] == 3
    assert transition_counts["negative_control_passed->negative_control_passed"] == 9
    assert (
        transition_counts["algorithm_failed->passed"]
        + transition_counts["algorithm_failed->capability_limited_passed"]
        == 11
    )

    old_failure_rows = [
        row for row in matrix["rows"].values() if row["before_status"] == "algorithm_failed"
    ]
    assert len(old_failure_rows) == 11
    assert {row["after_status"] for row in old_failure_rows} <= {
        "passed",
        "capability_limited_passed",
    }
    assert all(row["baseline_source_command"].startswith("git show ") for row in old_failure_rows)


@pytest.mark.parametrize(
    ("before_status", "after_status"),
    [
        ("passed", "capability_limited_passed"),
        ("passed", "algorithm_failed"),
        ("partial_passed", "capability_limited_passed"),
        ("partial_passed", "passed"),
        ("negative_control_passed", "passed"),
        ("negative_control_passed", "capability_limited_passed"),
        ("algorithm_failed", "partial_passed"),
    ],
)
def test_illegal_baseline_transitions_fail_instead_of_fabricating_before_status(
    before_status: str,
    after_status: str,
) -> None:
    baseline = validation_module._load_immutable_assets44_baseline()
    reports = _legal_current_reports(baseline)
    model_id = next(
        model_id
        for model_id, status in baseline.status_by_model.items()
        if status == before_status
    )
    reports[model_id]["status"] = after_status

    with pytest.raises(ValueError, match="illegal baseline transition"):
        validation_module._before_after_matrix(reports, baseline=baseline)


def test_baseline_partial_and_negative_sets_are_stable_in_legal_matrix() -> None:
    baseline = validation_module._load_immutable_assets44_baseline()
    reports = _legal_current_reports(baseline)

    matrix = validation_module._before_after_matrix(reports, baseline=baseline)
    unchanged_counts = Counter(
        row["before_status"]
        for row in matrix["rows"].values()
        if row["before_status"] == row["after_status"]
    )

    assert unchanged_counts["partial_passed"] == 3
    assert unchanged_counts["negative_control_passed"] == 9
