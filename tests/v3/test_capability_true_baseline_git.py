from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_retargeter.robotics.v3.failure_analysis import (
    BASELINE_ROBOT_IDS,
    DEFAULT_BASELINE_COMMIT,
    DEFAULT_BASELINE_LEDGER_PATH,
    DEFAULT_BASELINE_SUMMARY_PATH,
    DEFAULT_BEFORE_AFTER_PATH,
    EXPECTED_BASELINE_STATUS_COUNTS,
    BaselineGitObjectError,
    build_baseline_failure_ledger,
    build_capability_before_after,
    build_true_baseline_summary,
)


def test_true_baseline_summary_reads_pinned_git_commit_and_counts() -> None:
    baseline = build_true_baseline_summary()

    assert baseline["source_access"] == "git_object"
    assert baseline["baseline_commit"] == DEFAULT_BASELINE_COMMIT
    assert baseline["model_count"] == 44
    assert baseline["status_counts"] == EXPECTED_BASELINE_STATUS_COUNTS
    assert tuple(baseline["algorithm_failure_robot_ids"]) == BASELINE_ROBOT_IDS
    assert baseline["reports"]["apptronik_apollo_mjcf"]["status"] == "passed"
    assert baseline["reports"]["atlas_drc_urdf"]["status"] == "algorithm_failed"
    assert baseline["reports"]["berkeley_humanoid_mjcf_direct"]["status"] == "partial_passed"
    assert baseline["reports"]["bolt_urdf"]["status"] == "negative_control_passed"


def test_true_baseline_git_object_access_fails_closed() -> None:
    with pytest.raises(BaselineGitObjectError):
        build_true_baseline_summary(source_commit="0" * 40)

    with pytest.raises(BaselineGitObjectError):
        build_true_baseline_summary(artifact_root="artifacts/does_not_exist")


def test_before_after_uses_real_baseline_instead_of_fabricated_partial(tmp_path: Path) -> None:
    baseline = build_true_baseline_summary()
    reports = {}
    for robot_id, row in baseline["reports"].items():
        after_status = row["status"]
        if row["status"] == "algorithm_failed":
            after_status = "capability_limited_passed"
        reports[robot_id] = {
            "status": after_status,
            "status_reason": f"synthetic after status {after_status}",
            "task_certificate_summary": {"status": "synthetic"},
        }
    reports["apptronik_apollo_mjcf"]["status"] = "capability_limited_passed"

    current_summary = tmp_path / "summary.json"
    current_summary.write_text(json.dumps({"reports": reports}, indent=2, sort_keys=True) + "\n")

    before_after = build_capability_before_after(current_summary_path=current_summary)
    apptronik = before_after["rows"]["apptronik_apollo_mjcf"]
    atlas = before_after["rows"]["atlas_drc_urdf"]

    assert apptronik["before_status"] == "passed"
    assert apptronik["after_status"] == "capability_limited_passed"
    assert apptronik["transition"] == "passed->capability_limited_passed"
    assert apptronik["transition_allowed"] is False
    assert atlas["before_status"] == "algorithm_failed"
    assert atlas["after_status"] == "capability_limited_passed"
    assert atlas["transition"] == "algorithm_failed->capability_limited_passed"
    assert atlas["transition_allowed"] is True
    assert before_after["transition_validation"]["status"] == "failed"
    assert before_after["transition_validation"]["invalid_transition_counts"] == {
        "passed->capability_limited_passed": 1
    }


def test_committed_true_baseline_artifacts_match_git_objects() -> None:
    baseline_summary = json.loads(DEFAULT_BASELINE_SUMMARY_PATH.read_text())
    baseline_ledger = json.loads(DEFAULT_BASELINE_LEDGER_PATH.read_text())
    before_after = json.loads(DEFAULT_BEFORE_AFTER_PATH.read_text())

    assert baseline_summary == build_true_baseline_summary()
    assert baseline_ledger == build_baseline_failure_ledger()
    assert before_after == build_capability_before_after()
    assert before_after["baseline_counts"] == EXPECTED_BASELINE_STATUS_COUNTS
    assert before_after["row_count"] == 44
