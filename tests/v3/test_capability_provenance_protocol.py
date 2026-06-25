from __future__ import annotations

import json
from pathlib import Path

from scripts.write_capability_provenance import (
    validate_acceptance_ledger,
    validate_environment_metadata,
    validate_required_artifact_files,
)


def _valid_environment() -> dict:
    return {
        "schema_version": 1,
        "source_code_commit": "b84f12000599e41af34d3e2a9ace19410cc870be",
        "artifact_commit": "b84f12000599e41af34d3e2a9ace19410cc870be",
        "source_code_commit_remote_resolvable": True,
        "source_code_commit_is_artifact_commit_ancestor": True,
        "source_worktree_clean_before_run": True,
        "source_worktree_clean_after_run": True,
        "core_diff_after_source_commit": [],
        "git_status_short": "",
        "seed": 0,
    }


def _valid_acceptance_ledger() -> dict:
    return {
        "schema_version": 1,
        "audit_name": "retargeting_v3_step2_3_1_capability",
        "command": (
            "python scripts/audit_retargeting_v3_capability.py "
            "--artifact-dir artifacts/retargeting_v3_step2_capability"
        ),
        "returncode": 0,
        "status": "passed",
        "stdout": "Step 2.3 capability red-team audit PASS\n",
        "stderr": "",
        "source_code_commit": "b84f12000599e41af34d3e2a9ace19410cc870be",
        "time_utc": "2026-06-25T00:00:00Z",
    }


def test_environment_rejects_unresolvable_source_commit() -> None:
    environment = _valid_environment()
    environment["source_code_commit_remote_resolvable"] = False

    failures = validate_environment_metadata(environment)

    assert any("remote_resolvable" in failure for failure in failures)


def test_environment_rejects_source_not_artifact_ancestor() -> None:
    environment = _valid_environment()
    environment["source_code_commit_is_artifact_commit_ancestor"] = False

    failures = validate_environment_metadata(environment)

    assert any("artifact_commit_ancestor" in failure for failure in failures)


def test_environment_rejects_core_drift_after_source_commit() -> None:
    environment = _valid_environment()
    environment["core_diff_after_source_commit"] = [
        "soma_retargeter/robotics/v3/profile.py",
        "tests/v3/test_capability_profile_status_policy.py",
    ]

    failures = validate_environment_metadata(environment)

    assert any("core_diff_after_source_commit" in failure for failure in failures)


def test_environment_rejects_dirty_source_worktree() -> None:
    environment = _valid_environment()
    environment["source_worktree_clean_after_run"] = False
    environment["git_status_short"] = " M soma_retargeter/robotics/v3/profile.py"

    failures = validate_environment_metadata(environment)

    assert any("source_worktree_clean_after_run" in failure for failure in failures)
    assert any("git_status_short" in failure for failure in failures)


def test_required_artifact_files_reject_missing_pytest_junit_and_lfs_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "test_results").mkdir()

    failures = validate_required_artifact_files(artifact_dir)

    assert any("test_results/pytest.txt" in failure for failure in failures)
    assert any("test_results/junit.xml" in failure for failure in failures)
    assert any("test_results/pytest_summary.json" in failure for failure in failures)
    assert any("lfs_state.json" in failure for failure in failures)


def test_required_artifact_files_accept_present_pytest_junit_summary_and_lfs_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    test_results = artifact_dir / "test_results"
    test_results.mkdir(parents=True)
    (test_results / "pytest.txt").write_text("1 passed\n")
    (test_results / "junit.xml").write_text("<testsuite />\n")
    (test_results / "pytest_summary.json").write_text(json.dumps({"status": "passed"}) + "\n")
    (artifact_dir / "lfs_state.json").write_text(json.dumps({"fsck_returncode": 0}) + "\n")

    assert validate_required_artifact_files(artifact_dir) == []


def test_acceptance_ledger_rejects_stale_step22_assets44_ledger() -> None:
    ledger = _valid_acceptance_ledger()
    ledger["audit_name"] = "retargeting_v3_step2_2_assets44"
    ledger["command"] = (
        "python scripts/audit_retargeting_v3_assets44.py "
        "--artifact-dir artifacts/retargeting_v3_step2_assets44"
    )
    ledger["stdout"] = "Step 2.2 Assets44 audit PASS\n"

    failures = validate_acceptance_ledger(ledger)

    assert any("audit_retargeting_v3_capability.py" in failure for failure in failures)
    assert any("Step 2.3" in failure or "Step2.3.1" in failure for failure in failures)


def test_acceptance_ledger_accepts_current_capability_audit() -> None:
    assert validate_acceptance_ledger(_valid_acceptance_ledger()) == []
