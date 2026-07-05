from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_candidate_status_stays_separate_from_legacy_runtime_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    gate = read_json(artifact_dir / "gate_reconciliation_v3_report.json")
    summary = read_json(artifact_dir / "quality_summary.json")

    assert summary["runtime_quality_passed_count"] == 0
    assert gate["candidate_release_gates_not_counted_as_legacy_runtime_passed"] is True
    assert gate["runtime_quality_passed_count_uses_legacy_gates_only"] is True
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_4_audit_rejects_candidate_pass_without_hard_safety(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    validation = read_json(artifact_dir / "release_quality_v2_validation_matrix.json")
    validation["rows"][0]["candidate_status"] = "release_quality_candidate_passed"
    validation["rows"][0]["hard_safety_status"]["passed"] = False
    write_json(artifact_dir / "release_quality_v2_validation_matrix.json", validation)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["validation_matrix"] >= 1
