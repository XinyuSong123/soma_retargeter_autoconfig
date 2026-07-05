from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import (
    REQUIRED_ARTIFACT_FILES,
    run_audit,
)
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_required_artifact_schema_is_declared() -> None:
    required = set(REQUIRED_ARTIFACT_FILES)

    assert "release_quality_v2_validation_matrix.json" in required
    assert "candidate_warned_deep_audit.json" in required
    assert "release_quality_v2_blocker_taxonomy.json" in required
    assert "release_quality_v2_stress_test.json" in required
    assert "release_quality_v2_promotion_readiness.json" in required
    assert "gate_reconciliation_v3_report.json" in required
    assert "normalization_integrity_v2_report.json" in required


def test_step4_4_fixture_writes_required_artifact_set(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS_RC"
    assert result.blocking_count == 0
    for relative in REQUIRED_ARTIFACT_FILES:
        assert (artifact_dir / relative).exists(), relative


def test_step4_4_expands_candidate_warned_without_legacy_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    gate = read_json(artifact_dir / "gate_reconciliation_v3_report.json")
    readiness = read_json(artifact_dir / "release_quality_v2_promotion_readiness.json")

    assert summary["rows_below_candidate_warn_gate"] > summary["step4_3_rows_below_candidate_warn_gate"]
    assert summary["runtime_quality_passed_count"] == 0
    assert gate["candidate_release_gates_not_counted_as_legacy_runtime_passed"] is True
    assert readiness["decision"] == "keep_diagnostic_only"
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0
