from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import REQUIRED_ARTIFACT_FILES, run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_passing_fixture


def test_step4_2_required_artifact_schema_is_declared() -> None:
    required = set(REQUIRED_ARTIFACT_FILES)
    assert "orientation_policy_runtime_scoring_matrix.json" in required
    assert "active_vs_diagnostic_policy_matrix.json" in required
    assert "gate_reconciliation_report.json" in required
    assert "scoring_normalization_audit.json" in required
    assert "runtime_scoring_delta_vs_step4_1.json" in required
    assert "orientation_policy_runtime_impact_report.json" in required


def test_step4_2_fixture_writes_required_artifact_set(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.blocking_count == 0
    for relative in REQUIRED_ARTIFACT_FILES:
        assert (artifact_dir / relative).exists(), relative
    summary = read_json(artifact_dir / "quality_summary.json")
    assert summary["base_step4_1_final_head"]
    assert summary["orientation_policy_active_for_scoring"] is True
