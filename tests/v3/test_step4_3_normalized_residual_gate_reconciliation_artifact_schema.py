from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import (
    EXPECTED_CANDIDATES,
    REQUIRED_ARTIFACT_FILES,
    run_audit,
)
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_required_artifact_schema_is_declared() -> None:
    required = set(REQUIRED_ARTIFACT_FILES)
    assert "normalized_residual_scale_audit.json" in required
    assert "gate_semantics_audit.json" in required
    assert "normalization_candidate_matrix.json" in required
    assert "gate_candidate_matrix.json" in required
    assert "normalization_policy_selection.json" in required
    assert "gate_reconciliation_v2_report.json" in required
    assert "runtime_scoring_delta_vs_step4_2.json" in required


def test_step4_3_fixture_writes_required_artifact_set(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.blocking_count == 0
    for relative in REQUIRED_ARTIFACT_FILES:
        assert (artifact_dir / relative).exists(), relative
    summary = read_json(artifact_dir / "quality_summary.json")
    assert summary["base_step4_2_final_head"]
    assert summary["legacy_gates_unchanged"] is True
    assert summary["candidate_release_gates_defined"] is True


def test_step4_3_candidate_set_is_complete(tmp_path: Path) -> None:
    artifact_dir, _baseline_dir, _source_root = write_passing_fixture(tmp_path)
    candidates = read_json(artifact_dir / "normalization_candidate_matrix.json")

    assert EXPECTED_CANDIDATES.issubset({row["candidate_id"] for row in candidates["rows"]})
