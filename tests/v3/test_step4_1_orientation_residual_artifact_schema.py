from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import REQUIRED_ARTIFACT_FILES, run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_passing_fixture


def test_step4_1_required_artifact_set_is_declared() -> None:
    required = set(REQUIRED_ARTIFACT_FILES)
    assert "orientation_frame_semantics_matrix.json" in required
    assert "orientation_residual_math_audit.json" in required
    assert "orientation_offset_candidate_matrix.json" in required
    assert "orientation_policy_selection.json" in required
    assert "quality_delta_vs_step4_0.json" in required
    assert "orientation_delta_vs_step4_0.json" in required


def test_step4_1_fixture_writes_required_schema(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.blocking_count == 0
    for relative in REQUIRED_ARTIFACT_FILES:
        assert (artifact_dir / relative).exists(), relative
    summary = read_json(artifact_dir / "quality_summary.json")
    assert summary["base_step4_0_final_head"]
    assert summary["primary_quality_breakthrough"] is True

