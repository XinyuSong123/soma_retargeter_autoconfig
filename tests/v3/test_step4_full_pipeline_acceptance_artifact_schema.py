from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_full_pipeline_acceptance import REQUIRED_ARTIFACT_FILES, run_audit
from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_passing_fixture


def test_step4_required_artifact_schema_is_declared() -> None:
    required = set(REQUIRED_ARTIFACT_FILES)
    assert "full_pipeline_matrix.json" in required
    assert "trajectory_export_manifest.json" in required
    assert "orientation_residual_taxonomy.json" in required
    assert "normalization_audit.json" in required
    assert "temporal_continuity_matrix.json" in required
    assert "support_contact_diagnostics.json" in required
    assert "collision_proxy_diagnostics.json" in required


def test_step4_fixture_writes_required_artifact_set(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS_RC"
    for relative in REQUIRED_ARTIFACT_FILES:
        assert (artifact_dir / relative).exists(), relative
    assert read_json(artifact_dir / "quality_summary.json")["base_step3_4_final_head"]
