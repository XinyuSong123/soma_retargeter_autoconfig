from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_full_pipeline_acceptance import run_audit
from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_json, write_passing_fixture


def test_step4_normalization_audit_keeps_raw_residuals(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    normalization = read_json(artifact_dir / "normalization_audit.json")

    assert normalization["global_normalization_v2_enabled"] is True
    assert normalization["normalization_hides_raw_regression"] is False
    assert all("raw_task_residual_p95" in row for row in normalization["rows"])


def test_step4_normalization_rejects_robot_specific_denominator(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    normalization = read_json(artifact_dir / "normalization_audit.json")
    normalization["robot_specific_denominator_count"] = 1
    normalization["rows"][0]["residual_denominator_robot_specific"] = True
    write_json(artifact_dir / "normalization_audit.json", normalization)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)
    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["normalization_integrity"] >= 1
