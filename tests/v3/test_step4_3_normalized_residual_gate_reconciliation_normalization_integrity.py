from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_rejects_denominator_inflation(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    scale = read_json(artifact_dir / "normalized_residual_scale_audit.json")
    scale["denominator_inflation_detected"] = True
    write_json(artifact_dir / "normalized_residual_scale_audit.json", scale)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["normalization_integrity"] >= 1


def test_step4_3_rejects_normalization_hiding_raw_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    scale = read_json(artifact_dir / "normalized_residual_scale_audit.json")
    scale["normalization_hides_raw_residual_regression"] = True
    write_json(artifact_dir / "normalized_residual_scale_audit.json", scale)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["normalization_integrity"] >= 1
