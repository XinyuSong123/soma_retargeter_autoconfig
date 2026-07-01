from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_scale_audit_explains_row_local_saturation(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    scale = read_json(artifact_dir / "normalized_residual_scale_audit.json")

    assert scale["denominator_scope"] == "row_local_legacy_metric"
    assert scale["denominator_follows_current_row_max"] is True
    assert scale["row_local_denominator_saturation_detected"] is True
    assert scale["raw_residual_always_retained"] is True
    assert scale["task_class_dominance"]["dominant_semantic_counts"]
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_3_rejects_missing_task_dominance_audit(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    scale = read_json(artifact_dir / "normalized_residual_scale_audit.json")
    scale["task_class_dominance"] = {}
    write_json(artifact_dir / "normalized_residual_scale_audit.json", scale)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["normalized_residual_scale_audit"] >= 1
