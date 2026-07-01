from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_rejects_legacy_pass_from_candidate_only_gate(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    model = read_json(artifact_dir / "model_matrix.json")
    model["rows"][0]["runtime_quality_status"] = "runtime_quality_passed"
    model["rows"][0]["quality_classification"] = "runtime_quality_passed"
    write_json(artifact_dir / "model_matrix.json", model)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_passed_count"] = 1
    summary["runtime_quality_warned_count"] = 31
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["runtime_quality_label_honesty"] >= 1


def test_step4_3_rejects_pass_diagnostic_only_status(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["release_candidate_status"] = "PASS_DIAGNOSTIC_ONLY"
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1
