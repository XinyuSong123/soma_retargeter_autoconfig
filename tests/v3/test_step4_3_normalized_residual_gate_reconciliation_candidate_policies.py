from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_candidate_policy_selection_is_global_and_breaks_warn_gate(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    selection = read_json(artifact_dir / "normalization_policy_selection.json")
    gate = read_json(artifact_dir / "gate_reconciliation_v2_report.json")

    assert selection["normalization_policy_selected"] == "candidate_1_fixed_global_body_scale_normalization"
    assert selection["gate_policy_selected"] == "candidate_6_release_quality_gate_v2_candidate"
    assert selection["robot_specific_tuning_used"] is False
    assert gate["rows_below_candidate_warn_gate"]
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_3_rejects_model_specific_candidate_threshold(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    candidates = read_json(artifact_dir / "normalization_candidate_matrix.json")
    candidates["rows"][0]["uses_model_id_threshold"] = True
    write_json(artifact_dir / "normalization_candidate_matrix.json", candidates)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["candidate_policies"] >= 1
