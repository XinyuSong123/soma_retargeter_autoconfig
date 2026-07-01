from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_gate_semantics_keep_candidate_separate(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    semantics = read_json(artifact_dir / "gate_semantics_audit.json")

    assert semantics["legacy_gates_unchanged"] is True
    assert semantics["active_parent_relative_residual_needs_new_metric_name"] is True
    assert semantics["candidate_gates_do_not_replace_legacy_runtime_quality_passed"] is True
    assert "hard_safety_gates" in semantics["gate_categories"]
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_3_rejects_gate_semantics_that_replace_legacy_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    semantics = read_json(artifact_dir / "gate_semantics_audit.json")
    semantics["candidate_gates_do_not_replace_legacy_runtime_quality_passed"] = False
    write_json(artifact_dir / "gate_semantics_audit.json", semantics)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["gate_semantics"] >= 1
