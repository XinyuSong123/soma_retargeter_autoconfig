from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json, write_passing_fixture


def test_step4_2_gate_reconciliation_explains_zero_pass_rows(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    gate = read_json(artifact_dir / "gate_reconciliation_report.json")

    assert gate["pass_gate_thresholds_unchanged"] is True
    assert gate["why_no_pass_if_zero"]
    assert gate["per_gate_blocker_counts"]["normalized_task_residual_p95_above_pass_gate"] == 32
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_2_rejects_weakened_gate_report(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    gate = read_json(artifact_dir / "gate_reconciliation_report.json")
    gate["pass_gate_thresholds_unchanged"] = False
    gate["gates_weakened"] = True
    write_json(artifact_dir / "gate_reconciliation_report.json", gate)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["gate_reconciliation"] >= 1
