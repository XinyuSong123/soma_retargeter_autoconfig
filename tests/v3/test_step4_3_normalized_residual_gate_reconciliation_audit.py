from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import (
    write_blocked_fixture,
    write_passing_fixture,
)
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json


def test_step4_3_audit_accepts_release_quality_v2_breakthrough_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS_RC"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44


def test_step4_3_audit_accepts_complete_blocked_gate_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_blocked_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED_GATE_RECONCILIATION"
    assert result.blocking_count == 0


def test_step4_3_audit_rejects_missing_scale_audit(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / "normalized_residual_scale_audit.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["missing_required_artifacts"] >= 1


def test_step4_3_audit_rejects_pass_rc_without_reconciliation_impact(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "runtime_scoring_delta_vs_step4_2.json")
    delta["rows_below_candidate_warn_gate"] = 0
    delta["rows_below_candidate_pass_gate"] = 0
    delta["p95_normalized_task_residual_p95_delta"] = 0.0
    delta["primary_quality_breakthrough"] = False
    delta["improvements"] = []
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_2.json", delta)
    gate = read_json(artifact_dir / "gate_reconciliation_v2_report.json")
    gate["rows_below_candidate_warn_gate"] = []
    gate["rows_below_candidate_pass_gate"] = []
    write_json(artifact_dir / "gate_reconciliation_v2_report.json", gate)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1
