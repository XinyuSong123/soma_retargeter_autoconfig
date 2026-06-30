from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import (
    read_json,
    write_blocked_fixture,
    write_json,
    write_passing_fixture,
)


def test_step4_2_audit_accepts_active_scoring_breakthrough_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS_RC"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44


def test_step4_2_audit_accepts_complete_blocked_gate_status(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_blocked_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED_GATE_RECONCILIATION"
    assert result.blocking_count == 0


def test_step4_2_audit_rejects_missing_runtime_policy_matrix(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / "orientation_policy_runtime_scoring_matrix.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["missing_required_artifacts"] >= 1


def test_step4_2_audit_rejects_pass_rc_without_active_scoring_impact(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json")
    delta["p95_orientation_integrated_residual_delta"] = 0.0
    delta["p95_normalized_task_residual_p95_delta"] = 0.0
    delta["primary_quality_breakthrough"] = False
    delta["improvements"] = []
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1
