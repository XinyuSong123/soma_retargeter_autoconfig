from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_blocked_fixture, write_json, write_passing_fixture


def test_step4_1_audit_accepts_orientation_breakthrough_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS_RC"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44


def test_step4_1_audit_rejects_missing_orientation_semantics(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / "orientation_frame_semantics_matrix.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["missing_required_artifacts"] >= 1


def test_step4_1_audit_rejects_pass_rc_without_primary_breakthrough(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["primary_quality_breakthrough"] = False
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1


def test_step4_1_audit_accepts_complete_blocked_orientation_status(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_blocked_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED_ORIENTATION_SEMANTICS"
    assert result.blocking_count == 0


def test_step4_1_audit_rejects_pipeline_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_failed_count"] = 1
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["solver_backed_counts"] >= 1

