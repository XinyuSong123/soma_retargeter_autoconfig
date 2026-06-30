from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_pass_rc_requires_strict_quality_target(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    odelta = read_json(artifact_dir / "orientation_delta_vs_step4_0.json")
    odelta["p95_rotation_residual_p95_delta"] = -0.01
    odelta["accepted_breakthrough"] = False
    write_json(artifact_dir / "orientation_delta_vs_step4_0.json", odelta)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["rotation_dominant_residual_count"] = 27
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1


def test_step4_1_invalid_release_status_is_rejected(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["release_candidate_status"] = "PASS_DIAGNOSTIC_ONLY"
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1

