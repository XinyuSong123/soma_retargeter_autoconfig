from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_exports_and_temporal_are_finite(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    manifest = read_json(artifact_dir / "trajectory_export_manifest.json")
    temporal = read_json(artifact_dir / "temporal_continuity_matrix.json")

    assert manifest["row_count"] == 128
    assert all(row["finite_qpos"] is True for row in manifest["rows"])
    assert temporal["finite_count"] == 128


def test_step4_1_audit_rejects_nonfinite_export(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    manifest = read_json(artifact_dir / "trajectory_export_manifest.json")
    manifest["rows"][0]["finite_qpos"] = False
    write_json(artifact_dir / "trajectory_export_manifest.json", manifest)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["trajectory_exports"] >= 1

