from __future__ import annotations

from pathlib import Path

from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_passing_fixture


def test_step4_trajectory_manifest_has_finite_exports(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    manifest = read_json(artifact_dir / "trajectory_export_manifest.json")

    assert manifest["trajectory_exports_count"] == 64
    assert all(row["qpos_finite"] is True for row in manifest["rows"])
    assert all(row["nan_count"] == 0 and row["inf_count"] == 0 for row in manifest["rows"])
    assert all(row["export_hash"] for row in manifest["rows"])
