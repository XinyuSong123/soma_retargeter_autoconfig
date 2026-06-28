from __future__ import annotations

from pathlib import Path

from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_passing_fixture


def test_step4_contact_and_collision_diagnostics_are_finite_and_diagnostic_only(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    support = read_json(artifact_dir / "support_contact_diagnostics.json")
    collision = read_json(artifact_dir / "collision_proxy_diagnostics.json")

    assert support["row_count"] == 32
    assert collision["row_count"] == 32
    assert all(row["finite"] is True for row in support["rows"])
    assert all(row["finite"] is True for row in collision["rows"])
    assert all(row["collision_proxy_count"] == 0 for row in collision["rows"])
