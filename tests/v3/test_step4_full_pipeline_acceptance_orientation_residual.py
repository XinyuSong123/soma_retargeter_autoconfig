from __future__ import annotations

from pathlib import Path

from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_passing_fixture


def test_step4_orientation_taxonomy_records_breakthrough(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    taxonomy = read_json(artifact_dir / "orientation_residual_taxonomy.json")

    assert taxonomy["row_count"] == 32
    assert taxonomy["summary"]["accepted_breakthrough"] is True
    assert taxonomy["summary"]["p95_rotation_residual_p95"] < 1.20
    assert all(row["orientation_dominates"] for row in taxonomy["rows"])
