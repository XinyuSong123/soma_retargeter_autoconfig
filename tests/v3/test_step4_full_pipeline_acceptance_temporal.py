from __future__ import annotations

from pathlib import Path

from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_passing_fixture


def test_step4_temporal_continuity_matrix_is_diagnostic_and_finite(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    temporal = read_json(artifact_dir / "temporal_continuity_matrix.json")

    assert temporal["row_count"] == 32
    assert temporal["finite_count"] == 32
    assert all(row["finite"] is True for row in temporal["rows"])
    assert all(row["temporal_jump_count"] == 0 for row in temporal["rows"])
