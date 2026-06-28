from __future__ import annotations

from pathlib import Path

from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_passing_fixture


def test_step4_deterministic_rerun_covers_all_44_rows(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")

    assert deterministic["status"] == "passed"
    assert deterministic["deterministic"] is True
    assert deterministic["deterministic_compared_count"] == 44
    assert deterministic["deterministic_matched_count"] == 44
