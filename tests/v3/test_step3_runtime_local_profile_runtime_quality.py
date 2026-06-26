from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.runtime.v3.fleet_inventory import load_fleet_runtime_cases
from soma_retargeter.runtime.v3.runtime_local_profile import close_runtime_profile, write_profile_resolution_artifacts


def test_step3_1_runtime_profile_closure_is_terminal_for_all_rows(tmp_path: Path) -> None:
    cases = load_fleet_runtime_cases()
    closures = [close_runtime_profile(case, artifact_root=tmp_path) for case in cases]

    assert len(closures) == 44
    statuses = {closure.resolution_status for closure in closures}
    assert "runtime_model_load_failed" not in statuses
    assert "negative_control_rejected" in statuses
    assert all(
        closure.resolution_status in {
            "profile_match",
            "runtime_local_profile_generated",
            "runtime_local_profile_failed",
            "structured_partial_supported",
            "negative_control_rejected",
        }
        for closure in closures
    )


def test_step3_1_runtime_profile_closure_writes_matrix_and_summary(tmp_path: Path) -> None:
    cases = load_fleet_runtime_cases()
    closures = [close_runtime_profile(case, artifact_root=tmp_path) for case in cases]
    matrix, summary = write_profile_resolution_artifacts(artifact_root=tmp_path, closures=closures)

    assert matrix["row_count"] == 44
    assert summary["row_count"] == 44
    assert summary["negative_control_rejected_count"] == 9
    assert summary["structured_partial_supported_count"] == 3
    assert (tmp_path / "profile_resolution_matrix.json").exists()
    assert (tmp_path / "runtime_local_profile_summary.json").exists()
    for case in cases:
        payload = json.loads((tmp_path / "per_model" / case.model_id / "profile_resolution.json").read_text())
        assert payload["model_id"] == case.model_id
        assert not Path(payload["runtime_source_path"]).is_absolute()
