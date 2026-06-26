from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import run_full_fleet_runtime_quality


def test_step3_full_fleet_runner_writes_required_artifact_surface(tmp_path: Path) -> None:
    artifact_root = tmp_path / "runtime_quality"
    result = run_full_fleet_runtime_quality(
        artifact_root=artifact_root,
        step2_profile_root=Path("artifacts/retargeting_v3_step2_capability"),
        step3_shadow_root=Path("artifacts/retargeting_v3_step3_runtime_shadow"),
        lock=Path("assets/robot_zoo/robot_zoo_lock.json"),
        manifest=Path("assets/robot_zoo/robot_zoo_manifest.json"),
        clip_root=Path("assets/motions"),
        required_core_clips=[
            Path("assets/motions/bvh/Neutral_walk_forward_002__A057.bvh"),
            Path("assets/motions/bvh/wave_R_001__A428.bvh"),
            Path("assets/motions/bvh/body_stretch_1_004__A069.bvh"),
            Path("assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh"),
        ],
        short_max_frames=8,
        mid_max_frames=16,
        deterministic_rerun=True,
    )

    required = [
        "environment.json",
        "commands.txt",
        "source_inventory.json",
        "model_matrix.json",
        "profile_resolution_matrix.json",
        "runtime_local_profile_summary.json",
        "clip_inventory.json",
        "target_stream_matrix.json",
        "generic_smoke_matrix.json",
        "pipeline_backed_matrix.json",
        "quality_summary.json",
        "failure_matrix.json",
        "deterministic_rerun.json",
        "acceptance_ledger.json",
        "full_fleet_matrix.json",
        "pipeline_controls.json",
        "test_results/pytest.txt",
        "test_results/junit.xml",
        "test_results/pytest_summary.json",
    ]
    for relative in required:
        assert (artifact_root / relative).exists(), relative

    summary = json.loads((artifact_root / "quality_summary.json").read_text())
    matrix = json.loads((artifact_root / "model_matrix.json").read_text())
    assert result["verdict"] == "PASS"
    assert summary["in_scope_total"] == 44
    assert summary["full_humanoid_total"] == 32
    assert summary["partial_total"] == 3
    assert summary["negative_total"] == 9
    assert summary["status_counts"] == {"passed": 32, "partial_passed": 3, "negative_control_passed": 9}
    assert matrix["in_scope_total"] == 44
    assert len(matrix["rows"]) == 44


def test_step3_full_fleet_runner_records_negative_controls_without_promotion(tmp_path: Path) -> None:
    artifact_root = tmp_path / "runtime_quality"
    run_full_fleet_runtime_quality(
        artifact_root=artifact_root,
        step2_profile_root=Path("artifacts/retargeting_v3_step2_capability"),
        step3_shadow_root=Path("artifacts/retargeting_v3_step3_runtime_shadow"),
        lock=Path("assets/robot_zoo/robot_zoo_lock.json"),
        manifest=Path("assets/robot_zoo/robot_zoo_manifest.json"),
        clip_root=Path("assets/motions"),
        required_core_clips=[Path("assets/motions/bvh/Neutral_walk_forward_002__A057.bvh")],
        short_max_frames=4,
        mid_max_frames=8,
        deterministic_rerun=True,
    )
    matrix = json.loads((artifact_root / "model_matrix.json").read_text())
    negatives = [row for row in matrix["rows"] if row["profile_status"] == "negative_control_passed"]

    assert len(negatives) == 9
    for row in negatives:
        assert row["expected_capability"] == "negative_control"
        assert row["negative_control_status"] == "negative_control_rejected"
        assert row["humanoid_profile_generated"] is False
        assert row["promoted_to_runtime_quality"] is False
        assert row["override_allowed"] is False
