from __future__ import annotations

from pathlib import Path

from soma_retargeter.robotics.v3.failure_analysis import (
    BASELINE_ROBOT_IDS,
    build_baseline_failure_ledger,
)


FAILURE_DIR = Path("artifacts/retargeting_v3_step2_assets44/failures")

EXPECTED_ROBOT_IDS = (
    "atlas_drc_urdf",
    "atlas_v4_urdf",
    "booster_t1_mjcf",
    "booster_t1_urdf",
    "fourier_n1_mjcf",
    "jaxon_urdf",
    "mujoco_humanoid_mjcf",
    "pal_talos_mjcf_direct",
    "robotis_op3_mjcf",
    "talos_urdf",
    "valkyrie_urdf",
)

EXPECTED_FAILURES_BY_ROBOT = {
    "atlas_drc_urdf": 2,
    "atlas_v4_urdf": 1,
    "booster_t1_mjcf": 2,
    "booster_t1_urdf": 2,
    "fourier_n1_mjcf": 3,
    "jaxon_urdf": 26,
    "mujoco_humanoid_mjcf": 7,
    "pal_talos_mjcf_direct": 2,
    "robotis_op3_mjcf": 2,
    "talos_urdf": 1,
    "valkyrie_urdf": 2,
}


def test_capability_baseline_freezes_exact_robot_ids_and_counts() -> None:
    ledger = build_baseline_failure_ledger(FAILURE_DIR)

    assert BASELINE_ROBOT_IDS == EXPECTED_ROBOT_IDS
    assert tuple(ledger["frozen_robot_ids"]) == EXPECTED_ROBOT_IDS
    assert ledger["counts"]["robots"] == 11
    assert ledger["counts"]["failed_rows"] == 50
    assert ledger["counts"]["failure_rows_by_robot"] == EXPECTED_FAILURES_BY_ROBOT
    assert ledger["counts"]["failure_rows_by_metric_type"] == {
        "numerical_stability_gate": 5,
        "projection_position_normalized_residual": 43,
        "projection_rotation_normalized_residual": 2,
    }


def test_capability_baseline_keeps_every_failure_as_failed_evidence() -> None:
    ledger = build_baseline_failure_ledger(FAILURE_DIR)

    original_messages = {}
    for report in ledger["source_reports"]:
        original_messages[report["robot_id"]] = set(report["failures"])

    for row in ledger["failures"]:
        assert row["status"] == "failed"
        assert row["original_failure_message"] in original_messages[row["robot_id"]]
        assert row["metric_type"] in {
            "numerical_stability_gate",
            "projection_position_normalized_residual",
            "projection_rotation_normalized_residual",
        }
        assert row["threshold"]
        assert row["actual"]
        assert isinstance(row["active_coordinates"], list)
        assert len(row["active_joint_limits"]) == len(row["active_coordinates"])
        assert row["engine_jacobian_source"]["primary"] == "engine_relative_jacobian"
        assert row["engine_jacobian_source"]["source"] == "newton.eval_jacobian"
        assert "status" in row["solver"]
        assert "message" in row["solver"]
        assert row["semantic_source"] == "verified_semantic_map"
        assert isinstance(row["semantic_hash"], str)
        assert len(row["semantic_hash"]) == 64
        assert isinstance(row["chain_length"]["body_path_edges"], int)
        assert row["chain_length"]["body_path_edges"] >= 0
        assert "target_distance" in row
        assert "target_angle" in row
        assert row["rank"]["task_block"] in {"translation", "rotation", "translation+rotation"}


def test_capability_baseline_records_exact_projection_thresholds_and_residuals() -> None:
    ledger = build_baseline_failure_ledger(FAILURE_DIR)
    rows = {
        (row["robot_id"], row["motion"], row["task"]): row
        for row in ledger["failures"]
        if row["metric_type"].startswith("projection_")
    }

    jaxon_neutral = rows[("jaxon_urdf", "neutral", "left_hand")]
    assert jaxon_neutral["threshold"]["value"] == 0.001
    assert jaxon_neutral["actual"]["normalized_residual"] == 0.20336935196722078
    assert jaxon_neutral["actual"]["residual"] == 0.21676873480936135

    booster_torso = rows[("booster_t1_mjcf", "mixed_torso_rotation", "torso")]
    assert booster_torso["metric_type"] == "projection_rotation_normalized_residual"
    assert booster_torso["threshold"]["value"] == 0.08
    assert booster_torso["actual"]["normalized_residual"] == 0.08070212239751208
    assert booster_torso["actual"]["residual"] == 0.25353319485312825
    assert booster_torso["target_angle"] == 0.26925824035672535

    atlas_hand = rows[("atlas_drc_urdf", "crossed_body_reach", "left_hand")]
    assert atlas_hand["threshold"]["value"] == 0.12
    assert atlas_hand["actual"]["normalized_residual"] == 0.29757941765985046
    assert atlas_hand["actual"]["residual"] == 0.4720756454811152
    assert atlas_hand["active_coordinates"] == [3, 4, 5, 6, 7, 8, 9]


def test_capability_baseline_records_rank_thresholds_for_numerical_failures() -> None:
    ledger = build_baseline_failure_ledger(FAILURE_DIR)
    rows = {
        (row["robot_id"], row["motion"], row["task"]): row
        for row in ledger["failures"]
        if row["metric_type"] == "numerical_stability_gate"
    }

    assert set(rows) == {
        ("mujoco_humanoid_mjcf", "rank_stability", "left_foot"),
        ("mujoco_humanoid_mjcf", "rank_stability", "left_hand"),
        ("mujoco_humanoid_mjcf", "rank_stability", "right_foot"),
        ("mujoco_humanoid_mjcf", "rank_stability", "right_hand"),
        ("mujoco_humanoid_mjcf", "rank_stability", "torso"),
    }

    left_foot = rows[("mujoco_humanoid_mjcf", "rank_stability", "left_foot")]
    assert left_foot["threshold"]["criteria"]["engine_fd_normalized_error_p95_max"] == 0.02
    assert left_foot["actual"]["engine_fd_normalized_error_p95"] == 0.9861013004639585
    assert left_foot["actual"]["numerical_stability_gate_passed"] is False
    assert left_foot["rank"]["engine_rank_translation"] == 3
    assert left_foot["rank"]["fd_rank_translation"] == 3
    assert left_foot["rank"]["projector_distance_p95"] == 1.3344794978092242e-15

    left_hand = rows[("mujoco_humanoid_mjcf", "rank_stability", "left_hand")]
    assert left_hand["rank"]["engine_rank_rotation"] == 3
    assert left_hand["rank"]["fd_rank_rotation"] == 2
    assert left_hand["rank"]["rank_agreement_rate_rotation"] == 0.75
