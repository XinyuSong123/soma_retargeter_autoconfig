from __future__ import annotations

from soma_retargeter.robotics.v3.target_geometry_audit import audit_target_geometry_matrix


def _row_by_id(matrix: dict, model_id: str) -> dict:
    return {row["model_id"]: row for row in matrix["rows"]}[model_id]


def test_fourier_n1_variant_differential_records_runtime_target_geometry():
    matrix = audit_target_geometry_matrix(
        model_ids=("fourier_n1_urdf", "fourier_n1_mjcf"),
        backend="newton",
    )

    urdf = _row_by_id(matrix, "fourier_n1_urdf")
    mjcf = _row_by_id(matrix, "fourier_n1_mjcf")

    assert urdf["status"] == "runtime_loaded"
    assert mjcf["status"] == "runtime_loaded"
    assert urdf["sites"]["LeftHand"]["body"] == "left_hand_yaw_link"
    assert mjcf["sites"]["LeftHand"]["body"] == "left_end_effector_link"
    assert urdf["paths"]["left_hand"]["active_coordinate_count"] == 5
    assert mjcf["paths"]["left_hand"]["active_coordinate_count"] == 5
    assert urdf["sites"]["LeftHand"]["local_offset_norm"] > 0.13
    assert mjcf["sites"]["LeftHand"]["local_offset_norm"] < 0.031

    differential = matrix["differentials"]["fourier_n1"]
    assert differential["pair_status"] == "compared"
    assert differential["path_differences"]["left_hand"]["target_body_changed"] is True
    assert differential["site_differences"]["LeftHand"]["body_changed"] is True


def test_rpo_yaw_only_and_op3_fixed_torso_controls_are_explicit():
    matrix = audit_target_geometry_matrix(
        model_ids=("roboparty_rpo_local", "robotis_op3_mjcf"),
        backend="newton",
    )

    rpo = _row_by_id(matrix, "roboparty_rpo_local")
    op3 = _row_by_id(matrix, "robotis_op3_mjcf")

    assert rpo["paths"]["torso"]["active_coordinate_count"] == 1
    assert rpo["paths"]["torso"]["joint_types"] == ["revolute"]
    assert "yaw_only_torso_control" in rpo["classification_flags"]

    assert op3["paths"]["torso"]["active_coordinate_count"] == 0
    assert op3["paths"]["torso"]["desired_distance"] == 0.0
    assert "fixed_torso_rank_zero_control" in op3["classification_flags"]


def test_h1_g1_passing_controls_have_symmetric_distal_geometry():
    matrix = audit_target_geometry_matrix(
        model_ids=(
            "unitree_h1_urdf",
            "unitree_h1_mjcf",
            "unitree_g1_urdf",
            "unitree_g1_mjcf",
        ),
        backend="newton",
    )

    assert matrix["counts"]["runtime_loaded"] == 4
    for row in matrix["rows"]:
        assert row["status"] == "runtime_loaded"
        assert row["symmetry"]["arm_neutral_distance_delta"] < 0.02
        assert row["symmetry"]["leg_neutral_distance_delta"] < 0.02
        for semantic_name in ("LeftHand", "RightHand", "LeftFoot", "RightFoot"):
            assert row["sites"][semantic_name]["local_offset_norm"] > 1e-6
        for task in ("left_hand", "right_hand", "left_foot", "right_foot"):
            ratio = row["paths"][task]["desired_distance_to_chain_length_ratio"]
            assert ratio is not None
            assert 0.0 < ratio < 1.05
