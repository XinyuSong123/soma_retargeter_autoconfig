import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from soma_retargeter.robotics.reachability import rotation_vector_to_quat_xyzw
from soma_retargeter.tools.benchmark_retargeting import (
    _aggregate_motion_metrics,
    _legacy_runtime_retargeter_config,
    _profile_runtime_residual_metrics,
    _runtime_retargeter_config,
    _runtime_metrics_for_buffer,
    _select_motion_window,
    build_benchmark_gate_report,
    build_registry_coverage_report,
    main,
)


class TestBenchmarkRetargeting(unittest.TestCase):
    def test_legacy_runtime_config_does_not_use_compiled_profile(self):
        cfg = _legacy_runtime_retargeter_config("roboparty_rpo")
        self.assertNotIn("compiled_retarget_profile", cfg)
        self.assertNotIn("direction_tasks", cfg)
        self.assertNotIn("pole_vector_tasks", cfg)
        self.assertEqual(cfg["ik_map"]["Hips"]["t_weight"], 10.0)
        self.assertEqual(cfg["ik_map"]["Hips"]["r_weight"], 2.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["t_weight"], 0.5)
        self.assertEqual(cfg["ik_map"]["Chest"]["r_weight"], 0.5)

    def test_legacy_runtime_config_merges_contact_anchor_offsets(self):
        cfg = _legacy_runtime_retargeter_config("e3_v2")
        contact_cfg = cfg["contact_aware_foot_ik"]
        self.assertTrue(contact_cfg["enabled"])
        self.assertIn("anchor_offsets", contact_cfg)
        self.assertEqual(contact_cfg["anchor_offsets"]["left"]["toe"], [0.17, 0.0, -0.055])

    def test_pole_analytic_compare_mode_forces_only_pole_jacobians(self):
        self.assertIsNone(_runtime_retargeter_config("roboparty_rpo", "v2"))
        cfg = _runtime_retargeter_config("roboparty_rpo", "v2_pole_analytic")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_pole_analytic")
        self.assertTrue(cfg["pole_vector_tasks"])
        self.assertTrue(all(task["analytic_jacobian"] for task in cfg["pole_vector_tasks"]))
        self.assertTrue(any(not task["analytic_jacobian"] for task in cfg["direction_tasks"]))
        self.assertIn("force analytic pole-vector", cfg["pole_vector_tasks"][0]["jacobian_schedule_reason"])

    def test_no_pole_compare_mode_removes_pole_tasks_only(self):
        cfg = _runtime_retargeter_config("unitree_g1", "v2_no_pole")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_no_pole")
        self.assertEqual(cfg["pole_vector_tasks"], [])
        self.assertTrue(cfg["direction_tasks"])

    def test_projected_position_compare_mode_uses_profile_translation_basis(self):
        cfg = _runtime_retargeter_config("e3_v2", "v2_pos_projected")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_pos_projected")
        self.assertEqual(cfg["ik_map"]["LeftHand"]["t_weight"], 0.0)
        self.assertIn("rank-0", cfg["ik_map"]["LeftHand"]["v2_position_disabled_reason"])
        self.assertIn("v2_position_basis", cfg["ik_map"]["LeftFoot"])
        self.assertEqual(len(cfg["ik_map"]["LeftFoot"]["v2_position_basis"]), 3)
        self.assertEqual(len(cfg["ik_map"]["LeftFoot"]["v2_position_basis"][0]), 1)

    def test_pole_keep_compare_mode_filters_pole_tasks_by_semantic_selector(self):
        cfg = _runtime_retargeter_config("unitree_g1", "v2_pole_keep_hand")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_pole_keep_hand")
        self.assertTrue(cfg["direction_tasks"])
        self.assertEqual(
            [task["name"] for task in cfg["pole_vector_tasks"]],
            ["LeftForeArm_pole_vector", "RightForeArm_pole_vector"],
        )
        self.assertIn("pole selector hand", cfg["pole_vector_tasks"][0]["selection_reason"])

        foot_cfg = _runtime_retargeter_config("unitree_g1", "v2_pole_keep_foot")
        self.assertEqual(
            [task["name"] for task in foot_cfg["pole_vector_tasks"]],
            ["LeftShin_pole_vector", "RightShin_pole_vector"],
        )

        distal_cfg = _runtime_retargeter_config("unitree_g1", "v2_pole_keep_distal")
        self.assertEqual(
            [task["name"] for task in distal_cfg["pole_vector_tasks"]],
            [
                "LeftForeArm_pole_vector",
                "LeftShin_pole_vector",
                "RightForeArm_pole_vector",
                "RightShin_pole_vector",
            ],
        )

        proximal_cfg = _runtime_retargeter_config("unitree_g1", "v2_pole_keep_proximal")
        self.assertEqual(
            [task["name"] for task in proximal_cfg["pole_vector_tasks"]],
            [
                "LeftArm_pole_vector",
                "LeftLeg_pole_vector",
                "RightArm_pole_vector",
                "RightLeg_pole_vector",
            ],
        )

    def test_direction_analytic_compare_mode_filters_direction_tasks_by_selector(self):
        cfg = _runtime_retargeter_config("roboparty_rpo", "v2_direction_analytic_leg")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_direction_analytic_leg")
        analytic_by_name = {task["name"]: task["analytic_jacobian"] for task in cfg["direction_tasks"]}
        self.assertFalse(analytic_by_name["LeftArm_direction"])
        self.assertFalse(analytic_by_name["RightForeArm_direction"])
        self.assertTrue(analytic_by_name["LeftLeg_direction"])
        self.assertTrue(analytic_by_name["RightShin_direction"])
        self.assertTrue(all(not task["analytic_jacobian"] for task in cfg["pole_vector_tasks"]))
        self.assertIn("direction by selector leg", cfg["direction_tasks"][2]["jacobian_schedule_reason"])

        all_cfg = _runtime_retargeter_config("roboparty_rpo", "v2_direction_analytic_all")
        self.assertTrue(all(task["analytic_jacobian"] for task in all_cfg["direction_tasks"]))

    def test_iteration_compare_mode_changes_only_iteration_count(self):
        baseline = _runtime_retargeter_config("unitree_g1", "v2_pole_analytic")
        cfg = _runtime_retargeter_config("unitree_g1", "v2_iter4")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_iter4")
        self.assertEqual(cfg["ik_iterations"], 4)
        self.assertTrue(cfg["direction_tasks"])
        self.assertTrue(cfg["pole_vector_tasks"])
        self.assertTrue(all(task["analytic_jacobian"] for task in cfg["pole_vector_tasks"]))
        self.assertTrue(all(task["residual_mode"] == "tangent2" for task in cfg["pole_vector_tasks"]))
        self.assertEqual(cfg["pole_vector_tasks"][0]["weight"], baseline["pole_vector_tasks"][0]["weight"])

    def test_hand_weight_compare_mode_overrides_only_hand_position_weights(self):
        cfg = _runtime_retargeter_config("e3_v2", "v2_hand_w200")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_hand_w200")
        self.assertEqual(cfg["ik_map"]["LeftHand"]["t_weight"], 200.0)
        self.assertEqual(cfg["ik_map"]["RightHand"]["t_weight"], 200.0)
        self.assertNotEqual(cfg["ik_map"]["LeftFoot"]["t_weight"], 200.0)
        self.assertIn("benchmark experiment", cfg["ik_map"]["LeftHand"]["v2_position_weight_source"])

    def test_foot_weight_compare_mode_overrides_only_foot_position_weights(self):
        cfg = _runtime_retargeter_config("roboparty_rpo", "v2_foot_w1500")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_foot_w1500")
        self.assertEqual(cfg["ik_map"]["LeftFoot"]["t_weight"], 1500.0)
        self.assertEqual(cfg["ik_map"]["RightFoot"]["t_weight"], 1500.0)
        self.assertNotEqual(cfg["ik_map"]["LeftHand"]["t_weight"], 1500.0)
        self.assertIn("override foot position weight", cfg["ik_map"]["LeftFoot"]["v2_position_weight_source"])

    def test_hips_rotation_compare_mode_overrides_hips_rotation_weight(self):
        cfg = _runtime_retargeter_config("roboparty_rpo", "v2_hips_r2")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_hips_r2")
        self.assertEqual(cfg["ik_map"]["Hips"]["r_weight"], 2.0)
        self.assertIn("override hips rotation weight", cfg["ik_map"]["Hips"]["v2_rotation_weight_source"])

        combined = _runtime_retargeter_config("roboparty_rpo", "v2_hand_w10_hips_r2")
        self.assertEqual(combined["benchmark_compare_mode"], "v2_hand_w10_hips_r2")
        self.assertEqual(combined["ik_map"]["LeftHand"]["t_weight"], 10.0)
        self.assertEqual(combined["ik_map"]["Hips"]["r_weight"], 2.0)

    def test_weighted_pole_analytic_compare_mode_scales_pole_weights(self):
        baseline = _runtime_retargeter_config("unitree_g1", "v2_pole_analytic")
        scaled = _runtime_retargeter_config("unitree_g1", "v2_pole_analytic_w0.25")
        self.assertEqual(scaled["benchmark_compare_mode"], "v2_pole_analytic_w0.25")
        self.assertTrue(scaled["pole_vector_tasks"])
        self.assertTrue(all(task["analytic_jacobian"] for task in scaled["pole_vector_tasks"]))
        self.assertAlmostEqual(
            scaled["pole_vector_tasks"][0]["weight"],
            baseline["pole_vector_tasks"][0]["weight"] * 0.25,
        )
        self.assertAlmostEqual(
            scaled["pole_vector_tasks"][0]["normalized_weight"],
            baseline["pole_vector_tasks"][0]["normalized_weight"] * 0.25,
        )

    def test_tangent_pole_analytic_compare_mode_sets_residual_mode(self):
        cfg = _runtime_retargeter_config("unitree_g1", "v2_pole_tangent_analytic")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_pole_tangent_analytic")
        self.assertTrue(cfg["pole_vector_tasks"])
        self.assertTrue(all(task["analytic_jacobian"] for task in cfg["pole_vector_tasks"]))
        self.assertTrue(all(task["residual_mode"] == "tangent2" for task in cfg["pole_vector_tasks"]))
        self.assertIn("tangent-space", cfg["pole_vector_tasks"][0]["residual_mode_reason"])

    def test_combined_pole_analytic_and_hand_weight_compare_mode(self):
        cfg = _runtime_retargeter_config("unitree_g1", "v2_pole_analytic_w0.05_hand_w200")
        self.assertEqual(cfg["benchmark_compare_mode"], "v2_pole_analytic_w0.05_hand_w200")
        self.assertTrue(cfg["pole_vector_tasks"])
        self.assertTrue(all(task["analytic_jacobian"] for task in cfg["pole_vector_tasks"]))
        self.assertAlmostEqual(cfg["pole_vector_tasks"][0]["weight"], 0.5)
        self.assertEqual(cfg["ik_map"]["LeftHand"]["t_weight"], 200.0)
        self.assertEqual(cfg["ik_map"]["RightHand"]["t_weight"], 200.0)
        self.assertNotEqual(cfg["ik_map"]["LeftFoot"]["t_weight"], 200.0)
        self.assertIn("force analytic pole-vector", cfg["pole_vector_tasks"][0]["jacobian_schedule_reason"])
        self.assertIn("override hand position weight", cfg["ik_map"]["LeftHand"]["v2_position_weight_source"])

    def test_profile_runtime_residual_metrics_include_torso_leakage(self):
        profile = {
            "tasks": [
                {
                    "name": "torso_projected_relative_rotation",
                    "task_type": "projected_relative_rotation",
                    "target_site": "Chest",
                    "reference_site": "Hips",
                    "priority": 2,
                    "rotation_mask_or_basis": [[0.0], [0.0], [1.0]],
                    "enabled": True,
                },
                {
                    "name": "LeftHand_position",
                    "task_type": "position",
                    "target_site": "LeftHand",
                    "priority": 3,
                    "characteristic_length": 2.0,
                    "enabled": True,
                },
            ]
        }
        target = np.zeros((1, 3, 7), dtype=np.float64)
        target[:, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0])
        target[0, 1, 3:7] = rotation_vector_to_quat_xyzw(np.array([0.0, 0.0, 0.4]))
        target[0, 2, 0:3] = np.array([1.0, 0.0, 0.0])
        pipeline = type("P", (), {"input_targets": [target], "mapped_joints": ["Hips", "Chest", "LeftHand"]})()
        semantic_pose = {
            "Chest": {
                "position": np.zeros((1, 3), dtype=np.float64),
                "rotation": np.asarray([rotation_vector_to_quat_xyzw(np.array([0.2, 0.0, 0.4]))]),
            },
            "LeftHand": {
                "position": np.asarray([[1.2, 0.0, 0.0]], dtype=np.float64),
                "rotation": np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            },
        }

        metrics = _profile_runtime_residual_metrics(profile, pipeline, 0, semantic_pose)

        self.assertEqual(metrics["task_residual_by_type_priority"]["status"], "ok")
        self.assertIn("position:p3", metrics["task_residual_by_type_priority"]["groups"])
        self.assertAlmostEqual(metrics["task_residual_by_type_priority"]["groups"]["position:p3"]["value"], 0.1)
        self.assertEqual(metrics["torso_reachable_residual"]["status"], "ok")
        self.assertLess(metrics["torso_reachable_residual"]["value"], 0.01)
        self.assertEqual(metrics["torso_unreachable_residual"]["status"], "ok")
        self.assertGreater(metrics["torso_unreachable_residual"]["value"], 0.19)

    def test_select_motion_window_prefers_high_motion_segment(self):
        transforms = np.zeros((8, 1, 7), dtype=np.float64)
        transforms[:, :, 3] = 1.0
        transforms[4, 0, 0] = 10.0
        transforms[5, 0, 0] = 20.0
        transforms[6:, 0, 0] = 20.0
        animation = type("A", (), {"num_frames": 8, "local_transforms": transforms})()

        start, count, mode = _select_motion_window(animation, 3)

        self.assertEqual((start, count, mode), (3, 3, "max_motion_window"))

    def test_runtime_smoothness_metrics_split_root_from_actuated_joints(self):
        frames = np.zeros((3, 9), dtype=np.float32)
        frames[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        frames[1, 0] = 10.0
        frames[2, 0] = 20.0
        frames[1, 7] = 0.1
        frames[2, 7] = 0.2
        buffer = type("B", (), {"data": frames, "sample_rate": 10.0})()
        pipeline = type("P", (), {"ik_model": None})()

        with (
            mock.patch("soma_retargeter.tools.benchmark_retargeting._joint_limit_margin", return_value=0.0),
            mock.patch("soma_retargeter.tools.benchmark_retargeting._body_site_pose_trajectories", return_value={}),
        ):
            metrics = _runtime_metrics_for_buffer({"semantic_sites": {}}, pipeline, 0, buffer)

        self.assertAlmostEqual(metrics["velocity_p95"]["value"], 1.0)
        self.assertAlmostEqual(metrics["root_velocity_p95"]["value"], 100.0)
        self.assertEqual(metrics["velocity_p95"]["unit"], "actuated_joint_coord_per_s")
        self.assertEqual(metrics["root_velocity_p95"]["unit"], "root_coord_per_s")

    def test_runtime_tracking_metrics_skip_initialization_and_stabilization_targets(self):
        frames = np.zeros((2, 7), dtype=np.float32)
        frames[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buffer = type("B", (), {"data": frames, "sample_rate": 30.0})()
        actual_positions = np.asarray([[1.0, 2.0, 3.0], [1.5, 2.0, 3.0]], dtype=np.float64)
        target = np.zeros((4, 1, 7), dtype=np.float64)
        target[:, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        target[0:2, 0, 0:3] = np.asarray([[20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=np.float64)
        target[2:, 0, 0:3] = actual_positions
        pipeline = type(
            "P",
            (),
            {
                "ik_model": None,
                "input_targets": [target],
                "mapped_joints": ["LeftHand"],
                "num_initialization_frames": 1,
                "num_stabilization_frames": 1,
            },
        )()

        with (
            mock.patch("soma_retargeter.tools.benchmark_retargeting._joint_limit_margin", return_value=0.0),
            mock.patch(
                "soma_retargeter.tools.benchmark_retargeting._body_site_pose_trajectories",
                return_value={"LeftHand": {"position": actual_positions}},
            ),
        ):
            metrics = _runtime_metrics_for_buffer({"semantic_sites": {}}, pipeline, 0, buffer)

        self.assertEqual(metrics["hand_position_rmse"]["status"], "ok")
        self.assertAlmostEqual(metrics["hand_position_rmse"]["value"], 0.0)
        self.assertEqual(metrics["hand_position_rmse"]["axis_order"], ["x", "y", "z"])
        self.assertEqual(metrics["hand_position_rmse"]["sample_count"], 2)
        self.assertTrue(np.allclose(metrics["hand_position_rmse"]["axis_rmse"], [0.0, 0.0, 0.0]))
        self.assertIn("LeftHand", metrics["hand_position_rmse"]["by_semantic"])
        self.assertEqual(metrics["hand_position_rmse"]["by_semantic"]["LeftHand"]["sample_count"], 2)

    def test_runtime_tracking_metrics_include_axis_error_diagnostics(self):
        frames = np.zeros((2, 7), dtype=np.float32)
        frames[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buffer = type("B", (), {"data": frames, "sample_rate": 30.0})()
        actual_positions = np.asarray([[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]], dtype=np.float64)
        target = np.zeros((2, 1, 7), dtype=np.float64)
        target[:, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        target[:, 0, 0:3] = np.asarray([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64)
        pipeline = type("P", (), {"ik_model": None, "input_targets": [target], "mapped_joints": ["LeftHand"]})()

        with (
            mock.patch("soma_retargeter.tools.benchmark_retargeting._joint_limit_margin", return_value=0.0),
            mock.patch(
                "soma_retargeter.tools.benchmark_retargeting._body_site_pose_trajectories",
                return_value={"LeftHand": {"position": actual_positions}},
            ),
        ):
            metrics = _runtime_metrics_for_buffer({"semantic_sites": {}}, pipeline, 0, buffer)

        self.assertAlmostEqual(metrics["hand_position_rmse"]["value"], np.sqrt((1.0 + 4.0) / 2.0))
        self.assertTrue(np.allclose(metrics["hand_position_rmse"]["axis_rmse"], [np.sqrt(2.5), 0.0, 0.0]))
        self.assertTrue(np.allclose(metrics["hand_position_rmse"]["mean_error"], [1.5, 0.0, 0.0]))
        self.assertTrue(np.allclose(metrics["hand_position_rmse"]["p95_abs_error"], [1.95, 0.0, 0.0]))
        self.assertAlmostEqual(metrics["hand_position_rmse"]["by_semantic"]["LeftHand"]["value"], np.sqrt(2.5))
        self.assertTrue(
            np.allclose(metrics["hand_position_rmse"]["by_semantic"]["LeftHand"]["axis_rmse"], [np.sqrt(2.5), 0.0, 0.0])
        )

    def test_runtime_tracking_metrics_include_reachable_projection_diagnostics(self):
        frames = np.zeros((2, 7), dtype=np.float32)
        frames[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buffer = type("B", (), {"data": frames, "sample_rate": 30.0})()
        actual_positions = np.asarray([[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]], dtype=np.float64)
        target = np.zeros((2, 1, 7), dtype=np.float64)
        target[:, :, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        target[:, 0, 0:3] = np.asarray([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64)
        pipeline = type("P", (), {"ik_model": None, "input_targets": [target], "mapped_joints": ["LeftHand"]})()
        profile = {
            "semantic_sites": {},
            "chains": {
                "LeftHand": {
                    "translational_basis": [[0.0], [1.0], [0.0]],
                },
            },
        }

        with (
            mock.patch("soma_retargeter.tools.benchmark_retargeting._joint_limit_margin", return_value=0.0),
            mock.patch(
                "soma_retargeter.tools.benchmark_retargeting._body_site_pose_trajectories",
                return_value={"LeftHand": {"position": actual_positions}},
            ),
        ):
            metrics = _runtime_metrics_for_buffer(profile, pipeline, 0, buffer)

        self.assertAlmostEqual(metrics["hand_position_rmse"]["value"], np.sqrt((1.0 + 4.0) / 2.0))
        reachable = metrics["hand_reachable_position_rmse"]
        self.assertEqual(reachable["status"], "ok")
        self.assertEqual(reachable["projection_ranks"], [1])
        self.assertAlmostEqual(reachable["value"], 0.0)
        self.assertTrue(np.allclose(reachable["axis_rmse"], [0.0, 0.0, 0.0]))
        self.assertEqual(reachable["by_semantic"]["LeftHand"]["projection_rank"], 1)

    def test_aggregate_motion_metrics_preserves_tracking_axis_diagnostics(self):
        aggregated = _aggregate_motion_metrics(
            [
                {
                    "metrics": {
                        "hand_position_rmse": {
                            "status": "ok",
                            "value": 1.0,
                            "axis_rmse": [1.0, 0.0, 0.0],
                            "mean_error": [1.0, 0.0, 0.0],
                            "p95_abs_error": [1.0, 0.0, 0.0],
                            "axis_order": ["x", "y", "z"],
                            "sample_count": 1,
                            "by_semantic": {
                                "LeftHand": {
                                    "status": "ok",
                                    "value": 1.0,
                                    "axis_rmse": [1.0, 0.0, 0.0],
                                    "mean_error": [1.0, 0.0, 0.0],
                                    "p95_abs_error": [1.0, 0.0, 0.0],
                                    "axis_order": ["x", "y", "z"],
                                    "sample_count": 1,
                                }
                            },
                        }
                    }
                },
                {
                    "metrics": {
                        "hand_position_rmse": {
                            "status": "ok",
                            "value": 3.0,
                            "axis_rmse": [3.0, 0.0, 0.0],
                            "mean_error": [3.0, 0.0, 0.0],
                            "p95_abs_error": [3.0, 0.0, 0.0],
                            "axis_order": ["x", "y", "z"],
                            "sample_count": 3,
                            "by_semantic": {
                                "LeftHand": {
                                    "status": "ok",
                                    "value": 3.0,
                                    "axis_rmse": [3.0, 0.0, 0.0],
                                    "mean_error": [3.0, 0.0, 0.0],
                                    "p95_abs_error": [3.0, 0.0, 0.0],
                                    "axis_order": ["x", "y", "z"],
                                    "sample_count": 3,
                                }
                            },
                        }
                    }
                },
            ]
        )

        payload = aggregated["hand_position_rmse"]
        self.assertEqual(payload["value"], 2.0)
        self.assertEqual(payload["sample_count"], 4)
        self.assertTrue(np.allclose(payload["axis_rmse"], [2.5, 0.0, 0.0]))
        self.assertEqual(payload["by_semantic"]["LeftHand"]["sample_count"], 4)
        self.assertTrue(np.allclose(payload["by_semantic"]["LeftHand"]["axis_rmse"], [2.5, 0.0, 0.0]))

    def test_registry_coverage_report_marks_missing_and_incomplete_targets(self):
        report = build_registry_coverage_report(("roboparty_rpo", "unitree_g1", "unitree_g1_23dof", "unitree_g1_29dof", "e3_v2", "oli"))
        by_name = {entry["requested_name"]: entry for entry in report["robots"]}

        self.assertEqual(by_name["roboparty_rpo"]["status"], "ready")
        self.assertTrue(by_name["roboparty_rpo"]["paths"]["mjcf_path"]["exists"])
        self.assertTrue(by_name["roboparty_rpo"]["compiled_profile"]["exists"])
        self.assertEqual(by_name["unitree_g1"]["status"], "ready")
        self.assertTrue(by_name["unitree_g1"]["paths"]["mjcf_path"]["exists"])
        self.assertEqual(by_name["unitree_g1"]["blockers"], [])
        self.assertEqual(by_name["unitree_g1_23dof"]["resolved_name"], "unitree_g1_23dof")
        self.assertEqual(by_name["unitree_g1_23dof"]["status"], "ready")
        self.assertTrue(by_name["unitree_g1_23dof"]["paths"]["mjcf_path"]["exists"])
        self.assertTrue(by_name["unitree_g1_23dof"]["compiled_profile"]["exists"])
        self.assertEqual(by_name["unitree_g1_23dof"]["blockers"], [])
        self.assertEqual(by_name["unitree_g1_29dof"]["resolved_name"], "unitree_g1")
        self.assertEqual(by_name["unitree_g1_29dof"]["status"], "ready")
        self.assertTrue(by_name["unitree_g1_29dof"]["paths"]["mjcf_path"]["exists"])
        self.assertEqual(by_name["unitree_g1_29dof"]["blockers"], [])
        self.assertEqual(by_name["e3_v2"]["status"], "ready")
        self.assertEqual(by_name["e3_v2"]["asset_kind"], "synthetic_fixture")
        self.assertTrue(by_name["e3_v2"]["paths"]["mjcf_path"]["exists"])
        self.assertTrue(by_name["e3_v2"]["compiled_profile"]["exists"])
        self.assertEqual(by_name["e3_v2"]["blockers"], [])
        self.assertEqual(by_name["oli"]["status"], "missing_registration")
        self.assertEqual(by_name["oli"]["onboarding_report"]["status"], "missing_assets_or_registration")
        self.assertIn("mjcf_or_xml", by_name["oli"]["onboarding_report"]["required_files"])
        self.assertEqual(
            sorted(by_name["oli"]["onboarding_report"]["minimal_semantic_ik_map_template"].keys()),
            ["Chest", "Hips", "LeftFoot", "LeftHand", "RightFoot", "RightHand"],
        )
        self.assertIn(
            "python -m soma_retargeter.tools.autoconfigure_robot --robot oli --validate-only",
            by_name["oli"]["onboarding_report"]["next_commands"],
        )

    def test_benchmark_gate_report_flags_threshold_failures(self):
        result = {
            "robot": "fixture_bot",
            "compare_results": {
                "legacy": {
                    "metrics": {
                        "penetration": {"status": "ok", "value": 0.01},
                        "hand_position_rmse": {"status": "ok", "value": 1.0},
                        "runtime_seconds": {"motion_runtime": 1.0},
                    }
                },
                "v2": {
                    "metrics": {
                        "penetration": {"status": "ok", "value": 0.02},
                        "hand_position_rmse": {"status": "ok", "value": 1.04},
                        "runtime_seconds": {"motion_runtime": 1.4},
                    }
                },
            },
        }

        report = build_benchmark_gate_report([result], strict=True)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["strict"])
        gates = {gate["metric"]: gate for gate in report["robots"][0]["gates"]}
        self.assertEqual(gates["penetration"]["status"], "failed")
        self.assertEqual(gates["hand_position_rmse"]["status"], "passed")
        self.assertEqual(gates["runtime_seconds.motion_runtime"]["status"], "failed")
        self.assertEqual(gates["root_tilt"]["status"], "unavailable")

    def test_benchmark_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bench"
            rc = main([
                "--robots",
                "roboparty_rpo",
                "--compare",
                "legacy",
                "v2",
                "--output",
                str(out),
            ])

            self.assertEqual(rc, 0)
            for rel in (
                "benchmark_summary.json",
                "benchmark_frames.csv",
                "benchmark_gates.json",
                "environment.json",
                "commands.txt",
                "registry_coverage.json",
                "per_robot/roboparty_rpo.json",
            ):
                self.assertTrue((out / rel).exists(), rel)

            summary = json.loads((out / "benchmark_summary.json").read_text())
            self.assertEqual(summary["status"], "ok")
            self.assertIn("task_residual_by_type_priority", summary["metric_names"])
            self.assertEqual(summary["robots"], ["roboparty_rpo"])
            self.assertIn("registry_coverage", summary)
            self.assertIn("benchmark_gates", summary)
            coverage = json.loads((out / "registry_coverage.json").read_text())
            self.assertEqual(summary["registry_coverage"]["status_counts"], coverage["status_counts"])
            gates = json.loads((out / "benchmark_gates.json").read_text())
            self.assertEqual(summary["benchmark_gates"]["status_counts"], gates["status_counts"])

            per_robot = json.loads((out / "per_robot" / "roboparty_rpo.json").read_text())
            self.assertEqual(per_robot["profile_schema_version"], 2)
            self.assertIn("task_summary", per_robot)
            self.assertIn("chain_summary", per_robot)
            self.assertIn("collision_summary", per_robot)
            self.assertIn("proxy_count", per_robot["collision_summary"])
            self.assertIn("pair_count", per_robot["collision_summary"])
            self.assertIn("root_ground_summary", per_robot)
            self.assertIn("ground_height_source", per_robot["root_ground_summary"])
            self.assertIn("horizontal_scale", per_robot["root_ground_summary"])
            self.assertEqual(per_robot["metrics"]["hand_position_rmse"]["status"], "not_run")
            self.assertEqual(per_robot["metrics"]["hand_position_rmse"].get("reason"), None)

            with (out / "benchmark_frames.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["robot"], "roboparty_rpo")

    def test_benchmark_records_runtime_motion_metrics_when_motions_are_requested(self):
        runtime_payload = {
            "status": "ok",
            "runtime_seconds": {
                "bvh_load_runtime": 0.10,
                "pipeline_construct_runtime": 0.20,
                "target_setup_runtime": 0.30,
                "solve_runtime": 0.90,
                "metric_runtime": 0.05,
                "motion_runtime": 1.25,
                "motion_count": 2,
                "output_frame_count": 8,
                "solve_fps": 8.8888888889,
            },
            "solver_objectives": {
                "active_objectives": 12,
                "pole_vector_autodiff": 4,
                "autodiff_sparse_residual_dim": 12,
            },
            "motions": [
                {
                    "motion": "/tmp/fixture_a.bvh",
                    "frames": 4,
                    "sample_rate": 60.0,
                    "metrics": {
                        "velocity_p95": {"status": "ok", "value": 2.5},
                        "penetration": {"status": "ok", "value": 0.0},
                    },
                },
                {
                    "motion": "/tmp/fixture_b.bvh",
                    "frames": 4,
                    "sample_rate": 60.0,
                    "metrics": {
                        "velocity_p95": {"status": "ok", "value": 3.5},
                        "penetration": {"status": "ok", "value": 0.0},
                    },
                }
            ],
            "metrics": {
                "velocity_p95": {"status": "ok", "value": 3.0, "motion_count": 2, "aggregation": "mean"},
                "penetration": {"status": "ok", "value": 0.0, "motion_count": 2, "aggregation": "mean"},
                "fallback_counts": {"status": "ok", "pole_vector": [0]},
            },
        }
        captured_motion_counts = []

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            motion_dir = root / "motions"
            motion_dir.mkdir()
            (motion_dir / "fixture_a.bvh").write_text("HIERARCHY\n", encoding="utf-8")
            (motion_dir / "fixture_b.bvh").write_text("HIERARCHY\n", encoding="utf-8")
            (motion_dir / "fixture_c.bvh").write_text("HIERARCHY\n", encoding="utf-8")
            out = root / "bench"

            def fake_runtime(*args):
                captured_motion_counts.append(len(args[2]))
                return runtime_payload

            with mock.patch("soma_retargeter.tools.benchmark_retargeting._run_runtime_benchmark", side_effect=fake_runtime):
                rc = main([
                    "--robots",
                    "roboparty_rpo",
                    "--motions",
                    str(motion_dir),
                    "--max-motions",
                    "2",
                    "--output",
                    str(out),
                ])

            self.assertEqual(rc, 0)
            self.assertEqual(captured_motion_counts, [2, 2])
            summary = json.loads((out / "benchmark_summary.json").read_text())
            self.assertEqual(len(summary["resolved_motions"]), 2)
            per_robot = json.loads((out / "per_robot" / "roboparty_rpo.json").read_text())
            self.assertEqual(per_robot["motion_benchmark"]["status"], "ok")
            self.assertIn("legacy", per_robot["compare_results"])
            self.assertIn("v2", per_robot["compare_results"])
            self.assertEqual(per_robot["metrics"]["velocity_p95"]["status"], "ok")
            self.assertEqual(per_robot["metrics"]["velocity_p95"]["value"], 3.0)
            self.assertEqual(per_robot["metrics"]["runtime_seconds"]["motion_runtime"], 1.25)
            self.assertEqual(per_robot["metrics"]["runtime_seconds"]["target_setup_runtime"], 0.30)
            self.assertEqual(per_robot["metrics"]["runtime_seconds"]["solve_runtime"], 0.90)
            self.assertEqual(per_robot["metrics"]["runtime_seconds"]["output_frame_count"], 8)
            self.assertEqual(per_robot["compare_results"]["v2"]["motion_benchmark"]["solver_objectives"]["pole_vector_autodiff"], 4)

            with (out / "benchmark_frames.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            runtime_rows = [row for row in rows if row["motion"] == "/tmp/fixture_b.bvh" and row["metric"] == "velocity_p95"]
            self.assertEqual({row["compare_mode"] for row in runtime_rows}, {"legacy", "v2"})
            self.assertTrue(all(row["value"] == "3.5" for row in runtime_rows))

    def test_strict_gates_return_code_four_on_gate_failure(self):
        def fake_runtime(*args):
            compare_mode = args[4]
            value = 0.01 if compare_mode == "legacy" else 0.02
            return {
                "status": "ok",
                "runtime_seconds": {"motion_runtime": 1.0, "motion_count": 1},
                "solver_objectives": {"active_objectives": 3, "autodiff_sparse_residual_dim": 6},
                "motions": [],
                "metrics": {
                    "penetration": {"status": "ok", "value": value, "motion_count": 1},
                    "runtime_seconds": {"motion_runtime": 1.0},
                },
            }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            motion_dir = root / "motions"
            motion_dir.mkdir()
            (motion_dir / "fixture.bvh").write_text("HIERARCHY\n", encoding="utf-8")
            out = root / "bench"

            with mock.patch("soma_retargeter.tools.benchmark_retargeting._run_runtime_benchmark", side_effect=fake_runtime):
                rc = main([
                    "--robots",
                    "roboparty_rpo",
                    "--motions",
                    str(motion_dir),
                    "--strict-gates",
                    "--output",
                    str(out),
                ])

            self.assertEqual(rc, 4)
            gates = json.loads((out / "benchmark_gates.json").read_text())
            self.assertEqual(gates["status"], "failed")
            failure = json.loads((out / "failures" / "roboparty_rpo_gates.json").read_text())
            self.assertEqual(failure["failure_type"], "benchmark_gate")
            self.assertEqual(failure["robot"], "roboparty_rpo")
            self.assertIn("--strict-gates", failure["reproduction_command"])
            self.assertEqual([gate["metric"] for gate in failure["failed_gates"]], ["penetration"])
            self.assertEqual(failure["compare_solver_objectives"]["v2"]["autodiff_sparse_residual_dim"], 6)
            summary = json.loads((out / "benchmark_summary.json").read_text())
            self.assertEqual(summary["gate_failure_artifacts"][0]["path"], "failures/roboparty_rpo_gates.json")

    def test_benchmark_persists_failure_payload(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bench"
            rc = main([
                "--robots",
                "does_not_exist",
                "--output",
                str(out),
            ])

            self.assertEqual(rc, 1)
            failure = json.loads((out / "failures" / "does_not_exist.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["robot"], "does_not_exist")
            self.assertIn("exception", failure)
            self.assertIn("stack", failure)


if __name__ == "__main__":
    unittest.main()
