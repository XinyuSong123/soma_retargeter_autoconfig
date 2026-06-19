import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.robot_registry_parser import (
    _resolve_priority_weight_bands,
    build_runtime_retargeter_config,
    compiled_profile_cache_diagnostics,
    ensure_compiled_retarget_profile,
    get_profile_path,
    load_compiled_retarget_profile,
    validate_compiled_retarget_profile,
)


class TestContactConfig(unittest.TestCase):
    def test_contact_config_passthrough(self):
        raw = {
            "ik_map": {},
            "contact_aware_foot_ik": {"enabled": True, "toe_weight_stance": 0.8},
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertIn("contact_aware_foot_ik", cfg)
        self.assertTrue(cfg["contact_aware_foot_ik"]["enabled"])

    def test_ground_barrier_config_passthrough(self):
        raw = {
            "ik_map": {},
            "ground_barrier": {"enabled": True, "stance_weight": 2.0, "swing_weight": 0.25},
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertEqual(cfg["ground_barrier"]["stance_weight"], 2.0)
        self.assertEqual(cfg["ground_barrier"]["swing_weight"], 0.25)

    def test_backward_compat_missing_contact_config(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})
        self.assertNotIn("contact_aware_foot_ik", cfg)

    def test_virtual_sole_anchors_generate_contact_aware_config(self):
        raw = {
            "ik_map": {},
            "virtual_sole_anchors": {
                "enabled": True,
                "left": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0], "inner_edge": [0.0, 0.05, 0.0]},
                "right": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0], "outer_edge": [0.0, -0.05, 0.0]},
            },
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertEqual(cfg["contact_aware_foot_ik"]["contact_source"], "auto")
        self.assertIn("inner_edge", cfg["contact_aware_foot_ik"]["anchor_offsets"]["left"])
        self.assertIn("outer_edge", cfg["contact_aware_foot_ik"]["anchor_offsets"]["right"])

    def test_teacher_metadata_not_forwarded(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}, "teacher_refinement": {}, "g1_teacher_refined": True})
        self.assertNotIn("teacher_refinement", cfg)
        self.assertNotIn("g1_teacher_refined", cfg)

    def test_runtime_config_references_compiled_v2_profile(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})
        self.assertEqual(cfg["compiled_retarget_profile_schema_version"], 2)
        self.assertTrue(cfg["compiled_retarget_profile"].endswith("roboparty_rpo_compiled_retarget_profile_v2.json"))

    def test_compiled_v2_runtime_uses_bounded_iteration_default(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})
        self.assertEqual(cfg["ik_iterations"], 4)
        g1_cfg = build_runtime_retargeter_config("unitree_g1", {"ik_map": {}})
        self.assertEqual(g1_cfg["ik_iterations"], 8)

    def test_explicit_runtime_iterations_override_compiled_v2_default(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}, "ik_iterations": 24})
        self.assertEqual(cfg["ik_iterations"], 24)

    def test_compiled_v2_runtime_enables_joint_motion_limiter(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})
        self.assertTrue(cfg["joint_motion_limit_enabled"])
        self.assertEqual(cfg["joint_velocity_limit_fraction_per_second"], 2.0)
        self.assertEqual(cfg["joint_acceleration_limit_fraction_per_second2"], 40.0)

    def test_explicit_joint_motion_limiter_options_override_compiled_v2_default(self):
        cfg = build_runtime_retargeter_config(
            "roboparty_rpo",
            {
                "ik_map": {},
                "joint_motion_limit_enabled": False,
                "joint_velocity_limit_fraction_per_second": 1.5,
                "joint_acceleration_limit_fraction_per_second2": 12.0,
            },
        )
        self.assertFalse(cfg["joint_motion_limit_enabled"])
        self.assertEqual(cfg["joint_velocity_limit_fraction_per_second"], 1.5)
        self.assertEqual(cfg["joint_acceleration_limit_fraction_per_second2"], 12.0)

    def test_compiled_v2_tasks_disable_unreachable_legacy_objectives(self):
        raw = {
            "ik_map": {
                "Hips": "base_link",
                "Chest": "torso_link",
                "LeftArm": "left_arm_roll_link",
                "LeftHand": "left_elbow_yaw_link",
                "LeftFoot": "left_ankle_roll_link",
                "RightFoot": "right_ankle_roll_link",
            }
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertEqual(cfg["priority_weight_bands"], {"0": 10000.0, "1": 1000.0, "2": 100.0, "3": 10.0, "4": 1.0})
        self.assertEqual(cfg["priority_scheduler_diagnostics"], [])
        self.assertGreater(cfg["ik_map"]["Hips"]["t_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["Hips"]["t_weight"], 1000.0)
        self.assertEqual(cfg["ik_map"]["Hips"]["v2_position_priority"], 1)
        self.assertEqual(cfg["ik_map"]["Hips"]["v2_position_weight_source"], "compiled task normalized weight")
        self.assertEqual(cfg["ik_map"]["Hips"]["r_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["t_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["r_weight"], 1.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["v2_rotation_priority"], 2)
        self.assertEqual(cfg["ik_map"]["Chest"]["v2_rotation_weight_source"], "compiled task normalized weight")
        self.assertEqual(cfg["ik_map"]["Chest"]["v2_rotation_basis"], [[0.0], [0.0], [1.0]])
        self.assertEqual(cfg["ik_map"]["LeftArm"]["t_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["LeftArm"]["r_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["LeftHand"]["t_weight"], 50.0)
        self.assertEqual(cfg["ik_map"]["LeftHand"]["r_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["LeftFoot"]["t_weight"], 500.0)
        self.assertEqual(cfg["ik_map"]["RightFoot"]["t_weight"], 500.0)
        self.assertNotIn("v2_position_link_offset", cfg["ik_map"]["LeftHand"])
        g1_cfg = build_runtime_retargeter_config("unitree_g1", {"ik_map": {"LeftHand": "left_wrist_yaw_link"}})
        self.assertEqual(
            g1_cfg["ik_map"]["LeftHand"]["v2_position_link_offset"],
            [0.17635336870442034, -0.010919955391605394, 0.006463830307906945],
        )
        g1_full_cfg = build_runtime_retargeter_config(
            "unitree_g1",
            {
                "ik_map": {
                    "Chest": "torso_link",
                    "LeftFoot": "left_ankle_roll_link",
                    "RightFoot": "right_ankle_roll_link",
                }
            },
        )
        self.assertEqual(g1_full_cfg["ik_map"]["Chest"]["r_weight"], 50.0)
        self.assertEqual(g1_full_cfg["ik_map"]["LeftFoot"]["t_weight"], 1500.0)
        self.assertEqual(g1_full_cfg["ik_map"]["RightFoot"]["t_weight"], 1500.0)
        left_arm_direction = next(task for task in cfg["direction_tasks"] if task["name"] == "LeftArm_direction")
        self.assertEqual(left_arm_direction["reference_site"], "Chest")
        self.assertEqual(left_arm_direction["target_site"], "LeftArm")
        self.assertEqual(left_arm_direction["weight"], 10.0)
        self.assertEqual(left_arm_direction["priority_weight_band"], 10.0)
        self.assertEqual(left_arm_direction["weight_source"], "compiled task normalized weight")
        self.assertFalse(left_arm_direction["analytic_jacobian"])
        self.assertEqual(
            left_arm_direction["jacobian_schedule_reason"],
            "disabled for single-axis torso profile to preserve bounded tracking",
        )
        self.assertGreater(left_arm_direction["characteristic_length"], 0.0)
        g1_left_arm_direction = next(task for task in g1_full_cfg["direction_tasks"] if task["name"] == "LeftArm_direction")
        self.assertTrue(g1_left_arm_direction["analytic_jacobian"])
        self.assertEqual(
            g1_left_arm_direction["jacobian_schedule_reason"],
            "enabled for multi-axis torso profile",
        )
        left_forearm_pole = next(task for task in cfg["pole_vector_tasks"] if task["name"] == "LeftForeArm_pole_vector")
        self.assertEqual(left_forearm_pole["reference_site"], "LeftArm")
        self.assertEqual(left_forearm_pole["source_semantic"], "LeftForeArm")
        self.assertEqual(left_forearm_pole["target_site"], "LeftHand")
        self.assertEqual(left_forearm_pole["weight"], 10.0)
        self.assertFalse(left_forearm_pole["analytic_jacobian"])
        self.assertEqual(
            left_forearm_pole["jacobian_schedule_reason"],
            "disabled pending finite-difference validation of pole-vector analytic Jacobian",
        )
        self.assertEqual(left_forearm_pole["priority_weight_band"], 10.0)

    def test_compiled_profile_registry_path_and_validation(self):
        path = get_profile_path("roboparty_rpo", "compiled_retarget_profile")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

        profile = load_compiled_retarget_profile("roboparty_rpo")
        self.assertEqual(profile["schema_version"], 2)
        self.assertEqual(profile["quaternion_order"], "xyzw")
        self.assertEqual(validate_compiled_retarget_profile(profile), [])
        self.assertEqual(compiled_profile_cache_diagnostics("roboparty_rpo", profile), [])

        diagnostics = validate_compiled_retarget_profile({"schema_version": 1})
        self.assertTrue(any(item["code"] == "invalid_schema_version" for item in diagnostics))
        self.assertTrue(any(item["code"] == "missing_profile_key" for item in diagnostics))

    def test_compiled_profile_validation_reports_health_gate_failures(self):
        profile = load_compiled_retarget_profile("roboparty_rpo")
        profile["semantic_sites"]["Hips"]["local_rotation_xyzw"] = [0.0, 0.0, 0.0, 2.0]
        profile["chains"]["Hips"]["segment_lengths"] = [-0.1]
        profile["chains"]["Hips"]["total_length"] = 0.0
        profile["collision"]["margin"] = -0.1
        profile["collision"]["proxies"][0]["radius"] = 0.0
        profile["rest_frame_alignment"]["root_motion"]["horizontal_scale"] = float("inf")
        profile["chains"]["LeftHand"]["total_length"] = 1.0
        profile["chains"]["RightHand"]["total_length"] = 0.5
        profile["chains"]["LeftHand"]["segment_lengths"] = [1.0]
        profile["chains"]["RightHand"]["segment_lengths"] = [0.5]
        diagnostics = validate_compiled_retarget_profile(profile)
        codes = {item["code"] for item in diagnostics}
        self.assertIn("non_unit_site_quaternion", codes)
        self.assertIn("invalid_segment_length", codes)
        self.assertIn("invalid_chain_total_length", codes)
        self.assertIn("invalid_collision_margin", codes)
        self.assertIn("invalid_collision_proxy_radius", codes)
        self.assertIn("invalid_root_motion_value", codes)
        self.assertIn("non_finite_profile_value", codes)
        self.assertIn("symmetric_chain_length_mismatch", codes)
        self.assertIn("symmetric_segment_length_mismatch", codes)

    def test_compiled_profile_validation_requires_cache_fingerprints(self):
        profile = load_compiled_retarget_profile("roboparty_rpo")
        del profile["compiler_version"]
        profile["source_config_hash"] = ""
        diagnostics = validate_compiled_retarget_profile(profile)
        self.assertTrue(
            any(
                item["code"] == "missing_profile_key" and item["key"] == "compiler_version"
                for item in diagnostics
            )
        )
        self.assertTrue(
            any(
                item["code"] == "invalid_profile_fingerprint" and item["key"] == "source_config_hash"
                for item in diagnostics
            )
        )

    def test_compiled_profile_cache_diagnostics_report_stale_fingerprints(self):
        profile = load_compiled_retarget_profile("roboparty_rpo")
        profile["compiler_version"] = "old"
        profile["robot_fingerprint"] = "old"
        profile["source_config_hash"] = "old"
        diagnostics = compiled_profile_cache_diagnostics("roboparty_rpo", profile)
        codes = {item["code"] for item in diagnostics}
        self.assertIn("compiled_profile_compiler_version_stale", codes)
        self.assertIn("compiled_profile_robot_fingerprint_stale", codes)
        self.assertIn("compiled_profile_source_config_hash_stale", codes)

    def test_incomplete_profile_regeneration_does_not_overwrite_valid_cache(self):
        cached_profile = load_compiled_retarget_profile("roboparty_rpo")
        raw_config_path = get_profile_path("roboparty_rpo", "retargeter_config")
        self.assertIsNotNone(raw_config_path)
        incomplete_payload = {
            "robot_fingerprint": "missing-mjcf",
            "warnings": [{"code": "missing_mjcf_path"}],
        }

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "compiled_profile.json"
            io_utils.save_json(cache_path, cached_profile)
            with mock.patch(
                "soma_retargeter.robot_registry_parser.get_generated_compiled_profile_path",
                return_value=cache_path,
            ), mock.patch(
                "soma_retargeter.robot_registry_parser.get_robot_profile",
                return_value={"retargeter_config": raw_config_path},
            ), mock.patch(
                "soma_retargeter.robot_registry_parser.get_robot_mjcf_path",
                return_value=None,
            ), mock.patch(
                "soma_retargeter.robotics.morphology.analyze_mjcf_morphology",
                return_value=object(),
            ), mock.patch(
                "soma_retargeter.robotics.task_compiler.compile_retarget_profile",
                return_value=object(),
            ), mock.patch(
                "soma_retargeter.robot_registry_parser.profile_to_json_dict",
                return_value=incomplete_payload,
            ), mock.patch(
                "soma_retargeter.robotics.retarget_profile.write_profile_json",
            ) as write_mock:
                self.assertEqual(ensure_compiled_retarget_profile("roboparty_rpo", force=True), cache_path)
                write_mock.assert_not_called()
                self.assertEqual(io_utils.load_json(cache_path)["robot_fingerprint"], cached_profile["robot_fingerprint"])

    def test_priority_band_validation_reports_small_adjacent_ratio(self):
        bands, diagnostics = _resolve_priority_weight_bands(
            {"solver": {"priority_weight_bands": {"0": 100.0, "1": 50.0, "2": 10.0, "3": 1.0, "4": 0.1}}}
        )
        self.assertEqual(bands["0"], 100.0)
        self.assertTrue(any(item["code"] == "priority_band_ratio_too_small" for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
