import unittest
from soma_retargeter.robot_registry_parser import (
    build_runtime_retargeter_config,
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

    def test_compiled_v2_tasks_disable_unreachable_legacy_objectives(self):
        raw = {
            "ik_map": {
                "Hips": "base_link",
                "Chest": "torso_link",
                "LeftArm": "left_arm_roll_link",
                "LeftHand": "left_elbow_yaw_link",
            }
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertGreater(cfg["ik_map"]["Hips"]["t_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["Hips"]["r_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["t_weight"], 0.0)
        self.assertGreater(cfg["ik_map"]["Chest"]["r_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["Chest"]["v2_rotation_basis"], [[0.0], [0.0], [1.0]])
        self.assertEqual(cfg["ik_map"]["LeftArm"]["t_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["LeftArm"]["r_weight"], 0.0)
        self.assertGreater(cfg["ik_map"]["LeftHand"]["t_weight"], 0.0)
        self.assertEqual(cfg["ik_map"]["LeftHand"]["r_weight"], 0.0)

    def test_compiled_profile_registry_path_and_validation(self):
        path = get_profile_path("roboparty_rpo", "compiled_retarget_profile")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())

        profile = load_compiled_retarget_profile("roboparty_rpo")
        self.assertEqual(profile["schema_version"], 2)
        self.assertEqual(profile["quaternion_order"], "xyzw")
        self.assertEqual(validate_compiled_retarget_profile(profile), [])

        diagnostics = validate_compiled_retarget_profile({"schema_version": 1})
        self.assertTrue(any(item["code"] == "invalid_schema_version" for item in diagnostics))
        self.assertTrue(any(item["code"] == "missing_profile_key" for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
