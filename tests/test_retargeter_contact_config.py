import unittest
from unittest.mock import patch

from soma_retargeter.robot_registry_parser import build_runtime_retargeter_config


class TestContactConfig(unittest.TestCase):
    def test_contact_config_passthrough(self):
        raw = {
            "ik_map": {},
            "contact_aware_foot_ik": {"enabled": True, "toe_weight_stance": 0.8},
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertIn("contact_aware_foot_ik", cfg)
        self.assertTrue(cfg["contact_aware_foot_ik"]["enabled"])

    def test_generated_config_gets_virtual_sole_anchors_without_teacher_refinement(self):
        with patch(
            "soma_retargeter.teacher_refinement.sole_anchor_generator.generate_virtual_sole_anchors",
            side_effect=AssertionError("legacy teacher anchor path should not be called"),
        ):
            cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})

        self.assertIn("virtual_sole_anchors", cfg)
        self.assertTrue(cfg["virtual_sole_anchors"]["enabled"])
        self.assertIn("contact_aware_foot_ik", cfg)
        contact_cfg = cfg["contact_aware_foot_ik"]
        self.assertTrue(contact_cfg["enabled"])
        self.assertEqual(contact_cfg["contact_source"], "auto")
        for side in ("left", "right"):
            self.assertIn("toe", contact_cfg["anchor_offsets"][side])
            self.assertIn("heel", contact_cfg["anchor_offsets"][side])
            self.assertIn("inner_edge", contact_cfg["anchor_offsets"][side])
            self.assertIn("outer_edge", contact_cfg["anchor_offsets"][side])

    def test_virtual_sole_anchors_generate_contact_aware_config(self):
        raw = {
            "ik_map": {"LeftFoot": {"t_body": "left_foot", "t_weight": 4.0}, "RightFoot": {"t_body": "right_foot", "t_weight": 4.0}},
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
        self.assertGreater(cfg["contact_aware_foot_ik"]["toe_weight_stance"], 0.8)

    def test_teacher_metadata_not_forwarded(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}, "teacher_refinement": {}, "g1_teacher_refined": True})
        self.assertNotIn("teacher_refinement", cfg)
        self.assertNotIn("g1_teacher_refined", cfg)

    def test_raw_disabled_virtual_sole_anchors_are_respected(self):
        cfg = build_runtime_retargeter_config(
            "roboparty_rpo",
            {"ik_map": {}, "virtual_sole_anchors": {"enabled": False, "source": "disabled"}},
        )
        self.assertEqual(cfg["virtual_sole_anchors"]["source"], "disabled")
        self.assertNotIn("contact_aware_foot_ik", cfg)

    def test_explicit_contact_weights_override_derived_weights(self):
        raw = {
            "ik_map": {"LeftFoot": {"t_body": "left_foot", "t_weight": 4.0}, "RightFoot": {"t_body": "right_foot", "t_weight": 4.0}},
            "virtual_sole_anchors": {
                "enabled": True,
                "left": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
                "right": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
            },
            "contact_aware_foot_ik": {"enabled": True, "toe_weight_stance": 9.0},
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertEqual(cfg["contact_aware_foot_ik"]["toe_weight_stance"], 9.0)
        self.assertIn("anchor_offsets", cfg["contact_aware_foot_ik"])

    def test_contact_aware_ik_still_uses_manual_anchors_when_provided(self):
        raw = {
            "ik_map": {},
            "virtual_sole_anchors": {
                "enabled": True,
                "source": "manual_anchors",
                "left": {"toe": [1, 2, 3], "heel": [4, 5, 6]},
                "right": {"toe": [7, 8, 9], "heel": [10, 11, 12]},
            },
        }
        cfg = build_runtime_retargeter_config("roboparty_rpo", raw)
        self.assertEqual(cfg["contact_aware_foot_ik"]["anchor_offsets"]["left"]["toe"], [1, 2, 3])
        self.assertEqual(cfg["contact_aware_foot_ik"]["anchor_offsets"]["right"]["heel"], [10, 11, 12])

    def test_contact_aware_ik_skips_safely_when_anchors_cannot_be_generated(self):
        with patch("soma_retargeter.robot_registry_parser._generate_virtual_sole_anchors_for_runtime", return_value=None):
            cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})
        self.assertNotIn("contact_aware_foot_ik", cfg)


if __name__ == "__main__":
    unittest.main()
