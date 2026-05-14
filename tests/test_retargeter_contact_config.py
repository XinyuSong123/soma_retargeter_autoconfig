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

    def test_raw_disabled_virtual_sole_anchors_are_respected(self):
        cfg = build_runtime_retargeter_config(
            "roboparty_rpo",
            {"ik_map": {}, "virtual_sole_anchors": {"enabled": False, "source": "disabled"}},
        )
        self.assertEqual(cfg["virtual_sole_anchors"]["source"], "disabled")
        self.assertNotIn("contact_aware_foot_ik", cfg)


if __name__ == "__main__":
    unittest.main()
