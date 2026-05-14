import unittest
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

    def test_backward_compat_missing_contact_config(self):
        cfg = build_runtime_retargeter_config("roboparty_rpo", {"ik_map": {}})
        self.assertNotIn("contact_aware_foot_ik", cfg)


if __name__ == "__main__":
    unittest.main()
