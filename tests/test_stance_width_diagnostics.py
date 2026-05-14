import unittest

import numpy as np

from soma_retargeter.pipelines.stance_width_diagnostics import (
    apply_debug_options_to_ik_map,
    build_stance_width_report,
)


def _identity_transforms(width, frames=3):
    transforms = np.zeros((frames, 3, 7), dtype=np.float32)
    transforms[:, :, 6] = 1.0
    transforms[:, 1, 0] = -width * 0.5
    transforms[:, 2, 0] = width * 0.5
    return transforms


class TestStanceWidthDiagnostics(unittest.TestCase):
    def test_report_detects_scaled_target_wider_than_source(self):
        report = build_stance_width_report(
            source_soma_transforms=_identity_transforms(0.2),
            source_soma_names=["Hips", "LeftFoot", "RightFoot"],
            scaled_target_transforms=_identity_transforms(0.5),
            scaled_target_names=["Hips", "LeftFoot", "RightFoot"],
            robot_fk_transforms=_identity_transforms(0.5),
            robot_fk_names=["Hips", "LeftFoot", "RightFoot"],
            ik_map={},
        )

        self.assertGreater(report["ratio_scaled_to_source"], 2.0)

    def test_report_detects_robot_fk_wider_than_scaled_target(self):
        report = build_stance_width_report(
            source_soma_transforms=_identity_transforms(0.2),
            source_soma_names=["Hips", "LeftFoot", "RightFoot"],
            scaled_target_transforms=_identity_transforms(0.2),
            scaled_target_names=["Hips", "LeftFoot", "RightFoot"],
            robot_fk_transforms=_identity_transforms(0.6),
            robot_fk_names=["Hips", "LeftFoot", "RightFoot"],
            ik_map={},
        )

        self.assertGreater(report["ratio_robot_to_scaled"], 2.0)

    def test_disable_foot_rotation_tracking_sets_effective_weight_to_zero(self):
        ik_map = {
            "LeftFoot": {"t_body": "left_foot", "r_body": "left_foot", "t_weight": 1.0, "r_weight": 0.7},
            "RightFoot": {"t_body": "right_foot", "r_body": "right_foot", "t_weight": 1.0, "r_weight": 0.8},
            "Chest": {"t_body": "torso", "r_body": "torso", "t_weight": 1.0, "r_weight": 0.4},
        }

        effective, summary = apply_debug_options_to_ik_map(
            ik_map,
            {"disable_foot_rotation_tracking": True},
        )

        self.assertTrue(summary["enabled"])
        self.assertEqual(effective["LeftFoot"]["r_weight"], 0.0)
        self.assertEqual(effective["RightFoot"]["r_weight"], 0.0)
        self.assertEqual(effective["Chest"]["r_weight"], 0.4)

    def test_debug_options_off_by_default_preserve_ik_map(self):
        ik_map = {
            "LeftFoot": {"t_body": "left_foot", "r_body": "left_foot", "t_weight": 1.0, "r_weight": 0.7},
            "RightFoot": {"t_body": "right_foot", "r_body": "right_foot", "t_weight": 1.0, "r_weight": 0.8},
        }

        effective, summary = apply_debug_options_to_ik_map(ik_map, {})

        self.assertFalse(summary["enabled"])
        self.assertEqual(effective, ik_map)
        self.assertIsNot(effective, ik_map)


if __name__ == "__main__":
    unittest.main()
