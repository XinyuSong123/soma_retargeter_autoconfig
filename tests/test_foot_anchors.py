import unittest

import numpy as np

from soma_retargeter.foot_anchors import _anchors_from_bbox, generate_virtual_sole_anchors


class TestFootAnchors(unittest.TestCase):
    def test_manual_anchors_exact_pass_through(self):
        manual = {
            "left": {
                "sole_center": [1.0, 2.0, 3.0],
                "toe": [1.1, 2.0, 3.0],
                "heel": [0.9, 2.0, 3.0],
                "inner_edge": [1.0, 1.9, 3.0],
                "outer_edge": [1.0, 2.1, 3.0],
            },
            "right": {
                "sole_center": [4.0, 5.0, 6.0],
                "toe": [4.1, 5.0, 6.0],
                "heel": [3.9, 5.0, 6.0],
                "inner_edge": [4.0, 5.1, 6.0],
                "outer_edge": [4.0, 4.9, 6.0],
            },
        }
        anchors = generate_virtual_sole_anchors(
            "roboparty_rpo",
            {"links": {"left_foot": "missing_left", "right_foot": "missing_right"}},
            profile={"foot": {"manual_anchors": manual}},
        )
        self.assertTrue(anchors["enabled"])
        self.assertEqual(anchors["source"], "manual_anchors")
        self.assertEqual(anchors["left"], manual["left"])
        self.assertEqual(anchors["right"], manual["right"])

    def test_bbox_axis_inference_synthetic_box(self):
        points = np.array(
            [
                [x, y, z]
                for x in (-0.2, 0.3)
                for y in (-0.05, 0.07)
                for z in (-0.03, 0.02)
            ],
            dtype=np.float64,
        )
        anchors = _anchors_from_bbox(
            points,
            side="left",
            forward_local=np.array([1.0, 0.0, 0.0]),
            lateral_local=np.array([0.0, -1.0, 0.0]),
            up_local=np.array([0.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(anchors["toe"][0], 0.3)
        self.assertAlmostEqual(anchors["heel"][0], -0.2)
        self.assertAlmostEqual(anchors["sole_center"][2], -0.03)
        self.assertAlmostEqual(anchors["inner_edge"][1], -0.05)
        self.assertAlmostEqual(anchors["outer_edge"][1], 0.07)

    def test_no_silent_fixed_manual_fallback_for_unregistered_robot(self):
        anchors = generate_virtual_sole_anchors(
            "unregistered_test_robot",
            {"links": {"left_foot": "missing_left", "right_foot": "missing_right"}},
            profile={"foot": {"sole_anchor_mode": "bbox"}},
        )
        self.assertFalse(anchors["enabled"])
        self.assertEqual(anchors["source"], "disabled")
        self.assertTrue(any("no manual anchors" in warning for warning in anchors["warnings"]))


if __name__ == "__main__":
    unittest.main()
