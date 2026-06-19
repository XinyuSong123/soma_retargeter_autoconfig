import unittest

import numpy as np

from soma_retargeter.robotics.human_to_robot_scaler import (
    HumanToRobotScaler,
    LegacyHumanToRobotScaler,
    SegmentLocalTargetBuilder,
)


class TestSegmentLocalTargetBuilder(unittest.TestCase):
    def test_legacy_scaler_alias_keeps_v1_import_path_explicit(self):
        self.assertIs(LegacyHumanToRobotScaler, HumanToRobotScaler)

    def test_recursive_segment_lengths_do_not_scale_children_geocentrically(self):
        builder = SegmentLocalTargetBuilder(
            joint_names=["Hips", "Chest", "Hand"],
            parent_indices=[-1, 0, 1],
            segment_lengths=np.array([0.0, 2.0, 1.0]),
        )
        source = np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [11.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )

        out = builder.compute_positions(source)
        self.assertTrue(np.allclose(out[0], [10.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(out[1], [12.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(out[2], [12.0, 1.0, 0.0]))

    def test_batch_and_single_frame_share_the_same_math(self):
        builder = SegmentLocalTargetBuilder(
            joint_names=["Root", "Tip"],
            parent_indices=[-1, 0],
            segment_lengths=np.array([0.0, 0.5]),
            root_scale=np.array([2.0, 1.0, 1.0]),
        )
        frames = np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
                [[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )

        batched = builder.compute_positions_batch(frames)
        self.assertTrue(np.allclose(batched[0], builder.compute_positions(frames[0])))
        self.assertTrue(np.allclose(batched[1], builder.compute_positions(frames[1])))
        self.assertTrue(np.allclose(batched[0, 0], [2.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(batched[0, 1], [2.0, 0.5, 0.0]))

    def test_invalid_or_degenerate_segments_are_rejected(self):
        with self.assertRaises(ValueError):
            SegmentLocalTargetBuilder(["Root", "Tip"], [-1, 0], np.array([0.0, -1.0]))

        builder = SegmentLocalTargetBuilder(["Root", "Tip"], [-1, 0], np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            builder.compute_positions(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def test_transform_builder_preserves_rotation_components(self):
        builder = SegmentLocalTargetBuilder(["Root", "Tip"], [-1, 0], np.array([0.0, 1.0]))
        transforms = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.9],
            ],
            dtype=np.float64,
        )
        out = builder.compute_transforms(transforms)
        self.assertTrue(np.allclose(out[:, 3:7], transforms[:, 3:7]))
        self.assertTrue(np.allclose(out[1, 0:3], [1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
