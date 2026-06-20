import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
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

    def test_human_to_robot_scaler_uses_segment_local_profile_mode(self):
        identity = [0.0, 0.0, 0.0, 1.0]
        skeleton = Skeleton(
            3,
            ["Hips", "Chest", "LeftHand"],
            [-1, 0, 1],
            np.array(
                [
                    [10.0, 0.0, 0.0, *identity],
                    [1.0, 0.0, 0.0, *identity],
                    [0.0, 3.0, 0.0, *identity],
                ],
                dtype=np.float32,
            ),
        )
        instance = SkeletonInstance(skeleton, wp.vec3(1.0, 1.0, 1.0), wp.transform_identity())

        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "scaler.json"
            profile_path = Path(td) / "profile.json"
            config_path.write_text(json.dumps({
                "robot_type": "fixture",
                "human_height_assumption": 1.0,
                "joint_scales": {"Hips": 1.0, "Chest": 10.0, "LeftHand": 10.0},
                "joint_parents": {"Hips": "", "Chest": "Hips", "LeftHand": "Chest"},
                "joint_offsets": {
                    "Hips": [[0.0, 0.0, 0.0], identity],
                    "Chest": [[0.0, 0.0, 0.0], identity],
                    "LeftHand": [[0.0, 0.0, 0.0], identity],
                    "LeftToe": [[0.0, 0.0, 0.0], identity],
                    "RightToe": [[0.0, 0.0, 0.0], identity],
                },
            }))
            profile_path.write_text(json.dumps({
                "schema_version": 2,
                "chains": {
                    "Chest": {"total_length": 2.0, "semantic_edge_length": 20.0},
                    "LeftHand": {"total_length": 1.0, "semantic_edge_length": 10.0},
                },
            }))

            scaler = HumanToRobotScaler(skeleton, 1.0, config_path)
            scaler.enable_segment_local_from_profile(profile_path)
            out = scaler.compute_effectors_from_skeleton(instance, True)

        self.assertEqual(scaler.mode, "segment_local")
        self.assertTrue(np.allclose(out[0, 0:3], [10.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(out[1, 0:3], [12.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(out[2, 0:3], [12.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
