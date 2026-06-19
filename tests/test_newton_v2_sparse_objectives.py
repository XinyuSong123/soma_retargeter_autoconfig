import unittest

import numpy as np

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robotics.reachability import quat_xyzw_to_rotation_vector, rotation_vector_to_quat_xyzw


class TestNewtonV2SparseObjectives(unittest.TestCase):
    def test_build_target_mapping_keeps_all_effectors_but_skips_zero_weight_objectives(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.robot_builder = type("B", (), {"body_label": ["base", "hand", "torso"]})()

        class Skeleton:
            def joint_index(self, name):
                return {"Hips": 0, "LeftHand": 1, "Chest": 2}[name]

        cfg = {
            "ik_map": {
                "Hips": {"t_body": "base", "r_body": "base", "t_weight": 10.0, "r_weight": 0.0},
                "LeftHand": {"t_body": "hand", "r_body": "hand", "t_weight": 1.0, "r_weight": 0.0},
                "Chest": {
                    "t_body": "torso",
                    "r_body": "torso",
                    "t_weight": 0.0,
                    "r_weight": 0.5,
                    "v2_rotation_basis": [[0.0], [0.0], [1.0]],
                },
            }
        }

        mapped_joints, mapped_indices, pos_data, rot_data, link_by_joint = pipe._build_target_mapping(None, Skeleton(), cfg)

        self.assertEqual(mapped_joints, ["Hips", "LeftHand", "Chest"])
        self.assertEqual(mapped_indices, [0, 1, 2])
        self.assertEqual(pos_data, [(0, 0, 10.0), (1, 1, 1.0)])
        self.assertEqual(rot_data[0][:3], (2, 2, 0.5))
        self.assertTrue(np.allclose(rot_data[0][3], [[0.0], [0.0], [1.0]]))
        self.assertEqual(link_by_joint, {"Hips": 0, "LeftHand": 1, "Chest": 2})

    def test_build_target_mapping_extracts_v2_direction_tasks(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.robot_builder = type("B", (), {"body_label": ["base", "torso", "arm"]})()

        class Skeleton:
            def joint_index(self, name):
                return {"Hips": 0, "Chest": 1, "LeftArm": 2}[name]

        cfg = {
            "ik_map": {
                "Hips": {"t_body": "base", "r_body": "base", "t_weight": 10.0, "r_weight": 0.0},
                "Chest": {"t_body": "torso", "r_body": "torso", "t_weight": 0.0, "r_weight": 0.5},
                "LeftArm": {"t_body": "arm", "r_body": "arm", "t_weight": 0.0, "r_weight": 0.0},
            },
            "direction_tasks": [
                {"name": "LeftArm_direction", "reference_site": "Chest", "target_site": "LeftArm", "weight": 10.0}
            ],
        }

        pipe._build_target_mapping(None, Skeleton(), cfg)

        self.assertEqual(
            pipe.mapped_body_link_direction_data,
            [(1, 2, 1, 2, 10.0, "LeftArm_direction")],
        )

    def test_rotation_target_projection_drops_unreachable_components(self):
        source = rotation_vector_to_quat_xyzw(np.array([0.25, -0.5, 0.75]))
        projected = NewtonPipeline._project_rotation_target(source, np.array([[0.0], [0.0], [1.0]]))
        rotvec = quat_xyzw_to_rotation_vector(projected)
        self.assertAlmostEqual(rotvec[0], 0.0, places=6)
        self.assertAlmostEqual(rotvec[1], 0.0, places=6)
        self.assertAlmostEqual(rotvec[2], 0.75, places=6)

    def test_direction_target_uses_normalized_reference_to_child_vector(self):
        direction = NewtonPipeline._direction_between_targets(
            np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 5.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        )
        self.assertTrue(np.allclose(direction, [0.0, 0.6, 0.8]))
        self.assertTrue(
            np.allclose(
                NewtonPipeline._direction_between_positions([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                [0.0, 0.0, 0.0],
            )
        )


if __name__ == "__main__":
    unittest.main()
