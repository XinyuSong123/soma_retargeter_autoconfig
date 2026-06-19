import unittest

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


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
                "Chest": {"t_body": "torso", "r_body": "torso", "t_weight": 0.0, "r_weight": 0.5},
            }
        }

        mapped_joints, mapped_indices, pos_data, rot_data, link_by_joint = pipe._build_target_mapping(None, Skeleton(), cfg)

        self.assertEqual(mapped_joints, ["Hips", "LeftHand", "Chest"])
        self.assertEqual(mapped_indices, [0, 1, 2])
        self.assertEqual(pos_data, [(0, 0, 10.0), (1, 1, 1.0)])
        self.assertEqual(rot_data, [(2, 2, 0.5)])
        self.assertEqual(link_by_joint, {"Hips": 0, "LeftHand": 1, "Chest": 2})


if __name__ == "__main__":
    unittest.main()
