import unittest
import numpy as np

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


class _DummyObjective:
    def __init__(self):
        self.link_offset = (0.0, 0.0, 0.0)


class TestNewtonContactModes(unittest.TestCase):
    def test_missing_left_right_foot_skips_objectives(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_aware_foot_ik_enabled = True
        pipe.contact_aware_foot_ik = {
            "enabled": True,
            "anchor_offsets": {
                "left": {"toe": [0, 0, 0], "heel": [0, 0, 0]},
                "right": {"toe": [0, 0, 0], "heel": [0, 0, 0]},
            },
        }
        pipe.mapped_joints = ["Hips"]
        pipe.mapped_body_link_pos_data = [(0, 1.0)]
        out = pipe._create_contact_aware_objectives(1, [])
        self.assertEqual(out, [])

    def test_contact_source_prefers_npz_in_auto(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_source = "auto"
        pipe.contact_aware_foot_ik_enabled = True
        pipe.contact_aware_foot_ik = {"contact_score_smoothing_window": 1}
        pipe.initialization_pose = None
        pipe.num_initialization_frames = 0
        pipe.num_stabilization_frames = 0
        pipe.max_frames = -1
        pipe.input_targets = []
        pipe.input_sample_rates = []
        pipe.input_contact_scores = []

        pipe.human_robot_scaler = type("S", (), {"compute_effectors_from_buffer": lambda *_: np.zeros((2, 1, 7), dtype=np.float32)})()
        pipe.target_effector_indices = [0]

        class B:
            num_frames = 2
            sample_rate = 60
            foot_contacts = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)

        pipe.add_input_motions([B()], [], False)
        scores = pipe.input_contact_scores[0]
        self.assertEqual(float(scores["left_heel_contact_score"][0]), 1.0)
        self.assertEqual(float(scores["left_toe_contact_score"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
