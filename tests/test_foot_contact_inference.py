import unittest
import numpy as np
from soma_retargeter.pipelines.foot_contact_inference import contacts_from_npz_foot_contacts
from soma_retargeter.pipelines.foot_contact_inference import _contact_score_from_positions
from soma_retargeter.pipelines.foot_contact_inference import infer_contacts_from_animation_buffer


class TestFootContactInference(unittest.TestCase):
    def test_npz_contact_shape_and_range(self):
        frames = 20
        contacts = np.zeros((frames, 4), dtype=np.float32)
        contacts[5:12, :] = 1.0
        out = contacts_from_npz_foot_contacts(contacts, smoothing_window=5)
        self.assertEqual(len(out["left_toe_contact_score"]), frames)
        self.assertEqual(len(out["left_heel_contact_score"]), frames)
        self.assertEqual(len(out["right_toe_contact_score"]), frames)
        self.assertEqual(len(out["right_heel_contact_score"]), frames)
        for v in out.values():
            self.assertTrue(np.all(v >= 0.0))
            self.assertTrue(np.all(v <= 1.0))
            self.assertLess(np.max(np.abs(np.diff(v))), 1.0)

    def test_airborne_stationary_is_low_contact(self):
        pos = np.array([[0.0, 0.0, 0.5]] * 20, dtype=np.float32)
        score = _contact_score_from_positions(pos, ground_height=0.0, velocity_dt=1.0 / 60.0)
        self.assertLess(float(np.max(score)), 0.1)

    def test_grounded_moving_is_low_contact(self):
        pos = np.zeros((20, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, 1.0, 20)
        score = _contact_score_from_positions(pos, ground_height=0.0, velocity_dt=1.0 / 60.0)
        self.assertLess(float(np.max(score[1:])), 0.5)

    def test_infer_contacts_accepts_numpy_transform_rows(self):
        class Skeleton:
            def joint_index(self, name):
                return {
                    "LeftToeBase": 0,
                    "RightToeBase": 1,
                    "LeftFoot": 2,
                    "RightFoot": 3,
                }.get(name, -1)

        class Buffer:
            skeleton = Skeleton()
            num_frames = 4
            sample_rate = 60.0

            def compute_global_transforms(self, frame, root_tx=None):
                del root_tx
                x = frame * 0.001
                return np.array(
                    [
                        [x, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [x, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [x, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [x, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

        out = infer_contacts_from_animation_buffer(Buffer(), smoothing_window=1)
        self.assertEqual(set(out), {
            "left_toe_contact_score",
            "right_toe_contact_score",
            "left_heel_contact_score",
            "right_heel_contact_score",
        })
        self.assertGreater(float(out["left_toe_contact_score"][0]), 0.5)


if __name__ == "__main__":
    unittest.main()
