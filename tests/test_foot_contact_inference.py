import unittest
import numpy as np
from soma_retargeter.pipelines.foot_contact_inference import contacts_from_npz_foot_contacts
from soma_retargeter.pipelines.foot_contact_inference import _contact_score_from_positions


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


if __name__ == "__main__":
    unittest.main()
