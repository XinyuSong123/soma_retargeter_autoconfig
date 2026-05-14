import unittest
import numpy as np
from soma_retargeter.pipelines.foot_contact_inference import contacts_from_npz_foot_contacts


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


if __name__ == "__main__":
    unittest.main()
