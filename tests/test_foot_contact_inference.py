import unittest
import numpy as np
from soma_retargeter.pipelines.foot_contact_inference import (
    _contact_score_from_positions,
    contacts_from_npz_foot_contacts,
    infer_contacts_from_animation_buffer,
)


class _DummySkeleton:
    def __init__(self, names):
        self._names = list(names)

    def joint_index(self, name):
        try:
            return self._names.index(name)
        except ValueError:
            return -1


class _DummyBuffer:
    sample_rate = 60

    def __init__(self, names, frames):
        self.skeleton = _DummySkeleton(names)
        self._frames = np.asarray(frames, dtype=np.float32)
        self.num_frames = self._frames.shape[0]

    def compute_global_transforms(self, frame, root_tx=None):
        return self._frames[frame]


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

    def test_npz_contact_order_default_backward_compatible(self):
        contacts = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        out = contacts_from_npz_foot_contacts(contacts, smoothing_window=1)
        self.assertAlmostEqual(float(out["left_heel_contact_score"][0]), 0.1)
        self.assertAlmostEqual(float(out["left_toe_contact_score"][0]), 0.2)
        self.assertAlmostEqual(float(out["right_heel_contact_score"][0]), 0.3)
        self.assertAlmostEqual(float(out["right_toe_contact_score"][0]), 0.4)

    def test_npz_contact_order_custom_mapping(self):
        contacts = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        out = contacts_from_npz_foot_contacts(
            contacts,
            smoothing_window=1,
            contact_order=["right_toe", "left_toe", "right_heel", "left_heel"],
        )
        self.assertAlmostEqual(float(out["right_toe_contact_score"][0]), 0.1)
        self.assertAlmostEqual(float(out["left_toe_contact_score"][0]), 0.2)
        self.assertAlmostEqual(float(out["right_heel_contact_score"][0]), 0.3)
        self.assertAlmostEqual(float(out["left_heel_contact_score"][0]), 0.4)

    def test_source_foot_alias_resolution_uses_alternate_names(self):
        names = ["LToeMarker", "LHeelMarker", "RToeMarker", "RHeelMarker"]
        frames = np.zeros((8, 4, 3), dtype=np.float32)
        frames[:, 0] = [0.1, 0.0, 0.0]
        frames[:, 1] = [-0.1, 0.0, 0.0]
        frames[:, 2] = [0.1, 0.2, 0.0]
        frames[:, 3] = [-0.1, 0.2, 0.0]
        buffer = _DummyBuffer(names, frames)
        out = infer_contacts_from_animation_buffer(
            buffer,
            smoothing_window=1,
            source_foot_joint_aliases={
                "left_toe": ["LToeMarker"],
                "left_heel": ["LHeelMarker"],
                "right_toe": ["RToeMarker"],
                "right_heel": ["RHeelMarker"],
            },
        )
        self.assertEqual(set(out), {
            "left_toe_contact_score",
            "left_heel_contact_score",
            "right_toe_contact_score",
            "right_heel_contact_score",
        })

    def test_airborne_stationary_is_low_contact(self):
        pos = np.array([[0.0, 0.0, 0.5]] * 20, dtype=np.float32)
        score = _contact_score_from_positions(pos, ground_height=0.0, velocity_dt=1.0 / 60.0)
        self.assertLess(float(np.max(score)), 0.1)

    def test_airborne_stationary_is_low_contact_with_inferred_ground(self):
        names = ["LeftToeBase", "LeftFoot", "RightToeBase", "RightFoot"]
        frames = np.zeros((8, 4, 3), dtype=np.float32)
        frames[:, :, 2] = 0.5
        buffer = _DummyBuffer(names, frames)
        out = infer_contacts_from_animation_buffer(buffer, smoothing_window=1)
        self.assertLess(max(float(np.max(score)) for score in out.values()), 0.1)

    def test_grounded_moving_is_low_contact(self):
        pos = np.zeros((20, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, 1.0, 20)
        score = _contact_score_from_positions(pos, ground_height=0.0, velocity_dt=1.0 / 60.0)
        self.assertLess(float(np.max(score[1:])), 0.5)

    def test_numeric_contact_scales_override_adaptive_estimates(self):
        pos = np.zeros((20, 3), dtype=np.float32)
        pos[:, 0] = np.linspace(0.0, 0.02, 20)
        adaptive = _contact_score_from_positions(pos, ground_height=0.0, velocity_dt=1.0 / 60.0)
        explicit = _contact_score_from_positions(
            pos,
            ground_height=0.0,
            velocity_dt=1.0 / 60.0,
            contact_velocity_scale=10.0,
        )
        self.assertGreater(float(np.mean(explicit)), float(np.mean(adaptive)))


if __name__ == "__main__":
    unittest.main()
