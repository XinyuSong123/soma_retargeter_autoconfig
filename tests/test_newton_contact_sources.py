import unittest
import numpy as np
from unittest.mock import patch

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


class _DummyObjective:
    def __init__(self, **kwargs):
        self.link_index = kwargs["link_index"]
        self.link_offset = kwargs["link_offset"]
        self.target_positions = kwargs["target_positions"]
        self.weight = kwargs["weight"]
        self.targets = {}

    def set_weight(self, weight):
        self.weight = weight

    def set_target_position(self, env, target):
        self.targets[env] = target


def _make_contact_pipe(anchor_offsets):
    pipe = NewtonPipeline.__new__(NewtonPipeline)
    pipe.contact_source = "auto"
    pipe.contact_aware_foot_ik_enabled = True
    pipe.contact_aware_foot_ik = {
        "enabled": True,
        "anchor_offsets": anchor_offsets,
        "edge_weight_stance": 0.5,
        "edge_weight_swing": 0.05,
    }
    pipe.mapped_joints = ["LeftFoot", "RightFoot"]
    pipe.mapped_body_link_pos_data = [(11, 1.0), (12, 1.0)]
    return pipe


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

    def test_creates_toe_heel_and_edge_objectives_when_all_anchors_exist(self):
        anchors = {
            "left": {
                "toe": [0.1, 0.0, 0.0],
                "heel": [-0.1, 0.0, 0.0],
                "inner_edge": [0.0, -0.05, 0.0],
                "outer_edge": [0.0, 0.05, 0.0],
            },
            "right": {
                "toe": [0.1, 0.0, 0.0],
                "heel": [-0.1, 0.0, 0.0],
                "inner_edge": [0.0, 0.05, 0.0],
                "outer_edge": [0.0, -0.05, 0.0],
            },
        }
        pipe = _make_contact_pipe(anchors)

        with patch("soma_retargeter.pipelines.newton_pipeline.ik.IKObjectivePosition", _DummyObjective):
            out = pipe._create_contact_aware_objectives(1, [])

        self.assertEqual(len(out), 8)
        self.assertEqual(
            set(pipe.contact_objective_map),
            {
                "left_toe",
                "left_heel",
                "left_inner_edge",
                "left_outer_edge",
                "right_toe",
                "right_heel",
                "right_inner_edge",
                "right_outer_edge",
            },
        )
        self.assertEqual(pipe.contact_objective_map["left_inner_edge"]["stance"], 0.5)
        self.assertEqual(pipe.contact_objective_map["right_outer_edge"]["swing"], 0.05)

    def test_skips_edge_objectives_when_edge_anchors_are_missing(self):
        anchors = {
            "left": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
            "right": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
        }
        pipe = _make_contact_pipe(anchors)

        with patch("soma_retargeter.pipelines.newton_pipeline.ik.IKObjectivePosition", _DummyObjective):
            out = pipe._create_contact_aware_objectives(1, [])

        self.assertEqual(len(out), 4)
        self.assertEqual(set(pipe.contact_objective_map), {"left_toe", "left_heel", "right_toe", "right_heel"})

    def test_contact_source_none_creates_no_contact_objectives(self):
        anchors = {
            "left": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
            "right": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
        }
        pipe = _make_contact_pipe(anchors)
        pipe.contact_source = "none"

        with patch("soma_retargeter.pipelines.newton_pipeline.ik.IKObjectivePosition", _DummyObjective):
            out = pipe._create_contact_aware_objectives(1, [])

        self.assertEqual(out, [])
        self.assertEqual(pipe.contact_objective_map, {})

    def test_missing_contact_config_behaves_without_contact_objectives(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_source = "auto"
        pipe.contact_aware_foot_ik_enabled = False
        pipe.contact_aware_foot_ik = {}
        out = pipe._create_contact_aware_objectives(1, [])
        self.assertEqual(out, [])
        self.assertEqual(pipe.contact_objective_map, {})

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
