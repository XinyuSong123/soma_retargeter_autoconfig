import unittest
import numpy as np
from unittest.mock import patch

try:
    from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
except ModuleNotFoundError as exc:  # pragma: no cover - local env may omit Newton/Warp
    NewtonPipeline = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


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


def _full_anchor_offsets():
    return {
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


def _identity_foot_targets():
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@unittest.skipIf(NewtonPipeline is None, f"Newton/Warp unavailable: {_IMPORT_ERROR}")
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
        anchors = _full_anchor_offsets()
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

    def test_execute_raises_for_contact_aware_multi_env(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.input_targets = [
            np.zeros((1, 1, 7), dtype=np.float32),
            np.zeros((1, 1, 7), dtype=np.float32),
        ]
        pipe.ik_iterations = 1
        pipe.joint_limit_weight = 0.0
        pipe.smooth_joint_filter_weight = 0.0
        pipe.contact_aware_foot_ik_enabled = True
        pipe.source_type = "soma"
        pipe.target_type = "robot"
        pipe.post_processing_enabled = False
        pipe.initialization_pose = None
        pipe.num_initialization_frames = 0
        pipe.num_stabilization_frames = 0

        with (
            patch("soma_retargeter.pipelines.newton_pipeline.pipeline_utils.get_source_str_from_type", return_value="soma"),
            patch("soma_retargeter.pipelines.newton_pipeline.pipeline_utils.get_target_str_from_type", return_value="robot"),
        ):
            with self.assertRaisesRegex(ValueError, "requires batch_size=1"):
                pipe.execute()

    def test_toe_only_contact_activates_toe_not_edges(self):
        pipe = _make_contact_pipe(_full_anchor_offsets())
        with patch("soma_retargeter.pipelines.newton_pipeline.ik.IKObjectivePosition", _DummyObjective):
            pipe._create_contact_aware_objectives(1, [])

        pipe.input_contact_scores = [{
            "left_toe_contact_score": np.array([1.0], dtype=np.float32),
            "left_heel_contact_score": np.array([0.0], dtype=np.float32),
            "right_toe_contact_score": np.array([0.0], dtype=np.float32),
            "right_heel_contact_score": np.array([0.0], dtype=np.float32),
        }]
        pipe._update_contact_objectives_for_frame(0, 0, _identity_foot_targets(), [])

        self.assertTrue(pipe.contact_objective_map["left_toe"]["active"][0])
        self.assertFalse(pipe.contact_objective_map["left_heel"]["active"][0])
        self.assertFalse(pipe.contact_objective_map["left_inner_edge"]["active"][0])
        self.assertFalse(pipe.contact_objective_map["left_outer_edge"]["active"][0])

    def test_heel_only_contact_activates_heel_not_edges(self):
        pipe = _make_contact_pipe(_full_anchor_offsets())
        with patch("soma_retargeter.pipelines.newton_pipeline.ik.IKObjectivePosition", _DummyObjective):
            pipe._create_contact_aware_objectives(1, [])

        pipe.input_contact_scores = [{
            "left_toe_contact_score": np.array([0.0], dtype=np.float32),
            "left_heel_contact_score": np.array([1.0], dtype=np.float32),
            "right_toe_contact_score": np.array([0.0], dtype=np.float32),
            "right_heel_contact_score": np.array([0.0], dtype=np.float32),
        }]
        pipe._update_contact_objectives_for_frame(0, 0, _identity_foot_targets(), [])

        self.assertFalse(pipe.contact_objective_map["left_toe"]["active"][0])
        self.assertTrue(pipe.contact_objective_map["left_heel"]["active"][0])
        self.assertFalse(pipe.contact_objective_map["left_inner_edge"]["active"][0])
        self.assertFalse(pipe.contact_objective_map["left_outer_edge"]["active"][0])

    def test_toe_and_heel_contact_activate_edges(self):
        pipe = _make_contact_pipe(_full_anchor_offsets())
        with patch("soma_retargeter.pipelines.newton_pipeline.ik.IKObjectivePosition", _DummyObjective):
            pipe._create_contact_aware_objectives(1, [])

        pipe.input_contact_scores = [{
            "left_toe_contact_score": np.array([1.0], dtype=np.float32),
            "left_heel_contact_score": np.array([1.0], dtype=np.float32),
            "right_toe_contact_score": np.array([0.0], dtype=np.float32),
            "right_heel_contact_score": np.array([0.0], dtype=np.float32),
        }]
        pipe._update_contact_objectives_for_frame(0, 0, _identity_foot_targets(), [])

        self.assertTrue(pipe.contact_objective_map["left_toe"]["active"][0])
        self.assertTrue(pipe.contact_objective_map["left_heel"]["active"][0])
        self.assertTrue(pipe.contact_objective_map["left_inner_edge"]["active"][0])
        self.assertTrue(pipe.contact_objective_map["left_outer_edge"]["active"][0])


if __name__ == "__main__":
    unittest.main()
