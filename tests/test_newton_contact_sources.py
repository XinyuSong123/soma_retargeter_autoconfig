import unittest
import numpy as np
import warp as wp

from soma_retargeter.pipelines.ik_objectives import IKObjectivePerEnvGroundHeightBarrier, IKObjectiveSphereCollisionBarrier
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


class _DummyObjective:
    def __init__(self):
        self.link_offset = (0.0, 0.0, 0.0)


class TestNewtonContactModes(unittest.TestCase):
    def test_ground_barrier_residual_penalizes_only_penetration(self):
        obj = IKObjectivePerEnvGroundHeightBarrier(
            link_index=0,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            weights=wp.array(np.array([2.0], dtype=np.float32), dtype=wp.float32),
            ground_height=0.0,
            margin=0.1,
        )
        obj.bind_device(wp.get_device())
        problem_idx = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)

        body_q = wp.array([[wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity())]], dtype=wp.transform)
        residuals = wp.zeros((1, 1), dtype=wp.float32)
        obj.compute_residuals(body_q, None, None, residuals, 0, problem_idx)
        self.assertTrue(np.allclose(residuals.numpy(), [[1.0]]))

        body_q = wp.array([[wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())]], dtype=wp.transform)
        residuals = wp.zeros((1, 1), dtype=wp.float32)
        obj.compute_residuals(body_q, None, None, residuals, 0, problem_idx)
        self.assertTrue(np.allclose(residuals.numpy(), [[0.0]]))

    def test_sphere_collision_barrier_residual_penalizes_overlap_only(self):
        obj = IKObjectiveSphereCollisionBarrier(
            link_index_a=0,
            link_offset_a=wp.vec3(0.0, 0.0, 0.0),
            radius_a=0.2,
            link_index_b=1,
            link_offset_b=wp.vec3(0.0, 0.0, 0.0),
            radius_b=0.2,
            margin=0.1,
            weight=2.0,
        )
        obj.bind_device(wp.get_device())

        body_q = wp.array(
            [[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
            ]],
            dtype=wp.transform,
        )
        residuals = wp.zeros((1, 1), dtype=wp.float32)
        obj.compute_residuals(body_q, None, None, residuals, 0, None)
        self.assertTrue(np.allclose(residuals.numpy(), [[1.0]]))

        body_q = wp.array(
            [[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.7, 0.0, 0.0), wp.quat_identity()),
            ]],
            dtype=wp.transform,
        )
        residuals = wp.zeros((1, 1), dtype=wp.float32)
        obj.compute_residuals(body_q, None, None, residuals, 0, None)
        self.assertTrue(np.allclose(residuals.numpy(), [[0.0]]))

        obj.affects_dof_a = wp.array(np.array([1], dtype=np.uint8), dtype=wp.uint8)
        obj.affects_dof_b = wp.array(np.array([0], dtype=np.uint8), dtype=wp.uint8)
        joint_s = wp.array([[wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)]], dtype=wp.spatial_vector)
        jacobian = wp.zeros((1, 1, 1), dtype=wp.float32)
        model = type("M", (), {"joint_dof_count": 1})()
        body_q = wp.array(
            [[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
            ]],
            dtype=wp.transform,
        )
        obj.compute_jacobian_analytic(body_q, None, model, jacobian, joint_s, 0)
        self.assertTrue(np.allclose(jacobian.numpy(), [[[20.0]]]))

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
        summary = pipe.contact_score_summary[0]
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["source"], "npz_foot_contacts")
        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["channels"]["left_heel_contact_score"]["frame_count"], 2)
        self.assertAlmostEqual(summary["channels"]["left_heel_contact_score"]["active_fraction_0_5"], 0.5)

    def test_contact_score_summary_reports_unavailable_scores(self):
        summary = NewtonPipeline._summarize_contact_scores(None, "auto", 3)
        self.assertEqual(summary["status"], "unavailable")
        self.assertEqual(summary["source"], "auto")
        self.assertEqual(summary["smoothing_window"], 3)

    def test_explicit_contacts_are_prepended_for_initialization_frames(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_source = "auto"
        pipe.contact_aware_foot_ik = {"contact_score_smoothing_window": 1}
        pipe.num_initialization_frames = 2
        pipe.num_stabilization_frames = 1

        class B:
            foot_contacts = np.array([[0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

        scores = pipe._read_explicit_contact_scores(B())
        aligned = pipe._prepend_contact_scores_for_warmup(scores)
        self.assertEqual(len(aligned["left_heel_contact_score"]), 5)
        self.assertTrue(np.allclose(aligned["left_heel_contact_score"][:3], 0.25))
        self.assertEqual(float(aligned["right_toe_contact_score"][3]), 1.0)

    def test_contact_update_uses_per_env_weight(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_aware_foot_ik = {"contact_on_threshold": 0.6, "contact_off_threshold": 0.3}
        pipe.input_contact_scores = [{"left_toe_contact_score": np.array([1.0], dtype=np.float32)}]
        pipe.mapped_joints = ["LeftFoot", "RightFoot"]

        class Obj:
            link_offset = wp.vec3(0.0, 0.0, 0.0)

            def __init__(self):
                self.weights = []

            def set_target_position(self, env, value):
                pass

            def set_weight(self, env, value):
                self.weights.append((env, value))

        obj = Obj()
        pipe.contact_objective_map = {
            "left_toe": {
                "objective": obj,
                "score_key": "left_toe_contact_score",
                "stance": 0.8,
                "swing": 0.1,
                "active": [False],
                "locked": [None],
            }
        }

        frame_targets = np.zeros((2, 7), dtype=np.float32)
        frame_targets[:, 6] = 1.0
        pipe._update_contact_objectives_for_frame(0, 0, frame_targets, [])
        self.assertEqual(obj.weights, [(0, 0.8)])

    def test_contact_min_duration_holds_locked_target(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_aware_foot_ik = {
            "contact_on_threshold": 0.6,
            "contact_off_threshold": 0.3,
            "min_contact_frames": 2,
            "release_blend_frames": 0,
        }
        pipe.input_contact_scores = [{"left_toe_contact_score": np.array([1.0, 0.0, 0.0], dtype=np.float32)}]
        pipe.mapped_joints = ["LeftFoot", "RightFoot"]

        class Obj:
            link_offset = wp.vec3(0.0, 0.0, 0.0)

            def __init__(self):
                self.targets = []
                self.weights = []

            def set_target_position(self, env, value):
                self.targets.append(tuple(np.array(value, dtype=np.float32)))

            def set_weight(self, env, value):
                self.weights.append((env, value))

        obj = Obj()
        pipe.contact_objective_map = {
            "left_toe": {
                "objective": obj,
                "score_key": "left_toe_contact_score",
                "stance": 0.8,
                "swing": 0.1,
                "active": [False],
                "locked": [None],
                "age": [0],
                "release_remaining": [0],
                "release_total": [0],
                "release_start": [None],
            }
        }

        frame_targets = np.zeros((2, 7), dtype=np.float32)
        frame_targets[:, 6] = 1.0
        frame_targets[0, 0] = 1.0
        pipe._update_contact_objectives_for_frame(0, 0, frame_targets, [])
        frame_targets[0, 0] = 2.0
        pipe._update_contact_objectives_for_frame(0, 1, frame_targets, [])
        frame_targets[0, 0] = 3.0
        pipe._update_contact_objectives_for_frame(0, 2, frame_targets, [])

        self.assertEqual(obj.targets[0], (1.0, 0.0, 0.0))
        self.assertEqual(obj.targets[1], (1.0, 0.0, 0.0))
        self.assertEqual(obj.targets[2], (3.0, 0.0, 0.0))
        self.assertEqual(obj.weights, [(0, 0.8), (0, 0.8), (0, 0.1)])

    def test_contact_release_blends_from_locked_target(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.contact_aware_foot_ik = {
            "contact_on_threshold": 0.6,
            "contact_off_threshold": 0.3,
            "min_contact_frames": 1,
            "release_blend_frames": 2,
        }
        pipe.input_contact_scores = [{"left_toe_contact_score": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}]
        pipe.mapped_joints = ["LeftFoot", "RightFoot"]

        class Obj:
            link_offset = wp.vec3(0.0, 0.0, 0.0)

            def __init__(self):
                self.targets = []
                self.weights = []

            def set_target_position(self, env, value):
                self.targets.append(tuple(np.array(value, dtype=np.float32)))

            def set_weight(self, env, value):
                self.weights.append((env, value))

        obj = Obj()
        pipe.contact_objective_map = {
            "left_toe": {
                "objective": obj,
                "score_key": "left_toe_contact_score",
                "stance": 0.8,
                "swing": 0.2,
                "active": [False],
                "locked": [None],
                "age": [0],
                "release_remaining": [0],
                "release_total": [0],
                "release_start": [None],
            }
        }

        frame_targets = np.zeros((2, 7), dtype=np.float32)
        frame_targets[:, 6] = 1.0
        for frame, x in enumerate([1.0, 3.0, 5.0, 7.0]):
            frame_targets[0, 0] = x
            pipe._update_contact_objectives_for_frame(0, frame, frame_targets, [])

        self.assertEqual(obj.targets[0], (1.0, 0.0, 0.0))
        self.assertEqual(obj.targets[1], (1.0, 0.0, 0.0))
        self.assertEqual(obj.targets[2], (3.0, 0.0, 0.0))
        self.assertEqual(obj.targets[3], (7.0, 0.0, 0.0))
        self.assertEqual(obj.weights, [(0, 0.8), (0, 0.8), (0, 0.5), (0, 0.2)])

    def test_ground_barrier_objectives_use_contact_weight_bands(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.ground_barrier_enabled = True
        pipe.ground_barrier_config = {
            "ground_height": 0.0,
            "margin": 0.05,
            "stance_weight": 2.0,
            "swing_weight": 0.25,
        }
        pipe.contact_aware_foot_ik = {
            "contact_on_threshold": 0.6,
            "anchor_offsets": {
                "left": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
                "right": {"toe": [0.1, 0.0, 0.0], "heel": [-0.1, 0.0, 0.0]},
            },
        }
        pipe.mapped_body_link_by_joint = {"LeftFoot": 1, "RightFoot": 2}

        objectives = pipe._create_ground_barrier_objectives(2)
        self.assertEqual(len(objectives), 4)
        self.assertEqual(set(pipe.ground_barrier_objective_map), {"left_toe", "left_heel", "right_toe", "right_heel"})
        self.assertTrue(np.allclose(objectives[0].weights.numpy(), [0.25, 0.25]))

        class Obj:
            def __init__(self):
                self.weights = []

            def set_weight(self, env, value):
                self.weights.append((env, value))

        left_toe = Obj()
        left_heel = Obj()
        right_toe = Obj()
        right_heel = Obj()
        dummy_objectives = {
            "left_toe": left_toe,
            "left_heel": left_heel,
            "right_toe": right_toe,
            "right_heel": right_heel,
        }
        for key, obj in dummy_objectives.items():
            pipe.ground_barrier_objective_map[key]["objective"] = obj
        pipe.input_contact_scores = [{"left_toe_contact_score": np.array([1.0], dtype=np.float32)}]
        pipe._update_ground_barrier_objectives_for_frame(0, 0, objectives)
        self.assertEqual(left_toe.weights, [(0, 2.0)])
        self.assertEqual(left_heel.weights, [(0, 0.25)])


if __name__ == "__main__":
    unittest.main()
