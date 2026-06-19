import unittest

import numpy as np
import newton.ik as ik
import warp as wp

from soma_retargeter.pipelines.ik_objectives import IKObjectivePoleVector, IKObjectiveSphereCollisionBarrier
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robotics.reachability import quat_xyzw_to_rotation_vector, rotation_vector_to_quat_xyzw


class TestNewtonV2SparseObjectives(unittest.TestCase):
    def test_select_ik_jacobian_mode_uses_mixed_only_when_needed(self):
        class Objective:
            def __init__(self, analytic):
                self.analytic = analytic

            def supports_analytic(self):
                return self.analytic

        self.assertEqual(
            NewtonPipeline._select_ik_jacobian_mode([Objective(True), Objective(True)]),
            ik.IKJacobianType.ANALYTIC,
        )
        self.assertEqual(
            NewtonPipeline._select_ik_jacobian_mode([Objective(True), Objective(False)]),
            ik.IKJacobianType.MIXED,
        )

    def test_build_target_mapping_keeps_all_effectors_but_skips_zero_weight_objectives(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.robot_builder = type("B", (), {"body_label": ["base", "hand", "torso"]})()

        class Skeleton:
            def joint_index(self, name):
                return {"Hips": 0, "LeftHand": 1, "Chest": 2}[name]

        cfg = {
            "ik_map": {
                "Hips": {"t_body": "base", "r_body": "base", "t_weight": 10.0, "r_weight": 0.0},
                "LeftHand": {
                    "t_body": "hand",
                    "r_body": "hand",
                    "t_weight": 1.0,
                    "r_weight": 0.0,
                    "v2_position_link_offset": [0.2, 0.0, 0.0],
                },
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
        self.assertEqual(pos_data[0][:3], (0, 0, 10.0))
        self.assertTrue(np.allclose(pos_data[0][3], [0.0, 0.0, 0.0]))
        self.assertEqual(pos_data[1][:3], (1, 1, 1.0))
        self.assertTrue(np.allclose(pos_data[1][3], [0.2, 0.0, 0.0]))
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
                {
                    "name": "LeftArm_direction",
                    "reference_site": "Chest",
                    "target_site": "LeftArm",
                    "weight": 10.0,
                    "analytic_jacobian": True,
                }
            ],
        }

        pipe._build_target_mapping(None, Skeleton(), cfg)

        self.assertEqual(
            pipe.mapped_body_link_direction_data,
            [(1, 2, 1, 2, 10.0, "LeftArm_direction", True)],
        )

    def test_build_target_mapping_extracts_v2_pole_vector_tasks(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.robot_builder = type("B", (), {"body_label": ["upper", "forearm", "hand"]})()

        class Skeleton:
            def joint_index(self, name):
                return {"LeftArm": 0, "LeftForeArm": 1, "LeftHand": 2}[name]

        cfg = {
            "ik_map": {
                "LeftArm": {"t_body": "upper", "r_body": "upper", "t_weight": 0.0, "r_weight": 0.0},
                "LeftForeArm": {"t_body": "forearm", "r_body": "forearm", "t_weight": 0.0, "r_weight": 0.0},
                "LeftHand": {"t_body": "hand", "r_body": "hand", "t_weight": 1.0, "r_weight": 0.0},
            },
            "pole_vector_tasks": [
                {
                    "name": "LeftForeArm_pole_vector",
                    "reference_site": "LeftArm",
                    "source_semantic": "LeftForeArm",
                    "target_site": "LeftHand",
                    "weight": 10.0,
                    "analytic_jacobian": False,
                }
            ],
        }

        pipe._build_target_mapping(None, Skeleton(), cfg)

        self.assertEqual(
            pipe.mapped_body_link_pole_vector_data,
            [(0, 1, 2, 0, 1, 2, 10.0, "LeftForeArm_pole_vector", False)],
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

    def test_pole_vector_target_uses_bend_plane_normal_with_fallback(self):
        normal, used_fallback = NewtonPipeline._pole_normal_between_positions(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        )
        self.assertFalse(used_fallback)
        self.assertTrue(np.allclose(normal, [0.0, 0.0, 1.0]))

        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        normal, used_fallback = NewtonPipeline._pole_normal_between_positions(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            fallback,
        )
        self.assertTrue(used_fallback)
        self.assertTrue(np.allclose(normal, fallback))

    def test_pole_vector_analytic_jacobian_matches_finite_difference_for_body_masks(self):
        parent = np.array([0.4, -0.2, 0.1], dtype=np.float64)
        middle = np.array([1.1, 0.5, 0.4], dtype=np.float64)
        child = np.array([0.7, 1.4, 1.2], dtype=np.float64)
        linear_velocity = np.array([0.17, -0.11, 0.07], dtype=np.float64)
        angular_velocity = np.array([0.13, -0.19, 0.23], dtype=np.float64)
        weight = 2.0
        body_q = wp.array(
            [[
                wp.transform(wp.vec3(*parent), wp.quat_identity()),
                wp.transform(wp.vec3(*middle), wp.quat_identity()),
                wp.transform(wp.vec3(*child), wp.quat_identity()),
            ]],
            dtype=wp.transform,
        )
        joint_s = wp.array(
            [[
                wp.spatial_vector(
                    float(linear_velocity[0]),
                    float(linear_velocity[1]),
                    float(linear_velocity[2]),
                    float(angular_velocity[0]),
                    float(angular_velocity[1]),
                    float(angular_velocity[2]),
                )
            ]],
            dtype=wp.spatial_vector,
        )
        model = type("M", (), {"joint_dof_count": 1})()

        def body_velocity(position, affected):
            if not affected:
                return np.zeros(3, dtype=np.float64)
            return linear_velocity + np.cross(angular_velocity, position)

        def residual(p, m, ch):
            current = np.cross(m - p, ch - m)
            current = current / np.linalg.norm(current)
            return -weight * current

        cases = {
            "parent_only": (1, 0, 0),
            "middle_only": (0, 1, 0),
            "child_only": (0, 0, 1),
            "parent_middle": (1, 1, 0),
            "middle_child": (0, 1, 1),
            "all": (1, 1, 1),
        }
        eps = 1.0e-5
        for name, masks in cases.items():
            with self.subTest(name=name):
                objective = IKObjectivePoleVector(0, 1, 2, None, weight=weight, analytic_jacobian=True)
                objective.bind_device(wp.get_device())
                objective.affects_parent_dof = wp.array(np.array([masks[0]], dtype=np.uint8), dtype=wp.uint8)
                objective.affects_middle_dof = wp.array(np.array([masks[1]], dtype=np.uint8), dtype=wp.uint8)
                objective.affects_child_dof = wp.array(np.array([masks[2]], dtype=np.uint8), dtype=wp.uint8)
                jacobian = wp.zeros((1, 3, 1), dtype=wp.float32)

                objective.compute_jacobian_analytic(body_q, None, model, jacobian, joint_s, 0)
                wp.synchronize()

                parent_delta = body_velocity(parent, masks[0])
                middle_delta = body_velocity(middle, masks[1])
                child_delta = body_velocity(child, masks[2])
                residual_plus = residual(
                    parent + eps * parent_delta,
                    middle + eps * middle_delta,
                    child + eps * child_delta,
                )
                residual_minus = residual(
                    parent - eps * parent_delta,
                    middle - eps * middle_delta,
                    child - eps * child_delta,
                )
                finite_difference = (residual_plus - residual_minus) / (2.0 * eps)

                self.assertTrue(
                    np.allclose(jacobian.numpy()[0, :, 0], finite_difference, atol=5.0e-5),
                    f"{jacobian.numpy()[0, :, 0]} != {finite_difference}",
                )

    def test_pole_vector_analytic_buffers_follow_model_ancestor_dofs(self):
        class FakeModel:
            joint_count = 4
            body_count = 4
            joint_dof_count = 7
            joint_qd_start = wp.array([0, 1, 3, 4, 7], dtype=wp.int32)
            joint_child = wp.array([0, 1, 2, 3], dtype=wp.int32)
            joint_parent = wp.array([-1, 0, 1, 2], dtype=wp.int32)

        objective = IKObjectivePoleVector(1, 2, 3, None, weight=1.0, analytic_jacobian=True)
        objective.set_batch_layout(total_residuals=3, residual_offset=0, n_batch=1)
        objective.bind_device(wp.get_device())
        objective.init_buffers(FakeModel(), ik.IKJacobianType.ANALYTIC)

        self.assertEqual(objective.affects_parent_dof.numpy().tolist(), [1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(objective.affects_middle_dof.numpy().tolist(), [1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(objective.affects_child_dof.numpy().tolist(), [1, 1, 1, 1, 1, 1, 1])

    def test_collision_objectives_are_created_from_compiled_pairs_when_weighted(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.collision_weight = 3.0
        pipe.mapped_body_link_by_joint = {"Chest": 1, "LeftHand": 2, "RightHand": 3}
        pipe.compiled_collision_config = {
            "enabled": True,
            "margin": 0.05,
            "runtime_barrier": "sphere_pair",
            "proxies": [
                {"semantic": "Chest", "local_center": [0.0, 0.0, 0.0], "radius": 0.3},
                {"semantic": "LeftHand", "local_center": [0.0, 0.0, 0.0], "radius": 0.05},
                {"semantic": "RightHand", "local_center": [0.0, 0.0, 0.0], "radius": 0.05},
            ],
            "pairs": [
                {"a": "Chest", "b": "LeftHand", "margin": 0.04},
                {"a": "Chest", "b": "Missing", "margin": 0.04},
            ],
        }

        objectives = pipe._create_collision_objectives()
        self.assertEqual(len(objectives), 1)
        self.assertIsInstance(objectives[0], IKObjectiveSphereCollisionBarrier)
        self.assertEqual(objectives[0].link_index_a, 1)
        self.assertEqual(objectives[0].link_index_b, 2)
        self.assertAlmostEqual(objectives[0].radius_a, 0.3)
        self.assertAlmostEqual(objectives[0].radius_b, 0.05)
        self.assertAlmostEqual(objectives[0].margin, 0.04)
        self.assertAlmostEqual(objectives[0].weight, 3.0)
        self.assertEqual(pipe.collision_objective_report["created"], 1)
        self.assertEqual(pipe.collision_objective_report["skipped"], 1)

    def test_collision_objectives_default_to_disabled_when_weight_is_zero(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.collision_weight = 0.0
        pipe.compiled_collision_config = {"enabled": True, "pairs": [{"a": "Chest", "b": "LeftHand"}]}
        objectives = pipe._create_collision_objectives()
        self.assertEqual(objectives, [])
        self.assertFalse(pipe.collision_objective_report["enabled"])


if __name__ == "__main__":
    unittest.main()
