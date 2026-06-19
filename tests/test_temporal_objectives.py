import unittest

import newton.ik as ik
import numpy as np
import warp as wp

from soma_retargeter.pipelines.ik_objectives import IKTemporalJointRegularizer
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robot_registry_parser import build_runtime_retargeter_config


class _TemporalObjectiveModel:
    joint_coord_count = 3


class _TemporalScaleModel:
    joint_coord_count = 9
    joint_count = 2

    def __init__(self):
        self.joint_q_start = wp.array(np.array([0, 7, 9], dtype=np.int32), dtype=wp.int32)
        self.joint_qd_start = wp.array(np.array([0, 0, 2], dtype=np.int32), dtype=wp.int32)
        self.joint_dof_dim = wp.array(np.array([[0, 0], [0, 2]], dtype=np.int32), dtype=wp.int32)
        self.joint_limit_lower = wp.array(np.array([-1.0, -2.0], dtype=np.float32), dtype=wp.float32)
        self.joint_limit_upper = wp.array(np.array([1.0, 2.0], dtype=np.float32), dtype=wp.float32)


class _GuardModel:
    joint_coord_count = 8
    joint_count = 2

    def __init__(self):
        self.joint_q_start = wp.array(np.array([0, 7, 8], dtype=np.int32), dtype=wp.int32)
        self.joint_qd_start = wp.array(np.array([0, 0, 1], dtype=np.int32), dtype=wp.int32)
        self.joint_dof_dim = wp.array(np.array([[0, 0], [0, 1]], dtype=np.int32), dtype=wp.int32)
        self.joint_limit_lower = wp.array(np.array([-1.0], dtype=np.float32), dtype=wp.float32)
        self.joint_limit_upper = wp.array(np.array([1.0], dtype=np.float32), dtype=wp.float32)


class TestTemporalObjectives(unittest.TestCase):
    def test_velocity_residual_is_per_env_sample_rate_and_range_normalized(self):
        reference = wp.array(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            dtype=wp.float32,
        )
        scales = wp.array(
            np.array(
                [
                    [0.0, 2.0, 4.0],
                    [0.0, 10.0, 20.0],
                ],
                dtype=np.float32,
            ),
            dtype=wp.float32,
        )
        objective = IKTemporalJointRegularizer(reference, scales, weight=0.5, coord_masks=np.array([0.0, 1.0, 1.0]))
        objective.set_batch_layout(total_residuals=3, residual_offset=0, n_batch=2)
        objective.bind_device(wp.get_device())
        objective.init_buffers(_TemporalObjectiveModel(), ik.IKJacobianType.ANALYTIC)

        joint_q = wp.array(
            np.array(
                [
                    [9.0, 0.5, -0.25],
                    [9.0, 1.2, 0.5],
                ],
                dtype=np.float32,
            ),
            dtype=wp.float32,
        )
        residuals = wp.zeros((2, 3), dtype=wp.float32)
        jacobian = wp.zeros((2, 3, 3), dtype=wp.float32)
        problem_idx = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32)

        objective.compute_residuals(wp.zeros((2, 1), dtype=wp.transform), joint_q, _TemporalObjectiveModel(), residuals, 0, problem_idx)
        objective.compute_jacobian_analytic(None, joint_q, _TemporalObjectiveModel(), jacobian, None, 0)
        wp.synchronize()

        self.assertTrue(np.allclose(residuals.numpy(), [[0.0, 0.5, -0.5], [0.0, 1.0, -5.0]]))
        self.assertAlmostEqual(float(jacobian.numpy()[0, 1, 1]), 1.0)
        self.assertAlmostEqual(float(jacobian.numpy()[1, 2, 2]), 10.0)

    def test_acceleration_residual_uses_two_previous_frames(self):
        reference = wp.array(np.array([[1.0, 2.0]], dtype=np.float32), dtype=wp.float32)
        reference2 = wp.array(np.array([[0.5, 1.5]], dtype=np.float32), dtype=wp.float32)
        scales = wp.array(np.array([[2.0, 4.0]], dtype=np.float32), dtype=wp.float32)
        objective = IKTemporalJointRegularizer(
            reference,
            scales,
            reference_q2=reference2,
            weight=1.0,
            mode=IKTemporalJointRegularizer.MODE_ACCELERATION,
        )
        objective.set_batch_layout(total_residuals=2, residual_offset=0, n_batch=1)
        objective.bind_device(wp.get_device())
        objective.init_buffers(type("M", (), {"joint_coord_count": 2})(), ik.IKJacobianType.ANALYTIC)

        joint_q = wp.array(np.array([[1.75, 2.25]], dtype=np.float32), dtype=wp.float32)
        residuals = wp.zeros((1, 2), dtype=wp.float32)
        problem_idx = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32)

        objective.compute_residuals(wp.zeros((1, 1), dtype=wp.transform), joint_q, None, residuals, 0, problem_idx)
        wp.synchronize()

        self.assertTrue(np.allclose(residuals.numpy(), [[0.5, -1.0]]))

    def test_pipeline_temporal_scales_use_sample_rate_and_skip_floating_root(self):
        pipe = NewtonPipeline.__new__(NewtonPipeline)
        pipe.input_sample_rates = [50.0, 100.0]
        velocity, acceleration, masks = pipe._build_temporal_scale_matrices(_TemporalScaleModel(), 2)

        self.assertTrue(np.allclose(masks[:7], np.zeros(7)))
        self.assertEqual(float(masks[7]), 1.0)
        self.assertEqual(float(masks[8]), 1.0)
        self.assertAlmostEqual(float(velocity[0, 7]), 25.0)
        self.assertAlmostEqual(float(velocity[1, 7]), 50.0)
        self.assertAlmostEqual(float(acceleration[0, 8]), 625.0)
        self.assertAlmostEqual(float(acceleration[1, 8]), 2500.0)

    def test_runtime_config_passes_temporal_weights(self):
        cfg = build_runtime_retargeter_config(
            "roboparty_rpo",
            {
                "ik_map": {},
                "temporal_velocity_weight": 0.2,
                "temporal_acceleration_weight": 0.05,
            },
        )
        self.assertEqual(cfg["temporal_velocity_weight"], 0.2)
        self.assertEqual(cfg["temporal_acceleration_weight"], 0.05)

    def test_priority_guard_costs_ignore_root_and_track_joint_limit_margin(self):
        costs = NewtonPipeline._joint_limit_guard_costs(
            _GuardModel(),
            np.array(
                [
                    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.0],
                    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.95],
                ],
                dtype=np.float32,
            ),
            margin_fraction=0.2,
        )

        self.assertAlmostEqual(float(costs[0]), 0.0)
        self.assertGreater(float(costs[1]), 0.0)

    def test_priority_guard_zero_margin_tracks_hard_limit_penetration_only(self):
        costs = NewtonPipeline._joint_limit_guard_costs(
            _GuardModel(),
            np.array(
                [
                    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.99],
                    [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 1.1],
                ],
                dtype=np.float32,
            ),
            margin_fraction=0.0,
        )

        self.assertAlmostEqual(float(costs[0]), 0.0)
        self.assertGreater(float(costs[1]), 0.0)

    def test_priority_guard_rolls_back_only_when_protected_cost_regresses(self):
        self.assertTrue(
            np.array_equal(
                NewtonPipeline._priority_guard_should_rollback(
                    np.array([1.0, 1.0], dtype=np.float32),
                    np.array([1.01, 1.2], dtype=np.float32),
                    tolerance=0.05,
                    absolute_tolerance=0.0,
                ),
                np.array([False, True]),
            )
        )

    def test_priority_guard_cpu_check_only_required_for_soft_margin_guard(self):
        self.assertFalse(NewtonPipeline._priority_guard_requires_cpu_check(0.0))
        self.assertFalse(NewtonPipeline._priority_guard_requires_cpu_check(-0.1))
        self.assertTrue(NewtonPipeline._priority_guard_requires_cpu_check(0.08))

    def test_runtime_config_passes_priority_guard_options(self):
        cfg = build_runtime_retargeter_config(
            "roboparty_rpo",
            {
                "ik_map": {},
                "priority_residual_guard_enabled": False,
                "priority_residual_guard_tolerance": 0.2,
                "priority_residual_guard_absolute_tolerance": 0.01,
            },
        )
        self.assertFalse(cfg["priority_residual_guard_enabled"])
        self.assertEqual(cfg["priority_residual_guard_tolerance"], 0.2)
        self.assertEqual(cfg["priority_residual_guard_absolute_tolerance"], 0.01)


if __name__ == "__main__":
    unittest.main()
