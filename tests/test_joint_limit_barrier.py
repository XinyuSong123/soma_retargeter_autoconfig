import unittest

import numpy as np
import newton.ik as ik
import warp as wp

from soma_retargeter.pipelines.ik_objectives import (
    IKRangeNormalizedJointLimitBarrier,
    range_normalized_joint_limit_barrier,
)


class _FakeModel:
    def __init__(self, lower, upper):
        self.joint_count = 1
        self.joint_coord_count = 1
        self.joint_dof_count = 1
        self.joint_q_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32)
        self.joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32)
        self.joint_dof_dim = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=wp.int32)
        self.joint_limit_lower = wp.array(np.array([lower], dtype=np.float32), dtype=wp.float32)
        self.joint_limit_upper = wp.array(np.array([upper], dtype=np.float32), dtype=wp.float32)


class TestJointLimitBarrier(unittest.TestCase):
    def test_barrier_is_zero_inside_margin_and_monotonic_near_limits(self):
        lower = -2.0
        upper = 1.0
        margin = 0.1

        center, _ = range_normalized_joint_limit_barrier(-0.5, lower, upper, margin)
        low_inner, _ = range_normalized_joint_limit_barrier(-1.75, lower, upper, margin)
        low_outer, _ = range_normalized_joint_limit_barrier(-1.9, lower, upper, margin)
        high_inner, _ = range_normalized_joint_limit_barrier(0.75, lower, upper, margin)
        high_outer, _ = range_normalized_joint_limit_barrier(0.95, lower, upper, margin)

        self.assertEqual(center, 0.0)
        self.assertLess(low_outer, low_inner)
        self.assertGreater(abs(low_outer), abs(low_inner))
        self.assertGreater(high_outer, high_inner)
        self.assertGreater(abs(high_outer), abs(high_inner))

    def test_barrier_derivative_matches_numeric_difference(self):
        q = 0.93
        lower = 0.0
        upper = 1.0
        margin = 0.2
        eps = 1.0e-5
        value_plus, _ = range_normalized_joint_limit_barrier(q + eps, lower, upper, margin)
        value_minus, _ = range_normalized_joint_limit_barrier(q - eps, lower, upper, margin)
        _, derivative = range_normalized_joint_limit_barrier(q, lower, upper, margin)

        self.assertAlmostEqual((value_plus - value_minus) / (2.0 * eps), derivative, places=5)

    def test_barrier_skips_continuous_or_unbounded_limits(self):
        self.assertEqual(range_normalized_joint_limit_barrier(42.0, -np.inf, np.inf), (0.0, 0.0))
        self.assertEqual(range_normalized_joint_limit_barrier(42.0, -1.0e8, 1.0e8), (0.0, 0.0))
        self.assertEqual(range_normalized_joint_limit_barrier(0.0, 1.0, 1.0), (0.0, 0.0))

    def test_warp_objective_residual_and_analytic_jacobian_are_range_normalized(self):
        model = _FakeModel(0.0, 1.0)
        objective = IKRangeNormalizedJointLimitBarrier(
            model.joint_limit_lower,
            model.joint_limit_upper,
            weight=2.0,
            margin_fraction=0.2,
        )
        objective.set_batch_layout(total_residuals=1, residual_offset=0, n_batch=1)
        objective.bind_device(wp.get_device())
        objective.init_buffers(model, jacobian_mode=ik.IKJacobianType.ANALYTIC)

        joint_q = wp.array(np.array([[0.9]], dtype=np.float32), dtype=wp.float32)
        body_q = wp.zeros((1, 1), dtype=wp.transform)
        residuals = wp.zeros((1, 1), dtype=wp.float32)
        jacobian = wp.zeros((1, 1, 1), dtype=wp.float32)

        objective.compute_residuals(body_q, joint_q, model, residuals, 0, None)
        objective.compute_jacobian_analytic(body_q, joint_q, model, jacobian, None, 0)
        wp.synchronize()

        self.assertAlmostEqual(float(residuals.numpy()[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(jacobian.numpy()[0, 0, 0]), 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
