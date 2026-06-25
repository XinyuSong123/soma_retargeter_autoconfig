from __future__ import annotations

import numpy as np

from soma_retargeter.robotics.v3.projection_certificate import ProjectionCertificateEvidence, build_projection_certificate


def _passing_seed_consensus() -> dict:
    return {"checked": True, "passed": True, "start_count": 1}


def test_kkt_recompute_rejects_out_of_bounds_q_despite_provided_primal_true():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.zeros(3),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.array([[1.0], [0.0], [0.0]]),
            active_coordinates=[7],
            coordinate_values=np.array([2.0]),
            lower_bounds=np.array([-1.0]),
            upper_bounds=np.array([1.0]),
            seed_consensus=_passing_seed_consensus(),
            continuation_passed=True,
            kkt={
                "primal_feasible": True,
                "satisfied": True,
                "stationarity_inf_norm": 0.0,
                "stationarity_tolerance": 1e-7,
                "task_gradient": [0.0],
            },
        )
    ).to_json()

    assert cert["kkt"]["source"] == "independent_recompute"
    assert cert["kkt"]["raw"]["primal_feasible"] is True
    assert cert["kkt"]["primal_feasible"] is False
    assert cert["kkt"]["primal_violation_inf_norm"] == 1.0
    assert cert["gates"]["projected_gradient_kkt"] is False
    assert cert["certificate_class"] != "exact_reachable"


def test_kkt_recompute_uses_raw_jacobian_and_residual_not_provided_zero_gradient():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.zeros(3),
            projected=np.array([1.0, 0.0, 0.0]),
            seed=np.zeros(3),
            jacobian=np.array([[1.0], [0.0], [0.0]]),
            active_coordinates=[3],
            coordinate_values=np.array([1.0]),
            lower_bounds=np.array([-1.0]),
            upper_bounds=np.array([1.0]),
            seed_consensus=_passing_seed_consensus(),
            continuation_passed=True,
            kkt={
                "satisfied": True,
                "stationarity_inf_norm": 0.0,
                "stationarity_tolerance": 1e-7,
                "task_gradient": [0.0],
            },
        )
    ).to_json()

    assert cert["audit_evidence"]["task_gradient"] == [1.0]
    assert cert["kkt"]["gradient"] == [1.0]
    assert cert["kkt"]["provided_task_gradient_consistent"] is False
    assert cert["kkt"]["dual_feasible"] is False
    assert cert["gates"]["projected_gradient_kkt"] is False


def test_prior_gradient_is_serialized_and_ratio_does_not_require_nonzero_task_gradient():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.array([0.0, 1.0, 0.0]),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.array([[1.0], [0.0], [0.0]]),
            active_coordinates=[0],
            coordinate_values=np.array([0.0]),
            lower_bounds=np.array([-1.0]),
            upper_bounds=np.array([1.0]),
            seed_consensus=_passing_seed_consensus(),
            continuation_passed=True,
            prior_gradient=np.array([2e-7]),
        )
    ).to_json()

    assert cert["audit_evidence"]["task_gradient"] == [0.0]
    assert cert["audit_evidence"]["prior_gradient"] == [2e-07]
    assert cert["kkt"]["task_gradient_inf_norm"] == 0.0
    assert cert["kkt"]["prior_gradient_inf_norm"] == 2e-7
    assert cert["kkt"]["prior_cancellation_ratio"] == 2.0
