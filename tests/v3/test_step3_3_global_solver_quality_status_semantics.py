from __future__ import annotations

from soma_retargeter.runtime.v3.runtime_quality_gates import RUNTIME_QUALITY_FAILED, RUNTIME_QUALITY_PASSED, classify_runtime_quality
from soma_retargeter.tools import run_v3_full_fleet_runtime_quality as runner


LOW_RESIDUAL_METRICS = {
    "nan_count": 0,
    "inf_count": 0,
    "normalized_task_residual_p95": 0.01,
    "normalized_task_residual_max": 0.02,
    "joint_limit_violation_count": 0,
    "max_joint_limit_violation": 0.0,
    "joint_velocity_p95": 0.0,
    "joint_acceleration_p95": 0.0,
    "target_se3_orthogonality_error_max": 0.0,
    "solver_success_fraction": 1.0,
}


def test_step3_3_pass_requires_solver_backed_evidence() -> None:
    classification = classify_runtime_quality(LOW_RESIDUAL_METRICS, solver_backed=True, residual_only=False)

    assert classification["quality_classification"] == RUNTIME_QUALITY_PASSED
    assert classification["solver_backed"] is True
    assert classification["residual_only"] is False


def test_step3_3_joint_limit_above_global_tolerance_remains_hard_failure() -> None:
    metrics = dict(LOW_RESIDUAL_METRICS)
    metrics["joint_limit_violation_count"] = 1
    metrics["max_joint_limit_violation"] = 1e-3

    classification = classify_runtime_quality(metrics, solver_backed=True, residual_only=False)

    assert classification["quality_classification"] == RUNTIME_QUALITY_FAILED
    assert "joint_limit_violation_above_global_tolerance" in classification["failure_or_warning_reasons"]


def test_step3_3_runner_evidence_does_not_promote_missing_attempted_completed_flags() -> None:
    evidence = runner._smoke_quality_evidence(
        {
            "status": RUNTIME_QUALITY_PASSED,
            "solver_type": "generic_chain_projection_least_squares_smoke",
            "solver_backed": True,
            "residual_only": False,
            "quality_pass_allowed": True,
            "metrics": LOW_RESIDUAL_METRICS,
        }
    )

    assert evidence["status"] != RUNTIME_QUALITY_PASSED
    assert "solver_backed_smoke_attempted" in evidence["warning_reasons"]
