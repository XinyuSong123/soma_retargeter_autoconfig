from __future__ import annotations

from soma_retargeter.runtime.v3.runtime_quality_gates import RUNTIME_QUALITY_PASSED, RUNTIME_QUALITY_WARNED, classify_runtime_quality
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


def test_step3_4_pass_still_requires_solver_backed_evidence() -> None:
    classification = classify_runtime_quality(LOW_RESIDUAL_METRICS, solver_backed=True, residual_only=False)

    assert classification["quality_classification"] == RUNTIME_QUALITY_PASSED
    assert classification["solver_backed"] is True
    assert classification["residual_only"] is False


def test_step3_4_high_residual_remains_warned_not_passed() -> None:
    metrics = dict(LOW_RESIDUAL_METRICS)
    metrics["normalized_task_residual_p95"] = 0.7
    metrics["normalized_task_residual_max"] = 1.0

    classification = classify_runtime_quality(metrics, solver_backed=True, residual_only=False)

    assert classification["quality_classification"] == RUNTIME_QUALITY_WARNED
    assert "normalized_task_residual_p95_above_warn_gate" in classification["failure_or_warning_reasons"]


def test_step3_4_runner_evidence_does_not_promote_residual_only_pass() -> None:
    evidence = runner._smoke_quality_evidence(
        {
            "status": RUNTIME_QUALITY_PASSED,
            "solver_type": "runtime_model_fk_residual_evaluation_only",
            "solver_backed": False,
            "residual_only": True,
            "quality_pass_allowed": False,
            "metrics": LOW_RESIDUAL_METRICS,
        }
    )

    assert evidence["status"] != RUNTIME_QUALITY_PASSED
    assert "residual_only_fk_evaluation" in evidence["warning_reasons"]
