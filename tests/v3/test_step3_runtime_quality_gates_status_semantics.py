from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from soma_retargeter.runtime.v3.generic_smoke import run_full_humanoid_fk_smoke
from soma_retargeter.runtime.v3.runtime_quality_gates import (
    RESIDUAL_ONLY_SOLVER_TYPE,
    RUNTIME_EVALUATION_COMPLETED,
    RUNTIME_QUALITY_FAILED,
    RUNTIME_QUALITY_PASSED,
    RUNTIME_QUALITY_WARNED,
    classify_runtime_quality,
)


def _metrics(**overrides):
    metrics = {
        "nan_count": 0,
        "inf_count": 0,
        "output_finite": True,
        "normalized_task_residual_p95": 0.01,
        "normalized_task_residual_max": 0.02,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "joint_velocity_p95": 0.0,
        "joint_acceleration_p95": 0.0,
        "target_se3_orthogonality_error_max": 0.0,
        "foot_height_below_ground_count": 0,
        "solver_success_fraction": 1.0,
    }
    metrics.update(overrides)
    return metrics


def test_residual_only_smoke_cannot_be_runtime_quality_passed() -> None:
    result = classify_runtime_quality(
        _metrics(normalized_task_residual_p95=0.90, normalized_task_residual_max=1.0),
        solver_type=RESIDUAL_ONLY_SOLVER_TYPE,
        solver_backed=False,
    )

    assert result["classification"] == RUNTIME_QUALITY_WARNED
    assert result["quality_pass_allowed"] is False
    assert result["residual_only"] is True
    assert "normalized_task_residual_p95_above_warn_gate" in result["failure_or_warning_reasons"]


def test_low_residual_residual_only_is_evaluation_completed_not_passed() -> None:
    result = classify_runtime_quality(
        _metrics(),
        solver_type=RESIDUAL_ONLY_SOLVER_TYPE,
        solver_backed=False,
    )

    assert result["classification"] == RUNTIME_EVALUATION_COMPLETED
    assert result["classification"] != RUNTIME_QUALITY_PASSED
    assert result["quality_pass_allowed"] is False
    assert "runtime_quality_pass_requires_solver_backed_smoke" in result["classification_reason"]


def test_solver_backed_smoke_can_pass_only_when_all_gates_pass() -> None:
    result = classify_runtime_quality(
        _metrics(),
        solver_type="solver_backed_projection_smoke",
        solver_backed=True,
    )

    assert result["classification"] == RUNTIME_QUALITY_PASSED
    assert result["quality_pass_allowed"] is True


def test_severe_joint_limit_violation_is_runtime_quality_failed() -> None:
    result = classify_runtime_quality(
        _metrics(joint_limit_violation_count=1, max_joint_limit_violation=1e-3),
        solver_type="solver_backed_projection_smoke",
        solver_backed=True,
    )

    assert result["classification"] == RUNTIME_QUALITY_FAILED
    assert result["quality_pass_allowed"] is True
    assert "joint_limit_violation_above_global_tolerance" in result["failure_or_warning_reasons"]


def test_generic_fk_smoke_exposes_residual_only_classification_metrics() -> None:
    profile = {
        "semantic_sites": {
            "Hips": {
                "body_name": "pelvis",
                "local_position": [0.0, 0.0, 0.0],
                "local_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        }
    }
    case = SimpleNamespace(runtime_source_path="unused.xml", model_format="mjcf", profile=profile)
    target = np.repeat(np.eye(4, dtype=float)[None, :, :], 4, axis=0)
    target[:, 0, 3] = np.linspace(1.0, 4.0, 4)

    result = run_full_humanoid_fk_smoke(case, {"Hips": target}, adapter=_FakeAdapter(), profile=profile)
    payload = result.to_json()

    assert result.status == RUNTIME_QUALITY_WARNED
    assert payload["solver_type"] == RESIDUAL_ONLY_SOLVER_TYPE
    assert payload["solver_backed"] is False
    assert payload["residual_only"] is True
    assert payload["quality_pass_allowed"] is False
    assert payload["quality_classification"] == RUNTIME_QUALITY_WARNED
    assert payload["metrics"]["solver_type"] == RESIDUAL_ONLY_SOLVER_TYPE
    assert payload["metrics"]["quality_pass_allowed"] is False
    assert "classification_reason" in payload["metrics"]


class _FakeAdapter:
    nq = 1
    coordinate_info = []

    def neutral_q(self):
        return np.zeros(1, dtype=float)

    def forward_kinematics(self, q):
        return {"q": q}

    def site_transform(self, state, site):
        return np.eye(4, dtype=float)

    def close(self):
        raise AssertionError("provided adapter should not be owned by the smoke harness")
