"""Global Step 3.1.1 runtime quality gates.

These gates classify runtime quality evidence only. They intentionally do not
modify Step 2 capability thresholds or production retargeting configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


RUNTIME_QUALITY_PASSED = "runtime_quality_passed"
RUNTIME_QUALITY_WARNED = "runtime_quality_warned"
RUNTIME_QUALITY_FAILED = "runtime_quality_failed"
RUNTIME_EVALUATION_COMPLETED = "runtime_evaluation_completed"
PARTIAL_RUNTIME_PASSED = "partial_runtime_passed"
NEGATIVE_CONTROL_RUNTIME_PASSED = "negative_control_runtime_passed"
BLOCKED_SOURCE_OR_PROFILE = "blocked_source_or_profile"

RESIDUAL_ONLY_SOLVER_TYPE = "runtime_model_fk_residual_evaluation_only"
SOLVER_BACKED_REQUIRED_FOR_PASS_REASON = "runtime_quality_pass_requires_solver_backed_smoke"

FINAL_RUNTIME_STATUSES = {
    RUNTIME_QUALITY_PASSED,
    RUNTIME_QUALITY_WARNED,
    RUNTIME_QUALITY_FAILED,
    RUNTIME_EVALUATION_COMPLETED,
    PARTIAL_RUNTIME_PASSED,
    NEGATIVE_CONTROL_RUNTIME_PASSED,
    BLOCKED_SOURCE_OR_PROFILE,
}


@dataclass(frozen=True)
class RuntimeQualityGates:
    nan_count: int = 0
    inf_count: int = 0
    joint_limit_violation_severe_threshold: float = 1e-5
    normalized_task_residual_p95_pass: float = 0.15
    normalized_task_residual_p95_warn: float = 0.60
    normalized_task_residual_max_warn: float = 1.20
    joint_velocity_p95_warn: float = 0.25
    joint_acceleration_p95_warn: float = 0.75
    se3_orthogonality_error_max: float = 1e-8

    def to_json(self) -> dict[str, Any]:
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "joint_limit_violation_severe_threshold": self.joint_limit_violation_severe_threshold,
            "normalized_task_residual_p95_pass": self.normalized_task_residual_p95_pass,
            "normalized_task_residual_p95_warn": self.normalized_task_residual_p95_warn,
            "normalized_task_residual_max_warn": self.normalized_task_residual_max_warn,
            "joint_velocity_p95_warn": self.joint_velocity_p95_warn,
            "joint_acceleration_p95_warn": self.joint_acceleration_p95_warn,
            "se3_orthogonality_error_max": self.se3_orthogonality_error_max,
        }


GLOBAL_RUNTIME_QUALITY_GATES = RuntimeQualityGates()


def classify_runtime_quality(
    metrics: Mapping[str, Any],
    *,
    solver_backed: bool,
    residual_only: bool | None = None,
    solver_type: str | None = None,
    gates: RuntimeQualityGates = GLOBAL_RUNTIME_QUALITY_GATES,
) -> dict[str, Any]:
    """Classify one full-humanoid smoke result using global gates."""

    solver_type = solver_type or ("solver_backed_runtime_smoke" if solver_backed else RESIDUAL_ONLY_SOLVER_TYPE)
    if residual_only is None:
        residual_only = solver_type == RESIDUAL_ONLY_SOLVER_TYPE or not solver_backed

    gate_results: dict[str, bool] = {}
    reasons: list[str] = []

    nan_count = _as_int(metrics.get("nan_count", metrics.get("output_nan_count", 0)))
    inf_count = _as_int(metrics.get("inf_count", metrics.get("output_inf_count", 0)))
    residual_p95 = _as_float(metrics.get("normalized_task_residual_p95", 0.0))
    residual_max = _as_float(metrics.get("normalized_task_residual_max", 0.0))
    joint_limit_count = _as_int(metrics.get("joint_limit_violation_count", 0))
    max_joint_limit = _as_float(metrics.get("max_joint_limit_violation", 0.0))
    velocity_p95 = _as_float(metrics.get("joint_velocity_p95", 0.0))
    acceleration_p95 = _as_float(metrics.get("joint_acceleration_p95", 0.0))
    se3_orthogonality_error = _as_float(metrics.get("target_se3_orthogonality_error_max", 0.0))
    solver_success = _as_float(metrics.get("solver_success_fraction", 0.0))

    finite = all(
        value is not None and math.isfinite(float(value))
        for value in (
            nan_count,
            inf_count,
            residual_p95,
            residual_max,
            joint_limit_count,
            max_joint_limit,
            velocity_p95,
            acceleration_p95,
            se3_orthogonality_error,
            solver_success,
        )
    )
    gate_results["finite_metrics"] = finite
    if not finite:
        reasons.append("nonfinite_or_missing_runtime_metric")

    gate_results["nan_count"] = nan_count == gates.nan_count
    if not gate_results["nan_count"]:
        reasons.append("nan_count_nonzero")
    gate_results["inf_count"] = inf_count == gates.inf_count
    if not gate_results["inf_count"]:
        reasons.append("inf_count_nonzero")

    gate_results["solver_success_fraction"] = _ge_or_false(solver_success, 1.0) if solver_backed else not residual_only
    if solver_backed and not gate_results["solver_success_fraction"]:
        reasons.append("solver_success_fraction_below_one")

    gate_results["joint_limit_pass"] = joint_limit_count == 0 or _le_or_false(
        max_joint_limit, gates.joint_limit_violation_severe_threshold
    )
    if not gate_results["joint_limit_pass"]:
        reasons.append("joint_limit_violation_above_global_tolerance")

    gate_results["normalized_task_residual_p95_pass"] = _le_or_false(residual_p95, gates.normalized_task_residual_p95_pass)
    gate_results["normalized_task_residual_p95_warn"] = _le_or_false(residual_p95, gates.normalized_task_residual_p95_warn)
    gate_results["normalized_task_residual_max_warn"] = _le_or_false(residual_max, gates.normalized_task_residual_max_warn)
    if not gate_results["normalized_task_residual_p95_pass"]:
        reasons.append("normalized_task_residual_p95_above_pass_gate")
    if not gate_results["normalized_task_residual_p95_warn"]:
        reasons.append("normalized_task_residual_p95_above_warn_gate")
    if not gate_results["normalized_task_residual_max_warn"]:
        reasons.append("normalized_task_residual_max_above_warn_gate")

    gate_results["joint_velocity_p95_warn"] = _le_or_false(velocity_p95, gates.joint_velocity_p95_warn)
    if not gate_results["joint_velocity_p95_warn"]:
        reasons.append("joint_velocity_p95_above_warn_gate")
    gate_results["joint_acceleration_p95_warn"] = _le_or_false(acceleration_p95, gates.joint_acceleration_p95_warn)
    if not gate_results["joint_acceleration_p95_warn"]:
        reasons.append("joint_acceleration_p95_above_warn_gate")

    gate_results["se3_orthogonality_error_max"] = _le_or_false(
        se3_orthogonality_error, gates.se3_orthogonality_error_max
    )
    if not gate_results["se3_orthogonality_error_max"]:
        reasons.append("se3_orthogonality_error_above_global_gate")

    metric_reasons = list(reasons)
    quality_pass_allowed = bool(solver_backed and not residual_only)
    if not quality_pass_allowed:
        reasons.append(SOLVER_BACKED_REQUIRED_FOR_PASS_REASON)

    hard_failure = (
        not finite
        or not gate_results["nan_count"]
        or not gate_results["inf_count"]
        or not gate_results["joint_limit_pass"]
        or not gate_results["normalized_task_residual_max_warn"]
        or not gate_results["se3_orthogonality_error_max"]
    )
    pass_gates = (
        quality_pass_allowed
        and gate_results["finite_metrics"]
        and gate_results["nan_count"]
        and gate_results["inf_count"]
        and gate_results["solver_success_fraction"]
        and gate_results["joint_limit_pass"]
        and gate_results["normalized_task_residual_p95_pass"]
    )

    if pass_gates:
        status = RUNTIME_QUALITY_PASSED
    elif hard_failure:
        status = RUNTIME_QUALITY_FAILED
    elif residual_only:
        status = RUNTIME_QUALITY_WARNED if metric_reasons else RUNTIME_EVALUATION_COMPLETED
    else:
        status = RUNTIME_QUALITY_WARNED

    return {
        "classification": status,
        "quality_classification": status,
        "solver_type": solver_type,
        "quality_pass_allowed": quality_pass_allowed,
        "solver_backed": bool(solver_backed),
        "residual_only": bool(residual_only),
        "quality_gate_results": gate_results,
        "failure_or_warning_reasons": _dedupe(reasons),
        "classification_reason": ";".join(_dedupe(reasons)) if reasons else "all_global_runtime_quality_gates_satisfied",
        "gates": gates.to_json(),
    }


def combine_full_humanoid_classifications(classifications: list[str]) -> str:
    if not classifications:
        return RUNTIME_QUALITY_FAILED
    if any(value == RUNTIME_QUALITY_FAILED for value in classifications):
        return RUNTIME_QUALITY_FAILED
    if any(value == RUNTIME_QUALITY_WARNED for value in classifications):
        return RUNTIME_QUALITY_WARNED
    if all(value == RUNTIME_QUALITY_PASSED for value in classifications):
        return RUNTIME_QUALITY_PASSED
    return RUNTIME_EVALUATION_COMPLETED


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _le_or_false(value: float | int | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _ge_or_false(value: float | int | None, threshold: float) -> bool:
    return value is not None and value >= threshold
