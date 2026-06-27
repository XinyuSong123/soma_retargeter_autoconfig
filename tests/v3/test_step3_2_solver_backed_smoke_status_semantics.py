from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from soma_retargeter.runtime.v3.runtime_quality_gates import (
    RESIDUAL_ONLY_SOLVER_TYPE,
    RUNTIME_QUALITY_FAILED,
    RUNTIME_QUALITY_PASSED,
    SOLVER_BACKED_REQUIRED_FOR_PASS_REASON,
    classify_runtime_quality,
)
from soma_retargeter.tools import run_v3_full_fleet_runtime_quality as full_fleet_runner


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


def test_residual_only_full_humanoid_smoke_cannot_be_runtime_quality_passed_even_with_low_residuals() -> None:
    classification = classify_runtime_quality(
        LOW_RESIDUAL_METRICS,
        solver_type=RESIDUAL_ONLY_SOLVER_TYPE,
        solver_backed=False,
        residual_only=True,
    )

    assert classification["quality_classification"] != RUNTIME_QUALITY_PASSED
    assert classification["quality_pass_allowed"] is False
    assert classification["solver_backed"] is False
    assert classification["residual_only"] is True
    assert SOLVER_BACKED_REQUIRED_FOR_PASS_REASON in classification["failure_or_warning_reasons"]


def test_solver_backed_runtime_quality_pass_requires_attempted_completed_booleans() -> None:
    missing_attempt_evidence = _solver_smoke_summary()
    missing_attempt = full_fleet_runner._smoke_quality_evidence(missing_attempt_evidence)

    assert missing_attempt["status"] != RUNTIME_QUALITY_PASSED
    assert "solver_backed_smoke_attempted" in missing_attempt["warning_reasons"]
    assert "solver_backed_smoke_completed" in missing_attempt["warning_reasons"]


def test_solver_backed_runtime_quality_pass_requires_finite_metrics() -> None:
    nonfinite_evidence = _solver_smoke_summary(solver_attempted=True, solver_completed=True)
    nonfinite_evidence["metrics"]["normalized_task_residual_p95"] = float("nan")
    nonfinite = full_fleet_runner._smoke_quality_evidence(nonfinite_evidence)

    assert nonfinite["status"] == RUNTIME_QUALITY_FAILED
    assert any("finite" in reason for reason in nonfinite["warning_reasons"])


def test_solver_backed_runtime_quality_can_pass_with_completed_finite_generic_solver_evidence() -> None:
    complete = full_fleet_runner._smoke_quality_evidence(
        _solver_smoke_summary(solver_attempted=True, solver_completed=True)
    )
    assert complete["status"] == RUNTIME_QUALITY_PASSED


def test_step3_2_runtime_quality_gates_do_not_use_robot_specific_threshold_or_whitelist_tables() -> None:
    suspicious_tables: list[tuple[str, str]] = []
    for path in _relevant_runtime_tool_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [_target_name(target) for target in targets]
            table_name = next((name for name in names if name), "")
            if not _suspicious_gate_table_name(table_name):
                continue
            value = node.value
            if isinstance(value, ast.Dict) and _has_model_id_literal_key(value):
                suspicious_tables.append((str(path), table_name))

    assert suspicious_tables == []


def _solver_smoke_summary(
    *,
    solver_attempted: bool | None = None,
    solver_completed: bool | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "passed",
        "mode": "runtime_solver_smoke",
        "solver_type": "generic_solver_backed_runtime_projection",
        "solver_backed": True,
        "residual_only": False,
        "quality_pass_allowed": True,
        "metrics": dict(LOW_RESIDUAL_METRICS),
        "residuals": {"solver": "generic_solver_backed_runtime_projection"},
    }
    if solver_attempted is not None:
        payload["solver_backed_smoke_attempted"] = solver_attempted
    if solver_completed is not None:
        payload["solver_backed_smoke_completed"] = solver_completed
    return payload


def _relevant_runtime_tool_files() -> list[Path]:
    return [
        Path("soma_retargeter/runtime/v3/generic_smoke.py"),
        Path("soma_retargeter/runtime/v3/runtime_quality_gates.py"),
        Path("soma_retargeter/runtime/v3/fleet_harness.py"),
        Path("soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py"),
    ]


def _target_name(target: ast.expr) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _suspicious_gate_table_name(name: str) -> bool:
    lowered = name.lower()
    has_gate_term = any(term in lowered for term in ("threshold", "whitelist", "allowlist", "gate"))
    has_model_term = any(term in lowered for term in ("by_model", "by_robot", "model_id", "robot_type", "per_model"))
    return has_gate_term and has_model_term


def _has_model_id_literal_key(table: ast.Dict) -> bool:
    for key in table.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            value = key.value.lower()
            if any(token in value for token in ("unitree", "roboparty", "atlas", "jaxon", "h1", "g1")):
                return True
    return False
