from __future__ import annotations

import inspect

from soma_retargeter.tools import run_v3_full_fleet_runtime_quality as full_fleet_runner


def test_step3_2_runner_writes_solver_smoke_matrix_artifact_not_generic_only() -> None:
    source = inspect.getsource(full_fleet_runner.run_full_fleet_runtime_quality)

    assert "solver_smoke_matrix.json" in source


def test_step3_2_quality_summary_declares_solver_backed_smoke_counts() -> None:
    source = inspect.getsource(full_fleet_runner._quality_summary_payload)
    expected_count_fields = {
        "step3_2_solver_backed_smoke_attempted_count",
        "step3_2_solver_backed_smoke_completed_count",
        "step3_2_solver_backed_smoke_passed_count",
        "step3_2_solver_backed_smoke_failed_count",
        "step3_2_solver_backed_smoke_finite_metrics_count",
        "step3_2_solver_backed_runtime_quality_passed_count",
        "step3_2_residual_only_runtime_quality_passed_count",
    }

    for field in expected_count_fields:
        assert field in source


def test_step3_2_solver_smoke_rows_require_status_booleans_and_finite_metric_contract() -> None:
    source = inspect.getsource(full_fleet_runner)
    expected_row_fields = {
        "solver_backed_smoke_attempted",
        "solver_backed_smoke_completed",
        "solver_backed_smoke_metrics_finite",
        "solver_backed",
        "residual_only",
        "quality_pass_allowed",
        "runtime_quality_status",
        "failure_or_warning_reasons",
    }

    for field in expected_row_fields:
        assert field in source
