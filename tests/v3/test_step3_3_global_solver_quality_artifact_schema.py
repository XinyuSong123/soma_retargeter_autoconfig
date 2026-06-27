from __future__ import annotations

import inspect

from soma_retargeter.tools import run_v3_full_fleet_runtime_quality as runner


def test_step3_3_runner_writes_required_artifacts() -> None:
    source = inspect.getsource(runner.run_full_fleet_runtime_quality)

    for artifact in (
        "solver_config.json",
        "solver_diagnostics_matrix.json",
        "quality_delta_vs_step3_2.json",
        "solver_smoke_matrix.json",
    ):
        assert artifact in source


def test_step3_3_solver_diagnostics_schema_fields_are_declared() -> None:
    source = inspect.getsource(runner._solver_diagnostics_matrix_payload)

    for field in (
        "solver_config_hash",
        "solver_iteration_count_mean",
        "solver_iteration_count_p95",
        "solver_iteration_count_max",
        "solver_converged_frame_count",
        "solver_failed_frame_count",
        "line_search_count",
        "rollback_count",
        "pre_projection_joint_limit_violation_count",
        "post_projection_joint_limit_violation_count",
        "output_nan_count",
        "output_inf_count",
    ):
        assert field in source


def test_step3_3_artifact_root_is_guarded_against_baseline_overwrite() -> None:
    source = inspect.getsource(runner.run_full_fleet_runtime_quality)

    assert "baseline_artifact_dir" in source
    assert "must not overwrite the Step 3.2 baseline artifact tree" in source
