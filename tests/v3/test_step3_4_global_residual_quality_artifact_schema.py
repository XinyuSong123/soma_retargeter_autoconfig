from __future__ import annotations

import inspect

from soma_retargeter.tools import run_v3_full_fleet_runtime_quality as runner


def test_step3_4_runner_writes_required_artifacts() -> None:
    source = inspect.getsource(runner.run_full_fleet_runtime_quality)

    for artifact in (
        "quality_delta_vs_step3_3.json",
        "residual_taxonomy.json",
        "task_coverage_matrix.json",
        "anchor_reliability_matrix.json",
    ):
        assert artifact in source


def test_step3_4_solver_diagnostics_schema_fields_are_declared() -> None:
    source = inspect.getsource(runner._solver_diagnostics_matrix_payload)

    for field in (
        "raw_task_residual_p95",
        "task_anchor_count",
        "task_coverage_ratio",
        "anchor_reliability_score",
        "residual_denominator",
        "residual_denominator_robot_specific",
    ):
        assert field in source


def test_step3_4_artifact_root_is_guarded_against_step3_3_overwrite() -> None:
    source = inspect.getsource(runner.run_full_fleet_runtime_quality)

    assert "DEFAULT_STEP3_3_ARTIFACT_ROOT" in source
    assert "must not overwrite the closed Step 3.3 artifact tree" in source
