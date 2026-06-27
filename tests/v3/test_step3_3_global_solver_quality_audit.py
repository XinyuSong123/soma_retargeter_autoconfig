from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_step3_3_global_solver_quality import (
    EXPECTED_BASE_STEP3_2_FINAL_HEAD,
    run_audit,
)


def test_step3_3_audit_accepts_passing_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = _write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44


def test_step3_3_audit_rejects_missing_delta(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = _write_passing_fixture(tmp_path)
    (artifact_dir / "quality_delta_vs_step3_2.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_required_artifacts"] >= 1
    assert result.gate_counts["quality_delta_vs_step3_2"] >= 1


def test_step3_3_audit_rejects_no_failure_reduction(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = _write_passing_fixture(tmp_path)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_failed_count"] = 9
    _write_json(artifact_dir / "quality_summary.json", summary)
    delta = _read_json(artifact_dir / "quality_delta_vs_step3_2.json")
    delta["current_counts"]["runtime_quality_failed_count"] = 9
    delta["count_deltas"]["runtime_quality_failed_count"] = 0
    _write_json(artifact_dir / "quality_delta_vs_step3_2.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["quality_failure_reduction"] >= 1
    assert result.gate_counts["quality_delta_vs_step3_2"] >= 1


def test_step3_3_audit_rejects_solver_backed_count_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = _write_passing_fixture(tmp_path)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["solver_backed_count"] = 31
    _write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["solver_backed_counts"] >= 1


def test_step3_3_audit_rejects_negative_promotion(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = _write_passing_fixture(tmp_path)
    matrix = _read_json(artifact_dir / "model_matrix.json")
    row = next(item for item in matrix["rows"] if item["category"] == "negative_control")
    row["runtime_quality_status"] = "runtime_quality_passed"
    row["solver_backed"] = True
    row["solver_backed_smoke_attempted"] = True
    row["solver_backed_smoke_completed"] = True
    row["promoted_to_runtime_quality"] = True
    row["quality_evaluated"] = True
    _write_json(artifact_dir / "model_matrix.json", matrix)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["negative_and_partial_not_promoted"] >= 4


def _write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step3_3_global_solver_quality"
    baseline_dir = source_root / "artifacts/retargeting_v3_step3_2_solver_backed_smoke"
    (artifact_dir / "test_results").mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    rows = _model_rows()
    solver_rows = [row for row in _solver_rows(rows)]
    diagnostics = _diagnostic_rows(rows)
    summary = _summary()
    baseline_summary = _baseline_summary()
    delta = _delta(summary, baseline_summary)
    solver_config = _solver_config()

    _write_json(baseline_dir / "quality_summary.json", baseline_summary)
    _write_json(
        artifact_dir / "environment.json",
        {
            "schema_version": 1,
            "git_status_short": "",
            "source_code_commit": "a" * 40,
            "source_code_commit_remote_resolvable": True,
            "source_code_commit_is_artifact_commit_ancestor": True,
            "source_worktree_clean_before_run": True,
            "source_worktree_clean_after_run": True,
            "core_diff_after_source_commit": [],
        },
    )
    (artifact_dir / "commands.txt").write_text("PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_fleet_runtime_quality\n", encoding="utf-8")
    (artifact_dir / "test_results/pytest.txt").write_text("6 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text("<testsuite tests=\"6\" failures=\"0\"></testsuite>\n", encoding="utf-8")
    _write_json(artifact_dir / "test_results/pytest_summary.json", {"passed": 6, "failed": 0})
    _write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "in_scope_total": 44, "rows": rows})
    _write_json(artifact_dir / "solver_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": solver_rows})
    _write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": solver_rows})
    _write_json(artifact_dir / "solver_diagnostics_matrix.json", {"schema_version": 1, "row_count": 32, "solver_config_hash": "hash", "rows": diagnostics})
    _write_json(artifact_dir / "solver_config.json", solver_config)
    _write_json(artifact_dir / "quality_summary.json", summary)
    _write_json(artifact_dir / "quality_delta_vs_step3_2.json", delta)
    _write_json(artifact_dir / "pipeline_backed_matrix.json", {"schema_version": 1, "rows": [], "controls": {"rpo_present": True, "g1_present": True}})
    _write_json(artifact_dir / "acceptance_ledger.json", {"schema_version": 1, "verdict": "PASS", "base_step3_2_final_head": EXPECTED_BASE_STEP3_2_FINAL_HEAD, "solver_config_hash": "hash", "source_code_commit": "a" * 40})
    _write_json(artifact_dir / "deterministic_rerun.json", {"schema_version": 1, "status": "passed", "deterministic": True, "compared_count": 44, "matched_count": 44, "deterministic_compared_count": 44, "deterministic_matched_count": 44})
    return artifact_dir, baseline_dir, source_root


def _model_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(32):
        status = "runtime_quality_failed" if index < 8 else "runtime_quality_warned"
        rows.append({**_numeric_row(f"full_{index:02d}"), "category": "full_humanoid_profile", "source_status": "passed", "runtime_quality_status": status, "quality_classification": status, "solver_backed": True, "solver_backed_smoke_attempted": True, "solver_backed_smoke_completed": True, "residual_only": False})
    for index in range(3):
        rows.append({**_numeric_row(f"partial_{index:02d}"), "category": "partial_humanoid_profile", "source_status": "partial_passed", "runtime_quality_status": "partial_runtime_passed", "solver_backed": False, "solver_backed_smoke_attempted": False, "solver_backed_smoke_completed": False, "residual_only": False})
    for index in range(9):
        rows.append({**_numeric_row(f"negative_{index:02d}"), "category": "negative_control", "source_status": "negative_control_passed", "expected_capability": "negative_control", "runtime_quality_status": "negative_control_runtime_passed", "quality_classification": "negative_control_not_promoted", "solver_backed": False, "solver_backed_smoke_attempted": False, "solver_backed_smoke_completed": False, "residual_only": False, "promoted_to_runtime_quality": False, "quality_evaluated": False, "override_allowed": False, "humanoid_profile_generated": False})
    return rows


def _numeric_row(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "frame_count": 120,
        "solver_mode": "generic_chain_projection_least_squares_smoke",
        "sampled_frame_indices": [60],
        "normalized_task_residual_mean": 0.7,
        "normalized_task_residual_p95": 0.8,
        "normalized_task_residual_max": 1.0,
        "target_translation_error_mean": 0.1,
        "target_translation_error_p95": 0.2,
        "target_translation_error_max": 0.3,
        "target_rotation_error_mean": 0.1,
        "target_rotation_error_p95": 0.2,
        "target_rotation_error_max": 0.3,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "output_nan_count": 0,
        "output_inf_count": 0,
        "runtime_seconds": 0.1,
        "failure_or_warning_reasons": ["high_task_residual"],
        "runtime_quality_warning_reasons": ["high_task_residual"],
        "deterministic_hash_inputs": {"model_id": model_id},
    }


def _solver_rows(rows: list[dict]) -> list[dict]:
    return [
        {**row, "metrics": {"solver_iteration_mean": 4.0, "solver_iteration_p95": 5.0, "solver_iteration_max": 6.0, "solver_converged_frame_count": 1, "solver_failed_frame_count": 0, "line_search_count": 1, "rollback_count": 0, "pre_projection_joint_limit_violation_count": 1, "pre_projection_max_joint_limit_violation": 0.2, "post_projection_joint_limit_violation_count": 0, "post_projection_max_joint_limit_violation": 0.0, "projection_changed_coordinate_count": 1, "projection_repaired_frame_count": 1, "projection_delta_linf": 0.2, "projection_delta_l2": 0.2, "projection_delta_p95": 0.1, "projection_residual_worsened_count": 0, "nan_count": 0, "inf_count": 0, "runtime_seconds": 0.1}, "smoke_summary": {"task_diagnostics": []}}
        for row in rows
        if row["category"] == "full_humanoid_profile"
    ]


def _diagnostic_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["category"] != "full_humanoid_profile":
            continue
        out.append({**row, "solver_config_hash": "hash", "solver_iteration_count_mean": 4.0, "solver_iteration_count_p95": 5.0, "solver_iteration_count_max": 6.0, "solver_converged_frame_count": 1, "solver_failed_frame_count": 0, "line_search_count": 1, "rollback_count": 0, "pre_projection_joint_limit_violation_count": 1, "pre_projection_max_joint_limit_violation": 0.2, "post_projection_joint_limit_violation_count": 0, "post_projection_max_joint_limit_violation": 0.0, "projection_changed_coordinate_count": 1, "projection_repaired_frame_count": 1, "projection_delta_linf": 0.2, "projection_delta_l2": 0.2, "projection_delta_p95": 0.1, "projection_residual_worsened_count": 0})
    return out


def _summary() -> dict:
    return {**_baseline_summary(), "base_step3_2_final_head": EXPECTED_BASE_STEP3_2_FINAL_HEAD, "runtime_quality_failed_count": 8, "runtime_quality_warned_count": 24, "partial_runtime_passed_count": 3, "negative_control_runtime_passed_count": 9, "source_code_commit": "a" * 40}


def _baseline_summary() -> dict:
    return {"schema_version": 1, "row_count": 44, "in_scope_total": 44, "matrix_row_count": 44, "full_humanoid_total": 32, "partial_total": 3, "negative_total": 9, "status_counts": {"passed": 32, "partial_passed": 3, "negative_control_passed": 9}, "solver_backed_smoke_attempted_count": 32, "solver_backed_completed_count": 32, "solver_backed_count": 32, "residual_only_count": 0, "runtime_quality_passed_count": 0, "runtime_quality_warned_count": 23, "runtime_quality_failed_count": 9, "partial_runtime_passed_count": 3, "negative_control_runtime_passed_count": 9, "high_residual_warning_count": 32, "joint_limit_warning_count": 12, "joint_limit_smoke_warning_count": 10, "deterministic_compared_count": 44, "deterministic_matched_count": 44}


def _delta(summary: dict, baseline: dict) -> dict:
    keys = ("in_scope_total", "full_humanoid_total", "partial_total", "negative_total", "solver_backed_count", "residual_only_count", "runtime_quality_failed_count")
    baseline_counts = {key: baseline[key] for key in keys}
    current_counts = {key: summary[key] for key in keys}
    return {"schema_version": 1, "base_step3_2_final_head": EXPECTED_BASE_STEP3_2_FINAL_HEAD, "baseline_final_head": EXPECTED_BASE_STEP3_2_FINAL_HEAD, "current_source_commit": "a" * 40, "baseline_counts": baseline_counts, "current_counts": current_counts, "count_deltas": {key: current_counts[key] - baseline_counts[key] for key in keys}, "metric_distribution_deltas": {}, "regressions": [], "improvements": ["runtime_quality_failed_count_reduced"], "verdict": "PASS"}


def _solver_config() -> dict:
    return {"schema_version": 1, "base_step3_2_final_head": EXPECTED_BASE_STEP3_2_FINAL_HEAD, "global_config": True, "robot_specific_tuning": False, "solver_config_hash": "hash", "config": {"enable_global_quality_hardening": True, "project_joint_limits": True}}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
