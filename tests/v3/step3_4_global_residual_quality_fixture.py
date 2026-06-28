from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_step3_4_global_residual_quality import EXPECTED_BASE_STEP3_3_FINAL_HEAD


def write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step3_4_global_residual_quality"
    baseline_dir = source_root / "artifacts/retargeting_v3_step3_3_global_solver_quality"
    (artifact_dir / "test_results").mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    rows = model_rows()
    solver_rows = solver_smoke_rows(rows)
    diagnostics = diagnostic_rows(rows)
    summary = current_summary()
    baseline = baseline_summary()
    delta = quality_delta(summary, baseline)
    solver_config = step3_4_solver_config()
    coverage = task_coverage(rows)
    reliability = anchor_reliability(rows)
    taxonomy = residual_taxonomy(rows)

    write_json(baseline_dir / "quality_summary.json", baseline)
    write_json(
        artifact_dir / "environment.json",
        {
            "schema_version": 1,
            "git_status_short": "",
            "source_git_status_short_before_run": "",
            "source_code_commit": "a" * 40,
            "source_code_commit_remote_resolvable": True,
            "source_code_commit_is_artifact_commit_ancestor": True,
            "source_worktree_clean_before_run": True,
            "source_worktree_clean_after_run": True,
            "core_diff_after_source_commit": [],
        },
    )
    (artifact_dir / "commands.txt").write_text(
        "PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_fleet_runtime_quality --enable-global-residual-quality-hardening\n",
        encoding="utf-8",
    )
    (artifact_dir / "test_results/pytest.txt").write_text("8 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text("<testsuite tests=\"8\" failures=\"0\"></testsuite>\n", encoding="utf-8")
    write_json(artifact_dir / "test_results/pytest_summary.json", {"schema_version": 1, "passed": 8, "failed": 0})
    write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "in_scope_total": 44, "rows": rows})
    write_json(artifact_dir / "solver_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": solver_rows})
    write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": solver_rows})
    write_json(artifact_dir / "solver_diagnostics_matrix.json", {"schema_version": 1, "row_count": 32, "solver_config_hash": "hash", "rows": diagnostics})
    write_json(artifact_dir / "solver_config.json", solver_config)
    write_json(artifact_dir / "quality_summary.json", summary)
    write_json(artifact_dir / "quality_delta_vs_step3_3.json", delta)
    write_json(artifact_dir / "residual_taxonomy.json", taxonomy)
    write_json(artifact_dir / "task_coverage_matrix.json", coverage)
    write_json(artifact_dir / "anchor_reliability_matrix.json", reliability)
    write_json(artifact_dir / "pipeline_backed_matrix.json", {"schema_version": 1, "rows": [], "controls": {"rpo_present": True, "g1_present": True}})
    write_json(
        artifact_dir / "acceptance_ledger.json",
        {
            "schema_version": 1,
            "verdict": "PASS",
            "base_step3_3_final_head": EXPECTED_BASE_STEP3_3_FINAL_HEAD,
            "solver_config_hash": "hash",
            "source_code_commit": "a" * 40,
            "quality_delta_vs_step3_3": delta,
        },
    )
    write_json(
        artifact_dir / "deterministic_rerun.json",
        {
            "schema_version": 1,
            "status": "passed",
            "deterministic": True,
            "compared_count": 44,
            "matched_count": 44,
            "deterministic_compared_count": 44,
            "deterministic_matched_count": 44,
        },
    )
    return artifact_dir, baseline_dir, source_root


def model_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(32):
        rows.append(
            {
                **numeric_row(f"full_{index:02d}"),
                "category": "full_humanoid_profile",
                "source_status": "passed",
                "runtime_quality_status": "runtime_quality_warned",
                "quality_classification": "runtime_quality_warned",
                "solver_backed": True,
                "solver_backed_smoke_attempted": True,
                "solver_backed_smoke_completed": True,
                "residual_only": False,
                "runtime_quality_warning_reasons": [
                    "high_task_residual",
                    "normalized_task_residual_p95_above_pass_gate",
                    "normalized_task_residual_p95_above_warn_gate",
                ],
                "failure_or_warning_reasons": [
                    "high_task_residual",
                    "normalized_task_residual_p95_above_pass_gate",
                    "normalized_task_residual_p95_above_warn_gate",
                ],
            }
        )
    for index in range(3):
        rows.append(
            {
                **numeric_row(f"partial_{index:02d}"),
                "category": "partial_humanoid_profile",
                "source_status": "partial_passed",
                "runtime_quality_status": "partial_runtime_passed",
                "solver_backed": False,
                "solver_backed_smoke_attempted": False,
                "solver_backed_smoke_completed": False,
                "residual_only": False,
            }
        )
    for index in range(9):
        rows.append(
            {
                **numeric_row(f"negative_{index:02d}"),
                "category": "negative_control",
                "source_status": "negative_control_passed",
                "expected_capability": "negative_control",
                "runtime_quality_status": "negative_control_runtime_passed",
                "quality_classification": "negative_control_not_promoted",
                "solver_backed": False,
                "solver_backed_smoke_attempted": False,
                "solver_backed_smoke_completed": False,
                "residual_only": False,
                "promoted_to_runtime_quality": False,
                "quality_evaluated": False,
                "override_allowed": False,
                "humanoid_profile_generated": False,
            }
        )
    return rows


def numeric_row(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "frame_count": 120,
        "solver_mode": "generic_chain_projection_least_squares_smoke",
        "sampled_frame_indices": [60],
        "normalized_task_residual_mean": 0.5,
        "normalized_task_residual_p50": 0.55,
        "normalized_task_residual_p95": 0.7,
        "normalized_task_residual_max": 1.0,
        "task_residual_mean": 1.5,
        "task_residual_p50": 1.65,
        "task_residual_p95": 2.1,
        "task_residual_max": 3.0,
        "raw_task_residual_mean": 1.5,
        "raw_task_residual_p50": 1.65,
        "raw_task_residual_p95": 2.1,
        "raw_task_residual_max": 3.0,
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
        "task_anchor_count": 6,
        "task_anchor_semantic_counts": {"Chest": 3, "Hips": 3},
        "task_coverage_ratio": 1.0,
        "successful_task_coverage_ratio": 1.0,
        "anchor_reliability_score": 1.0,
        "anchor_rejection_reasons": [],
        "residual_normalization_version": "legacy_row_max_v1",
        "residual_normalization_formula": "residual / max(1.0, row_raw_residual_max)",
        "residual_denominator": 3.0,
        "residual_denominator_source": "current_row_raw_residual_max_with_global_floor_1",
        "residual_denominator_scope": "row_local_legacy_metric",
        "residual_denominator_units": "translation_meters_plus_rotation_radians",
        "residual_denominator_robot_specific": False,
        "residual_denominator_from_current_row_max": True,
        "runtime_seconds": 0.1,
        "deterministic_hash_inputs": {"model_id": model_id},
    }


def solver_smoke_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "metrics": {**numeric_row(row["model_id"]), "solver_success_fraction": 1.0, "solver_task_count": 5, "nan_count": 0, "inf_count": 0},
            "smoke_summary": {
                "metrics": {**numeric_row(row["model_id"]), "solver_success_fraction": 1.0, "solver_task_count": 5, "nan_count": 0, "inf_count": 0},
                "residuals": {
                    "task_coverage": {
                        "schema_version": 1,
                        "global_task_universe": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"],
                        "configured_task_order": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"],
                        "rows": [],
                        "summary": {"task_anchor_count": 6, "task_coverage_ratio": 1.0, "successful_task_coverage_ratio": 1.0, "available_task_count": 5, "attempted_task_count": 5, "successful_task_count": 5},
                    },
                    "anchor_reliability": {
                        "schema_version": 1,
                        "rows": [{"semantic": "Hips", "accepted": True, "rejection_reasons": []}],
                        "summary": {"anchor_reliability_score": 1.0, "accepted_anchor_count": 6, "rejected_anchor_count": 0, "rejection_reasons": []},
                    },
                },
                "task_diagnostics": [{"per_semantic": {"LeftHand": {"combined_residual": 2.0, "translation_residual": 0.5, "rotation_residual": 1.5}}}],
            },
        }
        for row in rows
        if row["category"] == "full_humanoid_profile"
    ]


def diagnostic_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "solver_config_hash": "hash",
            "solver_iteration_count_mean": 4.0,
            "solver_iteration_count_p95": 5.0,
            "solver_iteration_count_max": 6.0,
            "solver_converged_frame_count": 1,
            "solver_failed_frame_count": 0,
            "line_search_count": 1,
            "rollback_count": 0,
            "pre_projection_joint_limit_violation_count": 0,
            "pre_projection_max_joint_limit_violation": 0.0,
            "post_projection_joint_limit_violation_count": 0,
            "post_projection_max_joint_limit_violation": 0.0,
            "projection_changed_coordinate_count": 0,
            "projection_repaired_frame_count": 0,
            "projection_delta_linf": 0.0,
            "projection_delta_l2": 0.0,
            "projection_delta_p95": 0.0,
            "projection_residual_worsened_count": 0,
            "task_diagnostics": [{"per_semantic": {"LeftHand": {"combined_residual": 2.0, "translation_residual": 0.5, "rotation_residual": 1.5}}}],
        }
        for row in rows
        if row["category"] == "full_humanoid_profile"
    ]


def baseline_summary() -> dict:
    return {
        "schema_version": 1,
        "row_count": 44,
        "in_scope_total": 44,
        "matrix_row_count": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "status_counts": {"passed": 32, "partial_passed": 3, "negative_control_passed": 9},
        "solver_backed_smoke_attempted_count": 32,
        "solver_backed_completed_count": 32,
        "solver_backed_count": 32,
        "residual_only_count": 0,
        "runtime_quality_passed_count": 0,
        "runtime_quality_warned_count": 32,
        "runtime_quality_failed_count": 0,
        "partial_runtime_passed_count": 3,
        "negative_control_runtime_passed_count": 9,
        "high_residual_warning_count": 32,
        "joint_limit_warning_count": 1,
        "joint_limit_smoke_warning_count": 0,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def current_summary() -> dict:
    return {
        **baseline_summary(),
        "base_step3_3_final_head": EXPECTED_BASE_STEP3_3_FINAL_HEAD,
        "task_coverage_mean": 1.0,
        "task_coverage_min": 1.0,
        "anchor_reliability_mean": 1.0,
        "anchor_reliability_min": 1.0,
        "median_task_residual_p95": 2.1,
        "p95_task_residual_p95": 2.1,
        "max_task_residual_p95": 2.1,
        "median_normalized_task_residual_p95": 0.7,
        "p95_normalized_task_residual_p95": 0.7,
        "max_normalized_task_residual_p95": 0.7,
        "normalization_integrity_status": "legacy_row_max_recorded_raw_guarded",
        "source_code_commit": "a" * 40,
    }


def quality_delta(summary: dict, baseline: dict) -> dict:
    keys = tuple(baseline)
    baseline_counts = {key: baseline[key] for key in keys if isinstance(baseline.get(key), int)}
    current_counts = {key: summary[key] for key in keys if isinstance(baseline.get(key), int)}
    metric_delta = {
        "baseline": {"median": 4.0, "p95": 4.0, "max": 4.0},
        "current": {"median": 2.1, "p95": 2.1, "max": 2.1},
        "delta": {"median": -1.9, "p95": -1.9, "max": -1.9},
    }
    normalized_delta = {
        "baseline": {"median": 0.8, "p95": 0.8, "max": 0.8},
        "current": {"median": 0.7, "p95": 0.7, "max": 0.7},
        "delta": {"median": -0.1, "p95": -0.1, "max": -0.1},
    }
    return {
        "schema_version": 1,
        "baseline_final_head": EXPECTED_BASE_STEP3_3_FINAL_HEAD,
        "base_step3_3_final_head": EXPECTED_BASE_STEP3_3_FINAL_HEAD,
        "current_source_commit": "a" * 40,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": {key: current_counts[key] - baseline_counts[key] for key in baseline_counts},
        "metric_distribution_deltas": {
            "raw_task_residual_p95": metric_delta,
            "task_residual_p95": metric_delta,
            "normalized_task_residual_p95": normalized_delta,
        },
        "task_coverage_deltas": {"baseline": {"task_coverage_mean": 0.2, "task_coverage_min": 0.2}, "current": {"task_coverage_mean": 1.0, "task_coverage_min": 1.0}, "delta": {"task_coverage_mean": 0.8, "task_coverage_min": 0.8}},
        "anchor_reliability_deltas": {"baseline": {"evidence_available": False}, "current": {"anchor_reliability_mean": 1.0, "anchor_reliability_min": 1.0}, "delta": {"anchor_reliability_mean": None, "anchor_reliability_min": None}},
        "raw_vs_normalized_residual_deltas": {"raw_residual_regression_count": 0, "normalization_hides_raw_regression": False},
        "regressions": [],
        "improvements": ["raw_task_residual_p95_distribution_improved", "task_coverage_mean_increased"],
        "verdict": "PASS",
    }


def task_coverage(rows: list[dict]) -> dict:
    full = [row for row in rows if row["category"] == "full_humanoid_profile"]
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [{"model_id": row["model_id"], "category": row["category"], "task_coverage_ratio": 1.0, "task_anchor_count": 6, "task_rows": []} for row in full],
        "summary": {"task_coverage_mean": 1.0, "task_coverage_min": 1.0},
    }


def anchor_reliability(rows: list[dict]) -> dict:
    full = [row for row in rows if row["category"] == "full_humanoid_profile"]
    return {
        "schema_version": 1,
        "row_count": 32,
        "model_rows": [{"model_id": row["model_id"], "category": row["category"], "anchor_reliability_score": 1.0, "anchor_rejection_reasons": []} for row in full],
        "rows": [{"model_id": row["model_id"], "semantic": "Hips", "accepted": True, "rejection_reasons": []} for row in full],
        "summary": {"anchor_reliability_mean": 1.0, "anchor_reliability_min": 1.0},
    }


def residual_taxonomy(rows: list[dict]) -> dict:
    full = [row for row in rows if row["category"] == "full_humanoid_profile"]
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [{"model_id": row["model_id"], "blocker_buckets": ["rotation_residual_dominates"], "dominant_residual_semantic": "LeftHand", "dominant_residual_component": "rotation"} for row in full],
        "aggregate_buckets": {"rotation_residual_dominates": 32},
        "robot_specific_tuning_used": False,
    }


def step3_4_solver_config() -> dict:
    return {
        "schema_version": 1,
        "base_step3_3_final_head": EXPECTED_BASE_STEP3_3_FINAL_HEAD,
        "global_config": True,
        "robot_specific_tuning": False,
        "solver_config_hash": "hash",
        "config": {
            "enable_global_quality_hardening": True,
            "enable_global_residual_quality_hardening": True,
            "project_joint_limits": True,
            "task_order": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"],
        },
        "global_residual_quality_policy": {"enabled": True, "robot_specific_tuning": False},
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
