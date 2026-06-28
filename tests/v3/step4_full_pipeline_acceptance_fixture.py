from __future__ import annotations

import json
from pathlib import Path


EXPECTED_BASE_STEP3_4_FINAL_HEAD = "77e7c02393a6678ccab40cdb847021d7d94392c9"
SOURCE_COMMIT = "b" * 40
SOLVER_CONFIG_HASH = "step4-solver-config-hash"
PIPELINE_CONFIG_HASH = "step4-pipeline-config-hash"


def write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step4_full_pipeline_acceptance"
    baseline_dir = source_root / "artifacts/retargeting_v3_step3_4_global_residual_quality"
    (artifact_dir / "test_results").mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    rows = model_rows()
    full_rows = [row for row in rows if row["category"] == "full_humanoid_profile"]
    baseline = baseline_summary()
    summary = current_summary()
    delta = quality_delta(summary, baseline)

    write_json(baseline_dir / "quality_summary.json", baseline)
    write_json(
        artifact_dir / "environment.json",
        {
            "schema_version": 1,
            "git_status_short": "",
            "source_git_status_short_before_run": "",
            "source_code_commit": SOURCE_COMMIT,
            "source_code_commit_remote_resolvable": True,
            "source_code_commit_is_artifact_commit_ancestor": True,
            "source_worktree_clean_before_run": True,
            "source_worktree_clean_after_run": True,
            "core_diff_after_source_commit": [],
        },
    )
    (artifact_dir / "commands.txt").write_text(
        "PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_pipeline_acceptance "
        "--artifact-dir artifacts/retargeting_v3_step4_full_pipeline_acceptance "
        "--enable-solver-backed-generic-smoke\n",
        encoding="utf-8",
    )
    (artifact_dir / "test_results/pytest.txt").write_text("18 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text("<testsuite tests=\"18\" failures=\"0\"></testsuite>\n", encoding="utf-8")
    write_json(artifact_dir / "test_results/pytest_summary.json", {"schema_version": 1, "passed": 18, "failed": 0})

    write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "row_count": 44, "in_scope_total": 44, "rows": rows})
    write_json(artifact_dir / "clip_matrix.json", clip_matrix(full_rows))
    write_json(artifact_dir / "solver_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": smoke_rows(full_rows)})
    write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": smoke_rows(full_rows)})
    write_json(artifact_dir / "full_pipeline_matrix.json", {"schema_version": 1, "row_count": 44, "rows": rows})
    write_json(artifact_dir / "quality_summary.json", summary)
    write_json(artifact_dir / "quality_delta_vs_step3_4.json", delta)
    write_json(artifact_dir / "residual_taxonomy.json", residual_taxonomy(full_rows))
    write_json(artifact_dir / "orientation_residual_taxonomy.json", orientation_residual_taxonomy(full_rows))
    write_json(artifact_dir / "normalization_audit.json", normalization_audit(full_rows))
    write_json(artifact_dir / "task_coverage_matrix.json", task_coverage(full_rows))
    write_json(artifact_dir / "anchor_reliability_matrix.json", anchor_reliability(full_rows))
    write_json(artifact_dir / "trajectory_export_manifest.json", trajectory_export_manifest(full_rows))
    write_json(artifact_dir / "temporal_continuity_matrix.json", temporal_continuity(full_rows))
    write_json(artifact_dir / "support_contact_diagnostics.json", support_contact_diagnostics(full_rows))
    write_json(artifact_dir / "collision_proxy_diagnostics.json", collision_proxy_diagnostics(full_rows))
    write_json(artifact_dir / "pipeline_controls_reference.json", pipeline_controls_reference())
    write_json(artifact_dir / "solver_config.json", solver_config())
    write_json(artifact_dir / "pipeline_config.json", pipeline_config())
    write_json(artifact_dir / "solver_diagnostics_matrix.json", {"schema_version": 1, "row_count": 32, "rows": solver_diagnostics(full_rows)})
    write_json(artifact_dir / "red_team_report.json", {"schema_version": 1, "verdict": "PASS_RC", "blockers": []})
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
            "mismatched_model_ids": [],
        },
    )
    write_json(
        artifact_dir / "acceptance_ledger.json",
        {
            "schema_version": 1,
            "verdict": "PASS_RC",
            "release_candidate_status": "PASS_RC",
            "base_step3_4_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
            "source_code_commit": SOURCE_COMMIT,
            "solver_config_hash": SOLVER_CONFIG_HASH,
            "pipeline_config_hash": PIPELINE_CONFIG_HASH,
            "quality_delta_vs_step3_4": delta,
            "strict_step4_audit": {"status": "PASS_RC"},
            "ci": {"status": "passed", "jobs": ["step4-static-and-unit", "step4-artifact-audit"]},
        },
    )
    return artifact_dir, baseline_dir, source_root


def model_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(32):
        passed = index == 0
        rows.append(
            {
                **full_numeric_row(f"full_{index:02d}", passed=passed),
                "category": "full_humanoid_profile",
                "source_status": "passed",
                "runtime_quality_status": "runtime_quality_passed" if passed else "runtime_quality_warned",
                "release_candidate_row_status": "PASS_RC_ROW" if passed else "WARNED_WITH_BREAKTHROUGH",
                "quality_classification": "runtime_quality_passed" if passed else "runtime_quality_warned",
                "solver_backed_smoke_attempted": True,
                "solver_backed_smoke_completed": True,
                "solver_backed": True,
                "residual_only": False,
                "clip_count_attempted": 2,
                "clip_count_completed": 2,
                "trajectory_export_count": 2,
                "warning_reasons": [] if passed else ["high_task_residual", "rotation_residual_dominates"],
                "failure_reasons": [],
                "runtime_quality_warning_reasons": [] if passed else ["high_task_residual", "rotation_residual_dominates"],
                "failure_or_warning_reasons": [] if passed else ["high_task_residual", "rotation_residual_dominates"],
            }
        )
    for index in range(3):
        rows.append(
            {
                **non_full_row(f"partial_{index:02d}"),
                "category": "partial_humanoid_profile",
                "source_status": "partial_passed",
                "runtime_quality_status": "partial_runtime_passed",
                "release_candidate_row_status": "PARTIAL_NOT_RC",
            }
        )
    for index in range(9):
        rows.append(
            {
                **non_full_row(f"negative_{index:02d}"),
                "category": "negative_control",
                "source_status": "negative_control_passed",
                "expected_capability": "negative_control",
                "runtime_quality_status": "negative_control_runtime_passed",
                "release_candidate_row_status": "NEGATIVE_CONTROL_NOT_RC",
                "promoted_to_runtime_quality": False,
                "quality_evaluated": False,
                "override_allowed": False,
                "humanoid_profile_generated": False,
            }
        )
    return rows


def full_numeric_row(model_id: str, *, passed: bool) -> dict:
    normalized_p95 = 0.10 if passed else 0.42
    normalized_max = 0.14 if passed else 0.62
    raw_p95 = 1.60 if passed else 2.10
    raw_max = 2.00 if passed else 2.70
    rotation_p95 = 0.78 if passed else 0.86
    rotation_max = 0.92 if passed else 1.05
    translation_p95 = 0.12 if passed else 0.18
    translation_max = 0.16 if passed else 0.24
    return {
        "model_id": model_id,
        "frame_count": 120,
        "solver_mode": "full_pipeline_solver_backed_projection",
        "sampled_frame_indices": [0, 60, 119],
        "task_anchor_count": 6,
        "task_anchor_semantic_counts": {"Chest": 2, "Hips": 2, "LeftHand": 1, "RightHand": 1},
        "task_coverage_ratio": 1.0,
        "successful_task_coverage_ratio": 1.0,
        "anchor_reliability_score": 1.0,
        "anchor_rejection_reasons": [],
        "normalized_task_residual_mean": round(normalized_p95 * 0.70, 6),
        "normalized_task_residual_p50": round(normalized_p95 * 0.85, 6),
        "normalized_task_residual_p95": normalized_p95,
        "normalized_task_residual_max": normalized_max,
        "raw_task_residual_mean": round(raw_p95 * 0.70, 6),
        "raw_task_residual_p50": round(raw_p95 * 0.85, 6),
        "raw_task_residual_p95": raw_p95,
        "raw_task_residual_max": raw_max,
        "task_residual_mean": round(raw_p95 * 0.70, 6),
        "task_residual_p50": round(raw_p95 * 0.85, 6),
        "task_residual_p95": raw_p95,
        "task_residual_max": raw_max,
        "translation_residual_mean": round(translation_p95 * 0.70, 6),
        "translation_residual_p50": round(translation_p95 * 0.85, 6),
        "translation_residual_p95": translation_p95,
        "translation_residual_max": translation_max,
        "rotation_residual_mean": round(rotation_p95 * 0.70, 6),
        "rotation_residual_p50": round(rotation_p95 * 0.85, 6),
        "rotation_residual_p95": rotation_p95,
        "rotation_residual_max": rotation_max,
        "target_translation_error_mean": 0.03,
        "target_translation_error_p95": 0.08,
        "target_translation_error_max": 0.12,
        "target_rotation_error_mean": 0.08,
        "target_rotation_error_p95": 0.18,
        "target_rotation_error_max": 0.24,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "output_nan_count": 0,
        "output_inf_count": 0,
        "temporal_jump_count": 0,
        "velocity_p95": 0.55,
        "acceleration_p95": 1.1,
        "support_height_p95": 0.025,
        "collision_proxy_count": 0,
        "solver_config_hash": SOLVER_CONFIG_HASH,
        "pipeline_config_hash": PIPELINE_CONFIG_HASH,
        "export_hashes": [f"{model_id}-walk-export", f"{model_id}-turn-export"],
        "deterministic_hash_inputs": {"model_id": model_id, "clip_ids": ["walk_cycle", "turn_in_place"]},
        "residual_normalization_version": "global_chain_scale_v2",
        "residual_normalization_formula": "raw_residual / global_chain_scale",
        "residual_denominator": 5.0,
        "residual_denominator_source": "global_chain_scale_v2",
        "residual_denominator_scope": "global",
        "residual_denominator_units": "translation_meters_plus_rotation_radians",
        "residual_denominator_robot_specific": False,
        "residual_denominator_from_current_row_max": False,
        "runtime_seconds": 0.25,
    }


def non_full_row(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "frame_count": 120,
        "solver_backed_smoke_attempted": False,
        "solver_backed_smoke_completed": False,
        "solver_backed": False,
        "residual_only": False,
        "clip_count_attempted": 1,
        "clip_count_completed": 1,
        "trajectory_export_count": 0,
        "task_anchor_count": 0,
        "task_anchor_semantic_counts": {},
        "task_coverage_ratio": 0.0,
        "successful_task_coverage_ratio": 0.0,
        "anchor_reliability_score": 0.0,
        "normalized_task_residual_mean": 0.0,
        "normalized_task_residual_p50": 0.0,
        "normalized_task_residual_p95": 0.0,
        "normalized_task_residual_max": 0.0,
        "raw_task_residual_mean": 0.0,
        "raw_task_residual_p50": 0.0,
        "raw_task_residual_p95": 0.0,
        "raw_task_residual_max": 0.0,
        "task_residual_mean": 0.0,
        "task_residual_p50": 0.0,
        "task_residual_p95": 0.0,
        "task_residual_max": 0.0,
        "translation_residual_mean": 0.0,
        "translation_residual_p95": 0.0,
        "translation_residual_max": 0.0,
        "rotation_residual_mean": 0.0,
        "rotation_residual_p95": 0.0,
        "rotation_residual_max": 0.0,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "output_nan_count": 0,
        "output_inf_count": 0,
        "temporal_jump_count": 0,
        "velocity_p95": 0.0,
        "acceleration_p95": 0.0,
        "support_height_p95": 0.0,
        "collision_proxy_count": 0,
        "warning_reasons": [],
        "failure_reasons": [],
        "solver_config_hash": SOLVER_CONFIG_HASH,
        "pipeline_config_hash": PIPELINE_CONFIG_HASH,
        "export_hashes": [],
        "deterministic_hash_inputs": {"model_id": model_id},
    }


def baseline_summary() -> dict:
    return {
        "schema_version": 1,
        "base_step3_4_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
        "row_count": 44,
        "in_scope_total": 44,
        "matrix_row_count": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "clip_suite_count": 2,
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
        "rotation_dominant_residual_count": 31,
        "translation_dominant_residual_count": 1,
        "joint_limit_warning_count": 0,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
        "task_coverage_mean": 1.0,
        "task_coverage_min": 1.0,
        "anchor_reliability_mean": 1.0,
        "anchor_reliability_min": 1.0,
        "median_normalized_task_residual_p95": 0.70,
        "p95_normalized_task_residual_p95": 0.70,
        "max_normalized_task_residual_p95": 0.70,
        "median_raw_task_residual_p95": 2.10,
        "p95_raw_task_residual_p95": 2.10,
        "max_raw_task_residual_p95": 2.10,
        "median_rotation_residual_p95": 1.20,
        "p95_rotation_residual_p95": 1.20,
        "max_rotation_residual_p95": 1.25,
        "trajectory_exports_count": 64,
        "temporal_continuity_finite_count": 32,
        "support_contact_diagnostic_count": 32,
        "collision_proxy_diagnostic_count": 32,
        "release_candidate_status": "BLOCKED_RESIDUAL_QUALITY",
    }


def current_summary() -> dict:
    return {
        **baseline_summary(),
        "base_step3_4_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
        "source_code_commit": SOURCE_COMMIT,
        "runtime_quality_passed_count": 1,
        "runtime_quality_warned_count": 31,
        "high_residual_warning_count": 31,
        "median_normalized_task_residual_p95": 0.42,
        "p95_normalized_task_residual_p95": 0.42,
        "max_normalized_task_residual_p95": 0.62,
        "median_raw_task_residual_p95": 2.10,
        "p95_raw_task_residual_p95": 2.10,
        "max_raw_task_residual_p95": 2.70,
        "median_rotation_residual_p95": 0.86,
        "p95_rotation_residual_p95": 0.86,
        "max_rotation_residual_p95": 1.05,
        "release_candidate_status": "PASS_RC",
    }


def quality_delta(summary: dict, baseline: dict) -> dict:
    baseline_counts = {key: value for key, value in baseline.items() if isinstance(value, int)}
    current_counts = {key: summary[key] for key in baseline_counts}
    return {
        "schema_version": 1,
        "baseline_artifact_dir": "artifacts/retargeting_v3_step3_4_global_residual_quality",
        "current_artifact_dir": "artifacts/retargeting_v3_step4_full_pipeline_acceptance",
        "baseline_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
        "base_step3_4_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
        "current_source_commit": SOURCE_COMMIT,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": {key: current_counts[key] - baseline_counts[key] for key in baseline_counts},
        "metric_distribution_deltas": {
            "raw_task_residual_p95": metric_delta(2.10, 2.10),
            "normalized_task_residual_p95": metric_delta(0.70, 0.42),
            "rotation_residual_p95": metric_delta(1.20, 0.86),
            "p95_rotation_residual_p95": metric_delta(1.20, 0.86),
        },
        "orientation_residual_deltas": {
            "p95_rotation_residual_p95": {"baseline": 1.20, "current": 0.86, "delta": -0.34},
            "rotation_dominant_residual_count": {"baseline": 31, "current": 31, "delta": 0},
            "accepted_breakthrough": True,
        },
        "normalization_deltas": {
            "raw_residual_regression_count": 0,
            "normalization_hides_raw_regression": False,
            "global_normalization_v2_enabled": True,
        },
        "task_coverage_deltas": {"task_coverage_mean": {"baseline": 1.0, "current": 1.0, "delta": 0.0}},
        "anchor_reliability_deltas": {"anchor_reliability_mean": {"baseline": 1.0, "current": 1.0, "delta": 0.0}},
        "temporal_diagnostic_deltas": {"temporal_jump_count": {"baseline": 0, "current": 0, "delta": 0}},
        "regressions": [],
        "improvements": ["p95_rotation_residual_p95_distribution_improved", "runtime_quality_passed_count_increased"],
        "verdict": "PASS_RC",
    }


def metric_delta(baseline: float, current: float) -> dict:
    delta = round(current - baseline, 6)
    return {
        "baseline": {"median": baseline, "p95": baseline, "max": baseline},
        "current": {"median": current, "p95": current, "max": current},
        "delta": {"median": delta, "p95": delta, "max": delta},
    }


def smoke_rows(full_rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "metrics": {
                "solver_success_fraction": 1.0,
                "solver_task_count": 6,
                "nan_count": 0,
                "inf_count": 0,
                "normalized_task_residual_p95": row["normalized_task_residual_p95"],
                "rotation_residual_p95": row["rotation_residual_p95"],
            },
            "smoke_summary": {
                "mode": "full_pipeline_solver_backed_projection",
                "status": "passed",
                "metrics": {
                    "solver_success_fraction": 1.0,
                    "solver_task_count": 6,
                    "nan_count": 0,
                    "inf_count": 0,
                    "normalized_task_residual_p95": row["normalized_task_residual_p95"],
                    "rotation_residual_p95": row["rotation_residual_p95"],
                },
                "residual_only": False,
            },
        }
        for row in full_rows
    ]


def solver_diagnostics(full_rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "solver_iteration_count_mean": 4.0,
            "solver_iteration_count_p95": 6.0,
            "solver_iteration_count_max": 8.0,
            "solver_converged_frame_count": 120,
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
        }
        for row in full_rows
    ]


def clip_matrix(full_rows: list[dict]) -> dict:
    rows = [
        {
            "model_id": row["model_id"],
            "clip_id": clip_id,
            "status": "passed",
            "solver_backed": True,
            "output_nan_count": 0,
            "output_inf_count": 0,
            "trajectory_exported": True,
        }
        for row in full_rows
        for clip_id in ("walk_cycle", "turn_in_place")
    ]
    return {"schema_version": 1, "clip_suite_count": 2, "row_count": len(rows), "rows": rows}


def residual_taxonomy(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [
            {
                "model_id": row["model_id"],
                "blocker_buckets": [] if row["runtime_quality_status"] == "runtime_quality_passed" else ["rotation_residual_dominates"],
                "dominant_residual_semantic": "LeftHand",
                "dominant_residual_component": "rotation",
                "robot_specific_tuning_used": False,
            }
            for row in full_rows
        ],
        "aggregate_buckets": {"rotation_residual_dominates": 31},
        "robot_specific_tuning_used": False,
    }


def orientation_residual_taxonomy(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [
            {
                "model_id": row["model_id"],
                "rotation_residual_p95": row["rotation_residual_p95"],
                "translation_residual_p95": row["translation_residual_p95"],
                "orientation_dominates": row["rotation_residual_p95"] > row["translation_residual_p95"],
                "dominant_orientation_semantic": "LeftHand",
                "normalization_version": "global_chain_scale_v2",
            }
            for row in full_rows
        ],
        "summary": {
            "rotation_dominant_residual_count": 31,
            "p95_rotation_residual_p95": 0.86,
            "accepted_breakthrough": True,
            "robot_specific_tuning_used": False,
        },
    }


def normalization_audit(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "normalization_version": "global_chain_scale_v2",
        "global_normalization_v2_enabled": True,
        "normalization_hides_raw_regression": False,
        "raw_residual_regression_count": 0,
        "robot_specific_denominator_count": 0,
        "rows": [
            {
                "model_id": row["model_id"],
                "raw_task_residual_p95": row["raw_task_residual_p95"],
                "normalized_task_residual_p95": row["normalized_task_residual_p95"],
                "residual_denominator": row["residual_denominator"],
                "residual_denominator_robot_specific": False,
                "raw_regression_hidden": False,
            }
            for row in full_rows
        ],
    }


def task_coverage(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "task_coverage_ratio": 1.0,
                "successful_task_coverage_ratio": 1.0,
                "task_anchor_count": 6,
                "global_task_order": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"],
            }
            for row in full_rows
        ],
        "summary": {"task_coverage_mean": 1.0, "task_coverage_min": 1.0},
    }


def anchor_reliability(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "model_rows": [
            {"model_id": row["model_id"], "category": row["category"], "anchor_reliability_score": 1.0, "anchor_rejection_reasons": []}
            for row in full_rows
        ],
        "rows": [{"model_id": row["model_id"], "semantic": "Hips", "accepted": True, "rejection_reasons": []} for row in full_rows],
        "summary": {"anchor_reliability_mean": 1.0, "anchor_reliability_min": 1.0},
    }


def trajectory_export_manifest(full_rows: list[dict]) -> dict:
    rows = [
        {
            "model_id": row["model_id"],
            "clip_id": clip_id,
            "export_path": f"exports/{row['model_id']}_{clip_id}.npz",
            "export_hash": f"{row['model_id']}-{clip_id}-hash",
            "frame_count": 120,
            "finite": True,
            "nan_count": 0,
            "inf_count": 0,
            "qpos_finite": True,
            "timestamps_finite": True,
        }
        for row in full_rows
        for clip_id in ("walk_cycle", "turn_in_place")
    ]
    return {"schema_version": 1, "row_count": len(rows), "trajectory_exports_count": len(rows), "rows": rows}


def temporal_continuity(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "finite_count": 32,
        "rows": [
            {
                "model_id": row["model_id"],
                "finite": True,
                "temporal_jump_count": 0,
                "velocity_p95": row["velocity_p95"],
                "acceleration_p95": row["acceleration_p95"],
                "nan_count": 0,
                "inf_count": 0,
            }
            for row in full_rows
        ],
    }


def support_contact_diagnostics(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [
            {
                "model_id": row["model_id"],
                "finite": True,
                "support_height_p95": row["support_height_p95"],
                "foot_sliding_p95": 0.03,
                "contact_state_transitions": 4,
            }
            for row in full_rows
        ],
    }


def collision_proxy_diagnostics(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "row_count": 32,
        "rows": [
            {
                "model_id": row["model_id"],
                "finite": True,
                "collision_proxy_count": row["collision_proxy_count"],
                "collision_proxy_max_depth": 0.0,
            }
            for row in full_rows
        ],
    }


def pipeline_controls_reference() -> dict:
    return {
        "schema_version": 1,
        "controls": {
            "default_runtime_disabled_verified": True,
            "shadow_noop_verified": True,
            "override_explicit_only": True,
            "fingerprint_gate_enforced": True,
            "negative_controls_excluded": True,
            "artifact_paths_sanitized": True,
        },
    }


def solver_config() -> dict:
    return {
        "schema_version": 1,
        "base_step3_4_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
        "global_config": True,
        "robot_specific_tuning": False,
        "solver_config_hash": SOLVER_CONFIG_HASH,
        "config": {
            "enable_solver_backed_generic_smoke": True,
            "enable_orientation_residual_quality": True,
            "enable_global_normalization_v2": True,
            "project_joint_limits": True,
        },
    }


def pipeline_config() -> dict:
    return {
        "schema_version": 1,
        "base_step3_4_final_head": EXPECTED_BASE_STEP3_4_FINAL_HEAD,
        "pipeline_config_hash": PIPELINE_CONFIG_HASH,
        "robot_specific_tuning": False,
        "clip_suite": ["walk_cycle", "turn_in_place"],
        "trajectory_export_enabled": True,
        "temporal_continuity_enabled": True,
        "support_contact_diagnostics_enabled": True,
        "collision_proxy_diagnostics_enabled": True,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
