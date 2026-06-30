from __future__ import annotations

import json
from pathlib import Path


BASE_STEP4_1_FINAL_HEAD = "a" * 40
SOURCE_COMMIT = "b" * 40
SELECTED_POLICY = "parent_relative_runtime_inv_target"
PRODUCTION_POLICY = "world_runtime_inv_target"


def write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _write_fixture(tmp_path, release_status="PASS_RC", breakthrough=True)


def write_blocked_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _write_fixture(tmp_path, release_status="BLOCKED_GATE_RECONCILIATION", breakthrough=False)


def _write_fixture(tmp_path: Path, *, release_status: str, breakthrough: bool) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring"
    baseline_dir = source_root / "artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough"
    (artifact_dir / "test_results").mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    rows = model_rows()
    full_rows = [row for row in rows if row["category"] == "full_humanoid_profile"]
    summary = quality_summary(release_status=release_status, breakthrough=breakthrough)
    baseline = baseline_summary()
    orientation = orientation_matrix(full_rows, breakthrough=breakthrough)
    runtime_delta = runtime_scoring_delta(summary, breakthrough=breakthrough)
    gate_report = gate_reconciliation(full_rows)
    normalization = scoring_normalization()
    impact = impact_report(summary, runtime_delta, gate_report, normalization)

    write_json(baseline_dir / "quality_summary.json", baseline)
    write_json(baseline_dir / "full_pipeline_matrix.json", {"schema_version": 1, "row_count": 44, "rows": baseline_rows()})
    write_json(
        baseline_dir / "orientation_policy_selection.json",
        {
            "schema_version": 1,
            "global_selected_policy": {
                "policy": SELECTED_POLICY,
                "production_default_changed": False,
                "runtime_override_default_enabled": False,
                "robot_specific_tuning_used": False,
            },
        },
    )

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
    (artifact_dir / "commands.txt").write_text("PYTHONPATH=. python run_step4_2\n", encoding="utf-8")
    (artifact_dir / "test_results/pytest.txt").write_text("10 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text("<testsuite tests=\"10\" failures=\"0\"></testsuite>\n", encoding="utf-8")
    write_json(artifact_dir / "test_results/pytest_summary.json", {"schema_version": 1, "passed": 10, "failed": 0})

    write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "row_count": 44, "in_scope_total": 44, "rows": rows})
    write_json(artifact_dir / "full_pipeline_matrix.json", {"schema_version": 1, "row_count": 44, "rows": rows})
    write_json(artifact_dir / "clip_matrix.json", clip_matrix(full_rows, breakthrough=breakthrough))
    write_json(artifact_dir / "solver_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": smoke_rows(full_rows)})
    write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": smoke_rows(full_rows)})
    write_json(artifact_dir / "quality_summary.json", summary)
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json", runtime_delta)
    write_json(artifact_dir / "quality_delta_vs_step4_1.json", {"schema_version": 1, "runtime_scoring_delta_vs_step4_1": runtime_delta, "release_candidate_status": release_status})
    write_json(artifact_dir / "orientation_policy_runtime_impact_report.json", impact)
    write_json(artifact_dir / "orientation_policy_runtime_scoring_matrix.json", policy_runtime_matrix(full_rows, breakthrough=breakthrough))
    write_json(artifact_dir / "active_vs_diagnostic_policy_matrix.json", active_vs_diagnostic(full_rows, breakthrough=breakthrough))
    write_json(artifact_dir / "gate_reconciliation_report.json", gate_report)
    write_json(artifact_dir / "scoring_normalization_audit.json", normalization)
    write_json(artifact_dir / "orientation_integrated_residual_matrix.json", orientation)
    write_json(artifact_dir / "orientation_integrated_clip_consistency_matrix.json", clip_consistency(full_rows, breakthrough=breakthrough))
    write_json(artifact_dir / "pipeline_config.json", pipeline_config())
    write_json(artifact_dir / "solver_config.json", solver_config())
    write_json(artifact_dir / "trajectory_export_manifest.json", trajectory_export_manifest(full_rows))
    write_json(artifact_dir / "temporal_continuity_matrix.json", temporal_continuity(full_rows))
    write_json(artifact_dir / "support_contact_diagnostics.json", support_contact_diagnostics(full_rows))
    write_json(artifact_dir / "collision_proxy_diagnostics.json", collision_proxy_diagnostics(full_rows))
    write_json(artifact_dir / "pipeline_controls_reference.json", {"schema_version": 1, "controls": {"override_explicit_only": True}})
    write_json(artifact_dir / "red_team_report.json", {"schema_version": 1, "finding_count": 0, "checks": []})
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
    write_json(
        artifact_dir / "acceptance_ledger.json",
        {
            "schema_version": 1,
            "status": "PASS" if release_status == "PASS_RC" else "BLOCKED",
            "verdict": "PASS" if release_status == "PASS_RC" else "BLOCKED",
            "release_candidate_status": release_status,
            "base_step4_1_final_head": BASE_STEP4_1_FINAL_HEAD,
            "source_code_commit": SOURCE_COMMIT,
            "quality_summary": summary,
            "runtime_scoring_delta_vs_step4_1": runtime_delta,
            "deterministic_rerun": {"deterministic_compared_count": 44, "deterministic_matched_count": 44},
        },
    )
    return artifact_dir, baseline_dir, source_root


def baseline_summary() -> dict:
    return {
        "schema_version": 1,
        "source_code_commit": BASE_STEP4_1_FINAL_HEAD,
        "orientation_selected_policy": SELECTED_POLICY,
        "release_candidate_status": "PASS_RC",
        "primary_quality_breakthrough": True,
        "in_scope_total": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
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
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
        "median_raw_task_residual_p95": 4.0,
        "p95_raw_task_residual_p95": 4.0,
        "max_raw_task_residual_p95": 4.0,
        "median_normalized_task_residual_p95": 0.99,
        "p95_normalized_task_residual_p95": 0.99,
        "max_normalized_task_residual_p95": 1.0,
    }


def quality_summary(*, release_status: str, breakthrough: bool) -> dict:
    active = 3.5 if breakthrough else 4.0
    return {
        **baseline_summary(),
        "source_code_commit": SOURCE_COMMIT,
        "base_step4_1_final_head": BASE_STEP4_1_FINAL_HEAD,
        "release_candidate_status": release_status,
        "primary_quality_breakthrough": breakthrough,
        "orientation_policy_active_for_scoring": True,
        "active_runtime_scoring_orientation_policy": SELECTED_POLICY,
        "diagnostic_orientation_policy": SELECTED_POLICY,
        "production_default_orientation_policy": PRODUCTION_POLICY,
        "orientation_policy_production_default_changed": False,
        "runtime_override_default_enabled": False,
        "p95_orientation_integrated_residual": active,
        "median_orientation_integrated_residual": active,
        "max_orientation_integrated_residual": active,
        "raw_residual_regression_count": 0,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "trajectory_exports_count": 128,
        "temporal_continuity_finite_count": 128,
        "support_contact_diagnostic_count": 128,
        "collision_proxy_diagnostic_count": 128,
    }


def model_rows() -> list[dict]:
    rows = []
    for index in range(32):
        rows.append(full_row(f"full_{index:02d}"))
    for index in range(3):
        rows.append(non_full_row(f"partial_{index:02d}", "partial_humanoid_profile", "partial_runtime_passed"))
    for index in range(9):
        rows.append(non_full_row(f"negative_{index:02d}", "negative_control", "negative_control_runtime_passed"))
    return rows


def baseline_rows() -> list[dict]:
    rows = []
    for index in range(32):
        row = full_row(f"full_{index:02d}")
        row["raw_task_residual_p95"] = 4.0
        row["normalized_task_residual_p95"] = 0.99
        rows.append(row)
    for index in range(3):
        rows.append(non_full_row(f"partial_{index:02d}", "partial_humanoid_profile", "partial_runtime_passed"))
    for index in range(9):
        rows.append(non_full_row(f"negative_{index:02d}", "negative_control", "negative_control_runtime_passed"))
    return rows


def full_row(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "category": "full_humanoid_profile",
        "runtime_quality_status": "runtime_quality_warned",
        "quality_classification": "runtime_quality_warned",
        "solver_backed_smoke_attempted": True,
        "solver_backed_smoke_completed": True,
        "solver_backed": True,
        "residual_only": False,
        "normalized_task_residual_p95": 0.99,
        "normalized_task_residual_max": 1.0,
        "raw_task_residual_p95": 3.5,
        "raw_task_residual_max": 3.6,
        "orientation_integrated_residual_p95": 3.5,
        "orientation_integrated_residual_max": 3.6,
        "legacy_world_task_residual_p95": 4.0,
        "legacy_world_rotation_residual_p95": 2.1,
        "target_rotation_error_p95": 1.6,
        "rotation_residual_p95": 1.6,
        "translation_residual_p95": 1.9,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "output_nan_count": 0,
        "output_inf_count": 0,
        "active_runtime_scoring_orientation_policy": SELECTED_POLICY,
        "diagnostic_orientation_policy": SELECTED_POLICY,
        "production_default_orientation_policy": PRODUCTION_POLICY,
        "orientation_policy_active_for_scoring": True,
        "orientation_policy_production_default_changed": False,
        "runtime_override_default_enabled": False,
        "warning_reasons": ["normalized_task_residual_p95_above_pass_gate", "high_task_residual"],
        "failure_or_warning_reasons": ["normalized_task_residual_p95_above_pass_gate", "high_task_residual"],
    }


def non_full_row(model_id: str, category: str, status: str) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "runtime_quality_status": status,
        "solver_backed_smoke_attempted": False,
        "solver_backed_smoke_completed": False,
        "solver_backed": False,
        "residual_only": False,
        "normalized_task_residual_p95": 0.0,
        "normalized_task_residual_max": 0.0,
        "raw_task_residual_p95": 0.0,
        "raw_task_residual_max": 0.0,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "output_nan_count": 0,
        "output_inf_count": 0,
    }


def smoke_rows(full_rows: list[dict]) -> list[dict]:
    return [
        {
            "model_id": row["model_id"],
            "category": row["category"],
            "clip_id": "walk",
            "solver_backed": True,
            "solver_backed_smoke_attempted": True,
            "solver_backed_smoke_completed": True,
            "residual_only": False,
            "runtime_quality_status": "runtime_quality_warned",
            "quality_classification": "runtime_quality_warned",
            "metrics": {
                "normalized_task_residual_p95": 0.99,
                "normalized_task_residual_max": 1.0,
                "orientation_integrated_residual_p95": row["orientation_integrated_residual_p95"],
                "legacy_world_task_residual_p95": row["legacy_world_task_residual_p95"],
                "nan_count": 0,
                "inf_count": 0,
            },
            "smoke_summary": {
                "solver_backed": True,
                "solver_backed_smoke_completed": True,
                "quality_classification": "runtime_quality_warned",
                "metrics": {"normalized_task_residual_p95": 0.99, "normalized_task_residual_max": 1.0, "nan_count": 0, "inf_count": 0},
            },
        }
        for row in full_rows
    ]


def runtime_scoring_delta(summary: dict, *, breakthrough: bool) -> dict:
    base = baseline_summary()
    baseline_counts = {key: int(base.get(key, 0) or 0) for key in count_keys()}
    current_counts = {key: int(summary.get(key, 0) or 0) for key in count_keys()}
    active_delta = -0.5 if breakthrough else 0.0
    normalized_delta = 0.0
    return {
        "schema_version": 1,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": {key: current_counts[key] - baseline_counts[key] for key in count_keys()},
        "metric_distribution_deltas": {
            "orientation_integrated_residual_p95": metric_delta(4.0, 3.5 if breakthrough else 4.0),
            "normalized_task_residual_p95": metric_delta(0.99, 0.99),
            "raw_task_residual_p95": metric_delta(4.0, 3.5 if breakthrough else 4.0),
        },
        "runtime_quality_passed_count_delta": 0,
        "runtime_quality_warned_count_delta": 0,
        "high_residual_warning_count_delta": 0,
        "p95_normalized_task_residual_p95_delta": normalized_delta,
        "p95_orientation_integrated_residual_delta": active_delta,
        "raw_residual_regression_count": 0,
        "legacy_raw_residual_regression_count": 0,
        "normalization_hides_raw_residual_regression": False,
        "gate_blocker_taxonomy": {"normalized_task_residual_p95_above_pass_gate": 32},
        "per_model_deltas": [],
        "improvements": ["p95_orientation_integrated_residual_p95_distribution_improved"] if breakthrough else [],
        "primary_quality_breakthrough": breakthrough,
        "regressions": [],
    }


def count_keys() -> tuple[str, ...]:
    return (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_smoke_attempted_count",
        "solver_backed_completed_count",
        "solver_backed_count",
        "residual_only_count",
        "runtime_quality_passed_count",
        "runtime_quality_warned_count",
        "runtime_quality_failed_count",
        "partial_runtime_passed_count",
        "negative_control_runtime_passed_count",
        "high_residual_warning_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    )


def metric_delta(baseline: float, current: float) -> dict:
    delta = current - baseline
    return {
        "baseline": {"median": baseline, "p95": baseline, "max": baseline},
        "current": {"median": current, "p95": current, "max": current},
        "delta": {"median": delta, "p95": delta, "max": delta},
    }


def gate_reconciliation(full_rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "active_scoring_metrics": ["orientation_integrated_residual_p95", "normalized_task_residual_p95"],
        "diagnostic_only_metrics": ["legacy_world_task_residual_p95"],
        "runtime_quality_gate_inputs": ["solver_backed", "normalized_task_residual_p95"],
        "unchanged_gate_policy": "GLOBAL_RUNTIME_QUALITY_GATES unchanged from Step 4.1",
        "pass_gate_thresholds_unchanged": True,
        "warn_gate_thresholds_unchanged": True,
        "gate_thresholds": {"normalized_task_residual_p95_pass": 0.15},
        "per_gate_blocker_counts": {"normalized_task_residual_p95_above_pass_gate": 32},
        "rows_newly_passing": [],
        "rows_still_warned": [row["model_id"] for row in full_rows],
        "why_no_pass_if_zero": ["normalized_task_residual_p95_above_pass_gate"],
        "row_count": len(full_rows),
        "rows": [],
        "gates_weakened": False,
    }


def scoring_normalization() -> dict:
    return {
        "schema_version": 1,
        "selected_normalization": "legacy_row_max_v1_active_orientation_scoring",
        "raw_residual_always_retained": True,
        "normalization_robot_specific": False,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "raw_residual_regression_count": 0,
    }


def impact_report(summary: dict, runtime_delta: dict, gate_report: dict, normalization: dict) -> dict:
    return {
        "schema_version": 1,
        "release_candidate_status": summary["release_candidate_status"],
        "primary_quality_breakthrough": summary["primary_quality_breakthrough"],
        "selected_orientation_policy": SELECTED_POLICY,
        "orientation_policy_active_for_scoring": True,
        "orientation_policy_production_default_changed": False,
        "runtime_override_default_enabled": False,
        "runtime_scoring_delta_vs_step4_1": runtime_delta,
        "gate_reconciliation_summary": gate_report,
        "normalization_audit_result": normalization,
        "remaining_warned_rows": 32,
        "remaining_blockers": gate_report["why_no_pass_if_zero"],
    }


def orientation_matrix(full_rows: list[dict], *, breakthrough: bool) -> dict:
    rows = []
    active = 3.5 if breakthrough else 4.0
    for row in full_rows:
        rows.append(
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "runtime_quality_status": row["runtime_quality_status"],
                "solver_backed": True,
                "residual_only": False,
                "active_runtime_scoring_policy": SELECTED_POLICY,
                "diagnostic_orientation_policy": SELECTED_POLICY,
                "production_default_policy": PRODUCTION_POLICY,
                "active_for_scoring": True,
                "production_default_changed": False,
                "runtime_override_default_enabled": False,
                "normalized_task_residual_p95": 0.99,
                "raw_task_residual_p95": active,
                "orientation_integrated_residual_p95": active,
                "orientation_integrated_residual_max": active + 0.1,
                "legacy_world_task_residual_p95": 4.0,
                "active_rotation_residual_p95": 1.6,
                "legacy_world_rotation_residual_p95": 2.1,
                "translation_residual_p95": 1.9,
                "warning_reasons": ["normalized_task_residual_p95_above_pass_gate", "high_task_residual"],
            }
        )
    return {
        "schema_version": 1,
        "selected_policy": SELECTED_POLICY,
        "active_runtime_scoring_policy": SELECTED_POLICY,
        "production_default_policy": PRODUCTION_POLICY,
        "row_count": len(rows),
        "rows": rows,
        "distribution": {"median": active, "p95": active, "max": active + 0.1},
        "robot_specific_tuning_used": False,
    }


def policy_runtime_matrix(full_rows: list[dict], *, breakthrough: bool) -> dict:
    active = 3.5 if breakthrough else 4.0
    rows = [
        {
            "model_id": row["model_id"],
            "diagnostic_orientation_policy": SELECTED_POLICY,
            "active_runtime_scoring_policy": SELECTED_POLICY,
            "production_default_policy": PRODUCTION_POLICY,
            "policy_state": "scoring_active_explicit_opt_in",
            "active_for_scoring": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "orientation_integrated_residual_p95": active,
            "legacy_world_task_residual_p95": 4.0,
            "active_scoring_delta_vs_legacy_world": active - 4.0,
            "runtime_quality_status": "runtime_quality_warned",
        }
        for row in full_rows
    ]
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "active_runtime_scoring_policy": SELECTED_POLICY,
        "production_default_policy": PRODUCTION_POLICY,
        "production_default_changed": False,
        "runtime_override_default_enabled": False,
        "rows": rows,
    }


def active_vs_diagnostic(full_rows: list[dict], *, breakthrough: bool) -> dict:
    active = 3.5 if breakthrough else 4.0
    rows = [
        {
            "model_id": row["model_id"],
            "diagnostic_orientation_policy": SELECTED_POLICY,
            "active_runtime_scoring_policy": SELECTED_POLICY,
            "production_default_policy": PRODUCTION_POLICY,
            "active_for_step4_2_scoring": True,
            "diagnostic_only_in_step4_1": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "active_orientation_integrated_residual_p95": active,
            "legacy_world_task_residual_p95": 4.0,
            "delta_vs_legacy_world": active - 4.0,
        }
        for row in full_rows
    ]
    return {"schema_version": 1, "policy_roles": [], "row_count": len(rows), "rows": rows}


def clip_matrix(full_rows: list[dict], *, breakthrough: bool) -> dict:
    active = 3.5 if breakthrough else 4.0
    rows = []
    for row in full_rows:
        for clip in ("walk", "wave", "stretch", "pickup"):
            rows.append(
                {
                    "model_id": row["model_id"],
                    "category": row["category"],
                    "clip_id": clip,
                    "target_stream_status": "passed",
                    "per_clip_runtime_quality_status": "runtime_quality_warned",
                    "solver_backed": True,
                    "per_clip_residual_metrics": {
                        "orientation_integrated_residual_p95": active,
                        "legacy_world_task_residual_p95": 4.0,
                    },
                    "per_clip_orientation_residual_metrics": {
                        "active_runtime_scoring_orientation_policy": SELECTED_POLICY,
                        "orientation_policy_active_for_scoring": True,
                    },
                }
            )
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def clip_consistency(full_rows: list[dict], *, breakthrough: bool) -> dict:
    active = 3.5 if breakthrough else 4.0
    rows = [
        {
            "model_id": row["model_id"],
            "clip_id": clip,
            "runtime_quality_status": "runtime_quality_warned",
            "active_runtime_scoring_policy": SELECTED_POLICY,
            "active_for_scoring": True,
            "orientation_integrated_residual_p95": active,
            "legacy_world_task_residual_p95": 4.0,
            "delta_vs_legacy_world": active - 4.0,
            "clip_level_status": "active_scoring_improved" if breakthrough else "active_scoring_unchanged",
            "solver_backed": True,
        }
        for row in full_rows
        for clip in ("walk", "wave", "stretch", "pickup")
    ]
    return {"schema_version": 1, "row_count": len(rows), "selected_policy": SELECTED_POLICY, "rows": rows}


def solver_config() -> dict:
    return {
        "schema_version": 1,
        "step": "step4_2_orientation_policy_runtime_scoring",
        "base_step4_1_final_head": BASE_STEP4_1_FINAL_HEAD,
        "global_config": True,
        "robot_specific_tuning": False,
        "solver_config_hash": "step4-2-solver",
        "global_orientation_residual_policy": {
            "active_for_runtime_scoring": True,
            "active_runtime_scoring_policy": SELECTED_POLICY,
            "production_default_policy": PRODUCTION_POLICY,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
        },
    }


def pipeline_config() -> dict:
    return {
        "schema_version": 1,
        "step": "step4_2_orientation_policy_runtime_scoring",
        "base_step4_1_final_head": BASE_STEP4_1_FINAL_HEAD,
        "global_config": True,
        "robot_specific_tuning": False,
        "pipeline_config_hash": "step4-2-pipeline",
        "config": {
            "enable_parent_relative_orientation_runtime_scoring": True,
            "orientation_policy_production_default_changed": False,
            "runtime_override_default_enabled": False,
        },
    }


def trajectory_export_manifest(full_rows: list[dict]) -> dict:
    rows = [
        {"model_id": row["model_id"], "clip_id": clip, "finite_qpos": True, "nan_count": 0, "inf_count": 0, "export_hash": f"{row['model_id']}-{clip}"}
        for row in full_rows
        for clip in ("walk", "wave", "stretch", "pickup")
    ]
    return {"schema_version": 1, "row_count": len(rows), "rows": rows, "exports": rows}


def temporal_continuity(full_rows: list[dict]) -> dict:
    rows = [{"model_id": row["model_id"], "clip_id": clip, "finite": True, "finite_velocity": True, "finite_acceleration": True} for row in full_rows for clip in ("walk", "wave", "stretch", "pickup")]
    return {"schema_version": 1, "row_count": len(rows), "finite_count": len(rows), "rows": rows}


def support_contact_diagnostics(full_rows: list[dict]) -> dict:
    rows = [{"model_id": row["model_id"], "clip_id": clip, "finite": True} for row in full_rows for clip in ("walk", "wave", "stretch", "pickup")]
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def collision_proxy_diagnostics(full_rows: list[dict]) -> dict:
    rows = [{"model_id": row["model_id"], "clip_id": clip, "finite": True, "collision_proxy_count": 0} for row in full_rows for clip in ("walk", "wave", "stretch", "pickup")]
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
