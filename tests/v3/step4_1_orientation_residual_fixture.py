from __future__ import annotations

import json
from pathlib import Path


BASE_STEP4_0_FINAL_HEAD = "a" * 40
SOURCE_COMMIT = "b" * 40
SOLVER_CONFIG_HASH = "step4-1-solver-config-hash"
PIPELINE_CONFIG_HASH = "step4-1-pipeline-config-hash"
SELECTED_POLICY = "parent_relative_runtime_inv_target"


def write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _write_fixture(tmp_path, release_status="PASS_RC", breakthrough=True)


def write_blocked_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _write_fixture(tmp_path, release_status="BLOCKED_ORIENTATION_SEMANTICS", breakthrough=False)


def _write_fixture(tmp_path: Path, *, release_status: str, breakthrough: bool) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough"
    baseline_dir = source_root / "artifacts/retargeting_v3_step4_full_pipeline_acceptance"
    (artifact_dir / "test_results").mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    rows = model_rows(breakthrough=breakthrough)
    full_rows = [row for row in rows if row["category"] == "full_humanoid_profile"]
    base = baseline_summary()
    summary = quality_summary(release_status=release_status, breakthrough=breakthrough)
    qdelta = quality_delta(summary, breakthrough=breakthrough)
    odelta = orientation_delta(breakthrough=breakthrough)
    policy = orientation_policy_selection(breakthrough=breakthrough)

    write_json(baseline_dir / "quality_summary.json", base)
    write_json(baseline_dir / "full_pipeline_matrix.json", {"schema_version": 1, "row_count": 44, "rows": rows})
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
    (artifact_dir / "commands.txt").write_text("PYTHONPATH=. python run_step4_1\n", encoding="utf-8")
    (artifact_dir / "test_results/pytest.txt").write_text("10 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text("<testsuite tests=\"10\" failures=\"0\"></testsuite>\n", encoding="utf-8")
    write_json(artifact_dir / "test_results/pytest_summary.json", {"schema_version": 1, "passed": 10, "failed": 0})

    write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "row_count": 44, "in_scope_total": 44, "rows": rows})
    write_json(artifact_dir / "full_pipeline_matrix.json", {"schema_version": 1, "row_count": 44, "rows": rows})
    write_json(artifact_dir / "clip_matrix.json", clip_matrix(full_rows))
    write_json(artifact_dir / "solver_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": smoke_rows(full_rows)})
    write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "row_count": 32, "rows": smoke_rows(full_rows)})
    write_json(artifact_dir / "quality_summary.json", summary)
    write_json(artifact_dir / "quality_delta_vs_step4_0.json", qdelta)
    write_json(artifact_dir / "orientation_delta_vs_step4_0.json", odelta)
    write_json(artifact_dir / "release_candidate_impact_report.json", release_candidate_impact(summary, qdelta, odelta, policy))
    write_json(artifact_dir / "orientation_frame_semantics_matrix.json", orientation_frame_semantics(full_rows))
    write_json(artifact_dir / "orientation_residual_math_audit.json", orientation_math_audit(full_rows, breakthrough=breakthrough))
    write_json(artifact_dir / "orientation_offset_candidate_matrix.json", orientation_offset_candidates(breakthrough=breakthrough))
    write_json(artifact_dir / "orientation_policy_selection.json", policy)
    write_json(artifact_dir / "orientation_clip_consistency_matrix.json", orientation_clip_consistency(full_rows, breakthrough=breakthrough))
    write_json(artifact_dir / "normalization_audit.json", normalization_audit())
    write_json(artifact_dir / "trajectory_export_manifest.json", trajectory_export_manifest(full_rows))
    write_json(artifact_dir / "temporal_continuity_matrix.json", temporal_continuity(full_rows))
    write_json(artifact_dir / "support_contact_diagnostics.json", support_contact_diagnostics(full_rows))
    write_json(artifact_dir / "collision_proxy_diagnostics.json", collision_proxy_diagnostics(full_rows))
    write_json(artifact_dir / "pipeline_controls_reference.json", pipeline_controls_reference())
    write_json(artifact_dir / "solver_config.json", solver_config())
    write_json(artifact_dir / "pipeline_config.json", pipeline_config())
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
            "base_step4_0_final_head": BASE_STEP4_0_FINAL_HEAD,
            "source_code_commit": SOURCE_COMMIT,
            "quality_summary": summary,
            "quality_delta_vs_step4_0": qdelta,
            "orientation_delta_vs_step4_0": odelta,
            "orientation_policy_selection": policy,
            "deterministic_rerun": {"deterministic_compared_count": 44, "deterministic_matched_count": 44},
        },
    )
    return artifact_dir, baseline_dir, source_root


def model_rows(*, breakthrough: bool) -> list[dict]:
    rows: list[dict] = []
    selected_rotation = 1.65 if breakthrough else 2.91
    for index in range(32):
        rows.append(
            {
                "model_id": f"full_{index:02d}",
                "category": "full_humanoid_profile",
                "source_status": "passed",
                "runtime_quality_status": "runtime_quality_warned",
                "quality_classification": "runtime_quality_warned",
                "solver_backed_smoke_attempted": True,
                "solver_backed_smoke_completed": True,
                "solver_backed": True,
                "residual_only": False,
                "frame_count": 120,
                "clip_count_attempted": 4,
                "clip_count_completed": 4,
                "solver_clip_count_completed": 4,
                "trajectory_export_count": 4,
                "normalized_task_residual_mean": 0.90,
                "normalized_task_residual_p50": 0.95,
                "normalized_task_residual_p95": 0.99,
                "normalized_task_residual_max": 1.0,
                "raw_task_residual_mean": 3.0,
                "raw_task_residual_p50": 3.2,
                "raw_task_residual_p95": 3.55,
                "raw_task_residual_max": 4.0,
                "task_residual_mean": 3.0,
                "task_residual_p50": 3.2,
                "task_residual_p95": 3.55,
                "task_residual_max": 4.0,
                "translation_residual_mean": 1.0,
                "translation_residual_p95": 1.55,
                "translation_residual_max": 2.0,
                "rotation_residual_mean": selected_rotation * 0.8,
                "rotation_residual_p95": selected_rotation,
                "rotation_residual_max": selected_rotation + 0.1,
                "target_translation_error_mean": 0.0,
                "target_translation_error_p95": 0.0,
                "target_translation_error_max": 0.0,
                "target_rotation_error_mean": 0.0,
                "target_rotation_error_p95": 0.0,
                "target_rotation_error_max": 0.0,
                "joint_limit_violation_count": 0,
                "max_joint_limit_violation": 0.0,
                "output_nan_count": 0,
                "output_inf_count": 0,
                "task_anchor_count": 6,
                "task_anchor_semantic_counts": {"Chest": 2, "Hips": 2, "LeftHand": 1, "RightHand": 1},
                "task_coverage_ratio": 1.0,
                "successful_task_coverage_ratio": 1.0,
                "anchor_reliability_score": 1.0,
                "warning_reasons": ["high_task_residual", "rotation_residual_dominates"],
                "failure_or_warning_reasons": ["high_task_residual", "rotation_residual_dominates"],
                "runtime_quality_warning_reasons": ["high_task_residual", "rotation_residual_dominates"],
                "failure_reasons": [],
                "solver_config_hash": SOLVER_CONFIG_HASH,
                "pipeline_config_hash": PIPELINE_CONFIG_HASH,
            }
        )
    for index in range(3):
        rows.append(non_full_row(f"partial_{index:02d}", "partial_humanoid_profile", "partial_passed", "partial_runtime_passed"))
    for index in range(9):
        rows.append(non_full_row(f"negative_{index:02d}", "negative_control", "negative_control_passed", "negative_control_runtime_passed"))
    return rows


def non_full_row(model_id: str, category: str, source_status: str, runtime_status: str) -> dict:
    return {
        "model_id": model_id,
        "category": category,
        "source_status": source_status,
        "runtime_quality_status": runtime_status,
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


def baseline_summary() -> dict:
    return {
        "schema_version": 1,
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
        "rotation_dominant_residual_count": 27,
        "translation_dominant_residual_count": 5,
        "median_rotation_residual_p95": 2.50,
        "p95_rotation_residual_p95": 2.91,
        "max_rotation_residual_p95": 2.92,
        "median_raw_task_residual_p95": 3.55,
        "p95_raw_task_residual_p95": 4.26,
        "max_raw_task_residual_p95": 4.30,
        "median_normalized_task_residual_p95": 0.99,
        "p95_normalized_task_residual_p95": 0.999,
        "max_normalized_task_residual_p95": 1.0,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
        "release_candidate_status": "BLOCKED_RESIDUAL_QUALITY",
    }


def quality_summary(*, release_status: str, breakthrough: bool) -> dict:
    selected = 1.65 if breakthrough else 2.91
    return {
        **baseline_summary(),
        "base_step4_0_final_head": BASE_STEP4_0_FINAL_HEAD,
        "source_code_commit": SOURCE_COMMIT,
        "release_candidate_status": release_status,
        "primary_quality_breakthrough": breakthrough,
        "rotation_dominant_residual_count": 20 if breakthrough else 27,
        "translation_dominant_residual_count": 12 if breakthrough else 5,
        "median_rotation_residual_p95": selected,
        "p95_rotation_residual_p95": selected,
        "max_rotation_residual_p95": selected + 0.1,
        "normalization_reconstruction_mismatch_count": 0,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "trajectory_exports_count": 128,
        "temporal_continuity_finite_count": 128,
        "support_contact_diagnostic_count": 128,
        "collision_proxy_diagnostic_count": 128,
    }


def quality_delta(summary: dict, *, breakthrough: bool) -> dict:
    base = baseline_summary()
    baseline_counts = {key: int(value) for key, value in base.items() if isinstance(value, int)}
    current_counts = {key: int(summary.get(key, 0) or 0) for key in baseline_counts}
    return {
        "schema_version": 1,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": {key: current_counts[key] - baseline_counts[key] for key in baseline_counts},
        "metric_distribution_deltas": {
            "raw_task_residual_p95": metric_delta(4.26, 4.26),
            "normalized_task_residual_p95": metric_delta(0.999, 0.999),
        },
        "orientation_residual_deltas": orientation_delta(breakthrough=breakthrough),
        "normalization_deltas": {"raw_residual_regression_count": 0, "normalization_hides_raw_regression": False},
        "regressions": [],
        "improvements": ["p95_rotation_residual_p95_distribution_improved"] if breakthrough else [],
        "primary_quality_breakthrough": breakthrough,
        "verdict": "PASS_RC" if breakthrough else "BLOCKED",
    }


def orientation_delta(*, breakthrough: bool) -> dict:
    current = 1.65 if breakthrough else 2.91
    return {
        "schema_version": 1,
        "baseline_step4_0": {"median": 2.50, "p95": 2.91, "max": 2.92},
        "current_step4_1": {"median": current, "p95": current, "max": current + 0.1},
        "delta": {"median": current - 2.50, "p95": current - 2.91, "max": current + 0.1 - 2.92},
        "p95_rotation_residual_p95_delta": current - 2.91,
        "accepted_breakthrough": breakthrough,
        "raw_residual_regression_count": 0,
    }


def metric_delta(baseline: float, current: float) -> dict:
    delta = current - baseline
    return {
        "baseline": {"median": baseline, "p95": baseline, "max": baseline},
        "current": {"median": current, "p95": current, "max": current},
        "delta": {"median": delta, "p95": delta, "max": delta},
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
            "metrics": {"normalized_task_residual_p95": row["normalized_task_residual_p95"], "nan_count": 0, "inf_count": 0},
            "smoke_summary": {"metrics": {"normalized_task_residual_p95": row["normalized_task_residual_p95"], "nan_count": 0, "inf_count": 0}},
        }
        for row in full_rows
    ]


def clip_matrix(full_rows: list[dict]) -> dict:
    rows = [
        {"model_id": row["model_id"], "category": row["category"], "clip_id": clip, "target_stream_status": "passed", "solver_backed": True}
        for row in full_rows
        for clip in ("walk", "wave", "stretch", "pickup")
    ]
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def orientation_frame_semantics(full_rows: list[dict]) -> dict:
    rows = [
        {
            "model_id": row["model_id"],
            "clip_id": clip,
            "semantic_anchor": semantic,
            "runtime_body_name": f"{semantic}_body",
            "target_frame": "runtime_profile_target_world_frame",
            "runtime_frame": "runtime_model_site_world_frame",
            "source_frame": "BVH_source_semantic_world_frame",
            "quaternion_order": "xyzw",
            "sign_canonicalized": True,
            "rest_offset_policy": SELECTED_POLICY,
            "parent_relative": semantic != "Hips",
            "world_relative": semantic == "Hips",
            "axis_convention_notes": "right-handed z-up SO3 log-map",
            "validity_status": "valid",
            "warning_reasons": [],
        }
        for row in full_rows
        for clip in ("walk", "wave", "stretch", "pickup")
        for semantic in ("Hips", "Chest", "LeftHand", "RightHand")
    ]
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def orientation_math_audit(full_rows: list[dict], *, breakthrough: bool) -> dict:
    angle = 1.65 if breakthrough else 2.91
    rows = [
        {
            "model_id": row["model_id"],
            "clip_id": clip,
            "semantic_anchor": semantic,
            "q_runtime_xyzw": [0.0, 0.0, 0.0, 1.0],
            "q_target_xyzw": [0.0, 0.0, 0.0, 1.0],
            "q_delta_xyzw": [0.0, 0.0, 0.0, 1.0],
            "log_map_residual": [angle, 0.0, 0.0],
            "angle_radians": angle,
            "dominant_axis": "x",
            "finite": True,
        }
        for row in full_rows
        for clip in ("walk", "wave", "stretch", "pickup")
        for semantic in ("Hips", "Chest", "LeftHand", "RightHand")
    ]
    return {
        "schema_version": 1,
        "q_target_normalized": True,
        "q_runtime_normalized": True,
        "shortest_arc_sign_canonicalization": True,
        "row_count": len(rows),
        "finite_log_map_count": len(rows),
        "rows": rows,
    }


def orientation_offset_candidates(*, breakthrough: bool) -> dict:
    selected = 1.65 if breakthrough else 2.91
    candidates = [
        candidate("candidate_0_current_step4_policy", "world_no_offset_runtime_inv_target", 2.91),
        candidate("candidate_1_quaternion_sign_only", "world_no_offset_shortest_arc_sign_canonicalized", 2.91),
        candidate("candidate_2_world_delta_order_target_inv_runtime", "world_target_inv_runtime_log_order", 2.91),
        candidate("candidate_3_world_delta_order_runtime_inv_target", "world_runtime_inv_target_log_order", 2.91),
        candidate("candidate_4_parent_relative_delta", SELECTED_POLICY, selected),
        candidate("candidate_9_combined_global_best_policy", SELECTED_POLICY, selected),
    ]
    return {
        "schema_version": 1,
        "selected_policy": SELECTED_POLICY,
        "candidate_count": len(candidates),
        "candidate_policies": candidates,
        "row_count": len(candidates),
        "rows": candidates,
        "per_model_selected_rotation_residual_p95": {f"full_{index:02d}": selected for index in range(32)},
        "robot_specific_tuning_used": False,
    }


def candidate(candidate_id: str, policy: str, p95: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_policy": policy,
        "global_policy": True,
        "selection_eligible": True,
        "diagnostic_only": False,
        "sample_count": 128,
        "rotation_residual_distribution": {"mean": p95, "p50": p95, "p95": p95, "max": p95},
        "model_rotation_residual_p95_distribution": {"median": p95, "p95": p95, "max": p95},
        "robot_specific_tuning_used": False,
    }


def orientation_policy_selection(*, breakthrough: bool) -> dict:
    selected = 1.65 if breakthrough else 2.91
    return {
        "schema_version": 1,
        "candidate_policies": orientation_offset_candidates(breakthrough=breakthrough)["candidate_policies"],
        "global_selected_policy": {
            "candidate_id": "candidate_4_parent_relative_delta",
            "policy": SELECTED_POLICY,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning_used": False,
        },
        "selection_reason": "global parent-relative frame semantics reduce orientation residual",
        "why_policy_is_global": "same semantic parent map for all rows",
        "raw_residual_delta": metric_delta(4.26, 4.26),
        "rotation_residual_delta": {"delta_vs_step4_0": selected - 2.91},
        "pass_gate_impact": {"runtime_quality_gates_changed": False, "primary_quality_breakthrough_by_orientation_delta": breakthrough},
        "rejected_candidates": [],
        "risk_notes": ["diagnostic orientation residual only"],
        "raw_residual_regression_count": 0,
        "robot_specific_tuning_used": False,
    }


def orientation_clip_consistency(full_rows: list[dict], *, breakthrough: bool) -> dict:
    selected = 1.65 if breakthrough else 2.91
    rows = [
        {
            "model_id": row["model_id"],
            "clip_id": clip,
            "dominant_semantic_anchor": "LeftHand",
            "dominant_axis": "x",
            "selected_policy": SELECTED_POLICY,
            "selected_rotation_residual_distribution": {"mean": selected, "p50": selected, "p95": selected, "max": selected},
            "frame_convention_candidate_scores": {},
            "clip_level_status": "orientation_residual_improved" if breakthrough else "orientation_residual_unchanged",
        }
        for row in full_rows
        for clip in ("walk", "wave", "stretch", "pickup")
    ]
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def release_candidate_impact(summary: dict, qdelta: dict, odelta: dict, policy: dict) -> dict:
    return {
        "schema_version": 1,
        "release_candidate_status": summary["release_candidate_status"],
        "primary_quality_breakthrough": summary["primary_quality_breakthrough"],
        "quality_delta_vs_step4_0": qdelta,
        "orientation_delta_vs_step4_0": odelta,
        "orientation_policy_selection": policy["global_selected_policy"],
    }


def normalization_audit() -> dict:
    return {
        "schema_version": 1,
        "normalization_hides_raw_residual_regression": False,
        "denominator_inflation_detected": False,
        "orientation_policy_changes_normalization": False,
        "raw_residual_always_retained": True,
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


def pipeline_controls_reference() -> dict:
    return {"schema_version": 1, "controls": {"shadow_noop_verified": True, "override_explicit_only": True}}


def solver_config() -> dict:
    return {
        "schema_version": 1,
        "base_step4_0_final_head": BASE_STEP4_0_FINAL_HEAD,
        "global_config": True,
        "robot_specific_tuning": False,
        "solver_config_hash": SOLVER_CONFIG_HASH,
        "global_orientation_residual_policy": {"selected_policy": SELECTED_POLICY},
    }


def pipeline_config() -> dict:
    return {
        "schema_version": 1,
        "base_step4_0_final_head": BASE_STEP4_0_FINAL_HEAD,
        "global_config": True,
        "robot_specific_tuning": False,
        "pipeline_config_hash": PIPELINE_CONFIG_HASH,
        "config": {"enable_orientation_frame_semantics_audit": True},
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

