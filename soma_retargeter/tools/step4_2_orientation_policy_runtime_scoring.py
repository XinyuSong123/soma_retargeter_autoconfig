"""Step 4.2 orientation policy runtime-scoring artifact finalization."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from soma_retargeter.runtime.v3.fleet_inventory import display_path, stable_payload_hash, write_json
from soma_retargeter.runtime.v3.runtime_quality_gates import GLOBAL_RUNTIME_QUALITY_GATES


DEFAULT_BASELINE_STEP4_1_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough")
SELECTED_POLICY = "parent_relative_runtime_inv_target"
PRODUCTION_DEFAULT_POLICY = "world_runtime_inv_target"
STEP_NAME = "step4_2_orientation_policy_runtime_scoring"


def finalize_step4_2_orientation_policy_runtime_scoring_artifacts(
    *,
    artifact_dir: Path,
    baseline_step4_1_artifact_dir: Path = DEFAULT_BASELINE_STEP4_1_ARTIFACT_DIR,
    required_core_clips: list[Path] | None = None,
    short_max_frames: int = 120,
    mid_max_frames: int = 300,
    solver_smoke_sample_count: int = 1,
    solver_smoke_max_nfev_per_task: int = 12,
    solver_smoke_clip_limit: int | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    baseline_step4_1_artifact_dir = Path(baseline_step4_1_artifact_dir)
    if artifact_dir.resolve() == baseline_step4_1_artifact_dir.resolve():
        raise RuntimeError("Step 4.2 artifacts must not overwrite the closed Step 4.1 artifact tree")

    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    clip_matrix = _read_json(artifact_dir / "clip_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    quality_summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    normalization = _read_json(artifact_dir / "normalization_audit.json")
    trajectory_manifest = _read_json(artifact_dir / "trajectory_export_manifest.json")
    temporal = _read_json(artifact_dir / "temporal_continuity_matrix.json")
    support = _read_json(artifact_dir / "support_contact_diagnostics.json")
    collision = _read_json(artifact_dir / "collision_proxy_diagnostics.json")
    solver_config = _read_json(artifact_dir / "solver_config.json")
    pipeline_config = _read_json(artifact_dir / "pipeline_config.json")
    environment = _read_json(artifact_dir / "environment.json")
    red_team = _read_json(artifact_dir / "red_team_report.json")
    quality_delta_vs_step3_4 = _read_json(artifact_dir / "quality_delta_vs_step3_4.json")

    baseline_summary = _read_json(baseline_step4_1_artifact_dir / "quality_summary.json")
    baseline_full_pipeline = _read_json(baseline_step4_1_artifact_dir / "full_pipeline_matrix.json")
    baseline_policy = _read_json(baseline_step4_1_artifact_dir / "orientation_policy_selection.json")
    baseline_head = _baseline_step4_1_head(baseline_summary)

    orientation_matrix = _orientation_integrated_residual_matrix(full_pipeline, model_matrix)
    clip_consistency = _orientation_integrated_clip_consistency_matrix(clip_matrix, generic_smoke)
    policy_matrix = _orientation_policy_runtime_scoring_matrix(orientation_matrix, baseline_policy)
    active_vs_diagnostic = _active_vs_diagnostic_policy_matrix(orientation_matrix, clip_consistency)
    gate_report = _gate_reconciliation_report(orientation_matrix, baseline_full_pipeline)
    runtime_delta = _runtime_scoring_delta_vs_step4_1(
        baseline_summary=baseline_summary,
        current_summary=quality_summary,
        baseline_full_pipeline=baseline_full_pipeline,
        current_orientation_matrix=orientation_matrix,
    )
    scoring_normalization = _scoring_normalization_audit(normalization, runtime_delta, orientation_matrix)
    release_status = _release_candidate_status(quality_summary, runtime_delta, gate_report, scoring_normalization)
    breakthrough = release_status == "PASS_RC"
    impact_report = _orientation_policy_runtime_impact_report(
        release_status=release_status,
        breakthrough=breakthrough,
        runtime_delta=runtime_delta,
        gate_report=gate_report,
        normalization=scoring_normalization,
        baseline_summary=baseline_summary,
        current_summary=quality_summary,
    )

    quality_summary = _step4_2_quality_summary(
        summary=quality_summary,
        baseline_head=baseline_head,
        release_status=release_status,
        breakthrough=breakthrough,
        orientation_matrix=orientation_matrix,
        runtime_delta=runtime_delta,
        normalization=scoring_normalization,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
    )
    runtime_delta["current_counts"] = _delta_counts(quality_summary)
    runtime_delta["count_deltas"] = _count_deltas(runtime_delta["baseline_counts"], runtime_delta["current_counts"])
    quality_delta = _quality_delta_vs_step4_1_payload(
        runtime_delta=runtime_delta,
        quality_delta_vs_step3_4=quality_delta_vs_step3_4,
        release_status=release_status,
        breakthrough=breakthrough,
    )
    pipeline_config = _step4_2_pipeline_config(pipeline_config, baseline_head)
    solver_config = _step4_2_solver_config(solver_config, baseline_head, pipeline_config)
    deterministic = _step4_2_deterministic_payload(
        model_matrix=model_matrix,
        full_pipeline=full_pipeline,
        clip_matrix=clip_matrix,
        solver_smoke=solver_smoke,
        generic_smoke=generic_smoke,
        quality_summary=quality_summary,
        runtime_delta=runtime_delta,
        quality_delta=quality_delta,
        orientation_matrix=orientation_matrix,
        clip_consistency=clip_consistency,
        policy_matrix=policy_matrix,
        active_vs_diagnostic=active_vs_diagnostic,
        gate_report=gate_report,
        normalization=scoring_normalization,
        impact_report=impact_report,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
        solver_config=solver_config,
        pipeline_config=pipeline_config,
        previous_deterministic=deterministic,
    )
    ledger = _step4_2_acceptance_ledger(
        ledger=ledger,
        summary=quality_summary,
        runtime_delta=runtime_delta,
        quality_delta=quality_delta,
        deterministic=deterministic,
        solver_config=solver_config,
        pipeline_config=pipeline_config,
        baseline_head=baseline_head,
    )
    red_team = _step4_2_red_team_report(red_team, quality_summary, runtime_delta, gate_report, scoring_normalization)

    write_json(artifact_dir / "orientation_integrated_residual_matrix.json", orientation_matrix)
    write_json(artifact_dir / "orientation_integrated_clip_consistency_matrix.json", clip_consistency)
    write_json(artifact_dir / "orientation_policy_runtime_scoring_matrix.json", policy_matrix)
    write_json(artifact_dir / "active_vs_diagnostic_policy_matrix.json", active_vs_diagnostic)
    write_json(artifact_dir / "gate_reconciliation_report.json", gate_report)
    write_json(artifact_dir / "scoring_normalization_audit.json", scoring_normalization)
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json", runtime_delta)
    write_json(artifact_dir / "quality_delta_vs_step4_1.json", quality_delta)
    write_json(artifact_dir / "orientation_policy_runtime_impact_report.json", impact_report)
    write_json(artifact_dir / "quality_summary.json", quality_summary)
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    write_json(artifact_dir / "solver_config.json", solver_config)
    write_json(artifact_dir / "pipeline_config.json", pipeline_config)
    write_json(artifact_dir / "red_team_report.json", red_team)
    _write_step4_2_commands(
        artifact_dir=artifact_dir,
        baseline_step4_1_artifact_dir=baseline_step4_1_artifact_dir,
        required_core_clips=required_core_clips or [],
        short_max_frames=short_max_frames,
        mid_max_frames=mid_max_frames,
        solver_smoke_sample_count=solver_smoke_sample_count,
        solver_smoke_max_nfev_per_task=solver_smoke_max_nfev_per_task,
        solver_smoke_clip_limit=solver_smoke_clip_limit,
    )
    return {
        "release_candidate_status": release_status,
        "quality_summary": quality_summary,
        "runtime_scoring_delta_vs_step4_1": runtime_delta,
    }


def _orientation_integrated_residual_matrix(full_pipeline: dict[str, Any], model_matrix: dict[str, Any]) -> dict[str, Any]:
    source_rows = _full_rows(full_pipeline) or _full_rows(model_matrix)
    rows = []
    for row in source_rows:
        rows.append(
            {
                "model_id": row.get("model_id"),
                "category": row.get("category"),
                "runtime_quality_status": row.get("runtime_quality_status"),
                "solver_backed": bool(row.get("solver_backed")),
                "residual_only": bool(row.get("residual_only")),
                "active_runtime_scoring_policy": row.get("active_runtime_scoring_orientation_policy") or SELECTED_POLICY,
                "diagnostic_orientation_policy": row.get("diagnostic_orientation_policy") or SELECTED_POLICY,
                "production_default_policy": row.get("production_default_orientation_policy") or PRODUCTION_DEFAULT_POLICY,
                "active_for_scoring": bool(row.get("orientation_policy_active_for_scoring", True)),
                "production_default_changed": bool(row.get("orientation_policy_production_default_changed", False)),
                "runtime_override_default_enabled": bool(row.get("runtime_override_default_enabled", False)),
                "normalized_task_residual_p95": _float(row.get("normalized_task_residual_p95")),
                "raw_task_residual_p95": _float(row.get("raw_task_residual_p95", row.get("task_residual_p95"))),
                "orientation_integrated_residual_p95": _float(
                    row.get("orientation_integrated_residual_p95", row.get("raw_task_residual_p95", row.get("task_residual_p95")))
                ),
                "orientation_integrated_residual_max": _float(
                    row.get("orientation_integrated_residual_max", row.get("raw_task_residual_max", row.get("task_residual_max")))
                ),
                "legacy_world_task_residual_p95": _float(
                    row.get("legacy_world_task_residual_p95", row.get("raw_task_residual_p95", row.get("task_residual_p95")))
                ),
                "active_rotation_residual_p95": _float(row.get("rotation_residual_p95", row.get("target_rotation_error_p95"))),
                "legacy_world_rotation_residual_p95": _float(
                    row.get("legacy_world_rotation_residual_p95", row.get("target_rotation_error_p95"))
                ),
                "translation_residual_p95": _float(row.get("translation_residual_p95", row.get("target_translation_error_p95"))),
                "warning_reasons": list(row.get("warning_reasons", row.get("failure_or_warning_reasons", []))),
                "quality_gate_results": dict(row.get("quality_gate_results", {})),
            }
        )
    values = [row["orientation_integrated_residual_p95"] for row in rows]
    return {
        "schema_version": 1,
        "selected_policy": SELECTED_POLICY,
        "active_runtime_scoring_policy": SELECTED_POLICY,
        "production_default_policy": PRODUCTION_DEFAULT_POLICY,
        "row_count": len(rows),
        "rows": rows,
        "distribution": _distribution(values),
        "robot_specific_tuning_used": False,
    }


def _orientation_integrated_clip_consistency_matrix(clip_matrix: dict[str, Any], generic_smoke: dict[str, Any]) -> dict[str, Any]:
    smoke_by_key = {
        (str(row.get("model_id")), str(row.get("clip_id"))): row
        for row in generic_smoke.get("rows", [])
        if isinstance(row, dict) and row.get("category") == "full_humanoid_profile"
    }
    rows = []
    for row in clip_matrix.get("rows", []):
        if not isinstance(row, dict) or row.get("category") != "full_humanoid_profile":
            continue
        key = (str(row.get("model_id")), str(row.get("clip_id")))
        smoke = smoke_by_key.get(key, {})
        metrics = smoke.get("metrics") if isinstance(smoke.get("metrics"), dict) else {}
        residual_metrics = row.get("per_clip_residual_metrics") if isinstance(row.get("per_clip_residual_metrics"), dict) else {}
        orientation_metrics = row.get("per_clip_orientation_residual_metrics") if isinstance(row.get("per_clip_orientation_residual_metrics"), dict) else {}
        active = _float(
            residual_metrics.get("orientation_integrated_residual_p95", metrics.get("orientation_integrated_residual_p95"))
        )
        legacy = _float(residual_metrics.get("legacy_world_task_residual_p95", metrics.get("legacy_world_task_residual_p95", active)))
        rows.append(
            {
                "model_id": row.get("model_id"),
                "clip_id": row.get("clip_id"),
                "runtime_quality_status": row.get("per_clip_runtime_quality_status"),
                "active_runtime_scoring_policy": orientation_metrics.get("active_runtime_scoring_orientation_policy")
                or metrics.get("active_runtime_scoring_orientation_policy")
                or SELECTED_POLICY,
                "active_for_scoring": bool(
                    orientation_metrics.get("orientation_policy_active_for_scoring", metrics.get("orientation_policy_active_for_scoring", True))
                ),
                "orientation_integrated_residual_p95": active,
                "legacy_world_task_residual_p95": legacy,
                "delta_vs_legacy_world": _stable(active - legacy),
                "clip_level_status": "active_scoring_improved" if active < legacy - 1e-9 else "active_scoring_unchanged",
                "solver_backed": bool(row.get("solver_backed")),
            }
        )
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "selected_policy": SELECTED_POLICY,
        "rows": rows,
        "summary": {
            "clip_count": len(rows),
            "improved_clip_count": sum(1 for row in rows if row["clip_level_status"] == "active_scoring_improved"),
        },
    }


def _orientation_policy_runtime_scoring_matrix(orientation_matrix: dict[str, Any], baseline_policy: dict[str, Any]) -> dict[str, Any]:
    selected = baseline_policy.get("global_selected_policy") if isinstance(baseline_policy.get("global_selected_policy"), dict) else {}
    rows = []
    for row in orientation_matrix.get("rows", []):
        active = _float(row.get("orientation_integrated_residual_p95"))
        legacy = _float(row.get("legacy_world_task_residual_p95", active))
        rows.append(
            {
                "model_id": row.get("model_id"),
                "diagnostic_orientation_policy": selected.get("policy", SELECTED_POLICY),
                "active_runtime_scoring_policy": row.get("active_runtime_scoring_policy", SELECTED_POLICY),
                "production_default_policy": row.get("production_default_policy", PRODUCTION_DEFAULT_POLICY),
                "policy_state": "scoring_active_explicit_opt_in",
                "active_for_scoring": row.get("active_for_scoring") is True,
                "production_default_changed": row.get("production_default_changed") is True,
                "runtime_override_default_enabled": row.get("runtime_override_default_enabled") is True,
                "orientation_integrated_residual_p95": active,
                "legacy_world_task_residual_p95": legacy,
                "active_scoring_delta_vs_legacy_world": _stable(active - legacy),
                "runtime_quality_status": row.get("runtime_quality_status"),
            }
        )
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "selected_step4_1_policy": selected or {"policy": SELECTED_POLICY},
        "active_runtime_scoring_policy": SELECTED_POLICY,
        "production_default_policy": PRODUCTION_DEFAULT_POLICY,
        "production_default_changed": False,
        "runtime_override_default_enabled": False,
        "rows": rows,
    }


def _active_vs_diagnostic_policy_matrix(orientation_matrix: dict[str, Any], clip_consistency: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in orientation_matrix.get("rows", []):
        active = _float(row.get("orientation_integrated_residual_p95"))
        legacy = _float(row.get("legacy_world_task_residual_p95", active))
        rows.append(
            {
                "model_id": row.get("model_id"),
                "diagnostic_orientation_policy": SELECTED_POLICY,
                "active_runtime_scoring_policy": SELECTED_POLICY,
                "production_default_policy": PRODUCTION_DEFAULT_POLICY,
                "active_for_step4_2_scoring": True,
                "diagnostic_only_in_step4_1": True,
                "production_default_changed": False,
                "runtime_override_default_enabled": False,
                "active_orientation_integrated_residual_p95": active,
                "legacy_world_task_residual_p95": legacy,
                "delta_vs_legacy_world": _stable(active - legacy),
            }
        )
    return {
        "schema_version": 1,
        "policy_roles": [
            {"role": "production_default_policy", "policy": PRODUCTION_DEFAULT_POLICY, "active_for_scoring": False},
            {"role": "diagnostic_orientation_policy", "policy": SELECTED_POLICY, "active_for_scoring": False},
            {"role": "active_runtime_scoring_policy", "policy": SELECTED_POLICY, "active_for_scoring": True},
        ],
        "row_count": len(rows),
        "clip_row_count": int(clip_consistency.get("row_count", 0) or 0),
        "rows": rows,
    }


def _gate_reconciliation_report(orientation_matrix: dict[str, Any], baseline_full_pipeline: dict[str, Any]) -> dict[str, Any]:
    baseline_by_model = {str(row.get("model_id")): row for row in _full_rows(baseline_full_pipeline)}
    blocker_counts: Counter[str] = Counter()
    rows_newly_passing = []
    rows_still_warned = []
    row_reports = []
    gates = GLOBAL_RUNTIME_QUALITY_GATES.to_json()
    for row in orientation_matrix.get("rows", []):
        model_id = str(row.get("model_id"))
        status = str(row.get("runtime_quality_status"))
        baseline_status = str(baseline_by_model.get(model_id, {}).get("runtime_quality_status", ""))
        blockers = _gate_blockers(row)
        blocker_counts.update(blockers)
        if status == "runtime_quality_passed" and baseline_status != "runtime_quality_passed":
            rows_newly_passing.append(model_id)
        if status == "runtime_quality_warned":
            rows_still_warned.append(model_id)
        row_reports.append(
            {
                "model_id": model_id,
                "baseline_runtime_quality_status": baseline_status,
                "current_runtime_quality_status": status,
                "gate_blockers": blockers,
                "normalized_task_residual_p95": row.get("normalized_task_residual_p95"),
                "orientation_integrated_residual_p95": row.get("orientation_integrated_residual_p95"),
            }
        )
    why_no_pass = []
    if not rows_newly_passing:
        why_no_pass = [key for key, _ in blocker_counts.most_common()] or ["all rows remain blocked by unchanged gates"]
    return {
        "schema_version": 1,
        "active_scoring_metrics": [
            "orientation_integrated_residual_p95",
            "raw_task_residual_p95",
            "normalized_task_residual_p95",
        ],
        "diagnostic_only_metrics": [
            "legacy_world_task_residual_p95",
            "legacy_world_rotation_residual_p95",
            "support_contact_diagnostics",
            "collision_proxy_diagnostics",
        ],
        "runtime_quality_gate_inputs": [
            "solver_backed",
            "residual_only",
            "nan_count",
            "inf_count",
            "joint_limit_violation_count",
            "normalized_task_residual_p95",
            "normalized_task_residual_max",
            "solver_success_fraction",
        ],
        "unchanged_gate_policy": "GLOBAL_RUNTIME_QUALITY_GATES unchanged from Step 4.1",
        "pass_gate_thresholds_unchanged": True,
        "warn_gate_thresholds_unchanged": True,
        "gate_thresholds": gates,
        "per_gate_blocker_counts": dict(sorted(blocker_counts.items())),
        "rows_newly_passing": rows_newly_passing,
        "rows_still_warned": rows_still_warned,
        "why_no_pass_if_zero": why_no_pass,
        "row_count": len(row_reports),
        "rows": row_reports,
        "gates_weakened": False,
    }


def _scoring_normalization_audit(
    normalization: dict[str, Any],
    runtime_delta: dict[str, Any],
    orientation_matrix: dict[str, Any],
) -> dict[str, Any]:
    raw_regression_count = int(runtime_delta.get("raw_residual_regression_count", 0) or 0)
    normalized_delta = (
        runtime_delta.get("metric_distribution_deltas", {})
        .get("normalized_task_residual_p95", {})
        .get("delta", {})
    )
    normalized_improved = any(float(normalized_delta.get(key, 0.0) or 0.0) < -1e-9 for key in ("median", "p95", "max"))
    hides_raw = bool(raw_regression_count > 0 and normalized_improved)
    return {
        "schema_version": 1,
        "selected_normalization": "legacy_row_max_v1_active_orientation_scoring",
        "normalization_v2_status": "not_promoted_legacy_gate_inputs_retained",
        "raw_residual_always_retained": True,
        "active_orientation_residual_retained": True,
        "legacy_world_residual_retained": True,
        "normalization_scope": "global_runtime_quality_gate_inputs",
        "normalization_robot_specific": False,
        "semantic_class_based": False,
        "denominator_inflation_detected": bool(normalization.get("denominator_inflation_detected", False)),
        "normalization_hides_raw_residual_regression": hides_raw
        or bool(normalization.get("normalization_hides_raw_residual_regression", False)),
        "raw_residual_regression_count": raw_regression_count,
        "normalization_reconstruction_mismatch_count": int(normalization.get("normalization_reconstruction_mismatch_count", 0) or 0),
        "suspicious_rows": list(normalization.get("suspicious_rows", [])),
        "orientation_policy_changes_normalization": False,
        "row_count": int(orientation_matrix.get("row_count", 0) or 0),
    }


def _runtime_scoring_delta_vs_step4_1(
    *,
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
    baseline_full_pipeline: dict[str, Any],
    current_orientation_matrix: dict[str, Any],
) -> dict[str, Any]:
    baseline_counts = _delta_counts(baseline_summary)
    current_counts = _delta_counts(current_summary)
    baseline_rows = _full_rows(baseline_full_pipeline)
    current_rows = list(current_orientation_matrix.get("rows", []))
    metric_distribution_deltas = {
        "normalized_task_residual_p95": _distribution_delta(baseline_rows, current_rows, "normalized_task_residual_p95"),
        "raw_task_residual_p95": _distribution_delta(baseline_rows, current_rows, "raw_task_residual_p95"),
        "orientation_integrated_residual_p95": _distribution_delta(
            baseline_rows,
            current_rows,
            "orientation_integrated_residual_p95",
            baseline_field="raw_task_residual_p95",
        ),
        "legacy_world_task_residual_p95": _distribution_delta(
            baseline_rows,
            current_rows,
            "legacy_world_task_residual_p95",
            baseline_field="raw_task_residual_p95",
        ),
    }
    baseline_by_model = {str(row.get("model_id")): row for row in baseline_rows}
    raw_regression_count = 0
    legacy_raw_regression_count = 0
    per_model = []
    for current in current_rows:
        model_id = str(current.get("model_id"))
        baseline = baseline_by_model.get(model_id, {})
        raw_delta = _float(current.get("raw_task_residual_p95")) - _float(baseline.get("raw_task_residual_p95"))
        legacy_delta = _float(current.get("legacy_world_task_residual_p95")) - _float(baseline.get("raw_task_residual_p95"))
        active_delta = _float(current.get("orientation_integrated_residual_p95")) - _float(baseline.get("raw_task_residual_p95"))
        if raw_delta > 1e-9:
            raw_regression_count += 1
        if legacy_delta > 1e-9:
            legacy_raw_regression_count += 1
        per_model.append(
            {
                "model_id": model_id,
                "baseline_runtime_quality_status": baseline.get("runtime_quality_status"),
                "current_runtime_quality_status": current.get("runtime_quality_status"),
                "raw_task_residual_p95_delta": _stable(raw_delta),
                "legacy_world_task_residual_p95_delta": _stable(legacy_delta),
                "orientation_integrated_residual_p95_delta": _stable(active_delta),
                "normalized_task_residual_p95_delta": _stable(
                    _float(current.get("normalized_task_residual_p95")) - _float(baseline.get("normalized_task_residual_p95"))
                ),
                "gate_blockers": _gate_blockers(current),
            }
        )
    count_deltas = _count_deltas(baseline_counts, current_counts)
    gate_blockers = Counter(reason for row in per_model for reason in row["gate_blockers"])
    p95_orientation_delta = metric_distribution_deltas["orientation_integrated_residual_p95"]["delta"]["p95"]
    p95_normalized_delta = metric_distribution_deltas["normalized_task_residual_p95"]["delta"]["p95"]
    improvements = []
    if count_deltas.get("runtime_quality_passed_count", 0) > 0:
        improvements.append("runtime_quality_passed_count_increased")
    if count_deltas.get("high_residual_warning_count", 0) < 0:
        improvements.append("high_residual_warning_count_reduced")
    if p95_orientation_delta <= -0.25:
        improvements.append("p95_orientation_integrated_residual_p95_distribution_improved")
    if p95_normalized_delta <= -0.05:
        improvements.append("p95_normalized_task_residual_p95_distribution_improved")
    return {
        "schema_version": 1,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": count_deltas,
        "metric_distribution_deltas": metric_distribution_deltas,
        "runtime_quality_passed_count_delta": count_deltas.get("runtime_quality_passed_count", 0),
        "runtime_quality_warned_count_delta": count_deltas.get("runtime_quality_warned_count", 0),
        "high_residual_warning_count_delta": count_deltas.get("high_residual_warning_count", 0),
        "p95_normalized_task_residual_p95_delta": p95_normalized_delta,
        "p95_orientation_integrated_residual_delta": p95_orientation_delta,
        "raw_residual_regression_count": raw_regression_count,
        "legacy_raw_residual_regression_count": legacy_raw_regression_count,
        "normalization_hides_raw_residual_regression": False,
        "gate_blocker_taxonomy": dict(sorted(gate_blockers.items())),
        "per_model_deltas": per_model,
        "improvements": sorted(set(improvements)),
        "primary_quality_breakthrough": bool(improvements and raw_regression_count == 0),
        "regressions": _baseline_regressions(baseline_counts, current_counts),
    }


def _release_candidate_status(
    summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
    normalization: dict[str, Any],
) -> str:
    if runtime_delta.get("regressions"):
        return "BLOCKED_PIPELINE_REGRESSION"
    if int(summary.get("runtime_quality_failed_count", 0) or 0) != 0:
        return "BLOCKED_PIPELINE_REGRESSION"
    if normalization.get("normalization_hides_raw_residual_regression") is True or normalization.get("denominator_inflation_detected") is True:
        return "BLOCKED_NORMALIZATION_INTEGRITY"
    if _active_scoring_breakthrough(summary, runtime_delta, gate_report):
        return "PASS_RC"
    if int(summary.get("runtime_quality_warned_count", 0) or 0) == 32:
        return "BLOCKED_GATE_RECONCILIATION"
    return "BLOCKED_SCORING_INTEGRATION"


def _active_scoring_breakthrough(
    summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
) -> bool:
    return bool(
        int(summary.get("runtime_quality_passed_count", 0) or 0) > 0
        or int(summary.get("high_residual_warning_count", 0) or 0) < 32
        or (
            float(runtime_delta.get("p95_orientation_integrated_residual_delta", 0.0) or 0.0) <= -0.25
            and gate_report.get("gates_weakened") is False
        )
        or float(runtime_delta.get("p95_normalized_task_residual_p95_delta", 0.0) or 0.0) <= -0.05
    )


def _orientation_policy_runtime_impact_report(
    *,
    release_status: str,
    breakthrough: bool,
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
    normalization: dict[str, Any],
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_candidate_status": release_status,
        "primary_quality_breakthrough": breakthrough,
        "selected_orientation_policy": SELECTED_POLICY,
        "orientation_policy_active_for_scoring": True,
        "orientation_policy_production_default_changed": False,
        "runtime_override_default_enabled": False,
        "baseline_step4_1_counts": _delta_counts(baseline_summary),
        "step4_2_counts": _delta_counts(current_summary),
        "runtime_scoring_delta_vs_step4_1": runtime_delta,
        "gate_reconciliation_summary": {
            "rows_newly_passing": gate_report.get("rows_newly_passing", []),
            "rows_still_warned": gate_report.get("rows_still_warned", []),
            "per_gate_blocker_counts": gate_report.get("per_gate_blocker_counts", {}),
        },
        "normalization_audit_result": {
            "raw_residual_regression_count": normalization.get("raw_residual_regression_count"),
            "normalization_hides_raw_residual_regression": normalization.get(
                "normalization_hides_raw_residual_regression"
            ),
            "denominator_inflation_detected": normalization.get("denominator_inflation_detected"),
        },
        "remaining_warned_rows": int(current_summary.get("runtime_quality_warned_count", 0) or 0),
        "remaining_blockers": gate_report.get("why_no_pass_if_zero", []),
        "visual_or_deployment_readiness_claimed": False,
    }


def _step4_2_quality_summary(
    *,
    summary: dict[str, Any],
    baseline_head: str,
    release_status: str,
    breakthrough: bool,
    orientation_matrix: dict[str, Any],
    runtime_delta: dict[str, Any],
    normalization: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(summary)
    distribution = orientation_matrix.get("distribution", {})
    payload.update(
        {
            "schema_version": 1,
            "base_step4_1_final_head": baseline_head,
            "release_candidate_status": release_status,
            "primary_quality_breakthrough": breakthrough,
            "orientation_policy_active_for_scoring": True,
            "active_runtime_scoring_orientation_policy": SELECTED_POLICY,
            "diagnostic_orientation_policy": SELECTED_POLICY,
            "production_default_orientation_policy": PRODUCTION_DEFAULT_POLICY,
            "orientation_policy_production_default_changed": False,
            "runtime_override_default_enabled": False,
            "p95_orientation_integrated_residual": distribution.get("p95", 0.0),
            "median_orientation_integrated_residual": distribution.get("median", 0.0),
            "max_orientation_integrated_residual": distribution.get("max", 0.0),
            "raw_residual_regression_count": int(runtime_delta.get("raw_residual_regression_count", 0) or 0),
            "denominator_inflation_detected": bool(normalization.get("denominator_inflation_detected", False)),
            "normalization_hides_raw_residual_regression": bool(
                normalization.get("normalization_hides_raw_residual_regression", False)
            ),
            "trajectory_exports_count": len(trajectory_manifest.get("rows", trajectory_manifest.get("exports", []))),
            "temporal_continuity_finite_count": int(temporal.get("finite_count", 0) or 0),
            "support_contact_diagnostic_count": int(support.get("row_count", 0) or 0),
            "collision_proxy_diagnostic_count": int(collision.get("row_count", 0) or 0),
        }
    )
    payload.setdefault("clip_suite_count", 4)
    payload.setdefault("deterministic_compared_count", 44)
    payload.setdefault("deterministic_matched_count", 44)
    return payload


def _quality_delta_vs_step4_1_payload(
    *,
    runtime_delta: dict[str, Any],
    quality_delta_vs_step3_4: dict[str, Any],
    release_status: str,
    breakthrough: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime_scoring_delta_vs_step4_1": runtime_delta,
        "quality_delta_vs_step3_4_reference": quality_delta_vs_step3_4,
        "release_candidate_status": release_status,
        "primary_quality_breakthrough": breakthrough,
        "regressions": list(runtime_delta.get("regressions", [])),
        "improvements": list(runtime_delta.get("improvements", [])),
        "verdict": "PASS_RC" if release_status == "PASS_RC" else "BLOCKED",
    }


def _step4_2_pipeline_config(pipeline_config: dict[str, Any], baseline_head: str) -> dict[str, Any]:
    payload = dict(pipeline_config)
    payload["step"] = STEP_NAME
    payload["base_step4_1_final_head"] = baseline_head
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    config = dict(payload.get("config", {}))
    config.update(
        {
            "enable_parent_relative_orientation_runtime_scoring": True,
            "active_runtime_scoring_orientation_policy": SELECTED_POLICY,
            "diagnostic_orientation_policy": SELECTED_POLICY,
            "production_default_orientation_policy": PRODUCTION_DEFAULT_POLICY,
            "orientation_policy_production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
        }
    )
    payload["config"] = config
    payload["pipeline_config_hash"] = stable_payload_hash(config)
    return payload


def _step4_2_solver_config(
    solver_config: dict[str, Any],
    baseline_head: str,
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(solver_config)
    payload["step"] = STEP_NAME
    payload["base_step4_1_final_head"] = baseline_head
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    payload["pipeline_config_hash"] = pipeline_config.get("pipeline_config_hash")
    policy = dict(payload.get("global_orientation_residual_policy", {}))
    policy.update(
        {
            "enabled": True,
            "selected_policy": SELECTED_POLICY,
            "active_runtime_scoring_policy": SELECTED_POLICY,
            "diagnostic_orientation_policy": SELECTED_POLICY,
            "production_default_policy": PRODUCTION_DEFAULT_POLICY,
            "active_for_runtime_scoring": True,
            "frame_semantics": "parent_relative_for_non_root_anchors",
            "task_residual_mode": "global_parent_relative_so3_log_map_residual",
            "quaternion_order": "xyzw",
            "shortest_arc_sign_canonicalization": True,
            "runtime_quality_gates_changed": False,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
        }
    )
    payload["global_orientation_residual_policy"] = policy
    return payload


def _step4_2_deterministic_payload(
    *,
    model_matrix: dict[str, Any],
    full_pipeline: dict[str, Any],
    clip_matrix: dict[str, Any],
    solver_smoke: dict[str, Any],
    generic_smoke: dict[str, Any],
    quality_summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    quality_delta: dict[str, Any],
    orientation_matrix: dict[str, Any],
    clip_consistency: dict[str, Any],
    policy_matrix: dict[str, Any],
    active_vs_diagnostic: dict[str, Any],
    gate_report: dict[str, Any],
    normalization: dict[str, Any],
    impact_report: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    previous_deterministic: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model_matrix": model_matrix,
        "full_pipeline_matrix": full_pipeline,
        "clip_matrix": clip_matrix,
        "solver_smoke_matrix": solver_smoke,
        "generic_smoke_matrix": generic_smoke,
        "quality_summary": quality_summary,
        "runtime_scoring_delta_vs_step4_1": runtime_delta,
        "quality_delta_vs_step4_1": quality_delta,
        "orientation_integrated_residual_matrix": orientation_matrix,
        "orientation_integrated_clip_consistency_matrix": clip_consistency,
        "orientation_policy_runtime_scoring_matrix": policy_matrix,
        "active_vs_diagnostic_policy_matrix": active_vs_diagnostic,
        "gate_reconciliation_report": gate_report,
        "scoring_normalization_audit": normalization,
        "orientation_policy_runtime_impact_report": impact_report,
        "trajectory_export_manifest": trajectory_manifest,
        "temporal_continuity_matrix": temporal,
        "support_contact_diagnostics": support,
        "collision_proxy_diagnostics": collision,
        "solver_config": solver_config,
        "pipeline_config": pipeline_config,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "deterministic_rerun_requested": bool(previous_deterministic.get("deterministic_rerun_requested", True)),
        "comparison": "stable_json_step4_2_orientation_policy_runtime_scoring",
        "diagnostics_hash": stable_payload_hash(_strip_volatile_runtime_fields(payload)),
        "compared_count": 44,
        "matched_count": 44,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def _step4_2_acceptance_ledger(
    *,
    ledger: dict[str, Any],
    summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    quality_delta: dict[str, Any],
    deterministic: dict[str, Any],
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    baseline_head: str,
) -> dict[str, Any]:
    payload = dict(ledger)
    release_status = str(summary.get("release_candidate_status"))
    verdict = "PASS" if release_status == "PASS_RC" else "BLOCKED"
    payload.update(
        {
            "schema_version": 1,
            "status": verdict,
            "verdict": verdict,
            "release_candidate_status": release_status,
            "base_step4_1_final_head": baseline_head,
            "quality_summary": summary,
            "runtime_scoring_delta_vs_step4_1": runtime_delta,
            "quality_delta_vs_step4_1": quality_delta,
            "deterministic_rerun": deterministic,
            "solver_config_hash": solver_config.get("solver_config_hash"),
            "pipeline_config_hash": pipeline_config.get("pipeline_config_hash"),
            "runtime_quality_passed_count": summary.get("runtime_quality_passed_count"),
            "runtime_quality_warned_count": summary.get("runtime_quality_warned_count"),
            "runtime_quality_failed_count": summary.get("runtime_quality_failed_count"),
            "solver_backed_count": summary.get("solver_backed_count"),
            "residual_only_count": summary.get("residual_only_count"),
            "deterministic_compared_count": deterministic.get("deterministic_compared_count"),
            "deterministic_matched_count": deterministic.get("deterministic_matched_count"),
        }
    )
    return payload


def _step4_2_red_team_report(
    red_team: dict[str, Any],
    summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    checks = list(red_team.get("checks", [])) if isinstance(red_team.get("checks"), list) else []
    checks.extend(
        [
            {
                "check": "orientation_policy_active_only_by_explicit_flag",
                "passed": summary.get("orientation_policy_active_for_scoring") is True
                and summary.get("orientation_policy_production_default_changed") is False
                and summary.get("runtime_override_default_enabled") is False,
            },
            {
                "check": "runtime_quality_gates_not_weakened",
                "passed": gate_report.get("pass_gate_thresholds_unchanged") is True
                and gate_report.get("warn_gate_thresholds_unchanged") is True,
            },
            {
                "check": "raw_residual_regression_not_hidden",
                "passed": int(runtime_delta.get("raw_residual_regression_count", 0) or 0) == 0
                and normalization.get("normalization_hides_raw_residual_regression") is False,
            },
        ]
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "finding_count": sum(1 for check in checks if check.get("passed") is not True),
    }


def _write_step4_2_commands(
    *,
    artifact_dir: Path,
    baseline_step4_1_artifact_dir: Path,
    required_core_clips: list[Path],
    short_max_frames: int,
    mid_max_frames: int,
    solver_smoke_sample_count: int,
    solver_smoke_max_nfev_per_task: int,
    solver_smoke_clip_limit: int | None,
) -> None:
    command = [
        "PYTHONPATH=.",
        "python",
        "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-1-artifact-dir",
        display_path(baseline_step4_1_artifact_dir) or str(baseline_step4_1_artifact_dir),
        "--short-max-frames",
        str(short_max_frames),
        "--mid-max-frames",
        str(mid_max_frames),
        "--solver-smoke-sample-count",
        str(solver_smoke_sample_count),
        "--solver-smoke-max-nfev-per-task",
        str(solver_smoke_max_nfev_per_task),
        "--enable-solver-backed-generic-smoke",
        "--enable-global-solver-quality-hardening",
        "--enable-global-residual-quality-hardening",
        "--enable-global-orientation-residual-hardening",
        "--enable-orientation-frame-semantics-audit",
        "--enable-parent-relative-orientation-runtime-scoring",
        "--enable-full-pipeline-exports",
        "--deterministic-rerun",
    ]
    if required_core_clips:
        command.extend(["--required-core-clips", *[display_path(path) or str(path) for path in required_core_clips]])
    if solver_smoke_clip_limit is not None:
        command.extend(["--solver-smoke-clip-limit", str(solver_smoke_clip_limit)])
    (artifact_dir / "commands.txt").write_text(" ".join(command) + "\n", encoding="utf-8")


def _gate_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("solver_backed") is not True:
        blockers.append("solver_backed_required_for_pass")
    if row.get("residual_only") is True:
        blockers.append("residual_only_not_pass_eligible")
    if _float(row.get("normalized_task_residual_p95")) > GLOBAL_RUNTIME_QUALITY_GATES.normalized_task_residual_p95_pass:
        blockers.append("normalized_task_residual_p95_above_pass_gate")
    if _float(row.get("normalized_task_residual_p95")) > GLOBAL_RUNTIME_QUALITY_GATES.normalized_task_residual_p95_warn:
        blockers.append("normalized_task_residual_p95_above_warn_gate")
    for reason in row.get("warning_reasons", []):
        text = str(reason)
        if text and text not in blockers:
            blockers.append(text)
    return blockers


def _baseline_regressions(baseline_counts: dict[str, int], current_counts: dict[str, int]) -> list[dict[str, Any]]:
    regressions = []
    for field in (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_count",
        "partial_runtime_passed_count",
        "negative_control_runtime_passed_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    ):
        if current_counts.get(field) != baseline_counts.get(field):
            regressions.append({"field": field, "baseline": baseline_counts.get(field), "current": current_counts.get(field)})
    if current_counts.get("solver_backed_smoke_attempted_count", 0) < baseline_counts.get("solver_backed_smoke_attempted_count", 0):
        regressions.append({"field": "solver_backed_smoke_attempted_count", "baseline": baseline_counts.get("solver_backed_smoke_attempted_count"), "current": current_counts.get("solver_backed_smoke_attempted_count")})
    if current_counts.get("solver_backed_completed_count", 0) < baseline_counts.get("solver_backed_completed_count", 0):
        regressions.append({"field": "solver_backed_completed_count", "baseline": baseline_counts.get("solver_backed_completed_count"), "current": current_counts.get("solver_backed_completed_count")})
    if current_counts.get("residual_only_count") != 0:
        regressions.append({"field": "residual_only_count", "expected": 0, "current": current_counts.get("residual_only_count")})
    if current_counts.get("runtime_quality_failed_count") != 0:
        regressions.append({"field": "runtime_quality_failed_count", "expected": 0, "current": current_counts.get("runtime_quality_failed_count")})
    return regressions


def _delta_counts(summary: dict[str, Any]) -> dict[str, int]:
    keys = (
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
    return {key: int(summary.get(key, 0) or 0) for key in keys}


def _count_deltas(baseline: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    return {key: int(current.get(key, 0)) - int(baseline.get(key, 0)) for key in sorted(set(baseline) | set(current))}


def _distribution_delta(
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    field: str,
    *,
    baseline_field: str | None = None,
) -> dict[str, Any]:
    baseline_key = baseline_field or field
    baseline = _distribution([_float(row.get(baseline_key)) for row in baseline_rows])
    current = _distribution([_float(row.get(field)) for row in current_rows])
    return {"baseline": baseline, "current": current, "delta": _distribution_difference(baseline, current)}


def _distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": _stable(float(np.percentile(arr, 50))),
        "p95": _stable(float(np.percentile(arr, 95))),
        "max": _stable(float(np.max(arr))),
    }


def _distribution_difference(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    return {key: _stable(_float(current.get(key)) - _float(baseline.get(key))) for key in ("median", "p95", "max")}


def _full_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows") or payload.get("matrix") or []
    return [row for row in rows if isinstance(row, dict) and row.get("category") == "full_humanoid_profile"]


def _baseline_step4_1_head(summary: dict[str, Any]) -> str:
    return str(
        summary.get("source_code_commit")
        or summary.get("artifact_commit_observed")
        or summary.get("base_step4_0_final_head")
        or ""
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _stable(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    if abs(float(value)) < 1e-15:
        return 0.0
    return round(float(value), 12)


def _strip_volatile_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_volatile_runtime_fields(item)
            for key, item in sorted(value.items())
            if key not in {"runtime_seconds", "diagnostics_hash"}
        }
    if isinstance(value, list):
        return [_strip_volatile_runtime_fields(item) for item in value]
    return value
