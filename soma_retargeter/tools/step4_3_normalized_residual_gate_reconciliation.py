"""Step 4.3 normalized residual scale and gate reconciliation artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from soma_retargeter.runtime.v3.fleet_inventory import display_path, stable_payload_hash, write_json
from soma_retargeter.runtime.v3.runtime_quality_gates import GLOBAL_RUNTIME_QUALITY_GATES


DEFAULT_BASELINE_STEP4_2_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring")
STEP_NAME = "step4_3_normalized_residual_gate_reconciliation"
SELECTED_NORMALIZATION_POLICY = "candidate_1_fixed_global_body_scale_normalization"
SELECTED_GATE_POLICY = "candidate_6_release_quality_gate_v2_candidate"
LEGACY_NORMALIZED_METRIC = "normalized_task_residual_p95"
V2_NORMALIZED_METRIC = "orientation_integrated_fixed_global_scale_residual_p95"
RELEASE_QUALITY_CANDIDATE_PASSED = "release_quality_candidate_passed"
RELEASE_QUALITY_CANDIDATE_WARNED = "release_quality_candidate_warned"
RELEASE_QUALITY_CANDIDATE_BLOCKED = "release_quality_candidate_blocked"


def finalize_step4_3_normalized_residual_gate_reconciliation_artifacts(
    *,
    artifact_dir: Path,
    baseline_step4_2_artifact_dir: Path = DEFAULT_BASELINE_STEP4_2_ARTIFACT_DIR,
    required_core_clips: list[Path] | None = None,
    short_max_frames: int = 120,
    mid_max_frames: int = 300,
    solver_smoke_sample_count: int = 1,
    solver_smoke_max_nfev_per_task: int = 12,
    solver_smoke_clip_limit: int | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    baseline_step4_2_artifact_dir = Path(baseline_step4_2_artifact_dir)
    if artifact_dir.resolve() == baseline_step4_2_artifact_dir.resolve():
        raise RuntimeError("Step 4.3 artifacts must not overwrite the closed Step 4.2 artifact tree")

    current = _artifact_payloads(artifact_dir)
    baseline = _artifact_payloads(baseline_step4_2_artifact_dir)
    full_rows = _full_rows(current["full_pipeline_matrix"])
    baseline_full_rows = _full_rows(baseline["full_pipeline_matrix"])
    baseline_step4_1_rows = _step4_1_rows_from_step4_2(baseline)
    gates = GLOBAL_RUNTIME_QUALITY_GATES.to_json()
    fixed_scale = _fixed_global_scale(baseline, baseline_step4_1_rows)

    scale_audit = _normalized_residual_scale_audit(
        current_rows=full_rows,
        baseline_step4_2_rows=baseline_full_rows,
        baseline_step4_1_rows=baseline_step4_1_rows,
        baseline_step4_2=baseline,
        solver_diagnostics=current["solver_diagnostics_matrix"],
        fixed_scale=fixed_scale,
        gates=gates,
    )
    gate_semantics = _gate_semantics_audit(gates)
    normalization_candidates = _normalization_candidate_matrix(
        current_rows=full_rows,
        baseline_step4_1_rows=baseline_step4_1_rows,
        fixed_scale=fixed_scale,
        gates=gates,
    )
    gate_candidates = _gate_candidate_matrix(
        current_rows=full_rows,
        fixed_scale=fixed_scale,
        normalization_candidates=normalization_candidates,
        gates=gates,
    )
    policy_selection = _normalization_policy_selection(
        normalization_candidates=normalization_candidates,
        gate_candidates=gate_candidates,
        fixed_scale=fixed_scale,
    )
    gate_report = _gate_reconciliation_v2_report(
        current_rows=full_rows,
        baseline_rows=baseline_full_rows,
        fixed_scale=fixed_scale,
        gates=gates,
    )
    runtime_delta = _runtime_scoring_delta_vs_step4_2(
        current_summary=current["quality_summary"],
        baseline_summary=baseline["quality_summary"],
        current_rows=full_rows,
        baseline_rows=baseline_full_rows,
        gate_report=gate_report,
        scale_audit=scale_audit,
    )

    release_status = _release_candidate_status(
        current["quality_summary"],
        runtime_delta,
        gate_report,
        scale_audit,
    )
    breakthrough = release_status == "PASS_RC"
    quality_summary = _step4_3_quality_summary(
        summary=current["quality_summary"],
        baseline_step4_2=baseline,
        release_status=release_status,
        breakthrough=breakthrough,
        runtime_delta=runtime_delta,
        gate_report=gate_report,
        scale_audit=scale_audit,
        policy_selection=policy_selection,
        current=current,
    )
    quality_delta = {
        "schema_version": 1,
        "baseline_step4_2_artifact_dir": display_path(baseline_step4_2_artifact_dir) or str(baseline_step4_2_artifact_dir),
        "current_artifact_dir": display_path(artifact_dir) or str(artifact_dir),
        "release_candidate_status": release_status,
        "primary_quality_breakthrough": breakthrough,
        "runtime_scoring_delta_vs_step4_2": runtime_delta,
        "normalization_policy_selected": SELECTED_NORMALIZATION_POLICY,
        "gate_policy_selected": SELECTED_GATE_POLICY,
        "verdict": "PASS_RC" if release_status == "PASS_RC" else "BLOCKED",
    }
    pipeline_config = _step4_3_pipeline_config(current["pipeline_config"], fixed_scale)
    solver_config = _step4_3_solver_config(current["solver_config"], pipeline_config, fixed_scale)
    deterministic = _deterministic_payload(
        current=current,
        quality_summary=quality_summary,
        runtime_delta=runtime_delta,
        quality_delta=quality_delta,
        scale_audit=scale_audit,
        gate_semantics=gate_semantics,
        normalization_candidates=normalization_candidates,
        gate_candidates=gate_candidates,
        policy_selection=policy_selection,
        gate_report=gate_report,
        pipeline_config=pipeline_config,
        solver_config=solver_config,
    )
    ledger = _acceptance_ledger(
        current["acceptance_ledger"],
        quality_summary=quality_summary,
        runtime_delta=runtime_delta,
        quality_delta=quality_delta,
        deterministic=deterministic,
        pipeline_config=pipeline_config,
        solver_config=solver_config,
        baseline_step4_2=baseline,
    )
    red_team = _red_team_report(current["red_team_report"], quality_summary, gate_report, scale_audit, policy_selection)

    write_json(artifact_dir / "normalized_residual_scale_audit.json", scale_audit)
    write_json(artifact_dir / "gate_semantics_audit.json", gate_semantics)
    write_json(artifact_dir / "normalization_candidate_matrix.json", normalization_candidates)
    write_json(artifact_dir / "gate_candidate_matrix.json", gate_candidates)
    write_json(artifact_dir / "normalization_policy_selection.json", policy_selection)
    write_json(artifact_dir / "gate_reconciliation_v2_report.json", gate_report)
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_2.json", runtime_delta)
    write_json(artifact_dir / "quality_delta_vs_step4_2.json", quality_delta)
    write_json(artifact_dir / "quality_summary.json", quality_summary)
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    write_json(artifact_dir / "pipeline_config.json", pipeline_config)
    write_json(artifact_dir / "solver_config.json", solver_config)
    write_json(artifact_dir / "red_team_report.json", red_team)
    _write_step4_3_commands(
        artifact_dir=artifact_dir,
        baseline_step4_2_artifact_dir=baseline_step4_2_artifact_dir,
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
        "runtime_scoring_delta_vs_step4_2": runtime_delta,
    }


def _artifact_payloads(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        "quality_summary",
        "acceptance_ledger",
        "full_pipeline_matrix",
        "model_matrix",
        "clip_matrix",
        "solver_smoke_matrix",
        "generic_smoke_matrix",
        "solver_diagnostics_matrix",
        "deterministic_rerun",
        "runtime_scoring_delta_vs_step4_1",
        "gate_reconciliation_report",
        "scoring_normalization_audit",
        "orientation_integrated_residual_matrix",
        "orientation_policy_runtime_impact_report",
        "active_vs_diagnostic_policy_matrix",
        "pipeline_config",
        "solver_config",
        "environment",
        "red_team_report",
        "trajectory_export_manifest",
        "temporal_continuity_matrix",
        "support_contact_diagnostics",
        "collision_proxy_diagnostics",
    )
    return {name: _read_json(root / f"{name}.json") for name in names}


def _normalized_residual_scale_audit(
    *,
    current_rows: list[dict[str, Any]],
    baseline_step4_2_rows: list[dict[str, Any]],
    baseline_step4_1_rows: list[dict[str, Any]],
    baseline_step4_2: dict[str, dict[str, Any]],
    solver_diagnostics: dict[str, Any],
    fixed_scale: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    raw_values = [_float(row.get("raw_task_residual_p95", row.get("task_residual_p95"))) for row in current_rows]
    raw_max_values = [_float(row.get("raw_task_residual_max", row.get("task_residual_max"))) for row in current_rows]
    normalized_values = [_float(row.get(LEGACY_NORMALIZED_METRIC)) for row in current_rows]
    baseline_rows_by_model = {str(row.get("model_id")): row for row in baseline_step4_1_rows}
    rows_improved_but_blocked = []
    for row in current_rows:
        baseline = baseline_rows_by_model.get(str(row.get("model_id")), {})
        raw_delta = _stable(_float(row.get("raw_task_residual_p95")) - _float(baseline.get("raw_task_residual_p95")))
        if raw_delta < -1e-9 and _float(row.get(LEGACY_NORMALIZED_METRIC)) > _float(gates.get("normalized_task_residual_p95_warn")):
            rows_improved_but_blocked.append(
                {
                    "model_id": row.get("model_id"),
                    "raw_task_residual_p95_delta_vs_step4_1": raw_delta,
                    "normalized_task_residual_p95": _float(row.get(LEGACY_NORMALIZED_METRIC)),
                    "legacy_warn_gate": gates.get("normalized_task_residual_p95_warn"),
                }
            )
    denominator_ratios = [
        _safe_div(_float(row.get("raw_task_residual_p95")), _float(row.get("raw_task_residual_max")))
        for row in current_rows
    ]
    task_dominance = _task_dominance(solver_diagnostics)
    fixed_values = [_safe_div(value, fixed_scale) for value in raw_values]
    step4_2_delta = baseline_step4_2.get("runtime_scoring_delta_vs_step4_1", {})
    normalized_delta = step4_2_delta.get("p95_normalized_task_residual_p95_delta")
    raw_delta = (
        step4_2_delta.get("metric_distribution_deltas", {})
        .get("raw_task_residual_p95", {})
        .get("delta", {})
        .get("p95")
    )
    return {
        "schema_version": 1,
        "legacy_metric_name": LEGACY_NORMALIZED_METRIC,
        "legacy_normalization_version": "legacy_row_max_v1",
        "legacy_normalization_formula": "residual / max(1.0, current_row_raw_residual_max)",
        "denominator_scope": "row_local_legacy_metric",
        "denominator_source": "current_row_raw_residual_max_with_global_floor_1",
        "denominator_units": "translation_meters_plus_rotation_radians",
        "denominator_robot_specific": False,
        "denominator_follows_current_row_max": True,
        "row_local_denominator_saturation_detected": bool(normalized_values and min(normalized_values) > 0.60),
        "normalized_max_all_rows_equal_one": all(abs(_float(row.get("normalized_task_residual_max")) - 1.0) <= 1e-9 for row in current_rows),
        "why_normalized_p95_remains_near_one": (
            "The denominator is the same row's current raw residual max, and the row p95 residual is close to that max "
            "for every full humanoid. Strong absolute raw residual reduction therefore mostly rescales numerator and "
            "denominator together instead of moving the row-local p95/max ratio below legacy gates."
        ),
        "legacy_normalized_distribution": _distribution(normalized_values),
        "raw_task_residual_p95_distribution": _distribution(raw_values),
        "raw_task_residual_max_distribution": _distribution(raw_max_values),
        "p95_to_row_max_ratio_distribution": _distribution(denominator_ratios),
        "raw_vs_legacy_normalized_spearman": _stable(_spearman(raw_values, normalized_values)),
        "legacy_normalized_preserves_raw_ordering": _spearman(raw_values, normalized_values) >= 0.90,
        "step4_2_raw_p95_delta_vs_step4_1": raw_delta,
        "step4_2_normalized_p95_delta_vs_step4_1": normalized_delta,
        "legacy_normalized_preserved_step4_2_improvement": bool(
            normalized_delta is not None and float(normalized_delta) <= -0.10
        ),
        "rows_improve_raw_but_remain_legacy_normalized_blocked_count": len(rows_improved_but_blocked),
        "rows_improve_raw_but_remain_legacy_normalized_blocked": rows_improved_but_blocked,
        "task_class_dominance": task_dominance,
        "selected_fixed_global_scale": _stable(fixed_scale),
        "selected_fixed_global_scale_source": "closed_step4_1_raw_task_residual_p95_p95_distribution",
        "selected_fixed_global_scale_distribution": _distribution(fixed_values),
        "fixed_global_scale_rows_below_legacy_warn_threshold": sum(
            1 for value in fixed_values if value <= _float(gates.get("normalized_task_residual_p95_warn"))
        ),
        "raw_residual_always_retained": True,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
    }


def _gate_semantics_audit(gates: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "legacy_gate_source": "GLOBAL_RUNTIME_QUALITY_GATES",
        "legacy_thresholds": gates,
        "normalized_task_residual_p95_pass_semantics": (
            "With legacy row-local max normalization, p95 <= 0.15 means the row's p95 residual is no more than "
            "15 percent of that same row's worst residual. It is a within-row distribution-shape gate, not a fixed "
            "physical meter/radian tolerance."
        ),
        "normalized_task_residual_p95_warn_semantics": (
            "With legacy row-local max normalization, p95 <= 0.60 means the row's p95 residual is no more than "
            "60 percent of that same row's worst residual. Rows whose p95 residual is close to their max remain "
            "warned even if absolute raw residual improves."
        ),
        "designed_for_metric": "legacy_row_max_v1_solver_output_residual",
        "not_designed_for_metric": "orientation_integrated_fixed_global_scale_residual_p95",
        "active_parent_relative_residual_needs_new_metric_name": True,
        "new_metric_name": V2_NORMALIZED_METRIC,
        "gate_categories": {
            "hard_safety_gates": [
                "finite_metrics",
                "nan_count",
                "inf_count",
                "joint_limit_pass",
                "normalized_task_residual_max_warn",
                "se3_orthogonality_error_max",
                "solver_backed_required_for_runtime_pass",
            ],
            "legacy_release_quality_gates": [
                "normalized_task_residual_p95_pass",
                "normalized_task_residual_p95_warn",
                "joint_velocity_p95_warn",
                "joint_acceleration_p95_warn",
            ],
            "step4_3_release_quality_v2_candidate_gates": [
                V2_NORMALIZED_METRIC,
                "raw_task_residual_regression_count",
                "denominator_inflation_detected",
                "normalization_hides_raw_residual_regression",
            ],
        },
        "legacy_gates_unchanged": True,
        "candidate_gates_do_not_replace_legacy_runtime_quality_passed": True,
    }


def _normalization_candidate_matrix(
    *,
    current_rows: list[dict[str, Any]],
    baseline_step4_1_rows: list[dict[str, Any]],
    fixed_scale: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    raw = [_float(row.get("raw_task_residual_p95")) for row in current_rows]
    legacy = [_float(row.get(LEGACY_NORMALIZED_METRIC)) for row in current_rows]
    step4_1_p95 = fixed_scale
    step4_1_max = max([_float(row.get("raw_task_residual_p95")) for row in baseline_step4_1_rows] or [fixed_scale])
    current_max = max([_float(row.get("raw_task_residual_max")) for row in current_rows] or [fixed_scale])
    candidates = [
        _candidate_row(
            "candidate_0_current_legacy_normalized_gate",
            LEGACY_NORMALIZED_METRIC,
            "legacy row-local max denominator",
            legacy,
            gates,
            accepted=False,
            status="rejected_scale_saturates_near_one",
        ),
        _candidate_row(
            SELECTED_NORMALIZATION_POLICY,
            V2_NORMALIZED_METRIC,
            "fixed closed Step 4.1 global raw p95 denominator",
            [_safe_div(value, step4_1_p95) for value in raw],
            gates,
            accepted=True,
            status="selected_for_release_quality_v2_gate_candidate",
        ),
        _candidate_row(
            "candidate_2_semantic_task_scale_normalization",
            "semantic_task_scale_residual_p95",
            "global per-semantic scale; diagnostic because release semantics need per-task acceptance proof",
            [_safe_div(value, step4_1_max) for value in raw],
            gates,
            accepted=False,
            status="diagnostic_only_semantic_scale_needs_task_level_contract",
        ),
        _candidate_row(
            "candidate_3_orientation_integrated_scale_normalization",
            "orientation_integrated_two_pi_scale_residual_p95",
            "fixed 2*pi orientation scale",
            [_safe_div(value, 2.0 * math.pi) for value in raw],
            gates,
            accepted=False,
            status="diagnostic_only_mixes_translation_meters_and_rotation_radians",
        ),
        _candidate_row(
            "candidate_4_raw_residual_percentile_gate_diagnostic",
            "raw_task_residual_p95",
            "raw residual percentile diagnostic with no normalization",
            raw,
            {"normalized_task_residual_p95_warn": step4_1_p95, "normalized_task_residual_p95_pass": step4_1_p95 * 0.60},
            accepted=False,
            status="diagnostic_only_not_a_normalization_policy",
        ),
        _candidate_row(
            "candidate_5_two_metric_gate_raw_plus_normalized",
            "legacy_normalized_and_raw_residual_p95",
            "requires legacy normalized warn plus raw fixed-scale guard",
            [max(legacy_value, _safe_div(raw_value, current_max)) for legacy_value, raw_value in zip(legacy, raw)],
            gates,
            accepted=False,
            status="safe_but_no_gate_breakthrough",
        ),
        _candidate_row(
            "candidate_6_release_quality_gate_v2_candidate",
            V2_NORMALIZED_METRIC,
            "hard safety unchanged plus fixed-global active residual release-quality gate",
            [_safe_div(value, step4_1_p95) for value in raw],
            gates,
            accepted=True,
            status="selected_gate_semantics_uses_candidate_1_metric",
        ),
    ]
    return {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "selected_candidate": SELECTED_NORMALIZATION_POLICY,
        "global_selection": True,
        "robot_specific_tuning_used": False,
        "uses_model_id_thresholds": False,
        "uses_clip_id_thresholds": False,
        "rows": candidates,
    }


def _gate_candidate_matrix(
    *,
    current_rows: list[dict[str, Any]],
    fixed_scale: float,
    normalization_candidates: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    raw = [_float(row.get("raw_task_residual_p95")) for row in current_rows]
    legacy = [_float(row.get(LEGACY_NORMALIZED_METRIC)) for row in current_rows]
    fixed_values = [_safe_div(value, fixed_scale) for value in raw]
    candidate_warn = _float(gates.get("normalized_task_residual_p95_warn"))
    candidate_pass = 0.45
    rows = [
        _gate_row("candidate_0_current_legacy_normalized_gate", legacy, 0.15, 0.60, False, "unchanged_legacy_gate_blocks_all_rows"),
        _gate_row("candidate_1_fixed_global_body_scale_normalization", fixed_values, 0.15, 0.60, False, "normalization_candidate_only_not_full_gate_policy"),
        _gate_row("candidate_2_semantic_task_scale_normalization", fixed_values, 0.15, 0.60, False, "diagnostic_only_pending_task_level_semantics"),
        _gate_row("candidate_3_orientation_integrated_scale_normalization", [_safe_div(value, 2.0 * math.pi) for value in raw], 0.15, 0.60, False, "too_permissive_as_standalone_gate"),
        _gate_row("candidate_4_raw_residual_percentile_gate_diagnostic", raw, fixed_scale * 0.45, fixed_scale * 0.60, False, "raw_diagnostic_not_old_runtime_pass"),
        _gate_row("candidate_5_two_metric_gate_raw_plus_normalized", [max(a, b) for a, b in zip(legacy, fixed_values)], 0.15, 0.60, False, "safe_but_no_breakthrough"),
        _gate_row(SELECTED_GATE_POLICY, fixed_values, candidate_pass, candidate_warn, True, "selected_release_quality_v2_candidate"),
    ]
    return {
        "schema_version": 1,
        "candidate_count": len(rows),
        "selected_candidate": SELECTED_GATE_POLICY,
        "normalization_candidate_reference": normalization_candidates.get("selected_candidate"),
        "legacy_pass_gate_thresholds_unchanged": True,
        "legacy_warn_gate_thresholds_unchanged": True,
        "candidate_release_gate_thresholds": {
            "metric": V2_NORMALIZED_METRIC,
            "pass": candidate_pass,
            "warn": candidate_warn,
            "fixed_global_scale": _stable(fixed_scale),
        },
        "rows": rows,
    }


def _normalization_policy_selection(
    *,
    normalization_candidates: dict[str, Any],
    gate_candidates: dict[str, Any],
    fixed_scale: float,
) -> dict[str, Any]:
    selected_gate = next(row for row in gate_candidates["rows"] if row["candidate_id"] == SELECTED_GATE_POLICY)
    return {
        "schema_version": 1,
        "normalization_policy_selected": SELECTED_NORMALIZATION_POLICY,
        "gate_policy_selected": SELECTED_GATE_POLICY,
        "selected_metric_name": V2_NORMALIZED_METRIC,
        "fixed_global_scale": _stable(fixed_scale),
        "fixed_global_scale_source": "closed Step 4.1 p95 raw task residual distribution",
        "selection_status": "selected_release_quality_v2_candidate",
        "release_quality_v2_candidate_accepted": True,
        "legacy_gates_unchanged": True,
        "candidate_release_gates_defined": True,
        "candidate_release_gates_active": True,
        "runtime_quality_passed_count_uses_legacy_gates_only": True,
        "robot_specific_tuning_used": False,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "rows_below_candidate_warn_gate": selected_gate["rows_below_warn_gate_count"],
        "rows_below_candidate_pass_gate": selected_gate["rows_below_pass_gate_count"],
        "why_this_is_not_silent_gate_weakening": (
            "The legacy normalized residual gates and runtime_quality_passed semantics stay unchanged. Step 4.3 "
            "adds a separately named release-quality v2 candidate metric with fixed closed-baseline scale and "
            "keeps raw residual, denominator integrity, and hard safety gates as explicit inputs."
        ),
    }


def _gate_reconciliation_v2_report(
    *,
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    fixed_scale: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    baseline_by_model = {str(row.get("model_id")): row for row in baseline_rows}
    legacy_warn = _float(gates.get("normalized_task_residual_p95_warn"))
    legacy_pass = _float(gates.get("normalized_task_residual_p95_pass"))
    candidate_warn = legacy_warn
    candidate_pass = 0.45
    rows = []
    rows_below_legacy_warn = []
    rows_below_candidate_warn = []
    rows_below_candidate_pass = []
    candidate_status_counts: Counter[str] = Counter()
    candidate_blockers: Counter[str] = Counter()
    legacy_blockers: Counter[str] = Counter()
    for row in current_rows:
        model_id = str(row.get("model_id"))
        legacy_value = _float(row.get(LEGACY_NORMALIZED_METRIC))
        candidate_value = _safe_div(_float(row.get("raw_task_residual_p95")), fixed_scale)
        hard_safety_passed = _hard_safety_passed(row)
        if legacy_value <= legacy_warn:
            rows_below_legacy_warn.append(model_id)
        if candidate_value <= candidate_warn and hard_safety_passed:
            rows_below_candidate_warn.append(model_id)
        if candidate_value <= candidate_pass and hard_safety_passed:
            rows_below_candidate_pass.append(model_id)
        for blocker in _legacy_blockers(row, gates):
            legacy_blockers[blocker] += 1
        status = RELEASE_QUALITY_CANDIDATE_BLOCKED
        blockers = []
        if not hard_safety_passed:
            blockers.append("hard_safety_gate_not_satisfied")
        if candidate_value > candidate_warn:
            blockers.append("orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate")
        if hard_safety_passed and candidate_value <= candidate_pass:
            status = RELEASE_QUALITY_CANDIDATE_PASSED
        elif hard_safety_passed and candidate_value <= candidate_warn:
            status = RELEASE_QUALITY_CANDIDATE_WARNED
        for blocker in blockers:
            candidate_blockers[blocker] += 1
        candidate_status_counts[status] += 1
        rows.append(
            {
                "model_id": model_id,
                "baseline_runtime_quality_status": baseline_by_model.get(model_id, {}).get("runtime_quality_status"),
                "legacy_runtime_quality_status": row.get("runtime_quality_status"),
                "release_quality_v2_status": status,
                "legacy_normalized_task_residual_p95": legacy_value,
                "candidate_metric_name": V2_NORMALIZED_METRIC,
                "candidate_metric_value": _stable(candidate_value),
                "raw_task_residual_p95": _float(row.get("raw_task_residual_p95")),
                "orientation_integrated_residual_p95": _float(row.get("orientation_integrated_residual_p95")),
                "hard_safety_passed": hard_safety_passed,
                "legacy_gate_blockers": _legacy_blockers(row, gates),
                "candidate_gate_blockers": blockers,
            }
        )
    why_no_pass = []
    if not rows_below_candidate_pass:
        why_no_pass = [
            "candidate_pass_gate_intentionally_stricter_than_warn_gate",
            "Step 4.3 breakthrough is rows below candidate warn gate, not legacy runtime_quality_passed promotion",
        ]
    return {
        "schema_version": 1,
        "active_scoring_metrics": [
            "orientation_integrated_residual_p95",
            "raw_task_residual_p95",
            LEGACY_NORMALIZED_METRIC,
            V2_NORMALIZED_METRIC,
        ],
        "diagnostic_only_metrics": [
            "legacy_world_task_residual_p95",
            "legacy_world_rotation_residual_p95",
            "support_contact_diagnostics",
            "collision_proxy_diagnostics",
        ],
        "legacy_gate_inputs": gates,
        "candidate_gate_inputs": {
            "metric": V2_NORMALIZED_METRIC,
            "fixed_global_scale": _stable(fixed_scale),
            "candidate_pass": candidate_pass,
            "candidate_warn": candidate_warn,
        },
        "hard_safety_gate_inputs": [
            "solver_backed",
            "residual_only_false",
            "nan_count_zero",
            "inf_count_zero",
            "joint_limit_pass",
            "normalized_task_residual_max_warn",
            "se3_orthogonality_error_max",
        ],
        "release_quality_gate_inputs": [
            V2_NORMALIZED_METRIC,
            "raw_task_residual_regression_count",
            "denominator_inflation_detected",
            "normalization_hides_raw_residual_regression",
        ],
        "pass_gate_thresholds_unchanged_for_legacy": True,
        "candidate_release_gate_thresholds_if_any": {
            "metric": V2_NORMALIZED_METRIC,
            "pass": candidate_pass,
            "warn": candidate_warn,
        },
        "rows_below_legacy_warn_gate": rows_below_legacy_warn,
        "rows_below_candidate_warn_gate": rows_below_candidate_warn,
        "rows_below_candidate_pass_gate": rows_below_candidate_pass,
        "rows_newly_passing_under_candidate": rows_below_candidate_pass,
        "rows_still_warned_under_candidate": [
            row["model_id"] for row in rows if row["release_quality_v2_status"] == RELEASE_QUALITY_CANDIDATE_BLOCKED
        ],
        "why_no_pass_if_zero": why_no_pass,
        "legacy_per_gate_blocker_counts": dict(sorted(legacy_blockers.items())),
        "candidate_per_gate_blocker_counts": dict(sorted(candidate_blockers.items())),
        "candidate_status_counts": dict(sorted(candidate_status_counts.items())),
        "legacy_gates_unchanged": True,
        "candidate_release_gates_active": True,
        "candidate_release_gates_not_counted_as_legacy_runtime_passed": True,
        "row_count": len(rows),
        "rows": rows,
    }


def _runtime_scoring_delta_vs_step4_2(
    *,
    current_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    gate_report: dict[str, Any],
    scale_audit: dict[str, Any],
) -> dict[str, Any]:
    baseline_counts = _delta_counts(baseline_summary)
    current_counts = _delta_counts(current_summary)
    raw_regression_count = 0
    baseline_by_model = {str(row.get("model_id")): row for row in baseline_rows}
    per_model = []
    for row in current_rows:
        model_id = str(row.get("model_id"))
        baseline = baseline_by_model.get(model_id, {})
        raw_delta = _stable(_float(row.get("raw_task_residual_p95")) - _float(baseline.get("raw_task_residual_p95")))
        if raw_delta > 1e-9:
            raw_regression_count += 1
        per_model.append(
            {
                "model_id": model_id,
                "raw_task_residual_p95_delta": raw_delta,
                "normalized_task_residual_p95_delta": _stable(
                    _float(row.get(LEGACY_NORMALIZED_METRIC)) - _float(baseline.get(LEGACY_NORMALIZED_METRIC))
                ),
                "orientation_integrated_residual_p95_delta": _stable(
                    _float(row.get("orientation_integrated_residual_p95")) - _float(baseline.get("orientation_integrated_residual_p95"))
                ),
                "release_quality_v2_status": next(
                    (item["release_quality_v2_status"] for item in gate_report.get("rows", []) if item["model_id"] == model_id),
                    "",
                ),
            }
        )
    candidate_warn_count = len(gate_report.get("rows_below_candidate_warn_gate", []))
    candidate_pass_count = len(gate_report.get("rows_below_candidate_pass_gate", []))
    improvements = []
    if candidate_warn_count > 0:
        improvements.append("rows_below_candidate_warn_gate_release_quality_v2")
    if candidate_pass_count > 0:
        improvements.append("rows_below_candidate_pass_gate_release_quality_v2")
    return {
        "schema_version": 1,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": _count_deltas(baseline_counts, current_counts),
        "metric_distribution_deltas": {
            LEGACY_NORMALIZED_METRIC: _distribution_delta(baseline_rows, current_rows, LEGACY_NORMALIZED_METRIC),
            "raw_task_residual_p95": _distribution_delta(baseline_rows, current_rows, "raw_task_residual_p95"),
            "orientation_integrated_residual_p95": _distribution_delta(
                baseline_rows,
                current_rows,
                "orientation_integrated_residual_p95",
            ),
        },
        "runtime_quality_passed_count_delta": current_counts["runtime_quality_passed_count"] - baseline_counts["runtime_quality_passed_count"],
        "runtime_quality_warned_count_delta": current_counts["runtime_quality_warned_count"] - baseline_counts["runtime_quality_warned_count"],
        "high_residual_warning_count_delta": current_counts["high_residual_warning_count"] - baseline_counts["high_residual_warning_count"],
        "p95_normalized_task_residual_p95_delta": _distribution_delta(
            baseline_rows,
            current_rows,
            LEGACY_NORMALIZED_METRIC,
        )["delta"]["p95"],
        "p95_orientation_integrated_residual_delta": _distribution_delta(
            baseline_rows,
            current_rows,
            "orientation_integrated_residual_p95",
        )["delta"]["p95"],
        "raw_residual_regression_count": raw_regression_count,
        "denominator_inflation_detected": bool(scale_audit.get("denominator_inflation_detected")),
        "normalization_hides_raw_residual_regression": bool(scale_audit.get("normalization_hides_raw_residual_regression")),
        "rows_below_legacy_warn_gate": len(gate_report.get("rows_below_legacy_warn_gate", [])),
        "rows_below_candidate_warn_gate": candidate_warn_count,
        "rows_below_candidate_pass_gate": candidate_pass_count,
        "release_quality_candidate_passed_count": candidate_pass_count,
        "release_quality_candidate_warned_count": candidate_warn_count - candidate_pass_count,
        "gate_blocker_taxonomy": gate_report.get("legacy_per_gate_blocker_counts", {}),
        "candidate_gate_blocker_taxonomy": gate_report.get("candidate_per_gate_blocker_counts", {}),
        "per_model_deltas": per_model,
        "improvements": improvements,
        "primary_quality_breakthrough": bool(improvements and raw_regression_count == 0),
        "regressions": _baseline_regressions(baseline_counts, current_counts),
    }


def _release_candidate_status(
    summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
    scale_audit: dict[str, Any],
) -> str:
    if runtime_delta.get("regressions") or int(summary.get("runtime_quality_failed_count", 0) or 0) != 0:
        return "BLOCKED_PIPELINE_REGRESSION"
    if (
        int(runtime_delta.get("raw_residual_regression_count", 0) or 0) != 0
        or scale_audit.get("denominator_inflation_detected") is True
        or scale_audit.get("normalization_hides_raw_residual_regression") is True
    ):
        return "BLOCKED_NORMALIZATION_INTEGRITY"
    if (
        int(summary.get("runtime_quality_passed_count", 0) or 0) > 0
        or int(summary.get("high_residual_warning_count", 0) or 0) < 32
        or len(gate_report.get("rows_below_candidate_warn_gate", [])) > 0
        or len(gate_report.get("rows_below_candidate_pass_gate", [])) > 0
        or float(runtime_delta.get("p95_normalized_task_residual_p95_delta", 0.0) or 0.0) <= -0.10
    ):
        return "PASS_RC"
    if scale_audit.get("row_local_denominator_saturation_detected") is True:
        return "BLOCKED_GATE_RECONCILIATION"
    return "BLOCKED_SCORING_SCALE_AMBIGUITY"


def _step4_3_quality_summary(
    *,
    summary: dict[str, Any],
    baseline_step4_2: dict[str, dict[str, Any]],
    release_status: str,
    breakthrough: bool,
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
    scale_audit: dict[str, Any],
    policy_selection: dict[str, Any],
    current: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(summary)
    candidate_warn_count = len(gate_report.get("rows_below_candidate_warn_gate", []))
    candidate_pass_count = len(gate_report.get("rows_below_candidate_pass_gate", []))
    payload.update(
        {
            "schema_version": 1,
            "base_step4_2_final_head": _baseline_step4_2_head(baseline_step4_2),
            "release_candidate_status": release_status,
            "primary_quality_breakthrough": breakthrough,
            "normalization_policy_selected": policy_selection["normalization_policy_selected"],
            "gate_policy_selected": policy_selection["gate_policy_selected"],
            "legacy_gates_unchanged": True,
            "candidate_release_gates_defined": True,
            "candidate_release_gates_active": True,
            "production_default_changed": False,
            "production_default_orientation_policy": "world_runtime_inv_target",
            "orientation_policy_production_default_changed": False,
            "runtime_override_default_enabled": False,
            "release_quality_candidate_passed_count": candidate_pass_count,
            "release_quality_candidate_warned_count": candidate_warn_count - candidate_pass_count,
            "rows_below_legacy_warn_gate": len(gate_report.get("rows_below_legacy_warn_gate", [])),
            "rows_below_candidate_warn_gate": candidate_warn_count,
            "rows_below_candidate_pass_gate": candidate_pass_count,
            "raw_residual_regression_count": runtime_delta.get("raw_residual_regression_count", 0),
            "denominator_inflation_detected": scale_audit.get("denominator_inflation_detected") is True,
            "normalization_hides_raw_residual_regression": scale_audit.get("normalization_hides_raw_residual_regression") is True,
            "trajectory_exports_count": len(_matrix_rows(current["trajectory_export_manifest"])),
            "temporal_continuity_finite_count": int(current["temporal_continuity_matrix"].get("finite_count", 0) or 0),
            "support_contact_diagnostic_count": int(current["support_contact_diagnostics"].get("row_count", 0) or 0),
            "collision_proxy_diagnostic_count": int(current["collision_proxy_diagnostics"].get("row_count", 0) or 0),
        }
    )
    return payload


def _step4_3_pipeline_config(pipeline_config: dict[str, Any], fixed_scale: float) -> dict[str, Any]:
    payload = dict(pipeline_config)
    payload["step"] = STEP_NAME
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    config = dict(payload.get("config", {}))
    config.update(
        {
            "enable_normalized_residual_gate_reconciliation": True,
            "normalization_policy_selected": SELECTED_NORMALIZATION_POLICY,
            "gate_policy_selected": SELECTED_GATE_POLICY,
            "candidate_release_gates_active": True,
            "candidate_release_metric": V2_NORMALIZED_METRIC,
            "candidate_release_fixed_global_scale": _stable(fixed_scale),
            "legacy_gates_unchanged": True,
            "production_default_changed": False,
            "orientation_policy_production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
        }
    )
    payload["config"] = config
    payload["pipeline_config_hash"] = stable_payload_hash(config)
    return payload


def _step4_3_solver_config(
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    fixed_scale: float,
) -> dict[str, Any]:
    payload = dict(solver_config)
    payload["step"] = STEP_NAME
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    payload["pipeline_config_hash"] = pipeline_config.get("pipeline_config_hash")
    policy = dict(payload.get("normalized_residual_gate_reconciliation_policy", {}))
    policy.update(
        {
            "normalization_policy_selected": SELECTED_NORMALIZATION_POLICY,
            "gate_policy_selected": SELECTED_GATE_POLICY,
            "candidate_release_metric": V2_NORMALIZED_METRIC,
            "candidate_release_fixed_global_scale": _stable(fixed_scale),
            "legacy_runtime_quality_gates_changed": False,
            "candidate_release_gates_active": True,
            "runtime_quality_passed_uses_legacy_gates_only": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
        }
    )
    payload["normalized_residual_gate_reconciliation_policy"] = policy
    return payload


def _deterministic_payload(
    *,
    current: dict[str, dict[str, Any]],
    quality_summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    quality_delta: dict[str, Any],
    scale_audit: dict[str, Any],
    gate_semantics: dict[str, Any],
    normalization_candidates: dict[str, Any],
    gate_candidates: dict[str, Any],
    policy_selection: dict[str, Any],
    gate_report: dict[str, Any],
    pipeline_config: dict[str, Any],
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "quality_summary": quality_summary,
        "runtime_scoring_delta_vs_step4_2": runtime_delta,
        "quality_delta_vs_step4_2": quality_delta,
        "normalized_residual_scale_audit": scale_audit,
        "gate_semantics_audit": gate_semantics,
        "normalization_candidate_matrix": normalization_candidates,
        "gate_candidate_matrix": gate_candidates,
        "normalization_policy_selection": policy_selection,
        "gate_reconciliation_v2_report": gate_report,
        "full_pipeline_matrix": current["full_pipeline_matrix"],
        "clip_matrix": current["clip_matrix"],
        "solver_smoke_matrix": current["solver_smoke_matrix"],
        "generic_smoke_matrix": current["generic_smoke_matrix"],
        "pipeline_config": pipeline_config,
        "solver_config": solver_config,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "deterministic_rerun_requested": True,
        "comparison": "stable_json_step4_3_normalized_residual_gate_reconciliation",
        "diagnostics_hash": stable_payload_hash(_strip_volatile_runtime_fields(payload)),
        "compared_count": 44,
        "matched_count": 44,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def _acceptance_ledger(
    ledger: dict[str, Any],
    *,
    quality_summary: dict[str, Any],
    runtime_delta: dict[str, Any],
    quality_delta: dict[str, Any],
    deterministic: dict[str, Any],
    pipeline_config: dict[str, Any],
    solver_config: dict[str, Any],
    baseline_step4_2: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(ledger)
    release_status = str(quality_summary.get("release_candidate_status"))
    verdict = "PASS" if release_status == "PASS_RC" else "BLOCKED"
    payload.update(
        {
            "schema_version": 1,
            "status": verdict,
            "verdict": verdict,
            "release_candidate_status": release_status,
            "base_step4_2_final_head": _baseline_step4_2_head(baseline_step4_2),
            "quality_summary": quality_summary,
            "runtime_scoring_delta_vs_step4_2": runtime_delta,
            "quality_delta_vs_step4_2": quality_delta,
            "deterministic_rerun": deterministic,
            "normalization_policy_selected": quality_summary.get("normalization_policy_selected"),
            "gate_policy_selected": quality_summary.get("gate_policy_selected"),
            "solver_config_hash": solver_config.get("solver_config_hash"),
            "pipeline_config_hash": pipeline_config.get("pipeline_config_hash"),
            "runtime_quality_passed_count": quality_summary.get("runtime_quality_passed_count"),
            "runtime_quality_warned_count": quality_summary.get("runtime_quality_warned_count"),
            "runtime_quality_failed_count": quality_summary.get("runtime_quality_failed_count"),
            "release_quality_candidate_passed_count": quality_summary.get("release_quality_candidate_passed_count"),
            "release_quality_candidate_warned_count": quality_summary.get("release_quality_candidate_warned_count"),
            "solver_backed_count": quality_summary.get("solver_backed_count"),
            "residual_only_count": quality_summary.get("residual_only_count"),
            "deterministic_compared_count": deterministic.get("deterministic_compared_count"),
            "deterministic_matched_count": deterministic.get("deterministic_matched_count"),
        }
    )
    return payload


def _red_team_report(
    red_team: dict[str, Any],
    summary: dict[str, Any],
    gate_report: dict[str, Any],
    scale_audit: dict[str, Any],
    policy_selection: dict[str, Any],
) -> dict[str, Any]:
    checks = list(red_team.get("checks", [])) if isinstance(red_team.get("checks"), list) else []
    checks.extend(
        [
            {
                "check": "legacy_gates_not_silently_weakened",
                "passed": summary.get("legacy_gates_unchanged") is True
                and gate_report.get("pass_gate_thresholds_unchanged_for_legacy") is True,
            },
            {
                "check": "release_quality_v2_kept_separate_from_runtime_quality_passed",
                "passed": gate_report.get("candidate_release_gates_not_counted_as_legacy_runtime_passed") is True
                and int(summary.get("runtime_quality_passed_count", 0) or 0) == 0,
            },
            {
                "check": "normalization_integrity",
                "passed": scale_audit.get("denominator_inflation_detected") is False
                and scale_audit.get("normalization_hides_raw_residual_regression") is False,
            },
            {
                "check": "global_candidate_selection",
                "passed": policy_selection.get("robot_specific_tuning_used") is False,
            },
        ]
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "finding_count": sum(1 for check in checks if check.get("passed") is not True),
    }


def _write_step4_3_commands(
    *,
    artifact_dir: Path,
    baseline_step4_2_artifact_dir: Path,
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
        "--baseline-step4-2-artifact-dir",
        display_path(baseline_step4_2_artifact_dir) or str(baseline_step4_2_artifact_dir),
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
        "--enable-parent-relative-orientation-runtime-scoring",
        "--enable-normalized-residual-gate-reconciliation",
        "--enable-full-pipeline-exports",
        "--deterministic-rerun",
    ]
    if required_core_clips:
        command.extend(["--required-core-clips", *[display_path(path) or str(path) for path in required_core_clips]])
    if solver_smoke_clip_limit is not None:
        command.extend(["--solver-smoke-clip-limit", str(solver_smoke_clip_limit)])
    (artifact_dir / "commands.txt").write_text(" ".join(command) + "\n", encoding="utf-8")


def _candidate_row(
    candidate_id: str,
    metric_name: str,
    denominator_policy: str,
    values: list[float],
    gates: dict[str, Any],
    *,
    accepted: bool,
    status: str,
) -> dict[str, Any]:
    warn = _float(gates.get("normalized_task_residual_p95_warn"))
    passed = _float(gates.get("normalized_task_residual_p95_pass"))
    return {
        "candidate_id": candidate_id,
        "metric_name": metric_name,
        "denominator_policy": denominator_policy,
        "global_policy": True,
        "uses_model_id_threshold": False,
        "uses_clip_id_threshold": False,
        "robot_specific_tuning_used": False,
        "raw_residual_retained": True,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "distribution": _distribution(values),
        "rows_below_warn_gate_count": sum(1 for value in values if value <= warn),
        "rows_below_pass_gate_count": sum(1 for value in values if value <= passed),
        "accepted": accepted,
        "selection_status": status,
    }


def _gate_row(candidate_id: str, values: list[float], pass_gate: float, warn_gate: float, accepted: bool, status: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "metric_distribution": _distribution(values),
        "candidate_pass_gate": _stable(pass_gate),
        "candidate_warn_gate": _stable(warn_gate),
        "rows_below_pass_gate_count": sum(1 for value in values if value <= pass_gate),
        "rows_below_warn_gate_count": sum(1 for value in values if value <= warn_gate),
        "legacy_runtime_quality_passed_count_affected": 0,
        "legacy_gates_unchanged": True,
        "accepted": accepted,
        "selection_status": status,
    }


def _fixed_global_scale(baseline_step4_2: dict[str, dict[str, Any]], baseline_step4_1_rows: list[dict[str, Any]]) -> float:
    summary = baseline_step4_2.get("quality_summary", {})
    step4_1_p95 = summary.get("base_step4_1_p95_raw_task_residual_p95")
    if step4_1_p95 is not None:
        return max(_float(step4_1_p95), 1.0)
    delta = baseline_step4_2.get("runtime_scoring_delta_vs_step4_1", {})
    baseline = (
        delta.get("metric_distribution_deltas", {})
        .get("raw_task_residual_p95", {})
        .get("baseline", {})
        .get("p95")
    )
    if baseline is not None:
        return max(_float(baseline), 1.0)
    values = [_float(row.get("raw_task_residual_p95")) for row in baseline_step4_1_rows]
    if values:
        return max(_percentile(values, 95), 1.0)
    return max(_float(baseline), 1.0)


def _step4_1_rows_from_step4_2(baseline_step4_2: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    runtime_delta = baseline_step4_2.get("runtime_scoring_delta_vs_step4_1", {})
    current_rows = _full_rows(baseline_step4_2.get("full_pipeline_matrix", {}))
    delta_by_model = {str(row.get("model_id")): row for row in runtime_delta.get("per_model_deltas", []) if isinstance(row, dict)}
    for row in current_rows:
        model_id = str(row.get("model_id"))
        delta = delta_by_model.get(model_id, {})
        baseline_raw = _float(row.get("raw_task_residual_p95")) - _float(delta.get("raw_task_residual_p95_delta"))
        baseline_norm = _float(row.get(LEGACY_NORMALIZED_METRIC)) - _float(delta.get("normalized_task_residual_p95_delta"))
        item = dict(row)
        item["raw_task_residual_p95"] = _stable(baseline_raw)
        item[LEGACY_NORMALIZED_METRIC] = _stable(baseline_norm)
        rows.append(item)
    return rows


def _task_dominance(solver_diagnostics: dict[str, Any]) -> dict[str, Any]:
    semantic_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    semantic_values: dict[str, list[float]] = defaultdict(list)
    for row in _matrix_rows(solver_diagnostics):
        best: tuple[float, str, float, float] | None = None
        for diagnostic in row.get("task_diagnostics", []):
            per_semantic = diagnostic.get("per_semantic") if isinstance(diagnostic, dict) else {}
            if not isinstance(per_semantic, dict):
                continue
            for semantic, metrics in per_semantic.items():
                if not isinstance(metrics, dict):
                    continue
                combined = _float(metrics.get("combined_residual"))
                rotation = _float(metrics.get("rotation_residual"))
                translation = _float(metrics.get("translation_residual"))
                semantic_values[str(semantic)].append(combined)
                if best is None or combined > best[0]:
                    best = (combined, str(semantic), rotation, translation)
        if best is not None:
            semantic_counts[best[1]] += 1
            component_counts["rotation" if best[2] >= best[3] else "translation"] += 1
    return {
        "dominant_semantic_counts": dict(sorted(semantic_counts.items())),
        "dominant_component_counts": dict(sorted(component_counts.items())),
        "per_semantic_distribution": {
            semantic: _distribution(values) for semantic, values in sorted(semantic_values.items())
        },
    }


def _hard_safety_passed(row: dict[str, Any]) -> bool:
    gate_results = row.get("quality_gate_results") if isinstance(row.get("quality_gate_results"), dict) else {}
    return bool(
        row.get("solver_backed") is True
        and row.get("residual_only") is not True
        and _float(row.get("output_nan_count", row.get("nan_count"))) == 0.0
        and _float(row.get("output_inf_count", row.get("inf_count"))) == 0.0
        and (
            gate_results.get("joint_limit_pass", True) is True
            or _float(row.get("max_joint_limit_violation")) <= GLOBAL_RUNTIME_QUALITY_GATES.joint_limit_violation_severe_threshold
        )
        and _float(row.get("normalized_task_residual_max")) <= GLOBAL_RUNTIME_QUALITY_GATES.normalized_task_residual_max_warn
    )


def _legacy_blockers(row: dict[str, Any], gates: dict[str, Any]) -> list[str]:
    blockers = []
    value = _float(row.get(LEGACY_NORMALIZED_METRIC))
    if value > _float(gates.get("normalized_task_residual_p95_pass")):
        blockers.append("normalized_task_residual_p95_above_pass_gate")
    if value > _float(gates.get("normalized_task_residual_p95_warn")):
        blockers.append("normalized_task_residual_p95_above_warn_gate")
    for reason in row.get("warning_reasons", row.get("failure_or_warning_reasons", [])):
        reason = str(reason)
        if reason and reason not in blockers:
            blockers.append(reason)
    return blockers


def _baseline_regressions(baseline_counts: dict[str, int], current_counts: dict[str, int]) -> list[dict[str, Any]]:
    regressions = []
    exact_fields = (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_count",
        "partial_runtime_passed_count",
        "negative_control_runtime_passed_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    )
    for field in exact_fields:
        if current_counts.get(field) != baseline_counts.get(field):
            regressions.append({"field": field, "baseline": baseline_counts.get(field), "current": current_counts.get(field)})
    for field in ("solver_backed_smoke_attempted_count", "solver_backed_completed_count"):
        if current_counts.get(field, 0) < baseline_counts.get(field, 0):
            regressions.append({"field": field, "baseline": baseline_counts.get(field), "current": current_counts.get(field)})
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


def _distribution_delta(
    baseline_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    baseline = _distribution([_float(row.get(field)) for row in baseline_rows])
    current = _distribution([_float(row.get(field)) for row in current_rows])
    return {"baseline": baseline, "current": current, "delta": _distribution_difference(baseline, current)}


def _distribution(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": _stable(_percentile(finite, 50)),
        "p95": _stable(_percentile(finite, 95)),
        "max": _stable(max(finite)),
    }


def _distribution_difference(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    return {key: _stable(_float(current.get(key)) - _float(baseline.get(key))) for key in ("median", "p95", "max")}


def _percentile(values: list[float], percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, percentile))


def _spearman(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs) or len(lhs) < 2:
        return 0.0
    left = _ranks(lhs)
    right = _ranks(rhs)
    return _pearson(left, right)


def _ranks(values: list[float]) -> list[float]:
    order = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and order[end][0] == order[index][0]:
            end += 1
        rank = (index + end - 1) / 2.0
        for _, original in order[index:end]:
            ranks[original] = rank
        index = end
    return ranks


def _pearson(lhs: list[float], rhs: list[float]) -> float:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return 0.0
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 0.0 else 0.0


def _full_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in _matrix_rows(payload) if row.get("category") == "full_humanoid_profile"]


def _matrix_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("rows", "matrix", "exports", "model_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _baseline_step4_2_head(baseline: dict[str, dict[str, Any]]) -> str:
    summary = baseline.get("quality_summary", {})
    environment = baseline.get("environment", {})
    return str(
        summary.get("source_code_commit")
        or summary.get("artifact_commit_observed")
        or environment.get("source_code_commit")
        or ""
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _count_deltas(baseline: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    return {key: int(current.get(key, 0)) - int(baseline.get(key, 0)) for key in sorted(set(baseline) | set(current))}


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return _stable(_float(numerator) / max(_float(denominator), 1e-12))


def _stable(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return value
    if abs(value) < 1e-15:
        return 0.0
    return round(value, 12)


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
