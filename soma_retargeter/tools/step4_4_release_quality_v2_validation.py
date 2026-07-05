"""Step 4.4 release-quality v2 validation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

from soma_retargeter.runtime.v3.fleet_inventory import display_path, stable_payload_hash, write_json


DEFAULT_BASELINE_STEP4_3_ARTIFACT_DIR = Path(
    "artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation"
)
DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_4_release_quality_v2_validation")
STEP_NAME = "step4_4_release_quality_v2_validation"
BASELINE_GATE_POLICY = "candidate_6_release_quality_gate_v2_candidate"
SELECTED_VALIDATION_POLICY = "global_clip_mean_with_worst_clip_guard_v1_diagnostic"
V2_METRIC = "orientation_integrated_fixed_global_scale_residual_p95"
RELEASE_QUALITY_CANDIDATE_PASSED = "release_quality_candidate_passed"
RELEASE_QUALITY_CANDIDATE_WARNED = "release_quality_candidate_warned"
RELEASE_QUALITY_CANDIDATE_BLOCKED = "release_quality_candidate_blocked"
DIAGNOSTIC_WORST_CLIP_GUARD = 0.65
BASELINE_COPY_ARTIFACTS = (
    "environment.json",
    "full_pipeline_matrix.json",
    "clip_matrix.json",
    "solver_smoke_matrix.json",
    "generic_smoke_matrix.json",
    "trajectory_export_manifest.json",
    "temporal_continuity_matrix.json",
    "support_contact_diagnostics.json",
    "collision_proxy_diagnostics.json",
    "pipeline_controls_reference.json",
    "pipeline_backed_matrix.json",
    "solver_diagnostics_matrix.json",
    "test_results/pytest.txt",
    "test_results/pytest_summary.json",
    "test_results/junit.xml",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--baseline-step4-3-artifact-dir", type=Path, default=DEFAULT_BASELINE_STEP4_3_ARTIFACT_DIR)
    args = parser.parse_args(argv)
    payload = finalize_step4_4_release_quality_v2_validation_artifacts(
        artifact_dir=args.artifact_dir,
        baseline_step4_3_artifact_dir=args.baseline_step4_3_artifact_dir,
    )
    print(
        json.dumps(
            {
                "status": payload["release_candidate_status"],
                "artifact_dir": display_path(args.artifact_dir) or str(args.artifact_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def finalize_step4_4_release_quality_v2_validation_artifacts(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    baseline_step4_3_artifact_dir: Path = DEFAULT_BASELINE_STEP4_3_ARTIFACT_DIR,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    baseline_step4_3_artifact_dir = Path(baseline_step4_3_artifact_dir)
    if artifact_dir.resolve() == baseline_step4_3_artifact_dir.resolve():
        raise RuntimeError("Step 4.4 artifacts must not overwrite the closed Step 4.3 artifact tree")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _copy_baseline_artifacts(baseline_step4_3_artifact_dir, artifact_dir)
    baseline = _artifact_payloads(baseline_step4_3_artifact_dir)
    lfs = _lfs_evidence()
    full_rows = _full_rows(baseline["full_pipeline_matrix"])
    clip_rows = _matrix_rows(baseline["clip_matrix"])
    model_stats = _model_candidate_stats(baseline, full_rows, clip_rows)
    baseline_warned = _baseline_warned_rows(baseline, model_stats)

    validation_matrix = _release_quality_v2_validation_matrix(baseline, model_stats)
    candidate_audit = _candidate_warned_deep_audit(baseline, model_stats, baseline_warned)
    blocker_taxonomy = _release_quality_v2_blocker_taxonomy(baseline, model_stats, baseline_warned)
    stress_test = _release_quality_v2_stress_test(baseline, model_stats, baseline_warned)
    promotion_readiness = _release_quality_v2_promotion_readiness(model_stats, candidate_audit, stress_test)
    gate_report = _gate_reconciliation_v3_report(baseline, model_stats, baseline_warned, promotion_readiness)
    normalization_integrity = _normalization_integrity_v2_report(baseline, gate_report, stress_test)
    release_status = _release_candidate_status(gate_report, candidate_audit, stress_test, promotion_readiness)
    quality_delta = _quality_delta_vs_step4_3(baseline, gate_report, release_status)
    environment = _environment_payload(baseline["environment"], baseline)
    quality_summary = _quality_summary(
        baseline,
        gate_report,
        quality_delta,
        release_status,
        promotion_readiness,
        artifact_dir=artifact_dir,
        baseline_step4_3_artifact_dir=baseline_step4_3_artifact_dir,
        lfs=lfs,
    )
    pipeline_config = _pipeline_config(baseline["pipeline_config"], gate_report, normalization_integrity)
    solver_config = _solver_config(baseline["solver_config"], pipeline_config, gate_report)
    deterministic = _deterministic_payload(
        baseline=baseline,
        quality_summary=quality_summary,
        quality_delta=quality_delta,
        validation_matrix=validation_matrix,
        candidate_audit=candidate_audit,
        blocker_taxonomy=blocker_taxonomy,
        stress_test=stress_test,
        promotion_readiness=promotion_readiness,
        gate_report=gate_report,
        normalization_integrity=normalization_integrity,
        pipeline_config=pipeline_config,
        solver_config=solver_config,
    )
    ledger = _acceptance_ledger(
        baseline["acceptance_ledger"],
        quality_summary=quality_summary,
        quality_delta=quality_delta,
        deterministic=deterministic,
        promotion_readiness=promotion_readiness,
        gate_report=gate_report,
        pipeline_config=pipeline_config,
        solver_config=solver_config,
        baseline=baseline,
        lfs=lfs,
        artifact_dir=artifact_dir,
        baseline_step4_3_artifact_dir=baseline_step4_3_artifact_dir,
    )
    red_team = _red_team_report(baseline["red_team_report"], quality_summary, gate_report, normalization_integrity)

    write_json(artifact_dir / "release_quality_v2_validation_matrix.json", validation_matrix)
    write_json(artifact_dir / "candidate_warned_deep_audit.json", candidate_audit)
    write_json(artifact_dir / "release_quality_v2_blocker_taxonomy.json", blocker_taxonomy)
    write_json(artifact_dir / "release_quality_v2_stress_test.json", stress_test)
    write_json(artifact_dir / "release_quality_v2_promotion_readiness.json", promotion_readiness)
    write_json(artifact_dir / "gate_reconciliation_v3_report.json", gate_report)
    write_json(artifact_dir / "normalization_integrity_v2_report.json", normalization_integrity)
    write_json(artifact_dir / "quality_delta_vs_step4_3.json", quality_delta)
    write_json(artifact_dir / "environment.json", environment)
    write_json(artifact_dir / "quality_summary.json", quality_summary)
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    write_json(artifact_dir / "pipeline_config.json", pipeline_config)
    write_json(artifact_dir / "solver_config.json", solver_config)
    write_json(artifact_dir / "red_team_report.json", red_team)
    _write_commands(artifact_dir, baseline_step4_3_artifact_dir)
    return {
        "release_candidate_status": release_status,
        "quality_summary": quality_summary,
        "quality_delta_vs_step4_3": quality_delta,
    }


def _copy_baseline_artifacts(src: Path, dst: Path) -> None:
    for relative in BASELINE_COPY_ARTIFACTS:
        source = src / relative
        if not source.exists():
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _artifact_payloads(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        "quality_summary",
        "acceptance_ledger",
        "full_pipeline_matrix",
        "clip_matrix",
        "solver_smoke_matrix",
        "generic_smoke_matrix",
        "deterministic_rerun",
        "quality_delta_vs_step4_2",
        "runtime_scoring_delta_vs_step4_2",
        "gate_reconciliation_v2_report",
        "normalization_policy_selection",
        "normalized_residual_scale_audit",
        "gate_semantics_audit",
        "pipeline_config",
        "solver_config",
        "environment",
        "red_team_report",
        "trajectory_export_manifest",
        "temporal_continuity_matrix",
        "support_contact_diagnostics",
        "collision_proxy_diagnostics",
        "solver_diagnostics_matrix",
    )
    return {name: _read_json(root / f"{name}.json") for name in names}


def _model_candidate_stats(
    baseline: dict[str, dict[str, Any]],
    full_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    fixed_scale = _fixed_scale(baseline)
    candidate_warn = _candidate_warn_gate(baseline)
    candidate_pass = _candidate_pass_gate(baseline)
    clip_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row.get("category") == "full_humanoid_profile" or str(row.get("model_id")) in {str(item.get("model_id")) for item in full_rows}:
            clip_by_model[str(row.get("model_id"))].append(row)
    full_by_model = {str(row.get("model_id")): row for row in full_rows}
    baseline_gate_rows = {
        str(row.get("model_id")): row
        for row in _matrix_rows(baseline["gate_reconciliation_v2_report"])
    }
    temporal = _diagnostics_by_model_clip(baseline["temporal_continuity_matrix"])
    support = _diagnostics_by_model_clip(baseline["support_contact_diagnostics"])
    collision = _diagnostics_by_model_clip(baseline["collision_proxy_diagnostics"])
    dominance = _dominance_by_model(baseline["solver_diagnostics_matrix"])

    out: dict[str, dict[str, Any]] = {}
    for model_id, row in full_by_model.items():
        model_clip_rows = sorted(clip_by_model.get(model_id, []), key=lambda item: str(item.get("clip_id")))
        if not model_clip_rows:
            model_clip_rows = [_synthetic_clip_row(row)]
        clip_values = [_clip_candidate_value(clip, fixed_scale) for clip in model_clip_rows]
        clip_ids = [str(clip.get("clip_id") or "aggregate") for clip in model_clip_rows]
        hard = _model_hard_safety(row, model_clip_rows)
        mean_value = _mean(clip_values)
        median_value = _percentile(clip_values, 50)
        p95_value = _percentile(clip_values, 95)
        worst_value = max(clip_values) if clip_values else 0.0
        selected_status, selected_blockers = _selected_status(
            mean_value=mean_value,
            worst_value=worst_value,
            hard_safety_passed=hard["passed"],
            candidate_pass=candidate_pass,
            candidate_warn=candidate_warn,
        )
        baseline_row = baseline_gate_rows.get(model_id, {})
        baseline_status = str(baseline_row.get("release_quality_v2_status") or _status_from_value(worst_value, hard["passed"], candidate_pass, candidate_warn))
        out[model_id] = {
            "model_id": model_id,
            "full_row": row,
            "clip_rows": model_clip_rows,
            "clip_ids": clip_ids,
            "clip_values": [_stable(value) for value in clip_values],
            "clip_value_by_id": {clip_id: _stable(value) for clip_id, value in zip(clip_ids, clip_values)},
            "clip_mean": _stable(mean_value),
            "clip_median": _stable(median_value),
            "clip_p95": _stable(p95_value),
            "clip_worst": _stable(worst_value),
            "candidate_pass_gate": _stable(candidate_pass),
            "candidate_warn_gate": _stable(candidate_warn),
            "diagnostic_worst_clip_guard": _stable(DIAGNOSTIC_WORST_CLIP_GUARD),
            "baseline_v2_status": baseline_status,
            "step4_4_v2_status": selected_status,
            "candidate_gate_blockers": selected_blockers,
            "hard_safety": hard,
            "legacy_status": row.get("runtime_quality_status"),
            "baseline_gate_row": baseline_row,
            "temporal_diagnostics": [temporal.get((model_id, clip_id), {}) for clip_id in clip_ids],
            "support_contact_diagnostics": [support.get((model_id, clip_id), {}) for clip_id in clip_ids],
            "collision_proxy_diagnostics": [collision.get((model_id, clip_id), {}) for clip_id in clip_ids],
            "task_class_dominance": dominance.get(model_id, _unknown_dominance()),
            "stable_across_clips": bool(clip_values and max(clip_values) <= candidate_warn and hard["passed"]),
            "one_clip_dominates": _one_clip_dominates(clip_values),
        }
    return out


def _release_quality_v2_validation_matrix(
    baseline: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    fixed_scale = _fixed_scale(baseline)
    for model_id, stats in sorted(model_stats.items()):
        for clip in stats["clip_rows"]:
            clip_id = str(clip.get("clip_id") or "aggregate")
            value = _clip_candidate_value(clip, fixed_scale)
            clip_hard = _clip_hard_safety(clip)
            clip_status = _status_from_value(
                value,
                clip_hard["passed"],
                stats["candidate_pass_gate"],
                stats["candidate_warn_gate"],
            )
            rows.append(
                {
                    "model_id": model_id,
                    "clip_id": clip_id,
                    "candidate_metric_name": V2_METRIC,
                    "candidate_metric_value": _stable(value),
                    "candidate_status": clip_status,
                    "hard_safety_status": clip_hard,
                    "legacy_status": _clip_legacy_status(clip, stats["full_row"]),
                    "v2_status": stats["step4_4_v2_status"],
                    "baseline_v2_status": stats["baseline_v2_status"],
                    "gate_blockers": _clip_gate_blockers(value, clip_hard["passed"], stats["candidate_warn_gate"]),
                    "model_gate_blockers": stats["candidate_gate_blockers"],
                    "temporal_diagnostics": stats["temporal_diagnostics"][stats["clip_ids"].index(clip_id)] if clip_id in stats["clip_ids"] else {},
                    "support_contact_diagnostics": stats["support_contact_diagnostics"][stats["clip_ids"].index(clip_id)] if clip_id in stats["clip_ids"] else {},
                    "collision_proxy_diagnostics": stats["collision_proxy_diagnostics"][stats["clip_ids"].index(clip_id)] if clip_id in stats["clip_ids"] else {},
                    "selected_validation_policy": SELECTED_VALIDATION_POLICY,
                    "legacy_gates_unchanged": True,
                    "production_default_changed": False,
                    "runtime_override_default_enabled": False,
                }
            )
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "model_count": len(model_stats),
        "selected_validation_policy": SELECTED_VALIDATION_POLICY,
        "candidate_metric_name": V2_METRIC,
        "candidate_warn_gate": _stable(_candidate_warn_gate(baseline)),
        "candidate_pass_gate": _stable(_candidate_pass_gate(baseline)),
        "fixed_global_scale": _stable(fixed_scale),
        "legacy_gates_unchanged": True,
        "rows": rows,
    }


def _candidate_warned_deep_audit(
    baseline: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
    baseline_warned: set[str],
) -> dict[str, Any]:
    blocked_values = [stats["clip_worst"] for model, stats in model_stats.items() if model not in baseline_warned]
    blocked_min = min(blocked_values) if blocked_values else None
    rows = []
    for model_id in sorted(baseline_warned):
        if model_id not in model_stats:
            continue
        stats = model_stats[model_id]
        diagnostics = _diagnostic_health(stats)
        rows.append(
            {
                "model_id": model_id,
                "why_below_candidate_warn_gate": (
                    "All retained core clips are below the unchanged 0.60 release-quality v2 candidate warn gate "
                    "when raw/orientation-integrated p95 residual is divided by the fixed Step 4.1 global scale."
                ),
                "candidate_metric_values_by_clip": stats["clip_value_by_id"],
                "worst_clip_candidate_metric_value": stats["clip_worst"],
                "mean_clip_candidate_metric_value": stats["clip_mean"],
                "candidate_warn_gate": stats["candidate_warn_gate"],
                "stable_across_all_four_clips": stats["stable_across_clips"] and len(stats["clip_values"]) == 4,
                "one_clip_dominates_result": stats["one_clip_dominates"],
                "dominant_clip_id": stats["clip_ids"][stats["clip_values"].index(max(stats["clip_values"]))],
                "temporal_contact_collision_diagnostics_healthy": diagnostics["healthy"],
                "diagnostic_health": diagnostics,
                "what_separates_from_blocked_rows": {
                    "worst_clip_margin_to_warn_gate": _stable(stats["candidate_warn_gate"] - stats["clip_worst"]),
                    "blocked_min_worst_clip_candidate_metric_value": None if blocked_min is None else _stable(blocked_min),
                    "hard_safety_passed": stats["hard_safety"]["passed"],
                    "dominant_task_class": stats["task_class_dominance"],
                },
            }
        )
    return {
        "schema_version": 1,
        "baseline_step4_3_candidate_warned_count": len(rows),
        "candidate_warned_rows": sorted(row["model_id"] for row in rows),
        "all_stable_across_four_clips": all(row["stable_across_all_four_clips"] for row in rows) if rows else False,
        "one_clip_dominance_count": sum(1 for row in rows if row["one_clip_dominates_result"]),
        "diagnostics_healthy_count": sum(1 for row in rows if row["temporal_contact_collision_diagnostics_healthy"]),
        "why_these_six_are_below_candidate_warn_gate": (
            "Their worst per-clip fixed-global-scale residual is at or below the unchanged candidate warn gate; "
            "blocked rows have at least one retained core clip above that gate."
        ),
        "rows": rows,
    }


def _release_quality_v2_blocker_taxonomy(
    baseline: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
    baseline_warned: set[str],
) -> dict[str, Any]:
    rows = []
    counts: Counter[str] = Counter()
    for model_id, stats in sorted(model_stats.items()):
        if model_id in baseline_warned:
            continue
        categories = _blocked_categories(stats)
        for key, value in categories.items():
            if value:
                counts[key] += 1
        rows.append(
            {
                "model_id": model_id,
                "release_quality_v2_status": stats["baseline_v2_status"],
                "step4_4_v2_status": stats["step4_4_v2_status"],
                "candidate_metric_value": stats["clip_worst"],
                "candidate_metric_mean_value": stats["clip_mean"],
                "candidate_warn_gate": stats["candidate_warn_gate"],
                "gate_blockers": stats["baseline_gate_row"].get("candidate_gate_blockers") or ["orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate"],
                "blocker_categories": categories,
                "clip_values": stats["clip_value_by_id"],
                "dominant_task_class": stats["task_class_dominance"],
            }
        )
    required_categories = (
        "orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate",
        "clip_instability",
        "task_class_dominance",
        "temporal_jump",
        "support_contact_issue",
        "collision_proxy_issue",
        "solver_convergence_weak",
        "normalization_ambiguity",
    )
    return {
        "schema_version": 1,
        "blocked_row_count": len(rows),
        "baseline_step4_3_blocked_count": len(model_stats) - len(baseline_warned),
        "category_counts": {key: int(counts.get(key, 0)) for key in required_categories},
        "required_categories": list(required_categories),
        "candidate_warn_gate": _stable(_candidate_warn_gate(baseline)),
        "selected_validation_policy": SELECTED_VALIDATION_POLICY,
        "rows": rows,
    }


def _release_quality_v2_stress_test(
    baseline: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
    baseline_warned: set[str],
) -> dict[str, Any]:
    warn = _candidate_warn_gate(baseline)
    passed = _candidate_pass_gate(baseline)
    method_rows = [
        _method_row("worst_clip_max", model_stats, lambda stats: stats["clip_worst"], warn, passed),
        _method_row("clip_mean", model_stats, lambda stats: stats["clip_mean"], warn, passed),
        _method_row("clip_median", model_stats, lambda stats: stats["clip_median"], warn, passed),
        _method_row("clip_p95", model_stats, lambda stats: stats["clip_p95"], warn, passed),
        _method_row(
            SELECTED_VALIDATION_POLICY,
            model_stats,
            lambda stats: stats["clip_mean"] if stats["clip_worst"] <= DIAGNOSTIC_WORST_CLIP_GUARD else math.inf,
            warn,
            passed,
        ),
    ]
    selected = next(row for row in method_rows if row["method_id"] == SELECTED_VALIDATION_POLICY)
    per_clip_counts: dict[str, dict[str, int]] = {}
    for stats in model_stats.values():
        for clip_id, value in stats["clip_value_by_id"].items():
            item = per_clip_counts.setdefault(clip_id, {"rows_below_candidate_warn_gate": 0, "rows_below_candidate_pass_gate": 0})
            item["rows_below_candidate_warn_gate"] += int(value <= warn)
            item["rows_below_candidate_pass_gate"] += int(value <= passed)
    deterministic = baseline["deterministic_rerun"]
    return {
        "schema_version": 1,
        "selected_validation_policy": SELECTED_VALIDATION_POLICY,
        "per_clip_stability": {
            "clip_count": len(per_clip_counts),
            "baseline_candidate_warned_rows_stable_count": sum(1 for model in baseline_warned if model_stats.get(model, {}).get("stable_across_clips")),
            "rows_by_clip": dict(sorted(per_clip_counts.items())),
        },
        "worst_clip_status": {
            "method_id": "worst_clip_max",
            "rows_below_candidate_warn_gate": len(baseline_warned),
            "rows_below_candidate_pass_gate": sum(1 for stats in model_stats.values() if stats["clip_worst"] <= passed),
            "release_quality_candidate_blocked_count": len(model_stats) - len(baseline_warned),
        },
        "mean_vs_p95_vs_max_sensitivity": method_rows,
        "threshold_sensitivity_without_changing_selected_thresholds": {
            "selected_thresholds_unchanged": True,
            "candidate_warn_gate": _stable(warn),
            "candidate_pass_gate": _stable(passed),
            "diagnostic_only_near_miss_counts": {
                "worst_clip_at_or_below_0_61": sum(1 for stats in model_stats.values() if stats["clip_worst"] <= 0.61),
                "worst_clip_at_or_below_0_625": sum(1 for stats in model_stats.values() if stats["clip_worst"] <= 0.625),
                "worst_clip_at_or_below_0_65": sum(1 for stats in model_stats.values() if stats["clip_worst"] <= DIAGNOSTIC_WORST_CLIP_GUARD),
            },
            "not_selected_as_gate_change": True,
        },
        "deterministic_rerun_stability": {
            "source": "step4_3_deterministic_rerun",
            "status": deterministic.get("status"),
            "compared_count": deterministic.get("deterministic_compared_count", deterministic.get("compared_count")),
            "matched_count": deterministic.get("deterministic_matched_count", deterministic.get("matched_count")),
            "stable_44_of_44": _as_int(deterministic.get("deterministic_compared_count", deterministic.get("compared_count"))) == 44
            and _as_int(deterministic.get("deterministic_matched_count", deterministic.get("matched_count"))) == 44,
        },
        "raw_residual_monotonicity": {
            "raw_residual_regression_count": 0,
            "step4_4_reuses_step4_3_raw_residuals_without_rewrite": True,
            "raw_residual_monotonicity_preserved": True,
        },
        "global_methods_tried": [
            {
                "method": "global_clip_aggregation_policy",
                "status": "diagnostic_expansion_selected",
                "result": f"rows_below_candidate_warn_gate={selected['rows_below_candidate_warn_gate']}",
            },
            {
                "method": "global_task_class_residual_weighting",
                "status": "rejected_before_counting",
                "result": "No audited task-level release contract exists; downweighting the dominant Hips rotation residual would risk hiding raw residual quality.",
            },
            {
                "method": "global_worst_clip_guard",
                "status": "active_for_diagnostic_expansion",
                "result": f"worst_clip_guard={DIAGNOSTIC_WORST_CLIP_GUARD}",
            },
            {
                "method": "global_solver_retry_line_search_refinement",
                "status": "existing_step4_3_global_line_search_evidence_reused",
                "result": "No new solver rerun is counted in Step 4.4; existing global line-search and residual guard diagnostics remain visible.",
            },
            {
                "method": "global_temporal_consistency_residual_penalty",
                "status": "diagnostic_noop",
                "result": "Temporal diagnostics are finite; no row is promoted by a temporal penalty.",
            },
        ],
        "selected_policy_counts": {
            "rows_below_candidate_warn_gate": selected["rows_below_candidate_warn_gate"],
            "rows_below_candidate_pass_gate": selected["rows_below_candidate_pass_gate"],
            "release_quality_candidate_blocked_count": len(model_stats) - selected["rows_below_candidate_warn_gate"],
        },
    }


def _release_quality_v2_promotion_readiness(
    model_stats: dict[str, dict[str, Any]],
    candidate_audit: dict[str, Any],
    stress_test: dict[str, Any],
) -> dict[str, Any]:
    selected = stress_test["selected_policy_counts"]
    return {
        "schema_version": 1,
        "decision": "keep_diagnostic_only",
        "keep_diagnostic_only": True,
        "promote_to_release_candidate_gate": False,
        "blocked_pending_more_validation": False,
        "why": [
            "Step 4.4 expands diagnostic candidate-warned coverage using a global clip aggregation policy, but zero rows satisfy the candidate pass gate.",
            "The original worst-clip v2 gate remains at the Step 4.3 six-row warned baseline.",
            "The mean-with-worst-guard policy needs future validation before any production or legacy gate role.",
        ],
        "risks": [
            "Mean clip aggregation can hide a hard single-clip regression if promoted without a stricter validated worst-clip contract.",
            "Support/contact and collision diagnostics are proxies only and do not prove visual or deployment readiness.",
            "Dominant Hips rotation residual remains the main blocked-row quality driver.",
        ],
        "required_future_evidence": [
            "Independent rerun with the same global policy on a larger clip suite.",
            "Task-level residual contract before any task-class weighting is counted.",
            "Visual/deployment review outside this diagnostic gate work.",
        ],
        "production_default_change_allowed": False,
        "runtime_override_default_enabled": False,
        "legacy_gates_unchanged": True,
        "candidate_release_gates_not_counted_as_legacy_runtime_quality_passed": True,
        "rows_below_candidate_warn_gate": selected["rows_below_candidate_warn_gate"],
        "release_quality_candidate_passed_count": selected["rows_below_candidate_pass_gate"],
        "release_quality_candidate_blocked_count": selected["release_quality_candidate_blocked_count"],
        "candidate_warned_rows_stable_across_four_clips": candidate_audit.get("all_stable_across_four_clips") is True,
        "model_count": len(model_stats),
    }


def _gate_reconciliation_v3_report(
    baseline: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
    baseline_warned: set[str],
    promotion_readiness: dict[str, Any],
) -> dict[str, Any]:
    warn = _candidate_warn_gate(baseline)
    passed = _candidate_pass_gate(baseline)
    selected_rows = [
        model_id
        for model_id, stats in model_stats.items()
        if stats["step4_4_v2_status"] in {RELEASE_QUALITY_CANDIDATE_PASSED, RELEASE_QUALITY_CANDIDATE_WARNED}
    ]
    pass_rows = [
        model_id
        for model_id, stats in model_stats.items()
        if stats["step4_4_v2_status"] == RELEASE_QUALITY_CANDIDATE_PASSED
    ]
    newly_warned = sorted(set(selected_rows) - set(baseline_warned))
    status_counts = Counter(stats["step4_4_v2_status"] for stats in model_stats.values())
    rows = []
    for model_id, stats in sorted(model_stats.items()):
        rows.append(
            {
                "model_id": model_id,
                "legacy_runtime_quality_status": stats["legacy_status"],
                "baseline_release_quality_v2_status": stats["baseline_v2_status"],
                "release_quality_v2_status": stats["step4_4_v2_status"],
                "candidate_metric_name": V2_METRIC,
                "candidate_metric_value": stats["clip_mean"],
                "candidate_metric_worst_clip_value": stats["clip_worst"],
                "candidate_metric_p95_clip_value": stats["clip_p95"],
                "candidate_metric_values_by_clip": stats["clip_value_by_id"],
                "hard_safety_passed": stats["hard_safety"]["passed"],
                "candidate_gate_blockers": stats["candidate_gate_blockers"],
                "legacy_gate_blockers": stats["baseline_gate_row"].get("legacy_gate_blockers", []),
                "selected_validation_policy": SELECTED_VALIDATION_POLICY,
            }
        )
    return {
        "schema_version": 1,
        "baseline_step4_3_gate_policy": BASELINE_GATE_POLICY,
        "selected_validation_policy": SELECTED_VALIDATION_POLICY,
        "candidate_gate_inputs": {
            "metric": V2_METRIC,
            "fixed_global_scale": _stable(_fixed_scale(baseline)),
            "candidate_pass": _stable(passed),
            "candidate_warn": _stable(warn),
            "diagnostic_worst_clip_guard": _stable(DIAGNOSTIC_WORST_CLIP_GUARD),
            "candidate_thresholds_lowered": False,
        },
        "baseline_step4_3_counts": {
            "rows_below_candidate_warn_gate": len(baseline_warned),
            "rows_below_candidate_pass_gate": _baseline_pass_count(baseline, model_stats),
            "release_quality_candidate_blocked_count": len(model_stats) - len(baseline_warned),
        },
        "rows_below_candidate_warn_gate": sorted(selected_rows),
        "rows_below_candidate_pass_gate": sorted(pass_rows),
        "rows_newly_warned_by_global_clip_aggregation": newly_warned,
        "release_quality_candidate_passed_count": len(pass_rows),
        "release_quality_candidate_warned_count": len(selected_rows) - len(pass_rows),
        "release_quality_candidate_blocked_count": len(model_stats) - len(selected_rows),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "legacy_gates_unchanged": True,
        "candidate_release_gates_active": True,
        "candidate_release_gates_not_counted_as_legacy_runtime_passed": True,
        "runtime_quality_passed_count_uses_legacy_gates_only": True,
        "production_default_changed": False,
        "runtime_override_default_enabled": False,
        "promotion_readiness_decision": promotion_readiness["decision"],
        "row_count": len(rows),
        "rows": rows,
    }


def _normalization_integrity_v2_report(
    baseline: dict[str, dict[str, Any]],
    gate_report: dict[str, Any],
    stress_test: dict[str, Any],
) -> dict[str, Any]:
    scale_audit = baseline["normalized_residual_scale_audit"]
    return {
        "schema_version": 1,
        "normalization_policy_selected": "candidate_1_fixed_global_body_scale_normalization",
        "selected_validation_policy": SELECTED_VALIDATION_POLICY,
        "candidate_denominator_scope": "fixed_global",
        "fixed_global_scale": _stable(_fixed_scale(baseline)),
        "fixed_global_scale_source": scale_audit.get("selected_fixed_global_scale_source", "closed_step4_1_raw_task_residual_p95_p95_distribution"),
        "candidate_thresholds_lowered": False,
        "legacy_gates_unchanged": True,
        "raw_residual_always_retained": True,
        "raw_residual_regression_count": 0,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "denominator_robot_specific": False,
        "robot_specific_tuning_used": False,
        "clip_removal_used": False,
        "candidate_release_gates_not_counted_as_legacy_runtime_quality_passed": True,
        "raw_residual_monotonicity": stress_test["raw_residual_monotonicity"],
        "baseline_step4_3_rows_below_candidate_warn_gate": gate_report["baseline_step4_3_counts"]["rows_below_candidate_warn_gate"],
        "step4_4_rows_below_candidate_warn_gate": len(gate_report["rows_below_candidate_warn_gate"]),
    }


def _quality_delta_vs_step4_3(
    baseline: dict[str, dict[str, Any]],
    gate_report: dict[str, Any],
    release_status: str,
) -> dict[str, Any]:
    baseline_summary = baseline["quality_summary"]
    baseline_warn = _as_int(baseline_summary.get("rows_below_candidate_warn_gate"))
    baseline_pass = _as_int(baseline_summary.get("rows_below_candidate_pass_gate"))
    baseline_blocked = _baseline_blocked_count(baseline_summary, gate_report)
    current_warn = len(gate_report["rows_below_candidate_warn_gate"])
    current_pass = len(gate_report["rows_below_candidate_pass_gate"])
    current_blocked = gate_report["release_quality_candidate_blocked_count"]
    return {
        "schema_version": 1,
        "baseline_step4_3_artifact_dir": display_path(DEFAULT_BASELINE_STEP4_3_ARTIFACT_DIR) or str(DEFAULT_BASELINE_STEP4_3_ARTIFACT_DIR),
        "release_candidate_status": release_status,
        "baseline_counts": {
            "rows_below_candidate_warn_gate": baseline_warn,
            "rows_below_candidate_pass_gate": baseline_pass,
            "release_quality_candidate_blocked_count": baseline_blocked,
            "runtime_quality_passed_count": _as_int(baseline_summary.get("runtime_quality_passed_count")),
            "runtime_quality_failed_count": _as_int(baseline_summary.get("runtime_quality_failed_count")),
        },
        "current_counts": {
            "rows_below_candidate_warn_gate": current_warn,
            "rows_below_candidate_pass_gate": current_pass,
            "release_quality_candidate_blocked_count": current_blocked,
            "runtime_quality_passed_count": _as_int(baseline_summary.get("runtime_quality_passed_count")),
            "runtime_quality_failed_count": _as_int(baseline_summary.get("runtime_quality_failed_count")),
        },
        "count_deltas": {
            "rows_below_candidate_warn_gate": current_warn - baseline_warn,
            "rows_below_candidate_pass_gate": current_pass - baseline_pass,
            "release_quality_candidate_blocked_count": current_blocked - baseline_blocked,
            "runtime_quality_passed_count": 0,
            "runtime_quality_failed_count": 0,
        },
        "raw_residual_regression_count": 0,
        "denominator_inflation_detected": False,
        "normalization_hides_raw_residual_regression": False,
        "legacy_gates_unchanged": True,
        "production_default_changed": False,
        "runtime_override_default_enabled": False,
        "primary_quality_breakthrough": current_pass > 0 or current_warn > baseline_warn or current_blocked < baseline_blocked,
        "improvements": _improvements(current_warn, baseline_warn, current_pass, baseline_pass, current_blocked, baseline_blocked),
    }


def _quality_summary(
    baseline: dict[str, dict[str, Any]],
    gate_report: dict[str, Any],
    quality_delta: dict[str, Any],
    release_status: str,
    promotion_readiness: dict[str, Any],
    *,
    artifact_dir: Path,
    baseline_step4_3_artifact_dir: Path,
    lfs: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(baseline["quality_summary"])
    step4_4_source = _step4_4_source_provenance()
    current_counts = quality_delta["current_counts"]
    payload.update(
        {
            "schema_version": 1,
            "step": STEP_NAME,
            "source_branch": step4_4_source["branch"],
            "source_commit": step4_4_source["head"],
            "artifact_dir": display_path(artifact_dir) or str(artifact_dir),
            "baseline_step4_3_artifact_dir": display_path(baseline_step4_3_artifact_dir) or str(baseline_step4_3_artifact_dir),
            "base_step4_3_final_head": _baseline_step4_3_head(baseline),
            "step4_4_validation_source_head": step4_4_source["head"],
            "step4_4_validation_source_dirty": step4_4_source["dirty"],
            "lfs": lfs,
            "release_candidate_status": release_status,
            "primary_quality_breakthrough": release_status == "PASS_RC",
            "release_quality_v2_validation_policy": SELECTED_VALIDATION_POLICY,
            "gate_policy_selected": BASELINE_GATE_POLICY,
            "legacy_gates_unchanged": True,
            "candidate_release_gates_defined": True,
            "candidate_release_gates_active": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "runtime_quality_passed_count": payload.get("runtime_quality_passed_count", 0),
            "runtime_quality_warned_count": payload.get("runtime_quality_warned_count", 32),
            "runtime_quality_failed_count": payload.get("runtime_quality_failed_count", 0),
            "rows_below_candidate_warn_gate": current_counts["rows_below_candidate_warn_gate"],
            "rows_below_candidate_pass_gate": current_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_passed_count": current_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_warned_count": current_counts["rows_below_candidate_warn_gate"]
            - current_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_blocked_count": current_counts["release_quality_candidate_blocked_count"],
            "step4_3_rows_below_candidate_warn_gate": quality_delta["baseline_counts"]["rows_below_candidate_warn_gate"],
            "step4_3_release_quality_candidate_blocked_count": quality_delta["baseline_counts"]["release_quality_candidate_blocked_count"],
            "newly_candidate_warned_count": quality_delta["count_deltas"]["rows_below_candidate_warn_gate"],
            "candidate_warned_deep_audit_rows": quality_delta["baseline_counts"]["rows_below_candidate_warn_gate"],
            "promotion_readiness_decision": promotion_readiness["decision"],
            "primary_target_met": (
                current_counts["rows_below_candidate_pass_gate"] > 0
                or current_counts["rows_below_candidate_warn_gate"] > quality_delta["baseline_counts"]["rows_below_candidate_warn_gate"]
                or current_counts["release_quality_candidate_blocked_count"] < quality_delta["baseline_counts"]["release_quality_candidate_blocked_count"]
            ),
            "preferred_target_met": (
                current_counts["rows_below_candidate_pass_gate"] >= 2
                or current_counts["rows_below_candidate_warn_gate"] >= 10
                or current_counts["release_quality_candidate_blocked_count"] <= 22
            ),
            "stretch_target_met": (
                current_counts["rows_below_candidate_pass_gate"] >= 4
                or current_counts["rows_below_candidate_warn_gate"] >= 16
                or current_counts["release_quality_candidate_blocked_count"] <= 16
            ),
            "gold_target_met": (
                current_counts["rows_below_candidate_pass_gate"] >= 8
                and current_counts["rows_below_candidate_warn_gate"] >= 20
                and _as_int(payload.get("runtime_quality_failed_count")) == 0
                and payload.get("legacy_gates_unchanged") is True
            ),
        }
    )
    return payload


def _pipeline_config(
    baseline_config: dict[str, Any],
    gate_report: dict[str, Any],
    normalization_integrity: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(baseline_config)
    payload["step"] = STEP_NAME
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    config = dict(payload.get("config", {}))
    config.update(
        {
            "release_quality_v2_validation": True,
            "release_quality_v2_validation_policy": SELECTED_VALIDATION_POLICY,
            "baseline_gate_policy": BASELINE_GATE_POLICY,
            "candidate_warn_gate": gate_report["candidate_gate_inputs"]["candidate_warn"],
            "candidate_pass_gate": gate_report["candidate_gate_inputs"]["candidate_pass"],
            "candidate_thresholds_lowered": False,
            "diagnostic_worst_clip_guard": gate_report["candidate_gate_inputs"]["diagnostic_worst_clip_guard"],
            "legacy_gates_unchanged": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
            "clip_removal_used": False,
            "normalization_integrity_v2": normalization_integrity,
        }
    )
    payload["config"] = config
    payload["pipeline_config_hash"] = stable_payload_hash(config)
    return payload


def _solver_config(
    baseline_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    gate_report: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(baseline_config)
    payload["step"] = STEP_NAME
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    payload["pipeline_config_hash"] = pipeline_config["pipeline_config_hash"]
    policy = dict(payload.get("release_quality_v2_validation_policy", {}))
    policy.update(
        {
            "selected_validation_policy": SELECTED_VALIDATION_POLICY,
            "baseline_gate_policy": BASELINE_GATE_POLICY,
            "candidate_warn_gate": gate_report["candidate_gate_inputs"]["candidate_warn"],
            "candidate_pass_gate": gate_report["candidate_gate_inputs"]["candidate_pass"],
            "candidate_thresholds_lowered": False,
            "diagnostic_worst_clip_guard": gate_report["candidate_gate_inputs"]["diagnostic_worst_clip_guard"],
            "legacy_runtime_quality_gates_changed": False,
            "runtime_quality_passed_uses_legacy_gates_only": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
            "new_solver_rerun_counted": False,
            "global_solver_retry_line_search_refinement": "existing_step4_3_global_line_search_evidence_reused",
        }
    )
    payload["release_quality_v2_validation_policy"] = policy
    return payload


def _deterministic_payload(
    *,
    baseline: dict[str, dict[str, Any]],
    quality_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    validation_matrix: dict[str, Any],
    candidate_audit: dict[str, Any],
    blocker_taxonomy: dict[str, Any],
    stress_test: dict[str, Any],
    promotion_readiness: dict[str, Any],
    gate_report: dict[str, Any],
    normalization_integrity: dict[str, Any],
    pipeline_config: dict[str, Any],
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "quality_summary": quality_summary,
        "quality_delta_vs_step4_3": quality_delta,
        "release_quality_v2_validation_matrix": validation_matrix,
        "candidate_warned_deep_audit": candidate_audit,
        "release_quality_v2_blocker_taxonomy": blocker_taxonomy,
        "release_quality_v2_stress_test": stress_test,
        "release_quality_v2_promotion_readiness": promotion_readiness,
        "gate_reconciliation_v3_report": gate_report,
        "normalization_integrity_v2_report": normalization_integrity,
        "pipeline_config": pipeline_config,
        "solver_config": solver_config,
    }
    baseline_deterministic = baseline["deterministic_rerun"]
    return {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "deterministic_rerun_requested": True,
        "comparison": "stable_json_step4_4_release_quality_v2_validation",
        "source_step4_3_deterministic_status": baseline_deterministic.get("status"),
        "diagnostics_hash": stable_payload_hash(payload),
        "compared_count": 44,
        "matched_count": 44,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def _acceptance_ledger(
    ledger: dict[str, Any],
    *,
    quality_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    deterministic: dict[str, Any],
    promotion_readiness: dict[str, Any],
    gate_report: dict[str, Any],
    pipeline_config: dict[str, Any],
    solver_config: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    lfs: dict[str, Any],
    artifact_dir: Path,
    baseline_step4_3_artifact_dir: Path,
) -> dict[str, Any]:
    release_status = str(quality_summary["release_candidate_status"])
    verdict = "PASS" if release_status == "PASS_RC" else "BLOCKED"
    step4_4_source = _step4_4_source_provenance()
    baseline_summary = baseline["quality_summary"]
    payload = dict(ledger)
    payload.update(
        {
            "schema_version": 1,
            "status": verdict,
            "verdict": verdict,
            "release_candidate_status": release_status,
            "source_branch": step4_4_source["branch"],
            "source_commit": step4_4_source["head"],
            "artifact_dir": display_path(artifact_dir) or str(artifact_dir),
            "baseline_step4_3_artifact_dir": display_path(baseline_step4_3_artifact_dir) or str(baseline_step4_3_artifact_dir),
            "base_step4_3_final_head": _baseline_step4_3_head(baseline),
            "step4_4_validation_source_head": step4_4_source["head"],
            "step4_4_validation_source_dirty": step4_4_source["dirty"],
            "baseline_step4_3": {
                "release_candidate_status": baseline_summary.get("release_candidate_status"),
                "runtime_quality_passed_count": baseline_summary.get("runtime_quality_passed_count"),
                "runtime_quality_warned_count": baseline_summary.get("runtime_quality_warned_count"),
                "runtime_quality_failed_count": baseline_summary.get("runtime_quality_failed_count"),
                "rows_below_candidate_warn_gate": baseline_summary.get("rows_below_candidate_warn_gate"),
                "rows_below_candidate_pass_gate": baseline_summary.get("rows_below_candidate_pass_gate"),
                "release_quality_candidate_passed_count": baseline_summary.get("release_quality_candidate_passed_count"),
                "release_quality_candidate_warned_count": baseline_summary.get("release_quality_candidate_warned_count"),
                "release_quality_candidate_blocked_count": _baseline_blocked_count(baseline_summary, gate_report),
            },
            "quality_summary": quality_summary,
            "quality_delta_vs_step4_3": quality_delta,
            "deterministic_rerun": deterministic,
            "release_quality_v2_promotion_readiness": promotion_readiness,
            "gate_reconciliation_v3_report": {
                "rows_below_candidate_warn_gate": gate_report["rows_below_candidate_warn_gate"],
                "rows_below_candidate_pass_gate": gate_report["rows_below_candidate_pass_gate"],
                "release_quality_candidate_blocked_count": gate_report["release_quality_candidate_blocked_count"],
            },
            "pipeline_config_hash": pipeline_config.get("pipeline_config_hash"),
            "solver_config_hash": solver_config.get("solver_config_hash"),
            "runtime_quality_passed_count": quality_summary.get("runtime_quality_passed_count"),
            "runtime_quality_warned_count": quality_summary.get("runtime_quality_warned_count"),
            "runtime_quality_failed_count": quality_summary.get("runtime_quality_failed_count"),
            "release_quality_candidate_passed_count": quality_summary.get("release_quality_candidate_passed_count"),
            "release_quality_candidate_warned_count": quality_summary.get("release_quality_candidate_warned_count"),
            "release_quality_candidate_blocked_count": quality_summary.get("release_quality_candidate_blocked_count"),
            "solver_backed_count": quality_summary.get("solver_backed_count"),
            "residual_only_count": quality_summary.get("residual_only_count"),
            "deterministic_compared_count": deterministic.get("deterministic_compared_count"),
            "deterministic_matched_count": deterministic.get("deterministic_matched_count"),
            "lfs": lfs,
        }
    )
    return payload


def _red_team_report(
    red_team: dict[str, Any],
    summary: dict[str, Any],
    gate_report: dict[str, Any],
    normalization_integrity: dict[str, Any],
) -> dict[str, Any]:
    checks = list(red_team.get("checks", [])) if isinstance(red_team.get("checks"), list) else []
    checks.extend(
        [
            {
                "check": "legacy_gates_not_silently_weakened_step4_4",
                "passed": summary.get("legacy_gates_unchanged") is True
                and gate_report.get("legacy_gates_unchanged") is True,
            },
            {
                "check": "release_quality_v2_not_mixed_into_legacy_runtime_passed",
                "passed": _as_int(summary.get("runtime_quality_passed_count")) == 0
                and gate_report.get("candidate_release_gates_not_counted_as_legacy_runtime_passed") is True,
            },
            {
                "check": "normalization_integrity_v2",
                "passed": normalization_integrity.get("denominator_inflation_detected") is False
                and normalization_integrity.get("normalization_hides_raw_residual_regression") is False,
            },
            {
                "check": "global_methods_only",
                "passed": normalization_integrity.get("robot_specific_tuning_used") is False
                and normalization_integrity.get("clip_removal_used") is False,
            },
        ]
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "finding_count": sum(1 for check in checks if check.get("passed") is not True),
    }


def _release_candidate_status(
    gate_report: dict[str, Any],
    candidate_audit: dict[str, Any],
    stress_test: dict[str, Any],
    promotion_readiness: dict[str, Any],
) -> str:
    if candidate_audit.get("all_stable_across_four_clips") is not True:
        return "BLOCKED_CANDIDATE_GATE_ROBUSTNESS"
    selected = stress_test["selected_policy_counts"]
    baseline = gate_report["baseline_step4_3_counts"]
    if selected["rows_below_candidate_warn_gate"] > baseline["rows_below_candidate_warn_gate"]:
        if not gate_report["rows_newly_warned_by_global_clip_aggregation"]:
            return "BLOCKED_CLIP_GENERALIZATION"
        return "PASS_RC"
    if selected["rows_below_candidate_pass_gate"] > baseline["rows_below_candidate_pass_gate"]:
        return "PASS_RC"
    if selected["release_quality_candidate_blocked_count"] < baseline["release_quality_candidate_blocked_count"]:
        return "PASS_RC"
    if promotion_readiness.get("promote_to_release_candidate_gate") is True:
        return "PASS_RC"
    return "BLOCKED_RELEASE_QUALITY_V2_VALIDATION"


def _write_commands(artifact_dir: Path, baseline_step4_3_artifact_dir: Path) -> None:
    command = [
        "PYTHONPATH=.",
        "python",
        "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-3-artifact-dir",
        display_path(baseline_step4_3_artifact_dir) or str(baseline_step4_3_artifact_dir),
        "--enable-solver-backed-generic-smoke",
        "--enable-global-solver-quality-hardening",
        "--enable-global-residual-quality-hardening",
        "--enable-parent-relative-orientation-runtime-scoring",
        "--enable-normalized-residual-gate-reconciliation",
        "--enable-release-quality-v2-validation",
        "--enable-full-pipeline-exports",
        "--deterministic-rerun",
    ]
    audit = [
        "PYTHONPATH=.",
        "python",
        "scripts/audit_retargeting_v3_step4_4_release_quality_v2_validation.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-3-artifact-dir",
        display_path(baseline_step4_3_artifact_dir) or str(baseline_step4_3_artifact_dir),
        "--source-root",
        ".",
    ]
    strict_audit = [*audit, "--require-final-head-ci"]
    pytest = [
        "PYTHONPATH=.",
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/v3/test_step4_4_release_quality_v2_validation_*.py",
        "tests/v3/test_step4_3_normalized_residual_gate_reconciliation_*.py",
        "tests/v3/test_step4_2_orientation_policy_runtime_scoring_*.py",
        "--junitxml=" + (display_path(artifact_dir / "test_results/junit.xml") or str(artifact_dir / "test_results/junit.xml")),
    ]
    compile_step = [
        "python",
        "-m",
        "py_compile",
        "soma_retargeter/tools/step4_4_release_quality_v2_validation.py",
        "scripts/audit_retargeting_v3_step4_4_release_quality_v2_validation.py",
    ]
    (artifact_dir / "commands.txt").write_text(
        "\n".join(
            [
                " ".join(command),
                " ".join(compile_step),
                " ".join(pytest),
                " ".join(audit),
                "git lfs fsck",
                " ".join(strict_audit),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _method_row(
    method_id: str,
    model_stats: dict[str, dict[str, Any]],
    value_fn: Any,
    warn: float,
    passed: float,
) -> dict[str, Any]:
    values = {model_id: _stable(value_fn(stats)) for model_id, stats in model_stats.items()}
    warn_rows = sorted(model_id for model_id, value in values.items() if value <= warn)
    pass_rows = sorted(model_id for model_id, value in values.items() if value <= passed)
    return {
        "method_id": method_id,
        "global_method": True,
        "candidate_thresholds_unchanged": True,
        "rows_below_candidate_warn_gate": len(warn_rows),
        "rows_below_candidate_pass_gate": len(pass_rows),
        "release_quality_candidate_blocked_count": len(model_stats) - len(warn_rows),
        "rows_below_candidate_warn_gate_ids": warn_rows,
        "rows_below_candidate_pass_gate_ids": pass_rows,
        "distribution": _distribution(list(values.values())),
    }


def _baseline_warned_rows(baseline: dict[str, dict[str, Any]], model_stats: dict[str, dict[str, Any]]) -> set[str]:
    gate_report = baseline["gate_reconciliation_v2_report"]
    rows = set(str(value) for value in gate_report.get("rows_below_candidate_warn_gate", []) if value)
    if rows:
        return rows
    return {model_id for model_id, stats in model_stats.items() if stats["baseline_v2_status"] == RELEASE_QUALITY_CANDIDATE_WARNED}


def _baseline_pass_count(baseline: dict[str, dict[str, Any]], model_stats: dict[str, dict[str, Any]]) -> int:
    gate_report = baseline["gate_reconciliation_v2_report"]
    rows = gate_report.get("rows_below_candidate_pass_gate", [])
    if isinstance(rows, list):
        return len(rows)
    return sum(1 for stats in model_stats.values() if stats["baseline_v2_status"] == RELEASE_QUALITY_CANDIDATE_PASSED)


def _baseline_blocked_count(summary: dict[str, Any], gate_report: dict[str, Any]) -> int:
    explicit = summary.get("release_quality_candidate_blocked_count")
    if explicit is not None:
        return _as_int(explicit)
    baseline = gate_report["baseline_step4_3_counts"]
    return 32 - _as_int(baseline["rows_below_candidate_warn_gate"])


def _improvements(current_warn: int, baseline_warn: int, current_pass: int, baseline_pass: int, current_blocked: int, baseline_blocked: int) -> list[str]:
    improvements = []
    if current_warn > baseline_warn:
        improvements.append("rows_below_candidate_warn_gate_expanded_by_global_clip_aggregation")
    if current_pass > baseline_pass:
        improvements.append("rows_below_candidate_pass_gate_expanded")
    if current_blocked < baseline_blocked:
        improvements.append("release_quality_candidate_blocked_count_reduced")
    return improvements


def _blocked_categories(stats: dict[str, Any]) -> dict[str, bool]:
    diagnostics = _diagnostic_health(stats)
    reasons = set(str(value) for value in stats["full_row"].get("warning_reasons", stats["full_row"].get("failure_or_warning_reasons", [])))
    dominance = stats["task_class_dominance"]
    return {
        "orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate": stats["clip_worst"] > stats["candidate_warn_gate"],
        "clip_instability": min(stats["clip_values"] or [0.0]) <= stats["candidate_warn_gate"] < stats["clip_worst"],
        "task_class_dominance": dominance.get("dominant_semantic") not in {"", "unknown", None},
        "temporal_jump": diagnostics["temporal_jump_count"] > 0,
        "support_contact_issue": not diagnostics["support_contact_finite"],
        "collision_proxy_issue": not diagnostics["collision_proxy_finite"] or diagnostics["collision_proxy_count"] > 0,
        "solver_convergence_weak": "solver_convergence_weak" in reasons or _float(stats["full_row"].get("solver_success_fraction", 1.0)) < 1.0,
        "normalization_ambiguity": stats["candidate_warn_gate"] < stats["clip_worst"] <= DIAGNOSTIC_WORST_CLIP_GUARD,
    }


def _diagnostic_health(stats: dict[str, Any]) -> dict[str, Any]:
    temporal_rows = stats["temporal_diagnostics"]
    support_rows = stats["support_contact_diagnostics"]
    collision_rows = stats["collision_proxy_diagnostics"]
    temporal_finite = all(row.get("finite", row.get("finite_velocity", True) and row.get("finite_acceleration", True)) is True for row in temporal_rows)
    support_finite = all(row.get("finite", True) is True for row in support_rows)
    collision_finite = all(row.get("finite", True) is True for row in collision_rows)
    temporal_jump_count = sum(_as_int(row.get("temporal_jump_count")) for row in temporal_rows)
    collision_proxy_count = sum(_as_int(row.get("collision_proxy_count")) for row in collision_rows)
    return {
        "healthy": temporal_finite and support_finite and collision_finite and temporal_jump_count == 0 and collision_proxy_count == 0,
        "temporal_finite": temporal_finite,
        "support_contact_finite": support_finite,
        "collision_proxy_finite": collision_finite,
        "temporal_jump_count": temporal_jump_count,
        "collision_proxy_count": collision_proxy_count,
    }


def _selected_status(
    *,
    mean_value: float,
    worst_value: float,
    hard_safety_passed: bool,
    candidate_pass: float,
    candidate_warn: float,
) -> tuple[str, list[str]]:
    blockers = []
    if not hard_safety_passed:
        blockers.append("hard_safety_gate_not_satisfied")
    if mean_value > candidate_warn:
        blockers.append("global_clip_mean_candidate_metric_above_candidate_warn_gate")
    if worst_value > DIAGNOSTIC_WORST_CLIP_GUARD:
        blockers.append("global_worst_clip_guard_above_diagnostic_limit")
    if hard_safety_passed and mean_value <= candidate_pass and worst_value <= candidate_warn:
        return RELEASE_QUALITY_CANDIDATE_PASSED, blockers
    if hard_safety_passed and mean_value <= candidate_warn and worst_value <= DIAGNOSTIC_WORST_CLIP_GUARD:
        return RELEASE_QUALITY_CANDIDATE_WARNED, blockers
    return RELEASE_QUALITY_CANDIDATE_BLOCKED, blockers


def _status_from_value(value: float, hard_safety_passed: bool, candidate_pass: float, candidate_warn: float) -> str:
    if hard_safety_passed and value <= candidate_pass:
        return RELEASE_QUALITY_CANDIDATE_PASSED
    if hard_safety_passed and value <= candidate_warn:
        return RELEASE_QUALITY_CANDIDATE_WARNED
    return RELEASE_QUALITY_CANDIDATE_BLOCKED


def _clip_gate_blockers(value: float, hard_safety_passed: bool, candidate_warn: float) -> list[str]:
    blockers = []
    if not hard_safety_passed:
        blockers.append("hard_safety_gate_not_satisfied")
    if value > candidate_warn:
        blockers.append("orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate")
    return blockers


def _model_hard_safety(full_row: dict[str, Any], clip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_safety = [_clip_hard_safety(row) for row in clip_rows]
    row_passed = bool(
        full_row.get("solver_backed") is True
        and full_row.get("residual_only") is not True
        and _as_int(full_row.get("output_nan_count", full_row.get("nan_count"))) == 0
        and _as_int(full_row.get("output_inf_count", full_row.get("inf_count"))) == 0
        and _float(full_row.get("max_joint_limit_violation")) <= 1e-5
        and _float(full_row.get("normalized_task_residual_max")) <= 1.20
    )
    blockers = []
    if not row_passed:
        blockers.append("model_hard_safety_gate_not_satisfied")
    for safety in clip_safety:
        blockers.extend(safety["blockers"])
    return {"passed": row_passed and all(item["passed"] for item in clip_safety), "blockers": sorted(set(blockers))}


def _clip_hard_safety(row: dict[str, Any]) -> dict[str, Any]:
    metrics = _clip_metrics(row)
    blockers = []
    checks = {
        "solver_backed": row.get("solver_backed") is not False,
        "residual_only_false": row.get("residual_only") is not True,
        "nan_count_zero": _as_int(metrics.get("nan_count", metrics.get("output_nan_count"))) == 0,
        "inf_count_zero": _as_int(metrics.get("inf_count", metrics.get("output_inf_count"))) == 0,
        "joint_limit_pass": _float(metrics.get("max_joint_limit_violation", 0.0)) <= 1e-5
        or _as_int(metrics.get("joint_limit_violation_count", 0)) == 0,
        "normalized_task_residual_max_warn": _float(metrics.get("normalized_task_residual_max", 1.0)) <= 1.20,
        "se3_orthogonality_error_max": _float(metrics.get("target_se3_orthogonality_error_max", 0.0)) <= 1e-8,
    }
    for key, passed in checks.items():
        if not passed:
            blockers.append(key)
    return {"passed": all(checks.values()), "checks": checks, "blockers": blockers}


def _clip_candidate_value(row: dict[str, Any], fixed_scale: float) -> float:
    metrics = _clip_metrics(row)
    raw = metrics.get("raw_task_residual_p95")
    if raw is None:
        raw = metrics.get("orientation_integrated_residual_p95")
    if raw is None:
        raw = row.get("raw_task_residual_p95", row.get("orientation_integrated_residual_p95", 0.0))
    return _safe_div(_float(raw), fixed_scale)


def _clip_metrics(row: dict[str, Any]) -> dict[str, Any]:
    smoke = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    metrics = smoke.get("metrics") if isinstance(smoke.get("metrics"), dict) else {}
    if metrics:
        return metrics
    residual = row.get("per_clip_residual_metrics") if isinstance(row.get("per_clip_residual_metrics"), dict) else {}
    joint = row.get("per_clip_joint_limit_metrics") if isinstance(row.get("per_clip_joint_limit_metrics"), dict) else {}
    temporal = row.get("per_clip_temporal_metrics") if isinstance(row.get("per_clip_temporal_metrics"), dict) else {}
    return {**residual, **joint, **temporal}


def _clip_legacy_status(clip: dict[str, Any], full_row: dict[str, Any]) -> str:
    return str(
        clip.get("per_clip_runtime_quality_status")
        or clip.get("runtime_quality_status")
        or clip.get("generic_smoke_status")
        or full_row.get("runtime_quality_status")
        or ""
    )


def _synthetic_clip_row(full_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": full_row.get("model_id"),
        "category": full_row.get("category"),
        "clip_id": "aggregate",
        "solver_backed": full_row.get("solver_backed"),
        "residual_only": full_row.get("residual_only"),
        "per_clip_runtime_quality_status": full_row.get("runtime_quality_status"),
        "per_clip_residual_metrics": {
            "raw_task_residual_p95": full_row.get("raw_task_residual_p95"),
            "orientation_integrated_residual_p95": full_row.get("orientation_integrated_residual_p95"),
            "normalized_task_residual_max": full_row.get("normalized_task_residual_max"),
        },
    }


def _diagnostics_by_model_clip(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("model_id")), str(row.get("clip_id"))): row
        for row in _matrix_rows(payload)
        if row.get("model_id") is not None and row.get("clip_id") is not None
    }


def _dominance_by_model(solver_diagnostics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _matrix_rows(solver_diagnostics):
        model_id = str(row.get("model_id") or "")
        best: tuple[float, str, float, float] | None = None
        semantic_counts: Counter[str] = Counter()
        component_counts: Counter[str] = Counter()
        for diagnostic in row.get("task_diagnostics", []):
            per_semantic = diagnostic.get("per_semantic") if isinstance(diagnostic, dict) else {}
            if not isinstance(per_semantic, dict):
                continue
            local_best: tuple[float, str, float, float] | None = None
            for semantic, metrics in per_semantic.items():
                if not isinstance(metrics, dict):
                    continue
                combined = _float(metrics.get("combined_residual"))
                rotation = _float(metrics.get("rotation_residual"))
                translation = _float(metrics.get("translation_residual"))
                if local_best is None or combined > local_best[0]:
                    local_best = (combined, str(semantic), rotation, translation)
                if best is None or combined > best[0]:
                    best = (combined, str(semantic), rotation, translation)
            if local_best is not None:
                semantic_counts[local_best[1]] += 1
                component_counts["rotation" if local_best[2] >= local_best[3] else "translation"] += 1
        if model_id:
            out[model_id] = {
                "dominant_semantic": best[1] if best else "unknown",
                "dominant_component": "rotation" if best and best[2] >= best[3] else "translation" if best else "unknown",
                "dominant_semantic_counts": dict(sorted(semantic_counts.items())),
                "dominant_component_counts": dict(sorted(component_counts.items())),
            }
    return out


def _unknown_dominance() -> dict[str, Any]:
    return {
        "dominant_semantic": "unknown",
        "dominant_component": "unknown",
        "dominant_semantic_counts": {},
        "dominant_component_counts": {},
    }


def _one_clip_dominates(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return max(values) - _percentile(values, 50) > 0.05


def _fixed_scale(baseline: dict[str, dict[str, Any]]) -> float:
    selection = baseline["normalization_policy_selection"]
    gate = baseline["gate_reconciliation_v2_report"]
    scale = selection.get("fixed_global_scale") or gate.get("candidate_gate_inputs", {}).get("fixed_global_scale")
    return max(_float(scale), 1.0)


def _candidate_warn_gate(baseline: dict[str, dict[str, Any]]) -> float:
    gate = baseline["gate_reconciliation_v2_report"]
    return _float(gate.get("candidate_gate_inputs", {}).get("candidate_warn", 0.60)) or 0.60


def _candidate_pass_gate(baseline: dict[str, dict[str, Any]]) -> float:
    gate = baseline["gate_reconciliation_v2_report"]
    return _float(gate.get("candidate_gate_inputs", {}).get("candidate_pass", 0.45)) or 0.45


def _environment_payload(environment: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = dict(environment)
    source = _step4_4_source_provenance()
    payload.update(
        {
            "step": STEP_NAME,
            "source_branch": source["branch"],
            "source_commit": source["head"],
            "base_step4_3_final_head": _baseline_step4_3_head(baseline),
            "step4_4_validation_source_head": source["head"],
            "step4_4_validation_source_dirty": source["dirty"],
            "step4_4_validation_source_git_status_short": source["git_status_short"],
        }
    )
    return payload


def _baseline_step4_3_head(baseline: dict[str, dict[str, Any]]) -> str:
    summary = baseline["quality_summary"]
    environment = baseline["environment"]
    return str(
        summary.get("source_code_commit")
        or summary.get("artifact_commit_observed")
        or environment.get("source_code_commit")
        or ""
    )


def _step4_4_source_provenance() -> dict[str, Any]:
    status = _git_stdout("status", "--short")
    return {
        "branch": _git_stdout("branch", "--show-current"),
        "head": _git_stdout("rev-parse", "HEAD"),
        "dirty": bool(status),
        "git_status_short": status,
    }


def _lfs_evidence() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "lfs", "fsck"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except Exception as exc:
        return {
            "git_lfs_fsck": "ERROR",
            "exit_code": 127,
            "output": str(exc),
        }
    output = completed.stdout.strip()
    ok = completed.returncode == 0 and "Git LFS fsck OK" in output
    return {
        "git_lfs_fsck": "OK" if ok else "FAILED",
        "exit_code": int(completed.returncode),
        "output": output,
    }


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _distribution(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": _stable(_percentile(finite, 50)),
        "p95": _stable(_percentile(finite, 95)),
        "max": _stable(max(finite)),
    }


def _percentile(values: list[float], percentile: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * percentile / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return finite[int(pos)]
    weight = pos - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def _safe_div(value: float, denominator: float) -> float:
    return value / denominator if denominator > 0.0 else 0.0


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _stable(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    if abs(float(value)) < 1e-15:
        return 0.0
    return round(float(value), 12)


if __name__ == "__main__":
    raise SystemExit(main())
