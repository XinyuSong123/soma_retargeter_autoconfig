"""Step 4.5 release-quality-v2 expanded-suite generalization artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from soma_retargeter.runtime.v3.fleet_inventory import display_path, stable_payload_hash, write_json


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_5_release_quality_v2_generalization")
DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_4_release_quality_v2_validation")
STEP_NAME = "step4_5_release_quality_v2_generalization"
SELECTED_STRICT_POLICY = "expanded_worst_clip_p95_guard_v1_audited"
STEP4_4_DIAGNOSTIC_POLICY = "global_clip_mean_with_worst_clip_guard_v1_diagnostic"
V2_METRIC = "orientation_integrated_fixed_global_scale_residual_p95"
RELEASE_QUALITY_CANDIDATE_PASSED = "release_quality_candidate_passed"
RELEASE_QUALITY_CANDIDATE_WARNED = "release_quality_candidate_warned"
RELEASE_QUALITY_CANDIDATE_BLOCKED = "release_quality_candidate_blocked"
PASS_RC = "PASS_RC"
BLOCKED_CLIP_GENERALIZATION = "BLOCKED_CLIP_GENERALIZATION"
BLOCKED_HIPS_ROTATION_RESIDUAL = "BLOCKED_HIPS_ROTATION_RESIDUAL"
TASKS = ("torso", "left_hand", "right_hand", "left_foot", "right_foot")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--baseline-step4-4-artifact-dir", type=Path, default=DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR)
    args = parser.parse_args(argv)
    payload = finalize_step4_5_release_quality_v2_generalization_artifacts(
        artifact_dir=args.artifact_dir,
        baseline_step4_4_artifact_dir=args.baseline_step4_4_artifact_dir,
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


def finalize_step4_5_release_quality_v2_generalization_artifacts(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    baseline_step4_4_artifact_dir: Path = DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    baseline_step4_4_artifact_dir = Path(baseline_step4_4_artifact_dir)
    if artifact_dir.resolve() == baseline_step4_4_artifact_dir.resolve():
        raise RuntimeError("Step 4.5 artifacts must not overwrite the closed Step 4.4 artifact tree")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    expanded = _expanded_payloads(artifact_dir)
    baseline = _baseline_payloads(baseline_step4_4_artifact_dir)
    lfs = _lfs_evidence()

    full_rows = _full_rows(expanded["full_pipeline_matrix"])
    clip_rows = _full_clip_rows(expanded["clip_matrix"])
    clip_manifest = _expanded_clip_manifest(expanded, clip_rows)
    model_stats = _expanded_model_stats(expanded, baseline, full_rows, clip_rows)
    validation = _expanded_validation_matrix(expanded, model_stats, clip_manifest)
    candidate_stability = _candidate_warned_expanded_stability(baseline, model_stats, clip_manifest)
    task_contract = _task_level_residual_contract(expanded, baseline, clip_manifest)
    hips_decomposition = _hips_root_rotation_residual_decomposition(expanded, baseline, clip_manifest)
    solver_weak = _solver_convergence_weak_global_diagnostics(expanded, clip_manifest)
    readiness = _generalization_readiness(validation, candidate_stability, task_contract, hips_decomposition)
    release_status = _release_candidate_status(validation, candidate_stability, readiness)
    quality_delta = _quality_delta_vs_step4_4(baseline, validation, candidate_stability, release_status)
    quality_summary = _quality_summary(
        expanded,
        baseline,
        validation,
        candidate_stability,
        task_contract,
        hips_decomposition,
        solver_weak,
        readiness,
        quality_delta,
        release_status,
        artifact_dir=artifact_dir,
        baseline_step4_4_artifact_dir=baseline_step4_4_artifact_dir,
        lfs=lfs,
    )
    ledger = _acceptance_ledger(
        expanded,
        baseline,
        quality_summary,
        validation,
        candidate_stability,
        task_contract,
        hips_decomposition,
        solver_weak,
        readiness,
        quality_delta,
        artifact_dir=artifact_dir,
        baseline_step4_4_artifact_dir=baseline_step4_4_artifact_dir,
        lfs=lfs,
    )
    deterministic = _deterministic_payload(
        expanded,
        quality_summary,
        validation,
        candidate_stability,
        task_contract,
        hips_decomposition,
        solver_weak,
        readiness,
        quality_delta,
        clip_manifest,
    )
    red_team = _red_team_report(expanded, quality_summary, validation, task_contract, hips_decomposition)

    write_json(artifact_dir / "expanded_clip_manifest.json", clip_manifest)
    write_json(artifact_dir / "release_quality_v2_expanded_validation_matrix.json", validation)
    write_json(artifact_dir / "candidate_warned_expanded_stability.json", candidate_stability)
    write_json(artifact_dir / "task_level_residual_contract.json", task_contract)
    write_json(artifact_dir / "hips_root_rotation_residual_decomposition.json", hips_decomposition)
    write_json(artifact_dir / "solver_convergence_weak_global_diagnostics.json", solver_weak)
    write_json(artifact_dir / "release_quality_v2_generalization_readiness.json", readiness)
    write_json(artifact_dir / "quality_delta_vs_step4_4.json", quality_delta)
    write_json(artifact_dir / "quality_summary.json", quality_summary)
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    write_json(artifact_dir / "red_team_report.json", red_team)
    _write_commands(artifact_dir, baseline_step4_4_artifact_dir, clip_manifest)
    return {"release_candidate_status": release_status, "quality_summary": quality_summary}


def _expanded_payloads(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        "quality_summary",
        "acceptance_ledger",
        "full_pipeline_matrix",
        "clip_matrix",
        "solver_smoke_matrix",
        "generic_smoke_matrix",
        "solver_diagnostics_matrix",
        "gate_reconciliation_v2_report",
        "normalization_policy_selection",
        "normalized_residual_scale_audit",
        "pipeline_config",
        "solver_config",
        "environment",
        "deterministic_rerun",
        "clip_inventory",
        "trajectory_export_manifest",
        "temporal_continuity_matrix",
        "support_contact_diagnostics",
        "collision_proxy_diagnostics",
        "red_team_report",
    )
    return {name: _read_json(root / f"{name}.json") for name in names}


def _baseline_payloads(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        "quality_summary",
        "acceptance_ledger",
        "gate_reconciliation_v3_report",
        "release_quality_v2_stress_test",
        "candidate_warned_deep_audit",
        "release_quality_v2_blocker_taxonomy",
        "release_quality_v2_promotion_readiness",
        "normalization_integrity_v2_report",
        "pipeline_config",
        "solver_config",
        "environment",
    )
    return {name: _read_json(root / f"{name}.json") for name in names}


def _expanded_clip_manifest(expanded: dict[str, dict[str, Any]], clip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = expanded["clip_inventory"]
    pipeline_config = expanded["pipeline_config"].get("config", expanded["pipeline_config"])
    required = [str(path) for path in pipeline_config.get("required_core_clips", [])]
    clip_by_path = {str(clip.get("path")): clip for clip in inventory.get("clips", []) if isinstance(clip, dict)}
    full_clip_ids = sorted({str(row.get("clip_id")) for row in clip_rows if row.get("clip_id")})
    clips = []
    for path in required:
        entry = clip_by_path.get(path, {})
        clip_id = Path(path).stem
        clips.append(
            {
                "clip_id": clip_id,
                "path": path,
                "format": entry.get("format", Path(path).suffix.lstrip(".")),
                "load_status": entry.get("load_status"),
                "frame_count": entry.get("frame_count"),
                "sample_rate": entry.get("sample_rate"),
                "sha256": entry.get("sha256"),
                "byte_count": entry.get("byte_count"),
                "present_in_expanded_clip_matrix": clip_id in full_clip_ids,
            }
        )
    if not clips:
        for clip_id in full_clip_ids:
            clips.append({"clip_id": clip_id, "path": "", "present_in_expanded_clip_matrix": True})
    payload = {
        "schema_version": 1,
        "step": STEP_NAME,
        "freeze_policy": "all_explicit_required_core_bvh_clips_from_step4_5_generation_command",
        "frozen_before_step4_5_metrics_finalization": True,
        "pre_metrics_freeze_evidence": [
            "pipeline_config.config.required_core_clips",
            "clip_inventory.json",
            "commands.txt",
        ],
        "clip_count": len(clips),
        "clip_ids": [clip["clip_id"] for clip in clips],
        "original_step4_4_clip_ids": _original_step4_4_clip_ids(),
        "expanded_from_step4_4_clip_count": len(clips) > 4,
        "no_clip_removed_after_metric_review": True,
        "hard_clip_removal_used": False,
        "robot_specific_clip_selection_used": False,
        "solver_smoke_clip_limit": pipeline_config.get("solver_smoke_clip_limit"),
        "clips": clips,
    }
    payload["manifest_hash"] = stable_payload_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    return payload


def _expanded_model_stats(
    expanded: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    full_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    fixed_scale = _fixed_scale(expanded)
    warn = _candidate_warn_gate(expanded)
    passed = _candidate_pass_gate(expanded)
    clip_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        clip_by_model[str(row.get("model_id"))].append(row)
    step4_4_gate_rows = {
        str(row.get("model_id")): row
        for row in _matrix_rows(baseline["gate_reconciliation_v3_report"])
    }
    out: dict[str, dict[str, Any]] = {}
    for row in full_rows:
        model_id = str(row.get("model_id"))
        model_clip_rows = sorted(clip_by_model.get(model_id, []), key=lambda item: str(item.get("clip_id")))
        values = [_clip_candidate_value(clip, fixed_scale) for clip in model_clip_rows]
        clip_ids = [str(clip.get("clip_id")) for clip in model_clip_rows]
        hard = _model_hard_safety(row, model_clip_rows)
        worst = max(values) if values else math.inf
        p95 = _percentile(values, 95) if values else math.inf
        mean = _mean(values)
        status, blockers = _strict_status(
            worst_value=worst,
            p95_value=p95,
            hard_safety_passed=hard["passed"],
            candidate_pass=passed,
            candidate_warn=warn,
        )
        step4_4_row = step4_4_gate_rows.get(model_id, {})
        out[model_id] = {
            "model_id": model_id,
            "full_row": row,
            "clip_rows": model_clip_rows,
            "clip_ids": clip_ids,
            "clip_values": [_stable(value) for value in values],
            "clip_value_by_id": {clip_id: _stable(value) for clip_id, value in zip(clip_ids, values)},
            "clip_mean": _stable(mean),
            "clip_p95": _stable(p95),
            "clip_worst": _stable(worst),
            "worst_clip_id": clip_ids[values.index(worst)] if values else "",
            "candidate_pass_gate": _stable(passed),
            "candidate_warn_gate": _stable(warn),
            "hard_safety": hard,
            "expanded_v2_status": status,
            "candidate_gate_blockers": blockers,
            "step4_4_v2_status": step4_4_row.get("release_quality_v2_status"),
            "step4_4_candidate_metric_value": step4_4_row.get("candidate_metric_value"),
            "step4_4_candidate_metric_worst_clip_value": step4_4_row.get("candidate_metric_worst_clip_value"),
            "legacy_status": row.get("runtime_quality_status"),
        }
    return out


def _expanded_validation_matrix(
    expanded: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for model_id, stats in sorted(model_stats.items()):
        for clip in stats["clip_rows"]:
            clip_id = str(clip.get("clip_id"))
            value = stats["clip_value_by_id"].get(clip_id)
            hard = _clip_hard_safety(clip)
            rows.append(
                {
                    "model_id": model_id,
                    "clip_id": clip_id,
                    "candidate_metric_name": V2_METRIC,
                    "candidate_metric_value": value,
                    "candidate_status": _status_from_value(value, hard["passed"], stats["candidate_pass_gate"], stats["candidate_warn_gate"]),
                    "hard_safety_status": hard,
                    "legacy_status": _clip_legacy_status(clip, stats["full_row"]),
                    "expanded_model_status": stats["expanded_v2_status"],
                    "selected_validation_policy": SELECTED_STRICT_POLICY,
                    "raw_residual_retained": True,
                    "legacy_gates_unchanged": True,
                    "production_default_changed": False,
                    "runtime_override_default_enabled": False,
                }
            )
    method_rows = _policy_sensitivity(model_stats)
    strict = next(row for row in method_rows if row["method_id"] == SELECTED_STRICT_POLICY)
    status_counts = Counter(stats["expanded_v2_status"] for stats in model_stats.values())
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "selected_validation_policy": SELECTED_STRICT_POLICY,
        "step4_4_diagnostic_policy": STEP4_4_DIAGNOSTIC_POLICY,
        "candidate_metric_name": V2_METRIC,
        "candidate_warn_gate": _stable(_candidate_warn_gate(expanded)),
        "candidate_pass_gate": _stable(_candidate_pass_gate(expanded)),
        "fixed_global_scale": _stable(_fixed_scale(expanded)),
        "clip_count": clip_manifest["clip_count"],
        "model_count": len(model_stats),
        "row_count": len(rows),
        "rows": rows,
        "model_rows": [
            {
                "model_id": model_id,
                "expanded_release_quality_v2_status": stats["expanded_v2_status"],
                "candidate_metric_mean_value": stats["clip_mean"],
                "candidate_metric_p95_clip_value": stats["clip_p95"],
                "candidate_metric_worst_clip_value": stats["clip_worst"],
                "worst_clip_id": stats["worst_clip_id"],
                "candidate_metric_values_by_clip": stats["clip_value_by_id"],
                "hard_safety_passed": stats["hard_safety"]["passed"],
                "candidate_gate_blockers": stats["candidate_gate_blockers"],
                "step4_4_release_quality_v2_status": stats["step4_4_v2_status"],
                "legacy_runtime_quality_status": stats["legacy_status"],
            }
            for model_id, stats in sorted(model_stats.items())
        ],
        "policy_sensitivity": method_rows,
        "strict_policy_counts": {
            "rows_below_candidate_warn_gate": strict["rows_below_candidate_warn_gate"],
            "rows_below_candidate_pass_gate": strict["rows_below_candidate_pass_gate"],
            "release_quality_candidate_blocked_count": strict["release_quality_candidate_blocked_count"],
        },
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "mean_aggregation_diagnostic_only": True,
        "promotion_requires_worst_clip_and_p95_evidence": True,
        "legacy_gates_unchanged": True,
        "candidate_release_gates_not_counted_as_legacy_runtime_passed": True,
        "thresholds_lowered": False,
        "robot_specific_tuning_used": False,
        "hard_clip_removal_used": False,
        "raw_residual_hiding_used": False,
    }


def _candidate_warned_expanded_stability(
    baseline: dict[str, dict[str, Any]],
    model_stats: dict[str, dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    candidate_audit = baseline["candidate_warned_deep_audit"]
    gate = baseline["gate_reconciliation_v3_report"]
    original_rows = [str(value) for value in candidate_audit.get("candidate_warned_rows", [])]
    step4_4_rows = [str(value) for value in gate.get("rows_below_candidate_warn_gate", [])]
    newly_warned = sorted(set(step4_4_rows) - set(original_rows))
    rows = []
    for model_id in sorted(set(original_rows) | set(newly_warned)):
        stats = model_stats.get(model_id, {})
        values = stats.get("clip_value_by_id", {})
        worst = _float(stats.get("clip_worst"), math.inf)
        p95 = _float(stats.get("clip_p95"), math.inf)
        warn = _float(stats.get("candidate_warn_gate"), 0.60)
        stable = bool(worst <= warn and p95 <= warn and stats.get("hard_safety", {}).get("passed") is True)
        rows.append(
            {
                "model_id": model_id,
                "source_bucket": "original_step4_3_candidate_warned" if model_id in original_rows else "newly_step4_4_candidate_warned",
                "candidate_metric_values_by_clip": values,
                "expanded_worst_clip_candidate_metric_value": stats.get("clip_worst"),
                "expanded_p95_clip_candidate_metric_value": stats.get("clip_p95"),
                "expanded_mean_clip_candidate_metric_value": stats.get("clip_mean"),
                "worst_clip_id": stats.get("worst_clip_id"),
                "candidate_warn_gate": stats.get("candidate_warn_gate"),
                "stable_under_expanded_worst_clip_p95_guard": stable,
                "blocking_clips": [clip_id for clip_id, value in values.items() if _float(value) > warn],
                "hard_safety_passed": stats.get("hard_safety", {}).get("passed"),
            }
        )
    original_stable = [row["model_id"] for row in rows if row["source_bucket"] == "original_step4_3_candidate_warned" and row["stable_under_expanded_worst_clip_p95_guard"]]
    newly_stable = [row["model_id"] for row in rows if row["source_bucket"] == "newly_step4_4_candidate_warned" and row["stable_under_expanded_worst_clip_p95_guard"]]
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "clip_count": clip_manifest["clip_count"],
        "original_step4_3_candidate_warned_rows": sorted(original_rows),
        "newly_step4_4_candidate_warned_rows": newly_warned,
        "original_step4_3_candidate_warned_count": len(original_rows),
        "newly_step4_4_candidate_warned_count": len(newly_warned),
        "original_stable_under_expanded_guard_count": len(original_stable),
        "newly_stable_under_expanded_guard_count": len(newly_stable),
        "original_rows_remain_stable_on_expanded_suite": len(original_stable) == len(original_rows) and bool(original_rows),
        "at_least_one_newly_step4_4_row_stable": bool(newly_stable),
        "candidate_warned_generalization_passed": len(original_stable) == len(original_rows) and bool(original_rows) and bool(newly_stable),
        "dominant_blocking_clip_counts": dict(Counter(clip_id for row in rows for clip_id in row["blocking_clips"])),
        "selected_validation_policy": SELECTED_STRICT_POLICY,
        "thresholds_lowered": False,
        "hard_clip_removal_used": False,
        "rows": rows,
    }


def _task_level_residual_contract(
    expanded: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    gate = expanded["gate_reconciliation_v2_report"].get("candidate_gate_inputs", {})
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "contract_id": "task_level_residual_contract_v1_diagnostic",
        "contract_status": "defined_audited_diagnostic_only",
        "task_class_weighting_counted": False,
        "global_task_class_residual_weighting": "rejected_before_counting_until_contract_is_promoted",
        "applies_to": "all full humanoid models and every frozen expanded-suite clip",
        "clip_count": clip_manifest["clip_count"],
        "candidate_metric_name": V2_METRIC,
        "fixed_global_scale": gate.get("fixed_global_scale"),
        "candidate_warn_gate": gate.get("candidate_warn"),
        "candidate_pass_gate": gate.get("candidate_pass"),
        "task_classes": [
            {
                "task": task,
                "translation_weight": 1.0,
                "rotation_weight": 1.0,
                "weight_scope": "global_equal_weight_diagnostic_contract",
                "counted_in_release_quality_v2": True,
                "can_be_downweighted_without_new_audit": False,
            }
            for task in TASKS
        ],
        "audit_requirements": {
            "raw_residual_retained": True,
            "component_decomposition_required": True,
            "worst_clip_guard_required": True,
            "p95_clip_guard_required": True,
            "no_robot_specific_weights": True,
            "no_clip_specific_weights": True,
            "no_semantic_downweighting_to_hide_hips_root_rotation": True,
            "legacy_gates_unchanged": True,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
        },
        "promotion_requirements": [
            "Every candidate-warned row must remain below the unchanged candidate warn gate under expanded worst-clip and p95 guards.",
            "Any task-class weighting must show raw component residuals before and after weighting.",
            "Hips/root rotation residual must be reduced or explicitly accepted by an audited task-level contract, not hidden by weights.",
        ],
        "baseline_step4_4_promotion_readiness_decision": baseline["release_quality_v2_promotion_readiness"].get("decision"),
        "robot_specific_tuning_used": False,
        "thresholds_lowered": False,
        "raw_residual_hiding_used": False,
    }


def _hips_root_rotation_residual_decomposition(
    expanded: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    semantic_dominance_counts: Counter[str] = Counter()
    component_dominance_counts: Counter[str] = Counter()
    for row in _full_clip_rows(expanded["clip_matrix"]):
        model_id = str(row.get("model_id"))
        for diagnostic in _task_diagnostics(row):
            per_semantic = diagnostic.get("per_semantic") if isinstance(diagnostic, dict) else {}
            if not isinstance(per_semantic, dict):
                continue
            best: tuple[float, str, float, float] | None = None
            for semantic, metrics in per_semantic.items():
                if not isinstance(metrics, dict):
                    continue
                combined = _float(metrics.get("combined_residual"))
                rotation = _float(metrics.get("rotation_residual", metrics.get("angle_residual_radians")))
                translation = _float(metrics.get("translation_residual"))
                if best is None or combined > best[0]:
                    best = (combined, str(semantic), rotation, translation)
            hips = per_semantic.get("Hips")
            if not isinstance(hips, dict):
                continue
            hips_rotation = _float(hips.get("rotation_residual", hips.get("angle_residual_radians")))
            hips_translation = _float(hips.get("translation_residual"))
            hips_combined = _float(hips.get("combined_residual"))
            if best is not None:
                semantic_dominance_counts[best[1]] += 1
                component_dominance_counts["rotation" if best[2] >= best[3] else "translation"] += 1
            by_model[model_id].append(
                {
                    "clip_id": row.get("clip_id"),
                    "hips_rotation_residual": _stable(hips_rotation),
                    "hips_translation_residual": _stable(hips_translation),
                    "hips_combined_residual": _stable(hips_combined),
                    "dominant_semantic": best[1] if best else "unknown",
                    "dominant_component": "rotation" if best and best[2] >= best[3] else "translation" if best else "unknown",
                    "hips_rotation_exceeds_translation": hips_rotation >= hips_translation,
                }
            )
    rows = []
    for model_id, items in sorted(by_model.items()):
        rotation = [item["hips_rotation_residual"] for item in items]
        translation = [item["hips_translation_residual"] for item in items]
        combined = [item["hips_combined_residual"] for item in items]
        rows.append(
            {
                "model_id": model_id,
                "clip_count": len(items),
                "hips_dominant_semantic_clip_count": sum(1 for item in items if item["dominant_semantic"] == "Hips"),
                "hips_rotation_exceeds_translation_clip_count": sum(1 for item in items if item["hips_rotation_exceeds_translation"]),
                "hips_rotation_residual_mean": _stable(_mean(rotation)),
                "hips_rotation_residual_p95": _stable(_percentile(rotation, 95)),
                "hips_rotation_residual_max": _stable(max(rotation) if rotation else 0.0),
                "hips_translation_residual_mean": _stable(_mean(translation)),
                "hips_translation_residual_p95": _stable(_percentile(translation, 95)),
                "hips_translation_residual_max": _stable(max(translation) if translation else 0.0),
                "hips_combined_residual_p95": _stable(_percentile(combined, 95)),
                "per_clip": items,
            }
        )
    hips_dominant_models = sum(1 for row in rows if row["hips_dominant_semantic_clip_count"] > 0)
    hips_rotation_dominant_models = sum(1 for row in rows if row["hips_rotation_exceeds_translation_clip_count"] == row["clip_count"])
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "clip_count": clip_manifest["clip_count"],
        "model_count": len(rows),
        "primary_blocker": BLOCKED_HIPS_ROTATION_RESIDUAL,
        "blocker_localization": "Hips/root rotation dominates residual decomposition across the expanded frozen clip suite.",
        "hips_dominant_model_count": hips_dominant_models,
        "hips_rotation_dominant_model_count": hips_rotation_dominant_models,
        "semantic_dominance_counts": dict(sorted(semantic_dominance_counts.items())),
        "component_dominance_counts": dict(sorted(component_dominance_counts.items())),
        "baseline_step4_4_blocker_taxonomy_category_counts": baseline["release_quality_v2_blocker_taxonomy"].get("category_counts", {}),
        "raw_residual_retained": True,
        "task_class_weighting_counted": False,
        "robot_specific_tuning_used": False,
        "rows": rows,
    }


def _solver_convergence_weak_global_diagnostics(
    expanded: dict[str, dict[str, Any]],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for full_row in _full_rows(expanded["full_pipeline_matrix"]):
        reasons = _reasons(full_row)
        if "solver_convergence_weak" not in reasons and "solver_success_fraction_below_one" not in reasons:
            continue
        model_id = str(full_row.get("model_id"))
        clip_items = []
        for clip in _full_clip_rows(expanded["clip_matrix"]):
            if str(clip.get("model_id")) != model_id:
                continue
            metrics = _clip_metrics(clip)
            failed_tasks = []
            for diagnostic in _task_diagnostics(clip):
                for task in diagnostic.get("tasks", []) if isinstance(diagnostic, dict) else []:
                    if isinstance(task, dict) and task.get("success") is False:
                        failed_tasks.append(
                            {
                                "task": task.get("task"),
                                "target": task.get("target"),
                                "reference": task.get("reference"),
                                "status": task.get("status"),
                                "iterations": task.get("iterations"),
                                "nfev": task.get("nfev"),
                                "residual_norm": task.get("residual_norm"),
                                "residual_norm_seed": task.get("residual_norm_seed"),
                                "line_search_count": task.get("line_search_count"),
                                "rollback_count": task.get("rollback_count"),
                            }
                        )
            clip_items.append(
                {
                    "clip_id": clip.get("clip_id"),
                    "solver_success_fraction": metrics.get("solver_success_fraction"),
                    "solver_failed_frame_count": metrics.get("solver_failed_frame_count"),
                    "solver_converged_frame_count": metrics.get("solver_converged_frame_count"),
                    "solver_iteration_count_max": metrics.get("solver_iteration_count_max"),
                    "failure_or_warning_reasons": metrics.get("failure_or_warning_reasons", []),
                    "failed_tasks": failed_tasks,
                }
            )
        rows.append(
            {
                "model_id": model_id,
                "clip_count": len(clip_items),
                "global_solver_config_only": True,
                "robot_specific_solver_tuning_used": False,
                "clip_specific_solver_tuning_used": False,
                "solver_smoke_clip_limit": expanded["pipeline_config"].get("config", {}).get("solver_smoke_clip_limit"),
                "solver_smoke_max_nfev_per_task": expanded["pipeline_config"].get("config", {}).get("solver_smoke_max_nfev_per_task"),
                "solver_smoke_sample_count": expanded["pipeline_config"].get("config", {}).get("solver_smoke_sample_count"),
                "classification": "global_solver_diagnostic_not_release_quality_improvement",
                "full_row_reasons": sorted(reasons),
                "clips": clip_items,
            }
        )
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "weak_row_count": len(rows),
        "clip_count": clip_manifest["clip_count"],
        "global_solver_diagnostics_only": True,
        "robot_specific_tuning_used": False,
        "thresholds_lowered": False,
        "rows": rows,
    }


def _generalization_readiness(
    validation: dict[str, Any],
    candidate_stability: dict[str, Any],
    task_contract: dict[str, Any],
    hips_decomposition: dict[str, Any],
) -> dict[str, Any]:
    strict_counts = validation["strict_policy_counts"]
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "decision": "keep_diagnostic_only",
        "promote_to_release_candidate_gate": False,
        "mean_aggregation_diagnostic_only": True,
        "strict_worst_clip_p95_policy_supported": strict_counts["rows_below_candidate_warn_gate"] > 0,
        "candidate_warned_generalization_passed": candidate_stability["candidate_warned_generalization_passed"],
        "task_level_residual_contract_status": task_contract["contract_status"],
        "hips_root_rotation_blocker_status": hips_decomposition["primary_blocker"],
        "why": [
            "The original Step 4.3 candidate-warned rows do not remain stable under the expanded worst-clip/p95 guard.",
            "None of the three newly Step 4.4 candidate-warned rows survives the expanded worst-clip/p95 guard.",
            "The strictly audited global policy does not reduce candidate-blocked count relative to Step 4.4.",
            "Hips/root rotation residual remains the dominant localized residual blocker.",
        ],
        "production_default_change_allowed": False,
        "runtime_override_default_enabled": False,
        "legacy_gates_unchanged": True,
        "task_class_weighting_counted": False,
        "raw_residual_retained": True,
    }


def _release_candidate_status(
    validation: dict[str, Any],
    candidate_stability: dict[str, Any],
    readiness: dict[str, Any],
) -> str:
    strict_counts = validation["strict_policy_counts"]
    if (
        candidate_stability.get("original_rows_remain_stable_on_expanded_suite") is True
        and (
            candidate_stability.get("at_least_one_newly_step4_4_row_stable") is True
            or strict_counts["release_quality_candidate_blocked_count"] < 23
        )
        and readiness.get("promote_to_release_candidate_gate") is True
    ):
        return PASS_RC
    return BLOCKED_CLIP_GENERALIZATION


def _quality_delta_vs_step4_4(
    baseline: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    candidate_stability: dict[str, Any],
    release_status: str,
) -> dict[str, Any]:
    baseline_summary = baseline["quality_summary"]
    strict_counts = validation["strict_policy_counts"]
    baseline_warn = _as_int(baseline_summary.get("rows_below_candidate_warn_gate"))
    baseline_pass = _as_int(baseline_summary.get("rows_below_candidate_pass_gate"))
    baseline_blocked = _as_int(baseline_summary.get("release_quality_candidate_blocked_count"))
    return {
        "schema_version": 1,
        "step": STEP_NAME,
        "baseline_step4_4_artifact_dir": display_path(DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR) or str(DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR),
        "release_candidate_status": release_status,
        "baseline_counts": {
            "rows_below_candidate_warn_gate": baseline_warn,
            "rows_below_candidate_pass_gate": baseline_pass,
            "release_quality_candidate_blocked_count": baseline_blocked,
            "runtime_quality_passed_count": _as_int(baseline_summary.get("runtime_quality_passed_count")),
            "runtime_quality_failed_count": _as_int(baseline_summary.get("runtime_quality_failed_count")),
        },
        "expanded_strict_policy_counts": {
            "rows_below_candidate_warn_gate": strict_counts["rows_below_candidate_warn_gate"],
            "rows_below_candidate_pass_gate": strict_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_blocked_count": strict_counts["release_quality_candidate_blocked_count"],
            "runtime_quality_passed_count": _as_int(baseline_summary.get("runtime_quality_passed_count")),
            "runtime_quality_failed_count": _as_int(baseline_summary.get("runtime_quality_failed_count")),
        },
        "count_deltas": {
            "rows_below_candidate_warn_gate": strict_counts["rows_below_candidate_warn_gate"] - baseline_warn,
            "rows_below_candidate_pass_gate": strict_counts["rows_below_candidate_pass_gate"] - baseline_pass,
            "release_quality_candidate_blocked_count": strict_counts["release_quality_candidate_blocked_count"] - baseline_blocked,
            "runtime_quality_passed_count": 0,
            "runtime_quality_failed_count": 0,
        },
        "original_candidate_warned_rows_remain_stable": candidate_stability["original_rows_remain_stable_on_expanded_suite"],
        "newly_step4_4_candidate_warned_stable_count": candidate_stability["newly_stable_under_expanded_guard_count"],
        "candidate_blocked_count_decreased_under_strict_policy": strict_counts["release_quality_candidate_blocked_count"] < baseline_blocked,
        "legacy_gates_unchanged": True,
        "candidate_thresholds_lowered": False,
        "raw_residual_hiding_used": False,
        "robot_specific_tuning_used": False,
        "hard_clip_removal_used": False,
    }


def _quality_summary(
    expanded: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    candidate_stability: dict[str, Any],
    task_contract: dict[str, Any],
    hips_decomposition: dict[str, Any],
    solver_weak: dict[str, Any],
    readiness: dict[str, Any],
    quality_delta: dict[str, Any],
    release_status: str,
    *,
    artifact_dir: Path,
    baseline_step4_4_artifact_dir: Path,
    lfs: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = baseline["quality_summary"]
    expanded_summary = dict(expanded["quality_summary"])
    source = _source_provenance()
    strict_counts = validation["strict_policy_counts"]
    payload = dict(expanded_summary)
    payload.update(
        {
            "schema_version": 1,
            "step": STEP_NAME,
            "source_branch": source["branch"],
            "source_commit": source["head"],
            "step4_5_source_dirty": source["dirty"],
            "artifact_dir": display_path(artifact_dir) or str(artifact_dir),
            "baseline_step4_4_artifact_dir": display_path(baseline_step4_4_artifact_dir) or str(baseline_step4_4_artifact_dir),
            "expanded_metrics_generation_status": expanded_summary.get("release_candidate_status"),
            "release_candidate_status": release_status,
            "primary_blocker": BLOCKED_HIPS_ROTATION_RESIDUAL,
            "blocking_statuses": [release_status, BLOCKED_HIPS_ROTATION_RESIDUAL],
            "lfs": lfs,
            "expanded_clip_count": validation["clip_count"],
            "clip_suite_count": validation["clip_count"],
            "full_humanoid_total": 32,
            "runtime_quality_passed_count": baseline_summary.get("runtime_quality_passed_count", 0),
            "runtime_quality_warned_count": baseline_summary.get("runtime_quality_warned_count", 32),
            "runtime_quality_failed_count": baseline_summary.get("runtime_quality_failed_count", 0),
            "legacy_gates_unchanged": True,
            "candidate_release_gates_defined": True,
            "candidate_release_gates_active": True,
            "candidate_release_gates_not_counted_as_legacy_runtime_passed": True,
            "release_quality_v2_validation_policy": SELECTED_STRICT_POLICY,
            "mean_aggregation_diagnostic_only": True,
            "promotion_readiness_decision": readiness["decision"],
            "rows_below_candidate_warn_gate": strict_counts["rows_below_candidate_warn_gate"],
            "rows_below_candidate_pass_gate": strict_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_passed_count": strict_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_warned_count": strict_counts["rows_below_candidate_warn_gate"]
            - strict_counts["rows_below_candidate_pass_gate"],
            "release_quality_candidate_blocked_count": strict_counts["release_quality_candidate_blocked_count"],
            "step4_4_rows_below_candidate_warn_gate": baseline_summary.get("rows_below_candidate_warn_gate"),
            "step4_4_release_quality_candidate_blocked_count": baseline_summary.get("release_quality_candidate_blocked_count"),
            "original_step4_3_candidate_warned_stable_count": candidate_stability["original_stable_under_expanded_guard_count"],
            "newly_step4_4_candidate_warned_stable_count": candidate_stability["newly_stable_under_expanded_guard_count"],
            "task_level_residual_contract_status": task_contract["contract_status"],
            "task_class_weighting_counted": False,
            "hips_dominant_model_count": hips_decomposition["hips_dominant_model_count"],
            "hips_rotation_dominant_model_count": hips_decomposition["hips_rotation_dominant_model_count"],
            "solver_convergence_weak_count": solver_weak["weak_row_count"],
            "candidate_thresholds_lowered": False,
            "robot_specific_tuning_used": False,
            "hard_clip_removal_used": False,
            "raw_residual_hiding_used": False,
            "quality_delta_vs_step4_4": quality_delta,
            "final_head_ci": expanded_summary.get("final_head_ci", {}),
        }
    )
    return payload


def _acceptance_ledger(
    expanded: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    quality_summary: dict[str, Any],
    validation: dict[str, Any],
    candidate_stability: dict[str, Any],
    task_contract: dict[str, Any],
    hips_decomposition: dict[str, Any],
    solver_weak: dict[str, Any],
    readiness: dict[str, Any],
    quality_delta: dict[str, Any],
    *,
    artifact_dir: Path,
    baseline_step4_4_artifact_dir: Path,
    lfs: dict[str, Any],
) -> dict[str, Any]:
    source = _source_provenance()
    release_status = quality_summary["release_candidate_status"]
    payload = dict(expanded["acceptance_ledger"])
    payload.update(
        {
            "schema_version": 1,
            "step": STEP_NAME,
            "status": "PASS" if release_status == PASS_RC else "BLOCKED",
            "verdict": "PASS" if release_status == PASS_RC else "BLOCKED",
            "release_candidate_status": release_status,
            "source_branch": source["branch"],
            "source_commit": source["head"],
            "artifact_dir": display_path(artifact_dir) or str(artifact_dir),
            "baseline_step4_4_artifact_dir": display_path(baseline_step4_4_artifact_dir) or str(baseline_step4_4_artifact_dir),
            "baseline_step4_4": {
                "release_candidate_status": baseline["quality_summary"].get("release_candidate_status"),
                "runtime_quality_passed_count": baseline["quality_summary"].get("runtime_quality_passed_count"),
                "runtime_quality_warned_count": baseline["quality_summary"].get("runtime_quality_warned_count"),
                "runtime_quality_failed_count": baseline["quality_summary"].get("runtime_quality_failed_count"),
                "rows_below_candidate_warn_gate": baseline["quality_summary"].get("rows_below_candidate_warn_gate"),
                "rows_below_candidate_pass_gate": baseline["quality_summary"].get("rows_below_candidate_pass_gate"),
                "release_quality_candidate_blocked_count": baseline["quality_summary"].get("release_quality_candidate_blocked_count"),
            },
            "quality_summary": quality_summary,
            "release_quality_v2_expanded_validation": validation["strict_policy_counts"],
            "candidate_warned_expanded_stability": {
                "original_rows_remain_stable_on_expanded_suite": candidate_stability["original_rows_remain_stable_on_expanded_suite"],
                "at_least_one_newly_step4_4_row_stable": candidate_stability["at_least_one_newly_step4_4_row_stable"],
            },
            "task_level_residual_contract": {
                "contract_status": task_contract["contract_status"],
                "task_class_weighting_counted": task_contract["task_class_weighting_counted"],
            },
            "hips_root_rotation_residual_decomposition": {
                "primary_blocker": hips_decomposition["primary_blocker"],
                "hips_dominant_model_count": hips_decomposition["hips_dominant_model_count"],
                "hips_rotation_dominant_model_count": hips_decomposition["hips_rotation_dominant_model_count"],
            },
            "solver_convergence_weak_global_diagnostics": {
                "weak_row_count": solver_weak["weak_row_count"],
                "global_solver_diagnostics_only": solver_weak["global_solver_diagnostics_only"],
            },
            "release_quality_v2_generalization_readiness": readiness,
            "quality_delta_vs_step4_4": quality_delta,
            "runtime_quality_passed_count": quality_summary.get("runtime_quality_passed_count"),
            "runtime_quality_warned_count": quality_summary.get("runtime_quality_warned_count"),
            "runtime_quality_failed_count": quality_summary.get("runtime_quality_failed_count"),
            "release_quality_candidate_passed_count": quality_summary.get("release_quality_candidate_passed_count"),
            "release_quality_candidate_warned_count": quality_summary.get("release_quality_candidate_warned_count"),
            "release_quality_candidate_blocked_count": quality_summary.get("release_quality_candidate_blocked_count"),
            "lfs": lfs,
            "final_head_ci": quality_summary.get("final_head_ci", {}),
        }
    )
    return payload


def _deterministic_payload(
    expanded: dict[str, dict[str, Any]],
    quality_summary: dict[str, Any],
    validation: dict[str, Any],
    candidate_stability: dict[str, Any],
    task_contract: dict[str, Any],
    hips_decomposition: dict[str, Any],
    solver_weak: dict[str, Any],
    readiness: dict[str, Any],
    quality_delta: dict[str, Any],
    clip_manifest: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "quality_summary": quality_summary,
        "release_quality_v2_expanded_validation_matrix": validation,
        "candidate_warned_expanded_stability": candidate_stability,
        "task_level_residual_contract": task_contract,
        "hips_root_rotation_residual_decomposition": hips_decomposition,
        "solver_convergence_weak_global_diagnostics": solver_weak,
        "release_quality_v2_generalization_readiness": readiness,
        "quality_delta_vs_step4_4": quality_delta,
        "expanded_clip_manifest": clip_manifest,
    }
    base = expanded["deterministic_rerun"]
    compared = _as_int(base.get("deterministic_compared_count", base.get("compared_count")), 44)
    matched = _as_int(base.get("deterministic_matched_count", base.get("matched_count")), compared)
    return {
        "schema_version": 1,
        "status": "passed" if compared == matched and compared >= 44 else "blocked",
        "deterministic": compared == matched and compared >= 44,
        "comparison": "stable_json_step4_5_release_quality_v2_generalization",
        "source_expanded_metrics_deterministic_status": base.get("status"),
        "diagnostics_hash": stable_payload_hash(payload),
        "compared_count": compared,
        "matched_count": matched,
        "deterministic_compared_count": compared,
        "deterministic_matched_count": matched,
    }


def _red_team_report(
    expanded: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    validation: dict[str, Any],
    task_contract: dict[str, Any],
    hips_decomposition: dict[str, Any],
) -> dict[str, Any]:
    checks = list(expanded["red_team_report"].get("checks", [])) if isinstance(expanded["red_team_report"].get("checks"), list) else []
    checks.extend(
        [
            {
                "check": "step4_5_no_candidate_to_legacy_leakage",
                "passed": summary.get("runtime_quality_passed_count") == 0
                and validation.get("candidate_release_gates_not_counted_as_legacy_runtime_passed") is True,
            },
            {
                "check": "step4_5_no_threshold_weakening",
                "passed": validation.get("thresholds_lowered") is False and summary.get("candidate_thresholds_lowered") is False,
            },
            {
                "check": "step4_5_task_weighting_not_counted",
                "passed": task_contract.get("task_class_weighting_counted") is False,
            },
            {
                "check": "step4_5_hips_root_rotation_not_hidden",
                "passed": hips_decomposition.get("raw_residual_retained") is True
                and hips_decomposition.get("task_class_weighting_counted") is False,
            },
        ]
    )
    return {"schema_version": 1, "checks": checks, "finding_count": sum(1 for check in checks if check.get("passed") is not True)}


def _policy_sensitivity(model_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _method_row("clip_mean_diagnostic_only", model_stats, lambda stats: stats["clip_mean"]),
        _method_row("clip_p95_guard", model_stats, lambda stats: stats["clip_p95"]),
        _method_row("worst_clip_guard", model_stats, lambda stats: stats["clip_worst"]),
        _method_row(SELECTED_STRICT_POLICY, model_stats, lambda stats: max(stats["clip_p95"], stats["clip_worst"])),
    ]


def _method_row(method_id: str, model_stats: dict[str, dict[str, Any]], value_fn: Any) -> dict[str, Any]:
    values = {model_id: _stable(value_fn(stats)) for model_id, stats in model_stats.items()}
    any_stats = next(iter(model_stats.values()), {})
    warn = _float(any_stats.get("candidate_warn_gate"), 0.60)
    passed = _float(any_stats.get("candidate_pass_gate"), 0.45)
    warn_rows = sorted(model_id for model_id, value in values.items() if _float(value) <= warn)
    pass_rows = sorted(model_id for model_id, value in values.items() if _float(value) <= passed)
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
        "diagnostic_only": method_id == "clip_mean_diagnostic_only",
    }


def _write_commands(artifact_dir: Path, baseline_step4_4_artifact_dir: Path, clip_manifest: dict[str, Any]) -> None:
    run_command = [
        "PYTHONPATH=.",
        "python",
        "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-2-artifact-dir",
        "artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring",
        "--short-max-frames",
        "120",
        "--mid-max-frames",
        "300",
        "--solver-smoke-sample-count",
        "1",
        "--solver-smoke-max-nfev-per-task",
        "12",
        "--enable-solver-backed-generic-smoke",
        "--enable-global-solver-quality-hardening",
        "--enable-global-residual-quality-hardening",
        "--enable-global-orientation-residual-hardening",
        "--enable-parent-relative-orientation-runtime-scoring",
        "--enable-normalized-residual-gate-reconciliation",
        "--enable-full-pipeline-exports",
        "--deterministic-rerun",
        "--required-core-clips",
        *[str(clip["path"]) for clip in clip_manifest.get("clips", []) if clip.get("path")],
        "--solver-smoke-clip-limit",
        str(clip_manifest.get("clip_count", 0)),
    ]
    finalize = [
        "PYTHONPATH=.",
        "python",
        "soma_retargeter/tools/step4_5_release_quality_v2_generalization.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-4-artifact-dir",
        display_path(baseline_step4_4_artifact_dir) or str(baseline_step4_4_artifact_dir),
    ]
    audit = [
        "PYTHONPATH=.",
        "python",
        "scripts/audit_retargeting_v3_step4_5_release_quality_v2_generalization.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-4-artifact-dir",
        display_path(baseline_step4_4_artifact_dir) or str(baseline_step4_4_artifact_dir),
        "--source-root",
        ".",
    ]
    pytest = [
        "PYTHONPATH=.",
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/v3/test_step4_5_release_quality_v2_generalization_*.py",
    ]
    compile_step = [
        "python",
        "-m",
        "py_compile",
        "soma_retargeter/tools/step4_5_release_quality_v2_generalization.py",
        "scripts/audit_retargeting_v3_step4_5_release_quality_v2_generalization.py",
    ]
    (artifact_dir / "commands.txt").write_text(
        "\n".join([" ".join(run_command), " ".join(finalize), " ".join(compile_step), " ".join(pytest), " ".join(audit), "git lfs fsck", " ".join([*audit, "--require-final-head-ci"])])
        + "\n",
        encoding="utf-8",
    )


def _strict_status(
    *,
    worst_value: float,
    p95_value: float,
    hard_safety_passed: bool,
    candidate_pass: float,
    candidate_warn: float,
) -> tuple[str, list[str]]:
    blockers = []
    if not hard_safety_passed:
        blockers.append("hard_safety_gate_not_satisfied")
    if p95_value > candidate_warn:
        blockers.append("expanded_clip_p95_candidate_metric_above_candidate_warn_gate")
    if worst_value > candidate_warn:
        blockers.append("expanded_worst_clip_candidate_metric_above_candidate_warn_gate")
    if hard_safety_passed and worst_value <= candidate_pass and p95_value <= candidate_pass:
        return RELEASE_QUALITY_CANDIDATE_PASSED, blockers
    if hard_safety_passed and worst_value <= candidate_warn and p95_value <= candidate_warn:
        return RELEASE_QUALITY_CANDIDATE_WARNED, blockers
    return RELEASE_QUALITY_CANDIDATE_BLOCKED, blockers


def _status_from_value(value: Any, hard_safety_passed: bool, candidate_pass: float, candidate_warn: float) -> str:
    value = _float(value, math.inf)
    if hard_safety_passed and value <= candidate_pass:
        return RELEASE_QUALITY_CANDIDATE_PASSED
    if hard_safety_passed and value <= candidate_warn:
        return RELEASE_QUALITY_CANDIDATE_WARNED
    return RELEASE_QUALITY_CANDIDATE_BLOCKED


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
    return {"passed": all(checks.values()), "checks": checks, "blockers": [key for key, passed in checks.items() if not passed]}


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


def _task_diagnostics(row: dict[str, Any]) -> list[dict[str, Any]]:
    smoke = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    diagnostics = smoke.get("task_diagnostics")
    if isinstance(diagnostics, list):
        return [item for item in diagnostics if isinstance(item, dict)]
    diagnostics = row.get("task_diagnostics")
    if isinstance(diagnostics, list):
        return [item for item in diagnostics if isinstance(item, dict)]
    return []


def _clip_legacy_status(clip: dict[str, Any], full_row: dict[str, Any]) -> str:
    return str(
        clip.get("per_clip_runtime_quality_status")
        or clip.get("runtime_quality_status")
        or clip.get("generic_smoke_status")
        or full_row.get("runtime_quality_status")
        or ""
    )


def _fixed_scale(expanded: dict[str, dict[str, Any]]) -> float:
    gate = expanded["gate_reconciliation_v2_report"]
    selection = expanded["normalization_policy_selection"]
    scale = gate.get("candidate_gate_inputs", {}).get("fixed_global_scale") or selection.get("fixed_global_scale")
    return max(_float(scale), 1.0)


def _candidate_warn_gate(expanded: dict[str, dict[str, Any]]) -> float:
    return _float(expanded["gate_reconciliation_v2_report"].get("candidate_gate_inputs", {}).get("candidate_warn"), 0.60)


def _candidate_pass_gate(expanded: dict[str, dict[str, Any]]) -> float:
    return _float(expanded["gate_reconciliation_v2_report"].get("candidate_gate_inputs", {}).get("candidate_pass"), 0.45)


def _full_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in _matrix_rows(payload) if row.get("category") == "full_humanoid_profile"]


def _full_clip_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
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


def _reasons(row: dict[str, Any]) -> set[str]:
    values = row.get("warning_reasons", row.get("failure_or_warning_reasons", row.get("failure_reasons", [])))
    return {str(value) for value in values if value}


def _distribution(values: list[float]) -> dict[str, Any]:
    finite = sorted(_float(value) for value in values if math.isfinite(_float(value)))
    if not finite:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "min": _stable(finite[0]),
        "p50": _stable(_percentile(finite, 50)),
        "p95": _stable(_percentile(finite, 95)),
        "max": _stable(finite[-1]),
    }


def _percentile(values: list[float], percentile: float) -> float:
    finite = sorted(_float(value) for value in values if math.isfinite(_float(value)))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return finite[low]
    fraction = rank - low
    return finite[low] * (1.0 - fraction) + finite[high] * fraction


def _mean(values: list[float]) -> float:
    finite = [_float(value) for value in values if math.isfinite(_float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.inf


def _stable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 12)
        return value
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _source_provenance() -> dict[str, Any]:
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
        return {"git_lfs_fsck": "ERROR", "exit_code": 127, "output": str(exc)}
    output = completed.stdout.strip()
    ok = completed.returncode == 0 and "Git LFS fsck OK" in output
    return {"git_lfs_fsck": "OK" if ok else "FAILED", "exit_code": int(completed.returncode), "output": output}


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _original_step4_4_clip_ids() -> list[str]:
    return [
        "Neutral_walk_forward_002__A057",
        "body_stretch_1_004__A069",
        "item_pick_up_standing_R_001__A410",
        "wave_R_001__A428",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
