from __future__ import annotations

from pathlib import Path

from soma_retargeter.tools.step4_4_release_quality_v2_validation import (
    finalize_step4_4_release_quality_v2_validation_artifacts,
)
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import (
    write_passing_fixture as write_step4_3_passing_fixture,
)


def write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    step4_3_dir, _step4_2_dir, source_root = write_step4_3_passing_fixture(tmp_path)
    _shape_step4_3_baseline(step4_3_dir)
    artifact_dir = source_root / "artifacts/retargeting_v3_step4_4_release_quality_v2_validation"
    finalize_step4_4_release_quality_v2_validation_artifacts(
        artifact_dir=artifact_dir,
        baseline_step4_3_artifact_dir=step4_3_dir,
    )
    return artifact_dir, step4_3_dir, source_root


def _shape_step4_3_baseline(step4_3_dir: Path) -> None:
    selection = read_json(step4_3_dir / "normalization_policy_selection.json")
    scale = float(selection["fixed_global_scale"])
    warned_count = 6
    expanded_count = 3

    full = read_json(step4_3_dir / "full_pipeline_matrix.json")
    full_rows = [row for row in full["rows"] if row.get("category") == "full_humanoid_profile"]
    warned_models = {row["model_id"] for row in full_rows[:warned_count]}
    expanded_models = {row["model_id"] for row in full_rows[warned_count : warned_count + expanded_count]}
    blocked_models = {row["model_id"] for row in full_rows[warned_count:]}

    for row in full_rows:
        if row["model_id"] in warned_models:
            value = 0.50 * scale
        elif row["model_id"] in expanded_models:
            value = 0.62 * scale
        else:
            value = 0.70 * scale
        row["raw_task_residual_p95"] = value
        row["orientation_integrated_residual_p95"] = value
        row["raw_task_residual_max"] = value + 0.1
        row["orientation_integrated_residual_max"] = value + 0.1
    write_json(step4_3_dir / "full_pipeline_matrix.json", full)
    write_json(step4_3_dir / "model_matrix.json", full)

    clip = read_json(step4_3_dir / "clip_matrix.json")
    expanded_clip_values = [0.62 * scale, 0.59 * scale, 0.58 * scale, 0.58 * scale]
    clip_offsets: dict[str, int] = {}
    for row in clip["rows"]:
        model_id = row["model_id"]
        if model_id in warned_models:
            value = 0.50 * scale
        elif model_id in expanded_models:
            index = clip_offsets.get(model_id, 0)
            value = expanded_clip_values[index % len(expanded_clip_values)]
            clip_offsets[model_id] = index + 1
        else:
            value = 0.70 * scale
        row.setdefault("per_clip_residual_metrics", {})["orientation_integrated_residual_p95"] = value
        row["per_clip_residual_metrics"]["raw_task_residual_p95"] = value
    write_json(step4_3_dir / "clip_matrix.json", clip)

    gate = read_json(step4_3_dir / "gate_reconciliation_v2_report.json")
    gate["rows_below_candidate_warn_gate"] = sorted(warned_models)
    gate["rows_below_candidate_pass_gate"] = []
    gate["rows_newly_passing_under_candidate"] = []
    gate["rows_still_warned_under_candidate"] = sorted(blocked_models)
    gate["candidate_status_counts"] = {
        "release_quality_candidate_blocked": 26,
        "release_quality_candidate_warned": 6,
    }
    gate["candidate_per_gate_blocker_counts"] = {
        "orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate": 26
    }
    rows = []
    for row in full_rows:
        value = float(row["raw_task_residual_p95"]) / scale
        warned = row["model_id"] in warned_models
        rows.append(
            {
                "model_id": row["model_id"],
                "baseline_runtime_quality_status": "runtime_quality_warned",
                "legacy_runtime_quality_status": "runtime_quality_warned",
                "release_quality_v2_status": "release_quality_candidate_warned" if warned else "release_quality_candidate_blocked",
                "legacy_normalized_task_residual_p95": row.get("normalized_task_residual_p95", 0.99),
                "candidate_metric_name": "orientation_integrated_fixed_global_scale_residual_p95",
                "candidate_metric_value": value,
                "raw_task_residual_p95": row["raw_task_residual_p95"],
                "orientation_integrated_residual_p95": row["orientation_integrated_residual_p95"],
                "hard_safety_passed": True,
                "legacy_gate_blockers": ["normalized_task_residual_p95_above_warn_gate"],
                "candidate_gate_blockers": []
                if warned
                else ["orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate"],
            }
        )
    gate["rows"] = rows
    gate["row_count"] = len(rows)
    write_json(step4_3_dir / "gate_reconciliation_v2_report.json", gate)

    summary = read_json(step4_3_dir / "quality_summary.json")
    summary["rows_below_candidate_warn_gate"] = 6
    summary["rows_below_candidate_pass_gate"] = 0
    summary["release_quality_candidate_passed_count"] = 0
    summary["release_quality_candidate_warned_count"] = 6
    summary["release_quality_candidate_blocked_count"] = 26
    write_json(step4_3_dir / "quality_summary.json", summary)

    ledger = read_json(step4_3_dir / "acceptance_ledger.json")
    ledger["release_quality_candidate_passed_count"] = 0
    ledger["release_quality_candidate_warned_count"] = 6
    write_json(step4_3_dir / "acceptance_ledger.json", ledger)
