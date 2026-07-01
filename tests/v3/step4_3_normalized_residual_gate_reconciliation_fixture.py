from __future__ import annotations

import json
from pathlib import Path
import shutil

from soma_retargeter.tools.step4_3_normalized_residual_gate_reconciliation import (
    finalize_step4_3_normalized_residual_gate_reconciliation_artifacts,
)
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import (
    read_json,
    write_blocked_fixture as write_step4_2_blocked_fixture,
    write_json,
    write_passing_fixture as write_step4_2_passing_fixture,
)


def write_passing_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    step4_2_dir, _baseline_step4_1_dir, source_root = write_step4_2_passing_fixture(tmp_path)
    _make_candidate_breakthrough_scale(step4_2_dir)
    _write_solver_diagnostics(step4_2_dir)
    artifact_dir = source_root / "artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation"
    shutil.copytree(step4_2_dir, artifact_dir)
    finalize_step4_3_normalized_residual_gate_reconciliation_artifacts(
        artifact_dir=artifact_dir,
        baseline_step4_2_artifact_dir=step4_2_dir,
    )
    return artifact_dir, step4_2_dir, source_root


def write_blocked_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["release_candidate_status"] = "BLOCKED_GATE_RECONCILIATION"
    summary["primary_quality_breakthrough"] = False
    summary["rows_below_candidate_warn_gate"] = 0
    summary["rows_below_candidate_pass_gate"] = 0
    summary["release_quality_candidate_passed_count"] = 0
    summary["release_quality_candidate_warned_count"] = 0
    write_json(artifact_dir / "quality_summary.json", summary)

    ledger = read_json(artifact_dir / "acceptance_ledger.json")
    ledger["release_candidate_status"] = "BLOCKED_GATE_RECONCILIATION"
    ledger["status"] = "BLOCKED"
    ledger["verdict"] = "BLOCKED"
    write_json(artifact_dir / "acceptance_ledger.json", ledger)

    delta = read_json(artifact_dir / "runtime_scoring_delta_vs_step4_2.json")
    delta["rows_below_candidate_warn_gate"] = 0
    delta["rows_below_candidate_pass_gate"] = 0
    delta["release_quality_candidate_passed_count"] = 0
    delta["release_quality_candidate_warned_count"] = 0
    delta["primary_quality_breakthrough"] = False
    delta["improvements"] = []
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_2.json", delta)

    quality_delta = read_json(artifact_dir / "quality_delta_vs_step4_2.json")
    quality_delta["release_candidate_status"] = "BLOCKED_GATE_RECONCILIATION"
    quality_delta["primary_quality_breakthrough"] = False
    quality_delta["verdict"] = "BLOCKED"
    write_json(artifact_dir / "quality_delta_vs_step4_2.json", quality_delta)

    gate = read_json(artifact_dir / "gate_reconciliation_v2_report.json")
    gate["rows_below_candidate_warn_gate"] = []
    gate["rows_below_candidate_pass_gate"] = []
    gate["rows_newly_passing_under_candidate"] = []
    gate["rows_still_warned_under_candidate"] = [row["model_id"] for row in gate["rows"]]
    gate["candidate_status_counts"] = {"release_quality_candidate_blocked": 32}
    for row in gate["rows"]:
        row["release_quality_v2_status"] = "release_quality_candidate_blocked"
        row["candidate_gate_blockers"] = ["orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate"]
    write_json(artifact_dir / "gate_reconciliation_v2_report.json", gate)
    return artifact_dir, baseline_dir, source_root


def _make_candidate_breakthrough_scale(step4_2_dir: Path) -> None:
    delta_path = step4_2_dir / "runtime_scoring_delta_vs_step4_1.json"
    delta = read_json(delta_path)
    raw_delta = delta["metric_distribution_deltas"]["raw_task_residual_p95"]
    raw_delta["baseline"] = {"median": 7.0, "p95": 7.0, "max": 7.0}
    raw_delta["delta"] = {"median": -3.5, "p95": -3.5, "max": -3.5}
    write_json(delta_path, delta)
    ledger = read_json(step4_2_dir / "acceptance_ledger.json")
    ledger["runtime_scoring_delta_vs_step4_1"] = delta
    write_json(step4_2_dir / "acceptance_ledger.json", ledger)


def _write_solver_diagnostics(artifact_dir: Path) -> None:
    rows = []
    for row in read_json(artifact_dir / "full_pipeline_matrix.json")["rows"]:
        if row.get("category") != "full_humanoid_profile":
            continue
        rows.append(
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "task_diagnostics": [
                    {
                        "per_semantic": {
                            "Hips": {
                                "combined_residual": 3.5,
                                "rotation_residual": 2.0,
                                "translation_residual": 1.5,
                            },
                            "LeftHand": {
                                "combined_residual": 2.0,
                                "rotation_residual": 1.0,
                                "translation_residual": 1.0,
                            },
                        }
                    }
                ],
            }
        )
    write_json(artifact_dir / "solver_diagnostics_matrix.json", {"schema_version": 1, "row_count": len(rows), "rows": rows})


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json_file(path: Path) -> dict:
    return json.loads(read_text(path))
