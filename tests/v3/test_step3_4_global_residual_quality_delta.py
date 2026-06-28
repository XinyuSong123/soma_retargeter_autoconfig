from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import _quality_delta_vs_step3_3_payload
from tests.v3.step3_4_global_residual_quality_fixture import baseline_summary, current_summary


def test_step3_4_delta_passes_when_raw_residual_distribution_improves(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "quality_summary.json").write_text(json.dumps(baseline_summary()), encoding="utf-8")
    (baseline_dir / "model_matrix.json").write_text(json.dumps({"rows": [_row("full_00", 0.8, 4.0)]}), encoding="utf-8")
    (baseline_dir / "solver_smoke_matrix.json").write_text(
        json.dumps({"rows": [{"model_id": "full_00", "category": "full_humanoid_profile", "metrics": {"task_residual_p95": 4.0, "task_residual_max": 4.5, "solver_task_count": 1}}]}),
        encoding="utf-8",
    )
    model_matrix = {"rows": [_row("full_00", 0.7, 2.0)]}
    solver_smoke = {"rows": [{"model_id": "full_00", "category": "full_humanoid_profile", "metrics": {"task_residual_p95": 2.0, "task_residual_max": 3.0}}]}

    delta = _quality_delta_vs_step3_3_payload(
        baseline_artifact_dir=baseline_dir,
        current_artifact_dir=tmp_path / "current",
        model_matrix=model_matrix,
        solver_smoke_matrix=solver_smoke,
        quality_summary=current_summary(),
        task_coverage_matrix={"summary": {"task_coverage_mean": 1.0, "task_coverage_min": 1.0}},
        anchor_reliability_matrix={"summary": {"anchor_reliability_mean": 1.0, "anchor_reliability_min": 1.0}},
        source_commit="a" * 40,
    )

    assert delta["metric_distribution_deltas"]["raw_task_residual_p95"]["delta"]["median"] < 0
    assert delta["verdict"] == "PASS"


def test_step3_4_delta_blocks_without_residual_improvement(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "quality_summary.json").write_text(json.dumps(baseline_summary()), encoding="utf-8")
    (baseline_dir / "model_matrix.json").write_text(json.dumps({"rows": [_row("full_00", 0.8, 4.0)]}), encoding="utf-8")
    (baseline_dir / "solver_smoke_matrix.json").write_text(
        json.dumps({"rows": [{"model_id": "full_00", "category": "full_humanoid_profile", "metrics": {"task_residual_p95": 4.0, "task_residual_max": 4.5, "solver_task_count": 1}}]}),
        encoding="utf-8",
    )

    delta = _quality_delta_vs_step3_3_payload(
        baseline_artifact_dir=baseline_dir,
        current_artifact_dir=tmp_path / "current",
        model_matrix={"rows": [_row("full_00", 0.8, 4.0)]},
        solver_smoke_matrix={"rows": [{"model_id": "full_00", "category": "full_humanoid_profile", "metrics": {"task_residual_p95": 4.0, "task_residual_max": 4.5}}]},
        quality_summary=current_summary(),
        task_coverage_matrix={"summary": {"task_coverage_mean": 1.0, "task_coverage_min": 1.0}},
        anchor_reliability_matrix={"summary": {"anchor_reliability_mean": 1.0, "anchor_reliability_min": 1.0}},
        source_commit="a" * 40,
    )

    assert delta["verdict"] == "BLOCKED"


def _row(model_id: str, normalized_p95: float, raw_p95: float) -> dict:
    return {
        "model_id": model_id,
        "category": "full_humanoid_profile",
        "runtime_quality_status": "runtime_quality_warned",
        "solver_backed": True,
        "solver_backed_smoke_attempted": True,
        "solver_backed_smoke_completed": True,
        "residual_only": False,
        "normalized_task_residual_p95": normalized_p95,
        "normalized_task_residual_max": 1.0,
        "raw_task_residual_p95": raw_p95,
        "raw_task_residual_max": raw_p95 + 0.5,
        "task_residual_p95": raw_p95,
        "task_residual_max": raw_p95 + 0.5,
    }
