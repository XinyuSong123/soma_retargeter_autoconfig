from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import _quality_delta_vs_step3_2_payload


def test_step3_3_delta_counts_match_summary_and_baseline(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_summary = _summary(runtime_quality_failed_count=9, runtime_quality_warned_count=23)
    (baseline_dir / "quality_summary.json").write_text(json.dumps(baseline_summary), encoding="utf-8")
    current_summary = _summary(runtime_quality_failed_count=8, runtime_quality_warned_count=24)
    model_matrix = {"rows": [_full_row("a", 0.9, 0.2), _full_row("b", 0.8, 0.0)]}

    delta = _quality_delta_vs_step3_2_payload(
        baseline_artifact_dir=baseline_dir,
        current_artifact_dir=tmp_path / "current",
        model_matrix=model_matrix,
        quality_summary=current_summary,
        solver_diagnostics={"rows": []},
        source_commit="a" * 40,
    )

    assert delta["baseline_counts"]["runtime_quality_failed_count"] == 9
    assert delta["current_counts"]["runtime_quality_failed_count"] == 8
    assert delta["count_deltas"]["runtime_quality_failed_count"] == -1
    assert delta["verdict"] == "PASS"


def test_step3_3_delta_blocks_when_failure_count_not_reduced(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_summary = _summary(runtime_quality_failed_count=9, runtime_quality_warned_count=23)
    (baseline_dir / "quality_summary.json").write_text(json.dumps(baseline_summary), encoding="utf-8")

    delta = _quality_delta_vs_step3_2_payload(
        baseline_artifact_dir=baseline_dir,
        current_artifact_dir=tmp_path / "current",
        model_matrix={"rows": []},
        quality_summary=_summary(runtime_quality_failed_count=9, runtime_quality_warned_count=23),
        solver_diagnostics={"rows": []},
        source_commit="a" * 40,
    )

    assert delta["count_deltas"]["runtime_quality_failed_count"] == 0
    assert delta["verdict"] == "BLOCKED"


def _summary(*, runtime_quality_failed_count: int, runtime_quality_warned_count: int) -> dict:
    return {
        "in_scope_total": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "solver_backed_smoke_attempted_count": 32,
        "solver_backed_completed_count": 32,
        "solver_backed_count": 32,
        "residual_only_count": 0,
        "runtime_quality_passed_count": 0,
        "runtime_quality_warned_count": runtime_quality_warned_count,
        "runtime_quality_failed_count": runtime_quality_failed_count,
        "partial_runtime_passed_count": 3,
        "negative_control_runtime_passed_count": 9,
        "high_residual_warning_count": 32,
        "joint_limit_warning_count": 12,
        "joint_limit_smoke_warning_count": 10,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def _full_row(model_id: str, residual: float, joint_violation: float) -> dict:
    return {
        "model_id": model_id,
        "category": "full_humanoid_profile",
        "normalized_task_residual_p95": residual,
        "normalized_task_residual_max": 1.0,
        "joint_limit_violation_count": int(joint_violation > 0),
        "max_joint_limit_violation": joint_violation,
    }
