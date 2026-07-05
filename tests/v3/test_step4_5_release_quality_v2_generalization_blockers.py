from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_5_release_quality_v2_generalization")


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_step4_5_localizes_primary_blocker_to_hips_root_rotation() -> None:
    hips = read_json("hips_root_rotation_residual_decomposition.json")
    summary = read_json("quality_summary.json")

    assert summary["primary_blocker"] == "BLOCKED_HIPS_ROTATION_RESIDUAL"
    assert hips["primary_blocker"] == "BLOCKED_HIPS_ROTATION_RESIDUAL"
    assert hips["model_count"] == 32
    assert hips["hips_dominant_model_count"] == 32
    assert hips["hips_rotation_dominant_model_count"] == 32
    assert hips["raw_residual_retained"] is True
    assert hips["task_class_weighting_counted"] is False


def test_step4_5_solver_convergence_weak_is_global_only_diagnostic() -> None:
    solver = read_json("solver_convergence_weak_global_diagnostics.json")

    assert solver["weak_row_count"] == 1
    assert solver["global_solver_diagnostics_only"] is True
    assert solver["robot_specific_tuning_used"] is False
    row = solver["rows"][0]
    assert row["model_id"] == "draco3_urdf"
    assert row["clip_count"] == 11
    assert row["global_solver_config_only"] is True
    assert row["robot_specific_solver_tuning_used"] is False
    assert {task["task"] for clip in row["clips"] for task in clip["failed_tasks"]} == {"torso"}


def test_step4_5_preserves_step4_4_invariants_without_candidate_leakage() -> None:
    summary = read_json("quality_summary.json")
    delta = read_json("quality_delta_vs_step4_4.json")

    assert summary["release_candidate_status"] == "BLOCKED_CLIP_GENERALIZATION"
    assert summary["runtime_quality_passed_count"] == 0
    assert summary["runtime_quality_warned_count"] == 32
    assert summary["runtime_quality_failed_count"] == 0
    assert summary["candidate_release_gates_not_counted_as_legacy_runtime_passed"] is True
    assert summary["candidate_thresholds_lowered"] is False
    assert summary["robot_specific_tuning_used"] is False
    assert summary["hard_clip_removal_used"] is False
    assert delta["candidate_blocked_count_decreased_under_strict_policy"] is False
    assert delta["count_deltas"]["release_quality_candidate_blocked_count"] == 9

