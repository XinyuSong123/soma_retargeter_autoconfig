from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_5_release_quality_v2_generalization")


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_step4_5_expanded_manifest_freezes_all_loadable_bvh_clips() -> None:
    manifest = read_json("expanded_clip_manifest.json")
    validation = read_json("release_quality_v2_expanded_validation_matrix.json")

    assert manifest["clip_count"] == 11
    assert manifest["expanded_from_step4_4_clip_count"] is True
    assert manifest["hard_clip_removal_used"] is False
    assert manifest["robot_specific_clip_selection_used"] is False
    assert validation["clip_count"] == manifest["clip_count"]
    assert validation["row_count"] == 32 * manifest["clip_count"]


def test_step4_5_candidate_warned_rows_do_not_generalize_under_strict_guard() -> None:
    stability = read_json("candidate_warned_expanded_stability.json")
    readiness = read_json("release_quality_v2_generalization_readiness.json")

    assert stability["original_step4_3_candidate_warned_count"] == 6
    assert stability["newly_step4_4_candidate_warned_count"] == 3
    assert stability["original_stable_under_expanded_guard_count"] == 0
    assert stability["newly_stable_under_expanded_guard_count"] == 0
    assert stability["candidate_warned_generalization_passed"] is False
    assert readiness["decision"] == "keep_diagnostic_only"
    assert readiness["promote_to_release_candidate_gate"] is False
    assert readiness["mean_aggregation_diagnostic_only"] is True


def test_step4_5_task_contract_is_diagnostic_and_raw_residual_safe() -> None:
    contract = read_json("task_level_residual_contract.json")
    summary = read_json("quality_summary.json")

    assert contract["contract_status"] == "defined_audited_diagnostic_only"
    assert contract["task_class_weighting_counted"] is False
    assert {row["task"] for row in contract["task_classes"]} == {
        "torso",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
    }
    assert contract["audit_requirements"]["raw_residual_retained"] is True
    assert contract["audit_requirements"]["no_semantic_downweighting_to_hide_hips_root_rotation"] is True
    assert summary["task_class_weighting_counted"] is False
    assert summary["raw_residual_hiding_used"] is False

