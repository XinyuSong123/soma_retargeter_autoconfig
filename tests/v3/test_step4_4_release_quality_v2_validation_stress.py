from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import REQUIRED_GLOBAL_METHODS, run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_stress_test_records_global_methods_and_guards(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    stress = read_json(artifact_dir / "release_quality_v2_stress_test.json")

    assert stress["threshold_sensitivity_without_changing_selected_thresholds"]["selected_thresholds_unchanged"] is True
    assert stress["deterministic_rerun_stability"]["stable_44_of_44"] is True
    assert stress["raw_residual_monotonicity"]["raw_residual_monotonicity_preserved"] is True
    assert {row["method"] for row in stress["global_methods_tried"]} >= REQUIRED_GLOBAL_METHODS
    assert stress["selected_policy_counts"]["rows_below_candidate_warn_gate"] == 9
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_4_audit_rejects_candidate_warned_instability(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    candidate = read_json(artifact_dir / "candidate_warned_deep_audit.json")
    candidate["all_stable_across_four_clips"] = False
    write_json(artifact_dir / "candidate_warned_deep_audit.json", candidate)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["candidate_warned_deep_audit"] >= 1
