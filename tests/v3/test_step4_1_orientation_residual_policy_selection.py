from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_policy_selection_is_global(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    policy = read_json(artifact_dir / "orientation_policy_selection.json")

    assert policy["global_selected_policy"]["policy"] == "parent_relative_runtime_inv_target"
    assert policy["global_selected_policy"]["robot_specific_tuning_used"] is False
    assert policy["raw_residual_regression_count"] == 0


def test_step4_1_audit_rejects_robot_specific_selected_policy(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    policy = read_json(artifact_dir / "orientation_policy_selection.json")
    policy["global_selected_policy"]["robot_specific_tuning_used"] = True
    write_json(artifact_dir / "orientation_policy_selection.json", policy)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["orientation_policy_selection"] >= 1

