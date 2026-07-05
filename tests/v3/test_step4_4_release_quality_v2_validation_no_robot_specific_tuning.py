from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_audit_rejects_per_robot_solver_weights_in_config(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    solver = read_json(artifact_dir / "solver_config.json")
    solver["per_robot_solver_weights"] = {"full_00": 0.5}
    write_json(artifact_dir / "solver_config.json", solver)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["no_robot_specific_tuning"] >= 1


def test_step4_4_audit_rejects_row_local_candidate_denominator(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    normalization = read_json(artifact_dir / "normalization_integrity_v2_report.json")
    normalization["candidate_denominator_scope"] = "row_local"
    write_json(artifact_dir / "normalization_integrity_v2_report.json", normalization)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["normalization_integrity"] >= 1
