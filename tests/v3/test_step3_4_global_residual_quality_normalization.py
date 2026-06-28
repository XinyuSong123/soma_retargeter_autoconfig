from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step3_4_global_residual_quality import run_audit
from tests.v3.step3_4_global_residual_quality_fixture import read_json, write_json, write_passing_fixture


def test_step3_4_audit_rejects_suspicious_normalization_reconstruction(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    matrix = read_json(artifact_dir / "model_matrix.json")
    row = next(item for item in matrix["rows"] if item["category"] == "full_humanoid_profile")
    row["raw_task_residual_max"] = 6.0
    row["residual_denominator"] = 3.0
    row["normalized_task_residual_max"] = 1.0
    write_json(artifact_dir / "model_matrix.json", matrix)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["normalization_integrity"] >= 1


def test_step3_4_audit_rejects_robot_specific_denominator(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    diagnostics = read_json(artifact_dir / "solver_diagnostics_matrix.json")
    diagnostics["rows"][0]["residual_denominator_robot_specific"] = True
    write_json(artifact_dir / "solver_diagnostics_matrix.json", diagnostics)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["normalization_integrity"] >= 1
