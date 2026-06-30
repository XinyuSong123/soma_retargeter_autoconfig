from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json, write_passing_fixture


def test_step4_2_normalization_retains_raw_residual(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    normalization = read_json(artifact_dir / "scoring_normalization_audit.json")

    assert normalization["raw_residual_always_retained"] is True
    assert normalization["normalization_hides_raw_residual_regression"] is False
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_2_rejects_normalization_hiding_raw_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    normalization = read_json(artifact_dir / "scoring_normalization_audit.json")
    normalization["normalization_hides_raw_residual_regression"] = True
    write_json(artifact_dir / "scoring_normalization_audit.json", normalization)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["normalization_integrity"] >= 1
