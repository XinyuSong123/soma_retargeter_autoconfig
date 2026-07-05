from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_determinism_is_44_of_44(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")

    assert deterministic["status"] == "passed"
    assert deterministic["deterministic_compared_count"] == 44
    assert deterministic["deterministic_matched_count"] == 44
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_4_audit_rejects_deterministic_mismatch(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")
    deterministic["deterministic_matched_count"] = 43
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["deterministic_rerun"] >= 1
