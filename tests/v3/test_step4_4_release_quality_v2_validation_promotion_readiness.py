from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_promotion_readiness_is_diagnostic_only(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    readiness = read_json(artifact_dir / "release_quality_v2_promotion_readiness.json")

    assert readiness["decision"] == "keep_diagnostic_only"
    assert readiness["promote_to_release_candidate_gate"] is False
    assert readiness["production_default_change_allowed"] is False
    assert readiness["runtime_override_default_enabled"] is False
    assert readiness["legacy_gates_unchanged"] is True
    assert readiness["why"]
    assert readiness["risks"]
    assert readiness["required_future_evidence"]
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_4_audit_rejects_promotion_safety_mutation(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    readiness = read_json(artifact_dir / "release_quality_v2_promotion_readiness.json")
    readiness["production_default_change_allowed"] = True
    write_json(artifact_dir / "release_quality_v2_promotion_readiness.json", readiness)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["promotion_readiness"] >= 1
