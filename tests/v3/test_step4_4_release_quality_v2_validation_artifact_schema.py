from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_audit_output_has_required_super_goal_sections(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    payload = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).to_json()

    for key in (
        "status",
        "blocking_count",
        "finding_count",
        "findings",
        "quality_status",
        "invariants",
        "candidate_counts",
        "legacy_counts",
        "normalization_integrity",
        "candidate_stability",
        "blocker_taxonomy_summary",
        "final_head_ci",
        "lfs",
    ):
        assert key in payload
    assert payload["status"] == "PASS_RC"
    assert payload["invariants"]["row_count"] == 44


def test_step4_4_schema_records_source_and_artifact_provenance(tmp_path: Path) -> None:
    artifact_dir, _baseline_dir, _source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    ledger = read_json(artifact_dir / "acceptance_ledger.json")

    assert summary["artifact_dir"].endswith("retargeting_v3_step4_4_release_quality_v2_validation")
    assert summary["baseline_step4_3_artifact_dir"].endswith("retargeting_v3_step4_3_normalized_residual_gate_reconciliation")
    assert "source_branch" in summary
    assert "source_commit" in summary
    assert "baseline_step4_3" in ledger
