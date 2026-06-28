from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_full_pipeline_acceptance import run_audit
from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_json, write_passing_fixture


def test_step4_pass_rc_requires_runtime_pass_row(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_passed_count"] = 0
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)
    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["quality_delta_vs_step3_4"] >= 1 or result.gate_counts["release_candidate_status"] >= 1


def test_step4_blocked_status_is_valid_when_no_breakthrough(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["release_candidate_status"] = "BLOCKED_RESIDUAL_QUALITY"
    write_json(artifact_dir / "quality_summary.json", summary)
    ledger = read_json(artifact_dir / "acceptance_ledger.json")
    ledger["release_candidate_status"] = "BLOCKED_RESIDUAL_QUALITY"
    write_json(artifact_dir / "acceptance_ledger.json", ledger)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)
    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1
