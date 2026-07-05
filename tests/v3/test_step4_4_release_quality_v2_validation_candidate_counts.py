from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_candidate_counts_expand_without_legacy_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    delta = read_json(artifact_dir / "quality_delta_vs_step4_3.json")
    gate = read_json(artifact_dir / "gate_reconciliation_v3_report.json")

    assert summary["rows_below_candidate_warn_gate"] == 9
    assert summary["rows_below_candidate_warn_gate"] > summary["step4_3_rows_below_candidate_warn_gate"]
    assert summary["release_quality_candidate_blocked_count"] == 23
    assert summary["release_quality_candidate_blocked_count"] < summary["step4_3_release_quality_candidate_blocked_count"]
    assert summary["runtime_quality_passed_count"] == 0
    assert delta["count_deltas"]["rows_below_candidate_warn_gate"] == 3
    assert delta["count_deltas"]["release_quality_candidate_blocked_count"] == -3
    assert len(gate["rows_newly_warned_by_global_clip_aggregation"]) == 3
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_4_audit_rejects_candidate_count_mismatch(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["rows_below_candidate_warn_gate"] += 1
    (artifact_dir / "quality_summary.json").write_text(__import__("json").dumps(summary), encoding="utf-8")

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["gate_reconciliation_v3"] >= 1
