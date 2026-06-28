from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_full_pipeline_acceptance import run_audit
from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_json, write_passing_fixture


def test_step4_delta_records_step3_4_baseline_and_breakthrough(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "quality_delta_vs_step3_4.json")

    assert delta["baseline_final_head"] == "77e7c02393a6678ccab40cdb847021d7d94392c9"
    assert delta["count_deltas"]["runtime_quality_passed_count"] == 1
    assert delta["orientation_residual_deltas"]["accepted_breakthrough"] is True
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).status == "PASS_RC"


def test_step4_delta_blocks_count_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "quality_delta_vs_step3_4.json")
    delta["current_counts"]["solver_backed_count"] = 31
    delta["count_deltas"]["solver_backed_count"] = -1
    delta["regressions"] = [{"field": "solver_backed_count"}]
    write_json(artifact_dir / "quality_delta_vs_step3_4.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)
    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["quality_delta_vs_step3_4"] >= 1
