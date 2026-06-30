from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json, write_passing_fixture


def test_step4_2_runtime_delta_contains_required_impact_fields(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json")

    assert delta["runtime_quality_passed_count_delta"] == 0
    assert delta["high_residual_warning_count_delta"] == 0
    assert delta["p95_orientation_integrated_residual_delta"] <= -0.25
    assert delta["raw_residual_regression_count"] == 0
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_2_rejects_runtime_scoring_delta_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json")
    delta["regressions"] = [{"field": "solver_backed_count", "current": 31}]
    write_json(artifact_dir / "runtime_scoring_delta_vs_step4_1.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["runtime_scoring_delta_vs_step4_1"] >= 1
