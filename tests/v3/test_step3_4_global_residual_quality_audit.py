from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step3_4_global_residual_quality import run_audit
from tests.v3.step3_4_global_residual_quality_fixture import read_json, write_json, write_passing_fixture


def test_step3_4_audit_accepts_passing_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44


def test_step3_4_audit_rejects_missing_delta(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / "quality_delta_vs_step3_3.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_required_artifacts"] >= 1
    assert result.gate_counts["quality_delta_vs_step3_3"] >= 1


def test_step3_4_audit_rejects_no_residual_improvement_with_32_warnings(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "quality_delta_vs_step3_3.json")
    delta["metric_distribution_deltas"]["raw_task_residual_p95"]["delta"] = {"median": 0.0, "p95": 0.0, "max": 0.0}
    delta["metric_distribution_deltas"]["task_residual_p95"]["delta"] = {"median": 0.0, "p95": 0.0, "max": 0.0}
    delta["metric_distribution_deltas"]["normalized_task_residual_p95"]["delta"] = {"median": 0.0, "p95": 0.0, "max": 0.0}
    delta["improvements"] = []
    delta["verdict"] = "BLOCKED"
    write_json(artifact_dir / "quality_delta_vs_step3_3.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["residual_improvement"] >= 1


def test_step3_4_audit_rejects_core_invariant_regressions(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_failed_count"] = 1
    summary["solver_backed_count"] = 31
    summary["residual_only_count"] = 1
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["solver_backed_counts"] >= 1


def test_step3_4_audit_rejects_residual_only_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    matrix = read_json(artifact_dir / "model_matrix.json")
    row = next(item for item in matrix["rows"] if item["category"] == "full_humanoid_profile")
    row["runtime_quality_status"] = "runtime_quality_passed"
    row["residual_only"] = True
    row["normalized_task_residual_p95"] = 0.01
    write_json(artifact_dir / "model_matrix.json", matrix)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["solver_backed_counts"] >= 1 or result.gate_counts["runtime_quality_label_honesty"] >= 1


def test_step3_4_audit_rejects_partial_or_negative_promotion(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    matrix = read_json(artifact_dir / "model_matrix.json")
    row = next(item for item in matrix["rows"] if item["category"] == "negative_control")
    row["runtime_quality_status"] = "runtime_quality_passed"
    row["solver_backed"] = True
    row["solver_backed_smoke_attempted"] = True
    row["solver_backed_smoke_completed"] = True
    row["promoted_to_runtime_quality"] = True
    write_json(artifact_dir / "model_matrix.json", matrix)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["negative_and_partial_not_promoted"] >= 1


def test_step3_4_audit_rejects_missing_task_or_anchor_evidence(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / "task_coverage_matrix.json").unlink()
    (artifact_dir / "anchor_reliability_matrix.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_required_artifacts"] >= 2
    assert result.gate_counts["step3_4_evidence"] >= 1


def test_step3_4_audit_rejects_dirty_provenance(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    env = read_json(artifact_dir / "environment.json")
    env["git_status_short"] = " M soma_retargeter/runtime/v3/generic_smoke.py"
    write_json(artifact_dir / "environment.json", env)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["clean_provenance"] >= 1
