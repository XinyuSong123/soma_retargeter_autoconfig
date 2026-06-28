from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from scripts.audit_retargeting_v3_step4_full_pipeline_acceptance import run_audit
from tests.v3.step4_full_pipeline_acceptance_fixture import read_json, write_json, write_passing_fixture


def test_step4_audit_accepts_passing_full_pipeline_fixture(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status == "PASS_RC"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44


@pytest.mark.parametrize(
    "relative_path",
    [
        "full_pipeline_matrix.json",
        "trajectory_export_manifest.json",
        "orientation_residual_taxonomy.json",
        "normalization_audit.json",
    ],
)
def test_step4_audit_rejects_missing_required_step4_artifacts(tmp_path: Path, relative_path: str) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / relative_path).unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "missing_required_artifacts")


def test_step4_audit_rejects_no_quality_breakthrough(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def remove_breakthrough(row: dict) -> None:
        if row.get("category") != "full_humanoid_profile":
            return
        row["runtime_quality_status"] = "runtime_quality_warned"
        row["release_candidate_row_status"] = "WARNED_NO_BREAKTHROUGH"
        row["quality_classification"] = "runtime_quality_warned"
        row["normalized_task_residual_p95"] = 0.70
        row["normalized_task_residual_max"] = 0.90
        row["rotation_residual_p95"] = 1.20
        row["rotation_residual_max"] = 1.25
        row["warning_reasons"] = ["high_task_residual", "rotation_residual_dominates"]
        row["runtime_quality_warning_reasons"] = ["high_task_residual", "rotation_residual_dominates"]
        row["failure_or_warning_reasons"] = ["high_task_residual", "rotation_residual_dominates"]

    mutate_rows(artifact_dir, remove_breakthrough)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_passed_count"] = 0
    summary["runtime_quality_warned_count"] = 32
    summary["high_residual_warning_count"] = 32
    summary["median_normalized_task_residual_p95"] = 0.70
    summary["p95_normalized_task_residual_p95"] = 0.70
    summary["max_normalized_task_residual_p95"] = 0.90
    summary["median_rotation_residual_p95"] = 1.20
    summary["p95_rotation_residual_p95"] = 1.20
    summary["max_rotation_residual_p95"] = 1.25
    write_json(artifact_dir / "quality_summary.json", summary)
    delta = read_json(artifact_dir / "quality_delta_vs_step3_4.json")
    delta["metric_distribution_deltas"]["normalized_task_residual_p95"]["current"] = {"median": 0.70, "p95": 0.70, "max": 0.70}
    delta["metric_distribution_deltas"]["normalized_task_residual_p95"]["delta"] = {"median": 0.0, "p95": 0.0, "max": 0.0}
    for metric in ("rotation_residual_p95", "p95_rotation_residual_p95"):
        delta["metric_distribution_deltas"][metric]["current"] = {"median": 1.20, "p95": 1.20, "max": 1.20}
        delta["metric_distribution_deltas"][metric]["delta"] = {"median": 0.0, "p95": 0.0, "max": 0.0}
    delta["orientation_residual_deltas"]["p95_rotation_residual_p95"] = {"baseline": 1.20, "current": 1.20, "delta": 0.0}
    delta["orientation_residual_deltas"]["accepted_breakthrough"] = False
    delta["current_counts"]["runtime_quality_passed_count"] = 0
    delta["current_counts"]["runtime_quality_warned_count"] = 32
    delta["current_counts"]["high_residual_warning_count"] = 32
    delta["count_deltas"]["runtime_quality_passed_count"] = 0
    delta["count_deltas"]["runtime_quality_warned_count"] = 0
    delta["count_deltas"]["high_residual_warning_count"] = 0
    delta["improvements"] = []
    write_json(artifact_dir / "quality_delta_vs_step3_4.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "quality_breakthrough")


def test_step4_audit_rejects_runtime_quality_failed_count_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def fail_one_row(row: dict) -> None:
        if row.get("model_id") == "full_00":
            row["runtime_quality_status"] = "runtime_quality_failed"
            row["release_candidate_row_status"] = "BLOCKED_PIPELINE_REGRESSION_ROW"
            row["failure_reasons"] = ["synthetic_runtime_quality_failure"]

    mutate_rows(artifact_dir, fail_one_row)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_passed_count"] = 0
    summary["runtime_quality_warned_count"] = 31
    summary["runtime_quality_failed_count"] = 1
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "solver_backed_counts")


def test_step4_audit_rejects_solver_backed_count_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def remove_solver_backing(row: dict) -> None:
        if row.get("model_id") == "full_00":
            row["solver_backed"] = False
            row["solver_backed_smoke_completed"] = False

    mutate_rows(artifact_dir, remove_solver_backing)
    for relative in ("solver_smoke_matrix.json", "generic_smoke_matrix.json", "solver_diagnostics_matrix.json"):
        payload = read_json(artifact_dir / relative)
        payload["rows"][0]["solver_backed"] = False
        payload["rows"][0]["solver_backed_smoke_completed"] = False
        write_json(artifact_dir / relative, payload)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["solver_backed_count"] = 31
    summary["solver_backed_completed_count"] = 31
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "solver_backed_counts")


def test_step4_audit_rejects_residual_only_count_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def mark_residual_only(row: dict) -> None:
        if row.get("model_id") == "full_01":
            row["residual_only"] = True

    mutate_rows(artifact_dir, mark_residual_only)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["residual_only_count"] = 1
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "solver_backed_counts")


def test_step4_audit_rejects_residual_only_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def residual_only_pass(row: dict) -> None:
        if row.get("model_id") == "full_00":
            row["runtime_quality_status"] = "runtime_quality_passed"
            row["release_candidate_row_status"] = "PASS_RC_ROW"
            row["solver_backed"] = False
            row["solver_backed_smoke_completed"] = False
            row["residual_only"] = True
            row["normalized_task_residual_p95"] = 0.01
            row["normalized_task_residual_max"] = 0.02

    mutate_rows(artifact_dir, residual_only_pass)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["solver_backed_count"] = 31
    summary["solver_backed_completed_count"] = 31
    summary["residual_only_count"] = 1
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "runtime_quality_label_honesty")


def test_step4_audit_rejects_pass_row_without_solver_backed_evidence(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def pass_without_solver(row: dict) -> None:
        if row.get("model_id") == "full_00":
            row["runtime_quality_status"] = "runtime_quality_passed"
            row["solver_backed"] = False
            row["solver_backed_smoke_completed"] = False
            row["residual_only"] = False
            row["normalized_task_residual_p95"] = 0.01
            row["normalized_task_residual_max"] = 0.02

    mutate_rows(artifact_dir, pass_without_solver)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["solver_backed_count"] = 31
    summary["solver_backed_completed_count"] = 31
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "runtime_quality_label_honesty")


def test_step4_audit_rejects_partial_or_negative_promotion(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)

    def promote_non_full(row: dict) -> None:
        if row.get("model_id") in {"partial_00", "negative_00"}:
            row["runtime_quality_status"] = "runtime_quality_passed"
            row["release_candidate_row_status"] = "PASS_RC_ROW"
            row["solver_backed"] = True
            row["solver_backed_smoke_attempted"] = True
            row["solver_backed_smoke_completed"] = True
            row["promoted_to_runtime_quality"] = True
            row["quality_evaluated"] = True

    mutate_rows(artifact_dir, promote_non_full)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "negative_and_partial_not_promoted")


def test_step4_audit_rejects_suspicious_normalization_hiding_raw_residual_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    normalization = read_json(artifact_dir / "normalization_audit.json")
    normalization["normalization_hides_raw_regression"] = True
    normalization["raw_residual_regression_count"] = 1
    normalization["rows"][0]["raw_task_residual_p95"] = 3.50
    normalization["rows"][0]["normalized_task_residual_p95"] = 0.10
    normalization["rows"][0]["raw_regression_hidden"] = True
    write_json(artifact_dir / "normalization_audit.json", normalization)
    delta = read_json(artifact_dir / "quality_delta_vs_step3_4.json")
    delta["normalization_deltas"]["normalization_hides_raw_regression"] = True
    delta["normalization_deltas"]["raw_residual_regression_count"] = 1
    delta["regressions"] = ["normalization_hides_raw_residual_regression"]
    write_json(artifact_dir / "quality_delta_vs_step3_4.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "normalization_integrity")


def test_step4_audit_rejects_missing_export_finite_checks(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    manifest = read_json(artifact_dir / "trajectory_export_manifest.json")
    manifest["rows"][0].pop("finite")
    manifest["rows"][0].pop("qpos_finite")
    manifest["rows"][0]["nan_count"] = 1
    write_json(artifact_dir / "trajectory_export_manifest.json", manifest)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "trajectory_exports")


def test_step4_audit_rejects_missing_temporal_finite_checks(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    temporal = read_json(artifact_dir / "temporal_continuity_matrix.json")
    temporal["finite_count"] = 31
    temporal["rows"][0].pop("finite")
    temporal["rows"][0]["inf_count"] = 1
    write_json(artifact_dir / "temporal_continuity_matrix.json", temporal)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["temporal_continuity_finite_count"] = 31
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "temporal_continuity")


def test_step4_audit_rejects_dirty_provenance(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    environment = read_json(artifact_dir / "environment.json")
    environment["git_status_short"] = " M soma_retargeter/runtime/v3/full_pipeline.py"
    environment["source_worktree_clean_after_run"] = False
    environment["core_diff_after_source_commit"] = ["soma_retargeter/runtime/v3/full_pipeline.py"]
    write_json(artifact_dir / "environment.json", environment)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "clean_provenance")


def test_step4_audit_rejects_deterministic_mismatch(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")
    deterministic["status"] = "failed"
    deterministic["deterministic"] = False
    deterministic["matched_count"] = 43
    deterministic["deterministic_matched_count"] = 43
    deterministic["mismatched_model_ids"] = ["full_00"]
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["deterministic_matched_count"] = 43
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "deterministic_rerun")


def test_step4_audit_enforces_release_candidate_status_semantics(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["release_candidate_status"] = "PASS"
    write_json(artifact_dir / "quality_summary.json", summary)
    ledger = read_json(artifact_dir / "acceptance_ledger.json")
    ledger["verdict"] = "PASS"
    ledger["release_candidate_status"] = "PASS"
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    delta = read_json(artifact_dir / "quality_delta_vs_step3_4.json")
    delta["verdict"] = "PASS"
    write_json(artifact_dir / "quality_delta_vs_step3_4.json", delta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert_blocks(result, "release_candidate_status")


def mutate_rows(artifact_dir: Path, mutator: Callable[[dict], None]) -> None:
    for relative in ("model_matrix.json", "full_pipeline_matrix.json"):
        payload = read_json(artifact_dir / relative)
        for row in payload["rows"]:
            mutator(row)
        write_json(artifact_dir / relative, payload)


def assert_blocks(result: object, *expected_gates: str) -> None:
    payload = result.to_json()
    assert str(result.status).startswith("BLOCKED"), payload
    assert result.blocking_count > 0, payload
    for gate in expected_gates:
        assert result.gate_counts.get(gate, 0) >= 1, payload
