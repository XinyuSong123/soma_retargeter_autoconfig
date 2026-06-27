from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_step3_2_solver_backed_smoke import (
    EXPECTED_BASE_STEP3_1_1_FINAL_HEAD,
    run_audit,
)


def test_step3_2_solver_backed_smoke_audit_accepts_pass_fixture(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_step3_2_fixture(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, require_final_head_ci=True)

    assert result.status == "PASS"
    assert result.blocking_count == 0
    assert result.matrix_row_count == 44
    assert result.status_counts == {
        "negative_control_passed": 9,
        "partial_passed": 3,
        "passed": 32,
    }

    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    ledger.pop("final_head_ci")
    _write_json(artifact_dir / "acceptance_ledger.json", ledger)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary.pop("final_head_ci")
    _write_json(artifact_dir / "quality_summary.json", summary)

    result_without_strict_ci = run_audit(artifact_dir=artifact_dir, source_root=source_root)
    strict_result = run_audit(artifact_dir=artifact_dir, source_root=source_root, require_final_head_ci=True)

    assert result_without_strict_ci.status == "PASS"
    assert strict_result.status == "BLOCKED"
    assert strict_result.gate_counts["final_head_ci"] >= 1


def test_step3_2_solver_backed_smoke_audit_blocks_zero_completed(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_step3_2_fixture(tmp_path)
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    for row in model_matrix["rows"]:
        if row["category"] == "full_humanoid_profile":
            row["solver_backed_smoke_completed"] = False
            row["solver_backed"] = False
    _write_json(artifact_dir / "model_matrix.json", model_matrix)

    solver_matrix = _read_json(artifact_dir / "solver_smoke_matrix.json")
    for row in solver_matrix["rows"]:
        row["solver_backed_smoke_completed"] = False
        row["completed"] = False
        row["solver_backed"] = False
    _write_json(artifact_dir / "solver_smoke_matrix.json", solver_matrix)

    generic_matrix = _read_json(artifact_dir / "generic_smoke_matrix.json")
    for row in generic_matrix["rows"]:
        if row["category"] == "full_humanoid_profile":
            row["solver_backed"] = False
            row["solver_backed_smoke_completed"] = False
    _write_json(artifact_dir / "generic_smoke_matrix.json", generic_matrix)

    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["solver_backed_completed_count"] = 0
    summary["solver_backed_count"] = 0
    _write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["solver_backed_smoke_counts"] >= 2
    assert result.gate_counts["runtime_quality_label_honesty"] >= 1


def test_step3_2_solver_backed_smoke_audit_blocks_missing_solver_evidence(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_step3_2_fixture(tmp_path)
    (artifact_dir / "solver_smoke_matrix.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_required_artifacts"] == 1
    assert result.gate_counts["solver_evidence_present"] >= 1


def test_step3_2_solver_backed_smoke_audit_blocks_residual_only_pass(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_step3_2_fixture(tmp_path)
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    row = next(item for item in model_matrix["rows"] if item["category"] == "full_humanoid_profile")
    row["residual_only"] = True
    row["solver_mode"] = "generic_fk_residual_smoke"
    row["solver_backed_smoke_completed"] = False
    row["solver_backed"] = False
    row["runtime_quality_status"] = "runtime_quality_passed"
    _write_json(artifact_dir / "model_matrix.json", model_matrix)

    model_id = row["model_id"]
    solver_matrix = _read_json(artifact_dir / "solver_smoke_matrix.json")
    for solver_row in solver_matrix["rows"]:
        if solver_row["model_id"] == model_id:
            solver_row["residual_only"] = True
            solver_row["solver_mode"] = "generic_fk_residual_smoke"
            solver_row["solver_backed_smoke_completed"] = False
            solver_row["completed"] = False
            solver_row["solver_backed"] = False
    _write_json(artifact_dir / "solver_smoke_matrix.json", solver_matrix)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["runtime_quality_label_honesty"] >= 1


def test_step3_2_solver_backed_smoke_audit_blocks_negative_promotion(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_step3_2_fixture(tmp_path)
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    row = next(item for item in model_matrix["rows"] if item["category"] == "negative_control")
    row["runtime_quality_status"] = "runtime_quality_passed"
    row["final_step3_2_status"] = "runtime_quality_passed"
    row["solver_backed_smoke_attempted"] = True
    row["solver_backed_smoke_completed"] = True
    row["solver_backed"] = True
    row["promoted_to_runtime_quality"] = True
    row["quality_evaluated"] = True
    row["override_allowed"] = True
    row["humanoid_profile_generated"] = True
    _write_json(artifact_dir / "model_matrix.json", model_matrix)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["negative_and_partial_not_promoted"] >= 5


def _write_passing_step3_2_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step3_2_solver_backed_smoke"
    (artifact_dir / "test_results").mkdir(parents=True)

    rows = _model_rows()
    solver_rows = _solver_rows(rows)
    generic_rows = _generic_rows(rows)
    final_head_ci = _final_head_ci_payload()
    summary = {
        "schema_version": 1,
        "base_step3_1_1_final_head": EXPECTED_BASE_STEP3_1_1_FINAL_HEAD,
        "row_count": 44,
        "in_scope_total": 44,
        "matrix_row_count": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "status_counts": {
            "passed": 32,
            "partial_passed": 3,
            "negative_control_passed": 9,
        },
        "solver_backed_smoke_attempted_count": 32,
        "solver_backed_completed_count": 32,
        "solver_backed_failed_count": 0,
        "solver_backed_count": 32,
        "residual_only_count": 0,
        "runtime_quality_passed_count": 32,
        "runtime_quality_warned_count": 0,
        "runtime_quality_failed_count": 0,
        "partial_runtime_passed_count": 3,
        "negative_control_runtime_passed_count": 9,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
        "final_head_ci": final_head_ci,
    }
    _write_json(
        artifact_dir / "environment.json",
        {
            "schema_version": 1,
            "git_status_short": "",
            "source_code_commit": "a" * 40,
            "source_code_commit_remote_resolvable": True,
            "source_code_commit_is_artifact_commit_ancestor": True,
            "source_worktree_clean_before_run": True,
            "source_worktree_clean_after_run": True,
            "core_diff_after_source_commit": [],
        },
    )
    (artifact_dir / "commands.txt").write_text(
        "PYTHONPATH=. python scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py\n",
        encoding="utf-8",
    )
    (artifact_dir / "test_results/pytest.txt").write_text("5 passed\n", encoding="utf-8")
    _write_json(artifact_dir / "test_results/pytest_summary.json", {"passed": 5, "failed": 0})
    _write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "rows": rows})
    _write_json(artifact_dir / "solver_smoke_matrix.json", {"schema_version": 1, "rows": solver_rows})
    _write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "rows": generic_rows})
    _write_json(
        artifact_dir / "pipeline_backed_matrix.json",
        {
            "schema_version": 1,
            "status": "referenced",
            "source_artifact": "artifacts/retargeting_v3_step3_runtime_quality/pipeline_backed_matrix.json",
        },
    )
    _write_json(artifact_dir / "quality_summary.json", summary)
    _write_json(
        artifact_dir / "acceptance_ledger.json",
        {
            "schema_version": 1,
            "verdict": "PASS",
            "base_step3_1_1_final_head": EXPECTED_BASE_STEP3_1_1_FINAL_HEAD,
            "status_counts": summary["status_counts"],
            "solver_backed_completed_count": 32,
            "solver_backed_count": 32,
            "final_head_ci": final_head_ci,
        },
    )
    _write_json(
        artifact_dir / "deterministic_rerun.json",
        {
            "schema_version": 1,
            "status": "passed",
            "deterministic": True,
            "compared_count": 44,
            "matched_count": 44,
            "deterministic_compared_count": 44,
            "deterministic_matched_count": 44,
        },
    )
    return artifact_dir, source_root


def _model_rows() -> list[dict]:
    rows: list[dict] = []
    specs: list[tuple[str, str, str, str]] = [
        (f"humanoid_solver_{index:02d}", "full_humanoid_profile", "passed", "positive_humanoid")
        for index in range(32)
    ]
    specs.extend(
        (f"partial_solver_{index:02d}", "partial_humanoid_profile", "partial_passed", "partial_humanoid")
        for index in range(3)
    )
    specs.extend(
        (f"negative_solver_{index:02d}", "negative_control", "negative_control_passed", "negative_control")
        for index in range(9)
    )

    for model_id, category, source_status, expected_capability in specs:
        full = category == "full_humanoid_profile"
        row = {
            "model_id": model_id,
            "category": category,
            "source_status": source_status,
            "expected_capability": expected_capability,
            "solver_backed_smoke_attempted": full,
            "solver_backed_smoke_completed": full,
            "solver_backed": full,
            "solver_mode": "generic_solver_backed_projection_smoke" if full else "not_applicable",
            "residual_only": False,
            "solver_failure_reason": None,
            "warning_reasons": [],
            "sampled_frame_indices": [0, 30, 60, 90],
            "deterministic_hash_inputs": {"model_id": model_id, "clip": "canonical"},
            "frame_count": 120,
            "normalized_task_residual_mean": 0.001,
            "normalized_task_residual_p95": 0.002,
            "normalized_task_residual_max": 0.003,
            "target_translation_error_mean": 0.01,
            "target_translation_error_p95": 0.02,
            "target_translation_error_max": 0.03,
            "target_rotation_error_mean": 0.04,
            "target_rotation_error_p95": 0.05,
            "target_rotation_error_max": 0.06,
            "joint_limit_violation_count": 0,
            "max_joint_limit_violation": 0.0,
            "output_nan_count": 0,
            "output_inf_count": 0,
            "runtime_seconds": 1.25,
        }
        if full:
            row.update(
                {
                    "runtime_quality_status": "runtime_quality_passed",
                    "final_step3_2_status": "runtime_quality_passed",
                    "quality_classification": "runtime_quality_passed",
                    "quality_evaluated": True,
                }
            )
        elif category == "partial_humanoid_profile":
            row.update(
                {
                    "runtime_quality_status": "partial_runtime_passed",
                    "final_step3_2_status": "partial_runtime_passed",
                    "quality_classification": "partial_runtime_passed",
                    "quality_evaluated": True,
                }
            )
        else:
            row.update(
                {
                    "runtime_quality_status": "negative_control_runtime_passed",
                    "final_step3_2_status": "negative_control_runtime_passed",
                    "quality_classification": "negative_control_not_promoted",
                    "promoted_to_runtime_quality": False,
                    "quality_evaluated": False,
                    "override_allowed": False,
                    "humanoid_profile_generated": False,
                }
            )
        rows.append(row)
    assert len(rows) == 44
    return rows


def _solver_rows(model_rows: list[dict]) -> list[dict]:
    rows = []
    for row in model_rows:
        if row["category"] != "full_humanoid_profile":
            continue
        rows.append(
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "status": "runtime_quality_passed",
                "solver_backed_smoke_attempted": True,
                "solver_backed_smoke_completed": True,
                "attempted": True,
                "completed": True,
                "solver_backed": True,
                "solver_mode": "generic_solver_backed_projection_smoke",
                "residual_only": False,
            }
        )
    return rows


def _generic_rows(model_rows: list[dict]) -> list[dict]:
    rows = []
    for row in model_rows:
        full = row["category"] == "full_humanoid_profile"
        rows.append(
            {
                "model_id": row["model_id"],
                "category": row["category"],
                "status": row["runtime_quality_status"],
                "runtime_quality_status": row["runtime_quality_status"],
                "solver_backed_smoke_attempted": full,
                "solver_backed_smoke_completed": full,
                "solver_backed": full,
                "solver_mode": "generic_solver_backed_projection_smoke" if full else "not_applicable",
                "residual_only": False,
                "smoke_summary": {
                    "mode": "generic_solver_backed_projection_smoke" if full else "not_applicable",
                    "solver_type": "runtime_solver_projection" if full else "not_applicable",
                    "status": row["runtime_quality_status"],
                },
            }
        )
    return rows


def _final_head_ci_payload() -> dict:
    return {
        "workflow_run_id": "123456789",
        "head_sha": "c" * 40,
        "conclusion": "success",
        "job_conclusions": {
            "step3-2-artifact-audit": "success",
            "step3-2-static-and-unit": "success",
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
