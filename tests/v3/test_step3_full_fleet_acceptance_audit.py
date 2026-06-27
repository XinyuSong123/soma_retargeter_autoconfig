from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_step3_full_fleet import (
    ACCEPTANCE_GATES,
    EXPECTED_STATUS_COUNTS,
    REQUIRED_ARTIFACT_FILES,
    REQUIRED_PIPELINE_CONTROL_FLAGS,
    main as audit_main,
    run_audit,
)


def test_full_fleet_audit_accepts_complete_44_row_evidence(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, require_final_head_ci=True)

    assert result.status == "PASS"
    assert result.blocking_count == 0
    assert set(ACCEPTANCE_GATES) <= set(result.gate_counts)
    assert len(REQUIRED_ARTIFACT_FILES) == 13
    assert set(EXPECTED_STATUS_COUNTS) == {"passed", "partial_passed", "negative_control_passed"}
    assert set(REQUIRED_PIPELINE_CONTROL_FLAGS) == {
        "default_runtime_disabled_verified",
        "shadow_noop_verified",
        "override_explicit_only",
        "fingerprint_gate_enforced",
        "negative_controls_excluded",
        "artifact_paths_sanitized",
    }


def test_audit_blocks_missing_required_artifact_file(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    (artifact_dir / "pipeline_controls.json").unlink()
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, require_final_head_ci=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_required_artifacts"] == 1
    assert result.gate_counts["pipeline_controls_present"] >= 1


def test_audit_blocks_dirty_or_incomplete_provenance(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    environment = _read_json(artifact_dir / "environment.json")
    environment["git_status_short"] = "M soma_retargeter/runtime/v3/generic_smoke.py\n"
    environment["source_worktree_clean_after_run"] = False
    environment["core_diff_after_source_commit"] = ["soma_retargeter/runtime/v3/generic_smoke.py"]
    _write_json(artifact_dir / "environment.json", environment)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["clean_provenance"] >= 3


def test_audit_blocks_residual_only_smoke_labeled_runtime_quality_pass(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    smoke["rows"][0]["smoke_summary"]["mode"] = "generic_fk_residual_smoke"
    smoke["rows"][0]["smoke_summary"]["status"] = "passed"
    smoke["rows"][0]["smoke_summary"]["residuals"] = {"solver": "runtime_model_fk_residual_evaluation"}
    smoke["rows"][0]["smoke_summary"]["metrics"]["normalized_task_residual_p95"] = 0.91
    _write_json(artifact_dir / "generic_smoke_matrix.json", smoke)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["quality_failed_count"] = 0
    _write_json(artifact_dir / "quality_summary.json", summary)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["runtime_quality_label_honesty"] >= 1
    assert result.gate_counts["status_count_honesty"] >= 1


def test_audit_blocks_missing_final_head_ci_evidence(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    ledger.pop("final_head_ci")
    _write_json(artifact_dir / "acceptance_ledger.json", ledger)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary.pop("final_head_ci")
    _write_json(artifact_dir / "quality_summary.json", summary)
    handoff = source_root / "docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md"
    handoff.write_text("# Step 3.1 Agent F\n\nverdict = BLOCKED\n", encoding="utf-8")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, require_final_head_ci=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["final_head_ci"] >= 2


def test_audit_blocks_dishonest_status_and_smoke_count_mismatches(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["final_status_counts"]["runtime_quality_passed"] = 31
    summary["generic_smoke_success_count"] = 0
    _write_json(artifact_dir / "quality_summary.json", summary)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["status_count_honesty"] >= 2


def test_audit_blocks_non_44_matrix_and_wrong_32_3_9_counts(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    matrix = _read_json(artifact_dir / "full_fleet_matrix.json")
    matrix["matrix"].pop()
    _write_json(artifact_dir / "full_fleet_matrix.json", matrix)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["row_count"] = 43
    summary["status_counts"]["negative_control_passed"] = 8
    _write_json(artifact_dir / "quality_summary.json", summary)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["matrix_shape"] >= 1
    assert result.gate_counts["status_counts_32_3_9"] >= 1


def test_audit_blocks_rpo_g1_only_matrix_even_with_44_rows(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    matrix = _read_json(artifact_dir / "full_fleet_matrix.json")
    for index, row in enumerate(matrix["matrix"]):
        row["model_id"] = f"unitree_g1_clone_{index:02d}"
        row["robot_type"] = "unitree_g1"
    _write_json(artifact_dir / "full_fleet_matrix.json", matrix)
    summary = _read_json(artifact_dir / "quality_summary.json")
    summary["non_rpo_g1_row_count"] = 0
    _write_json(artifact_dir / "quality_summary.json", summary)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["non_rpo_g1_full_fleet"] >= 1


def test_audit_blocks_promoted_negative_controls(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    matrix = _read_json(artifact_dir / "full_fleet_matrix.json")
    negative_row = next(row for row in matrix["matrix"] if row["source_status"] == "negative_control_passed")
    negative_row["runtime_quality_status"] = "quality_passed"
    negative_row["promoted_to_runtime_quality"] = True
    negative_row["quality_evaluated"] = True
    negative_row["override_allowed"] = True
    _write_json(artifact_dir / "full_fleet_matrix.json", matrix)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["negative_controls_not_promoted"] >= 4


def test_audit_blocks_missing_nonfinite_or_misordered_quality_numbers(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    matrix = _read_json(artifact_dir / "full_fleet_matrix.json")
    matrix["matrix"][0].pop("target_translation_error_mean")
    matrix["matrix"][1]["target_rotation_error_max"] = float("nan")
    matrix["matrix"][2]["output_inf_count"] = 1
    matrix["matrix"][3]["target_translation_error_mean"] = 0.4
    matrix["matrix"][3]["target_translation_error_p95"] = 0.3
    matrix["matrix"][3]["target_translation_error_max"] = 0.2
    _write_json(artifact_dir / "full_fleet_matrix.json", matrix)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["quality_numeric_fields"] >= 4


def test_audit_blocks_absolute_path_leakage_and_sanitizes_report(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    leaked_path = "/" + "mnt/ssd1/song/Desktop/soma_retargeter_autoconfig/runtime_quality.py"
    (artifact_dir / "commands.txt").write_text(f"python {leaked_path}\n", encoding="utf-8")
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)
    serialized = json.dumps(result.to_json())

    assert result.status == "BLOCKED"
    assert result.gate_counts["absolute_path_leakage"] == 1
    assert "/" + "mnt/ssd1/song" not in serialized
    assert "${LOCAL_SOURCE_PATH}/runtime_quality.py" in serialized


def test_audit_blocks_missing_pipeline_controls(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    controls = _read_json(artifact_dir / "pipeline_controls.json")
    controls["controls"].pop("shadow_noop_verified")
    _write_json(artifact_dir / "pipeline_controls.json", controls)
    matrix = _read_json(artifact_dir / "full_fleet_matrix.json")
    matrix["matrix"][0]["control_modes"] = ["override_experimental"]
    matrix["matrix"][0]["legacy_default_unchanged"] = False
    _write_json(artifact_dir / "full_fleet_matrix.json", matrix)
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["pipeline_controls_present"] >= 1


def test_audit_blocks_missing_full_repo_pytest_caveat_and_classification(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    ledger["full_repo_pytest"] = {
        "status": "not_run",
        "classification": "unknown",
        "caveat": "",
    }
    _write_json(artifact_dir / "acceptance_ledger.json", ledger)
    handoff = source_root / "docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md"
    handoff.write_text("# Step 3.1 Agent F\n\nverdict = BLOCKED\n", encoding="utf-8")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["full_repo_pytest_caveat"] >= 3


def test_audit_blocks_stale_pass_verdict_when_live_gates_block(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    matrix = _read_json(artifact_dir / "full_fleet_matrix.json")
    matrix["matrix"][0]["frame_count"] = 0
    _write_json(artifact_dir / "full_fleet_matrix.json", matrix)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root)

    assert result.status == "BLOCKED"
    assert result.gate_counts["quality_numeric_fields"] >= 1
    assert result.gate_counts["stale_agent_f_verdict"] >= 2


def test_audit_cli_writes_report_and_returns_nonzero_for_blockers(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    (artifact_dir / "full_fleet_matrix.json").unlink()
    _set_declared_verdict(artifact_dir, source_root, "BLOCKED")
    output_json = artifact_dir / "audit_report.json"

    exit_code = audit_main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--source-root",
            str(source_root),
            "--output-json",
            str(output_json),
        ]
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "BLOCKED"
    assert payload["gate_counts"]["missing_required_artifacts"] == 1
    assert payload["gate_counts"]["matrix_shape"] >= 1


def _write_passing_evidence(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step3_runtime_quality"
    (artifact_dir / "test_results").mkdir(parents=True)
    _write_agent_f_handoff(source_root, "PASS")
    rows = _matrix_rows()
    smoke_rows = _generic_smoke_rows(rows)
    final_head_ci = _final_head_ci_payload()
    _write_json(
        artifact_dir / "environment.json",
        {
            "schema_version": 1,
            "git_status_short": "",
            "source_code_commit": "a" * 40,
            "artifact_commit": "b" * 40,
            "source_code_commit_remote_resolvable": True,
            "source_code_commit_is_artifact_commit_ancestor": True,
            "source_worktree_clean_before_run": True,
            "source_worktree_clean_after_run": True,
            "core_diff_after_source_commit": [],
        },
    )
    (artifact_dir / "commands.txt").write_text(
        "PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py\n",
        encoding="utf-8",
    )
    (artifact_dir / "test_results/pytest.txt").write_text("11 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text(
        '<testsuite tests="11" failures="0"></testsuite>\n',
        encoding="utf-8",
    )
    _write_json(artifact_dir / "test_results/pytest_summary.json", {"passed": 11, "failed": 0})
    _write_json(artifact_dir / "model_matrix.json", {"schema_version": 1, "rows": rows})
    _write_json(artifact_dir / "full_fleet_matrix.json", {"schema_version": 1, "matrix": rows})
    _write_json(artifact_dir / "target_stream_matrix.json", {"schema_version": 1, "row_count": 44, "rows": []})
    _write_json(artifact_dir / "generic_smoke_matrix.json", {"schema_version": 1, "row_count": 44, "rows": smoke_rows})
    _write_json(
        artifact_dir / "pipeline_backed_matrix.json",
        {"schema_version": 1, "status": "passed", "status_counts": {"passed": 10, "fail_closed": 2}, "rows": []},
    )
    _write_json(
        artifact_dir / "quality_summary.json",
        {
            "schema_version": 1,
            "row_count": 44,
            "in_scope_total": 44,
            "matrix_row_count": 44,
            "status_counts": dict(EXPECTED_STATUS_COUNTS),
            "final_status_counts": {
                "runtime_quality_passed": 32,
                "partial_runtime_passed": 3,
                "negative_control_runtime_passed": 9,
            },
            "generic_smoke_success_count": 32,
            "generic_smoke_failed_count": 0,
            "quality_failed_count": 0,
            "non_rpo_g1_row_count": 42,
            "final_head_ci": final_head_ci,
        },
    )
    _write_json(
        artifact_dir / "pipeline_controls.json",
        {
            "schema_version": 1,
            "controls": {flag: True for flag in REQUIRED_PIPELINE_CONTROL_FLAGS},
        },
    )
    _write_json(
        artifact_dir / "acceptance_ledger.json",
        {
            "schema_version": 1,
            "verdict": "PASS",
            "blocking_count": 0,
            "matrix_row_count": 44,
            "status_counts": dict(EXPECTED_STATUS_COUNTS),
            "final_head_ci": final_head_ci,
            "full_repo_pytest": {
                "status": "not_run",
                "classification": "not_run_scoped_caveat",
                "caveat": "Full repo pytest was not run; this is a scoped full-fleet acceptance audit in a concurrent worktree.",
                "command": None,
            },
        },
    )
    return artifact_dir, source_root


def _matrix_rows() -> list[dict]:
    rows = []
    model_specs = [
        ("roboparty_rpo_local", "roboparty_rpo", "passed", "positive_humanoid"),
        ("unitree_g1_mjcf", "unitree_g1", "passed", "positive_humanoid"),
    ]
    model_specs.extend(
        (f"humanoid_quality_{index:02d}", "humanoid", "passed", "positive_humanoid")
        for index in range(30)
    )
    model_specs.extend(
        (f"partial_quality_{index:02d}", "humanoid", "partial_passed", "partial_humanoid")
        for index in range(3)
    )
    model_specs.extend(
        (f"negative_control_{index:02d}", "non_humanoid", "negative_control_passed", "negative_control")
        for index in range(9)
    )
    for model_id, robot_type, status, expected_capability in model_specs:
        row = {
            "model_id": model_id,
            "robot_type": robot_type,
            "source_status": status,
            "expected_capability": expected_capability,
            "pipeline_control_id": f"controls/{model_id}.json",
            "control_modes": ["disabled", "shadow", "override_experimental"],
            "legacy_default_unchanged": True,
            "shadow_noop_verified": True,
            "override_explicit_only": True,
            "fingerprint_gate_enforced": True,
            "frame_count": 120,
            "target_translation_error_mean": 0.01,
            "target_translation_error_p95": 0.02,
            "target_translation_error_max": 0.03,
            "target_rotation_error_mean": 0.04,
            "target_rotation_error_p95": 0.05,
            "target_rotation_error_max": 0.06,
            "output_nan_count": 0,
            "output_inf_count": 0,
            "joint_limit_violation_count": 0,
            "max_joint_limit_violation": 0.0,
            "runtime_seconds": 1.25,
        }
        if status == "negative_control_passed":
            row.update(
                {
                    "final_step3_1_status": "negative_control_runtime_passed",
                    "runtime_quality_status": "negative_control_not_promoted",
                    "quality_classification": "negative_control_not_promoted",
                    "promoted_to_runtime_quality": False,
                    "quality_evaluated": False,
                    "override_allowed": False,
                    "humanoid_profile_generated": False,
                }
            )
        elif status == "partial_passed":
            row.update(
                {
                    "final_step3_1_status": "partial_runtime_passed",
                    "runtime_quality_status": "partial_runtime_passed",
                    "quality_classification": "partial_runtime_passed",
                    "quality_evaluated": True,
                }
            )
        else:
            row.update(
                {
                    "final_step3_1_status": "runtime_quality_passed",
                    "runtime_quality_status": "runtime_quality_passed",
                    "quality_classification": "runtime_quality_passed",
                    "quality_evaluated": True,
                }
            )
        rows.append(row)
    assert len(rows) == 44
    return rows


def _generic_smoke_rows(matrix_rows: list[dict]) -> list[dict]:
    rows = []
    for row in matrix_rows:
        if row["source_status"] == "negative_control_passed":
            status = "negative_control_rejected"
            category = "negative_control"
            solver_backed = False
        elif row["source_status"] == "partial_passed":
            status = "partial_runtime_passed"
            category = "partial_humanoid_profile"
            solver_backed = False
        else:
            status = "runtime_quality_passed"
            category = "full_humanoid_profile"
            solver_backed = True
        rows.append(
            {
                "model_id": row["model_id"],
                "category": category,
                "status": status,
                "runtime_quality_status": status,
                "quality_classification": status,
                "solver_type": "runtime_solver_projection",
                "solver_backed": solver_backed,
                "quality_pass_allowed": solver_backed,
                "smoke_summary": {
                    "status": status,
                    "mode": "runtime_solver_smoke",
                    "solver_type": "runtime_solver_projection",
                    "solver_backed": solver_backed,
                    "quality_pass_allowed": solver_backed,
                    "metrics": {
                        "normalized_task_residual_p95": 0.01,
                        "task_residual_p95": 0.02,
                        "joint_limit_violation_count": 0,
                        "max_joint_limit_violation": 0.0,
                    },
                    "residuals": {"solver": "runtime_solver_projection"},
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
            "full-fleet-static-and-unit": "success",
            "full-fleet-artifact-audit": "success",
            "lfs-and-snapshot-smoke": "success",
            "pipeline-backed-regression": "success",
            "quality-status-semantics": "success",
        },
    }


def _set_declared_verdict(artifact_dir: Path, source_root: Path, verdict: str) -> None:
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    ledger["verdict"] = verdict
    _write_json(artifact_dir / "acceptance_ledger.json", ledger)
    _write_agent_f_handoff(source_root, verdict)


def _write_agent_f_handoff(source_root: Path, verdict: str) -> None:
    handoff = source_root / "docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md"
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(
        f"""# Step 3.1 Agent F Red Team

verdict = {verdict}

final_head: `{"c" * 40}`
head_sha: `{"c" * 40}`
workflow_run_id: 123456789
conclusion=success
job conclusions: full-fleet-static-and-unit=success, full-fleet-artifact-audit=success, lfs-and-snapshot-smoke=success, pipeline-backed-regression=success, quality-status-semantics=success

full repo pytest classification: `not_run_scoped_caveat`
full repo pytest caveat: Full repo pytest was not run; this fixture covers scoped audit evidence only.

remaining blockers = {0 if verdict == "PASS" else 1}
""",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
