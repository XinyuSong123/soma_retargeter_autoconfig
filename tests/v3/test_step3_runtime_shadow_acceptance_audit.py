from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_step3_runtime_shadow import (
    GOAL_FALSE_POSITIVE_GATES,
    GOAL_FALSE_POSITIVE_TRACEABILITY,
    REQUIRED_CI_JOBS,
    REQUIRED_RUNTIME_CLIPS,
    REQUIRED_RUNTIME_ROBOTS,
    main as audit_main,
    run_audit,
)


SEMANTICS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")


def test_step3_red_team_gate_coverage_is_complete(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "PASS"
    assert result.blocking_count == 0
    assert set(GOAL_FALSE_POSITIVE_GATES) <= set(result.gate_counts)
    assert set(REQUIRED_CI_JOBS) == {
        "runtime-shadow-unit-tests",
        "runtime-shadow-lfs-smoke",
        "runtime-shadow-artifact-audit",
    }
    for false_positive, gates in GOAL_FALSE_POSITIVE_TRACEABILITY.items():
        assert gates, false_positive
        assert all(gate in result.gate_counts for gate in gates)


def test_audit_blocks_default_output_changed(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    shadow = _read_json(artifact_dir / "shadow_summary.json")
    shadow["matrix"][0]["output_equal_to_disabled_baseline"] = False
    shadow["matrix"][0]["output_diff_max"] = 1e-4
    _write_json(artifact_dir / "shadow_summary.json", shadow)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["default_output_changed"] >= 1


def test_audit_blocks_shadow_ik_input_mutation(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    shadow = _read_json(artifact_dir / "shadow_summary.json")
    shadow["matrix"][1]["ik_inputs_equal_to_disabled"] = False
    _write_json(artifact_dir / "shadow_summary.json", shadow)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["shadow_mutates_ik_inputs"] == 1


def test_audit_blocks_override_enabled_without_explicit_config(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    override = _read_json(artifact_dir / "override_smoke_summary.json")
    override["smoke_matrix"][0]["config_explicit"] = False
    _write_json(artifact_dir / "override_smoke_summary.json", override)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["override_without_explicit_config"] == 1


def test_audit_blocks_fingerprint_mismatch_silently_accepted(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    profile_resolution = _read_json(artifact_dir / "profile_resolution.json")
    profile_resolution["profiles"][1].update(
        {
            "fingerprint_match": False,
            "source_hash_match": False,
            "resolution_status": "passed",
            "warnings": [],
            "errors": [],
        }
    )
    _write_json(artifact_dir / "profile_resolution.json", profile_resolution)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["fingerprint_mismatch_silent"] == 1


def test_audit_blocks_partial_or_negative_profile_override(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    override = _read_json(artifact_dir / "override_smoke_summary.json")
    override["smoke_matrix"][0]["profile_status"] = "partial_passed"
    override["smoke_matrix"][1]["profile_status"] = "negative_control_passed"
    _write_json(artifact_dir / "override_smoke_summary.json", override)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["partial_or_negative_profile_override"] == 2


def test_audit_blocks_lfs_pointer_profile_or_snapshot(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    pointer_profile = source_root / "artifacts/retargeting_v3_step2_capability/per_robot/roboparty_rpo_local.json"
    pointer_profile.write_text(_lfs_pointer_text(), encoding="utf-8")
    snapshot = source_root / "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(_lfs_pointer_text(), encoding="utf-8")
    profile_resolution = _read_json(artifact_dir / "profile_resolution.json")
    profile_resolution["profiles"][1]["runtime_mjcf_path"] = "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml"
    _write_json(artifact_dir / "profile_resolution.json", profile_resolution)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["lfs_pointer_profile_or_snapshot"] == 2


def test_audit_blocks_missing_diagnostics(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    (artifact_dir / "shadow_summary.json").unlink()
    for path in artifact_dir.glob("per_clip/*/*/target_deltas.json"):
        path.unlink()
        break

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_diagnostics"] >= 2


def test_audit_blocks_nan_and_inf_in_diagnostics(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    target_deltas = next(artifact_dir.glob("per_clip/*/*/target_deltas.json"))
    payload = _read_json(target_deltas)
    payload["per_semantic"]["LeftHand"]["translation_delta_max"] = float("nan")
    payload["per_semantic"]["RightHand"]["rotation_delta_max"] = float("inf")
    _write_json(target_deltas, payload)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["diagnostics_nan_inf"] == 2


def test_audit_blocks_local_absolute_path_leakage_and_sanitizes_report(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    leaked_path = "/" + "mnt/ssd1/song/Desktop/soma_retargeter_autoconfig/private_runner.py"
    (artifact_dir / "commands.txt").write_text(
        f"python {leaked_path}\n",
        encoding="utf-8",
    )

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)
    serialized = json.dumps(result.to_json())

    assert result.status == "BLOCKED"
    assert result.gate_counts["local_absolute_path_leakage"] == 1
    assert "/" + "mnt/ssd1/song" not in serialized
    assert "${LOCAL_SOURCE_PATH}/private_runner.py" in serialized


def test_audit_blocks_step2_artifacts_mutated(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    environment = _read_json(artifact_dir / "environment.json")
    environment["step2_artifact_status_short"] = " M artifacts/retargeting_v3_step2_capability/summary.json"
    _write_json(artifact_dir / "environment.json", environment)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["step2_artifacts_mutated"] == 1


def test_audit_blocks_missing_ci_or_missing_required_ci_jobs(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    workflow = source_root / ".github/workflows/retargeting_v3_step3_runtime_shadow.yml"
    workflow.write_text("name: incomplete\njobs:\n  runtime-shadow-unit-tests:\n    runs-on: ubuntu-latest\n", encoding="utf-8")

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["missing_ci"] == 2


def test_audit_blocks_stale_agent_f_pass_fail_mismatch(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    shadow = _read_json(artifact_dir / "shadow_summary.json")
    shadow["matrix"][0]["output_equal_to_disabled_baseline"] = False
    _write_json(artifact_dir / "shadow_summary.json", shadow)
    handoff = source_root / "docs/retargeting_v3/subagents/step3_agent_f_red_team.md"
    handoff.write_text("# Step 3 Agent F Red Team\n\nverdict = PASS\nremaining blockers = 0\n", encoding="utf-8")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    ledger["verdict"] = "PASS"
    _write_json(artifact_dir / "acceptance_ledger.json", ledger)

    result = run_audit(artifact_dir=artifact_dir, source_root=source_root, skip_git_checks=True)

    assert result.status == "BLOCKED"
    assert result.gate_counts["default_output_changed"] == 1
    assert result.gate_counts["stale_agent_f_verdict"] >= 1


def test_audit_cli_writes_report_and_returns_nonzero_for_blockers(tmp_path: Path) -> None:
    artifact_dir, source_root = _write_passing_evidence(tmp_path)
    (artifact_dir / "profile_resolution.json").unlink()
    output_json = artifact_dir / "acceptance_ledger.json"

    exit_code = audit_main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--source-root",
            str(source_root),
            "--skip-git-checks",
            "--output-json",
            str(output_json),
        ]
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "BLOCKED"
    assert payload["gate_counts"]["missing_diagnostics"] >= 1


def _write_passing_evidence(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "repo"
    artifact_dir = source_root / "artifacts/retargeting_v3_step3_runtime_shadow"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "test_results").mkdir()
    _write_workflow(source_root)
    _write_agent_f_handoff(source_root, "PASS")
    _write_acceptance_doc(source_root, "PASS")
    _write_step2_profile(source_root, "roboparty_rpo_local", "passed")
    _write_step2_profile(source_root, "unitree_g1_mjcf", "passed")

    _write_json(
        artifact_dir / "environment.json",
        {
            "schema_version": 1,
            "source_code_commit": "fixture-source",
            "artifact_commit": "fixture-artifact",
            "git_status_short": "",
            "step2_artifact_status_short": "",
            "lfs_fsck": "OK",
        },
    )
    (artifact_dir / "commands.txt").write_text(
        "PYTHONPATH=. python scripts/audit_retargeting_v3_step3_runtime_shadow.py\n",
        encoding="utf-8",
    )
    _write_json(artifact_dir / "profile_resolution.json", _profile_resolution_payload())
    _write_json(artifact_dir / "shadow_summary.json", _shadow_summary_payload())
    _write_json(artifact_dir / "override_smoke_summary.json", _override_summary_payload())
    _write_json(
        artifact_dir / "test_results/pytest_summary.json",
        {"passed": 14, "failed": 0, "skipped": 0, "summary": "14 passed"},
    )
    (artifact_dir / "test_results/pytest.txt").write_text("14 passed\n", encoding="utf-8")
    (artifact_dir / "test_results/junit.xml").write_text(
        '<testsuite tests="14" failures="0"></testsuite>\n',
        encoding="utf-8",
    )
    for robot in REQUIRED_RUNTIME_ROBOTS:
        for clip in REQUIRED_RUNTIME_CLIPS:
            clip_dir = artifact_dir / "per_clip" / robot / clip
            clip_dir.mkdir(parents=True, exist_ok=True)
            _write_json(clip_dir / "target_deltas.json", _target_deltas_payload(robot, clip))
            _write_json(clip_dir / "pipeline_summary.json", _pipeline_summary_payload("shadow"))
    _write_json(artifact_dir / "acceptance_ledger.json", _acceptance_ledger_payload())
    return artifact_dir, source_root


def _write_workflow(source_root: Path) -> None:
    workflow = source_root / ".github/workflows/retargeting_v3_step3_runtime_shadow.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        """name: Retargeting V3 Step 3 Runtime Shadow
on:
  pull_request:
  push:
  workflow_dispatch:
jobs:
  runtime-shadow-unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: python -m pytest -q tests/v3/test_step3_runtime_shadow_acceptance_*.py
  runtime-shadow-lfs-smoke:
    runs-on: ubuntu-latest
    steps:
      - run: git lfs fsck
  runtime-shadow-artifact-audit:
    runs-on: ubuntu-latest
    steps:
      - run: python scripts/audit_retargeting_v3_step3_runtime_shadow.py
""",
        encoding="utf-8",
    )


def _write_agent_f_handoff(source_root: Path, verdict: str) -> None:
    handoff = source_root / "docs/retargeting_v3/subagents/step3_agent_f_red_team.md"
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(
        f"""# Step 3 Agent F Red Team

verdict = {verdict}

final_head: `fixture-head`
source_code_commit: `fixture-source`
artifact_commit: `fixture-artifact`
workflow_run_id: `fixture-workflow`
pytest summary: `14 passed`
remaining blockers = 0
""",
        encoding="utf-8",
    )


def _write_acceptance_doc(source_root: Path, verdict: str) -> None:
    doc = source_root / "docs/retargeting_v3/STEP3_RUNTIME_SHADOW_ACCEPTANCE.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        f"""# Step 3 Runtime Shadow Acceptance

Status: {verdict}

Fixture evidence records passing runtime-shadow red-team gates.
""",
        encoding="utf-8",
    )


def _write_step2_profile(source_root: Path, model_id: str, status: str) -> None:
    profile = source_root / "artifacts/retargeting_v3_step2_capability/per_robot" / f"{model_id}.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        profile,
        {
            "schema_version": 3,
            "status": status,
            "semantic_sites": {name: {"body_name": name, "source": "fixture"} for name in SEMANTICS},
            "rest_calibration": {"robot_neutral_site_transforms": {name: _identity() for name in SEMANTICS}},
            "canonical_projection_reports": {"motion_order": ["neutral"]},
            "task_certificate_summary": {"status": "available"},
        },
    )


def _profile_resolution_payload() -> dict:
    return {
        "schema_version": 1,
        "profiles": [
            {
                "robot_type": "roboparty_rpo",
                "profile_model_id": "roboparty_rpo_local",
                "profile_status": "passed",
                "profile_artifact_path": "artifacts/retargeting_v3_step2_capability/per_robot/roboparty_rpo_local.json",
                "runtime_mjcf_path": "soma_retargeter/configs/roboparty_rpo/model.xml",
                "runtime_fingerprint": "rpo-fingerprint",
                "profile_fingerprint": "rpo-fingerprint",
                "fingerprint_match": True,
                "source_hash_match": True,
                "strict_match_required": True,
                "resolution_status": "passed",
                "warnings": [],
                "errors": [],
            },
            {
                "robot_type": "unitree_g1",
                "profile_model_id": "unitree_g1_mjcf",
                "profile_status": "passed",
                "profile_artifact_path": "artifacts/retargeting_v3_step2_capability/per_robot/unitree_g1_mjcf.json",
                "runtime_mjcf_path": "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml",
                "runtime_fingerprint": "runtime-g1",
                "profile_fingerprint": "profile-g1",
                "fingerprint_match": False,
                "source_hash_match": False,
                "strict_match_required": False,
                "resolution_status": "shadow_fingerprint_skip",
                "warnings": ["fingerprint_mismatch"],
                "errors": [],
            },
        ],
    }


def _shadow_summary_payload() -> dict:
    return {
        "schema_version": 1,
        "matrix": [
            {
                "robot_type": robot,
                "clip_name": clip,
                "mode": "shadow",
                "frame_count": 120,
                "v3_targets_finite": True,
                "ik_inputs_equal_to_disabled": True,
                "output_equal_to_disabled_baseline": True,
                "output_diff_max": 0.0,
                "diagnostics_written": True,
                "diagnostics_deterministic": True,
                "fingerprint_status": "matched" if robot == "roboparty_rpo" else "shadow_fingerprint_skip",
            }
            for robot in REQUIRED_RUNTIME_ROBOTS
            for clip in REQUIRED_RUNTIME_CLIPS
        ],
    }


def _override_summary_payload() -> dict:
    return {
        "schema_version": 1,
        "smoke_matrix": [
            {
                "robot_type": "roboparty_rpo",
                "clip_name": clip,
                "mode": "override_experimental",
                "frame_count": 120,
                "output_finite": True,
                "joint_limit_violation_count": 0,
                "max_joint_limit_violation": 0.0,
                "diagnostics_written": True,
                "experimental_label": True,
                "config_explicit": True,
                "profile_model_id": "roboparty_rpo_local",
                "profile_status": "passed",
                "resolution_status": "passed",
            }
            for clip in REQUIRED_RUNTIME_CLIPS
        ]
        + [
            {
                "robot_type": "unitree_g1",
                "clip_name": clip,
                "mode": "override_experimental",
                "config_explicit": True,
                "profile_model_id": "unitree_g1_mjcf",
                "profile_status": "passed",
                "resolution_status": "fail_closed",
                "reason": "fingerprint_mismatch",
            }
            for clip in REQUIRED_RUNTIME_CLIPS
        ],
    }


def _target_deltas_payload(robot: str, clip: str) -> dict:
    return {
        "schema_version": 1,
        "robot_type": robot,
        "mode": "shadow",
        "clip_name": clip,
        "frame_count": 120,
        "semantic_names": list(SEMANTICS),
        "legacy_target_available": True,
        "v3_target_available": True,
        "per_semantic": {
            name: {
                "translation_delta_mean": 0.01,
                "translation_delta_max": 0.02,
                "translation_delta_p95": 0.015,
                "rotation_delta_mean": 0.01,
                "rotation_delta_max": 0.02,
                "rotation_delta_p95": 0.015,
                "finite_count": 120,
                "nan_count": 0,
                "skipped_reason": None,
            }
            for name in SEMANTICS
        },
        "root_policy": {"horizontal_scale": 1.0, "support_height_policy": "fixture"},
        "capability_policy": "exact",
    }


def _pipeline_summary_payload(mode: str) -> dict:
    return {
        "schema_version": 1,
        "mode": mode,
        "output_frame_count": 120,
        "joint_coord_count": 29,
        "nan_count": 0,
        "inf_count": 0,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "output_equal_to_disabled_baseline": True,
        "output_diff_max": 0.0,
        "runtime_seconds": 1.0,
    }


def _acceptance_ledger_payload() -> dict:
    return {
        "schema_version": 1,
        "verdict": "PASS",
        "status": "PASS",
        "blocking_count": 0,
        "final_head": "fixture-head",
        "source_code_commit": "fixture-source",
        "artifact_commit": "fixture-artifact",
        "workflow_run_id": "fixture-workflow",
        "pytest_summary": "14 passed",
        "smoke_matrix_summary": "RPO shadow/override passed; G1 shadow skipped on fingerprint mismatch and override fail-closed",
        "shadow_equality_result": "passed",
        "rpo_override_result": "passed",
        "g1_override_result": "fail_closed:fingerprint_mismatch",
        "remaining_blockers": [],
        "gates": {gate: "PASS" for gate in GOAL_FALSE_POSITIVE_GATES},
    }


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _lfs_pointer_text() -> str:
    return "\n".join(
        [
            "version https://git-lfs.github.com/spec/v1",
            "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "size 123",
        ]
    ) + "\n"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
