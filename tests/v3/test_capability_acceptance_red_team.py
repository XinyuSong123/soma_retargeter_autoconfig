from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_capability import (
    EXPECTED_BASELINE_COMMIT,
    EXPECTED_MODEL_COUNT,
    EXACT_PROJECTION_THRESHOLDS,
    audit_acceptance_ledger,
    audit_agent_f_handoff,
    audit_baseline_truth,
    audit_before_after_acceptance,
    audit_clean_provenance,
    audit_deterministic_rerun,
    audit_motion_status_policy,
    audit_negative_controls,
    audit_projection_reports,
    audit_required_artifact_files,
    audit_threshold_calibration,
    audit_workflow_config,
)


def _task(
    *,
    normalized_residual: float,
    task: str = "left_hand",
    motion: str = "crossed_body_reach",
    status: str = "converged/with_residual",
    kkt_certificate: dict | None = None,
    capability_certificate: dict | None = None,
) -> dict:
    payload = {
        "active_coordinates": [0, 1],
        "converged": True,
        "desired": [0.2, 0.0, 0.0],
        "projected": [0.0, 0.0, 0.0],
        "residual": normalized_residual,
        "normalized_residual": normalized_residual,
        "normalization_scale": 1.0,
        "status": status,
        "reference": "Chest",
        "target": "LeftHand" if "hand" in task else "Chest",
        "desired_source": "canonical_targets.transforms",
    }
    if kkt_certificate is not None:
        payload["kkt_certificate"] = kkt_certificate
    if capability_certificate is not None:
        payload["capability_certificate"] = capability_certificate
    payload["motion"] = motion
    payload["task"] = task
    return payload


def _report(*, task_payload: dict, status: str = "algorithm_failed", expected: str = "positive") -> dict:
    motion = task_payload.pop("motion")
    task = task_payload.pop("task")
    return {
        "status": status,
        "manifest_entry": {
            "id": "fixture_humanoid",
            "expected_capability": expected,
            "robot_class": "humanoid",
            "required": True,
        },
        "failures": [],
        "canonical_target_validation": {"failures": []},
        "canonical_projection_reports": {
            "motion_order": ["neutral", motion],
            "motions": {
                motion: {
                    "tasks": {
                        task: task_payload,
                    }
                }
            },
        },
    }


def _valid_kkt_certificate() -> dict:
    return {
        "certified": True,
        "stationarity_inf_norm": 0.0,
        "stationarity_tolerance": 1e-6,
        "complementarity_inf_norm": 0.0,
        "complementarity_tolerance": 1e-6,
        "primal_feasible": True,
        "dual_feasible": True,
        "task_gradient_inf_norm": 0.0,
        "prior_gradient_inf_norm": 0.0,
        "prior_cancellation_ratio": 0.0,
        "raw_evidence": {
            "q": [0.0, 0.0],
            "lower_bounds": [-1.0, -1.0],
            "upper_bounds": [1.0, 1.0],
            "task_jacobian": [[0.0, 0.0], [0.0, 0.0]],
            "task_residual": [0.2, 0.0],
            "prior_jacobian": [],
            "prior_residual": [],
        },
        "seed_consistency": {
            "checked": True,
            "status": "consistent",
            "max_task_cost_delta": 0.0,
            "tolerance": 1e-7,
            "seed_results": [
                {"accepted": True, "task_residual_norm": 0.2, "task_space_endpoint": [0.0, 0.0]},
                {"accepted": True, "task_residual_norm": 0.2, "task_space_endpoint": [0.0, 0.0]},
            ],
        },
    }


def _valid_capability_certificate_v2() -> dict:
    raw = _valid_kkt_certificate()["raw_evidence"]
    seed_results = [
        {
            "seed_index": 0,
            "accepted": True,
            "normalized_residual": 0.2,
            "final_task_vector": [0.0, 0.0, 0.0],
            "certificate_class": "capability_limited_rank",
        },
        {
            "seed_index": 1,
            "accepted": True,
            "normalized_residual": 0.2,
            "final_task_vector": [0.0, 0.0, 0.0],
            "certificate_class": "capability_limited_rank",
        },
    ]
    return {
        "schema_version": 2,
        "certificate_class": "capability_limited_rank",
        "passed": True,
        "gates": {
            "projected_gradient_kkt": True,
            "seed_consensus": True,
            "residual_explained": True,
            "continuation": True,
            "joint_limits": True,
            "numerical": True,
        },
        "kkt": {
            "satisfied": True,
            "stationarity_inf_norm": 0.0,
            "stationarity_tolerance": 1e-6,
            "complementarity_inf_norm": 0.0,
            "complementarity_tolerance": 1e-6,
            "complementarity_passed": True,
            "primal_feasible": True,
            "dual_feasible": True,
            "task_gradient_inf_norm": 0.0,
            "prior_gradient_inf_norm": 0.0,
            "prior_cancellation_ratio": 0.0,
            "raw": {"raw_evidence": raw},
        },
        "seed_consensus": {
            "checked": True,
            "passed": True,
            "task_space_tolerance": 1e-7,
            "seed_results": seed_results,
        },
        "audit_evidence": {
            "q_active": raw["q"],
            "lower_bounds": raw["lower_bounds"],
            "upper_bounds": raw["upper_bounds"],
            "relevant_task_jacobian": raw["task_jacobian"],
            "normalized_residual_vector": raw["task_residual"],
            "prior_gradient": [0.0, 0.0],
            "normalization_scale": 1.0,
        },
    }


def test_threshold_audit_locks_exact_global_values() -> None:
    payload = {
        "projection_quality": {
            "neutral_position_abs_m": EXACT_PROJECTION_THRESHOLDS["neutral"],
            "foot_rho_p": EXACT_PROJECTION_THRESHOLDS["foot"],
            "hand_rho_p": 0.13,
            "torso_rho_r": EXACT_PROJECTION_THRESHOLDS["torso"],
        }
    }

    failures = audit_threshold_calibration(payload, check_live_function=False)

    assert any("hand_rho_p" in failure and "0.12" in failure for failure in failures)


def test_over_threshold_residual_requires_schema_v2_capability_certificate() -> None:
    report = _report(task_payload=_task(normalized_residual=0.2))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("missing schema-v2 capability_certificate" in failure for failure in failures)


def test_valid_schema_v2_capability_certificate_is_accepted_for_over_threshold_residual() -> None:
    report = _report(
        task_payload=_task(normalized_residual=0.2, capability_certificate=_valid_capability_certificate_v2())
    )

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert failures == []


def test_schema_v2_capability_certificate_recomputes_raw_bounds() -> None:
    certificate = _valid_capability_certificate_v2()
    certificate["kkt"]["raw"]["raw_evidence"]["q"] = [2.0, 0.0]
    report = _report(task_payload=_task(normalized_residual=0.2, capability_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("primal_feasible" in failure and "independent recompute" in failure for failure in failures)


def test_kkt_certificate_rejects_prior_gradient_cancellation_and_seed_inconsistency() -> None:
    certificate = _valid_kkt_certificate()
    certificate["prior_cancellation_ratio"] = 0.75
    certificate["seed_consistency"] = {"checked": True, "status": "inconsistent"}
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("prior_cancellation_ratio" in failure for failure in failures)
    assert any("seed_consistency" in failure for failure in failures)


def test_valid_over_threshold_kkt_certificate_is_accepted_for_algorithm_failure() -> None:
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=_valid_kkt_certificate()))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert failures == []


def test_kkt_certificate_recomputes_primal_feasibility_from_raw_bounds() -> None:
    certificate = _valid_kkt_certificate()
    certificate["raw_evidence"]["q"] = [2.0, 0.0]
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("primal_feasible" in failure and "independent recompute" in failure for failure in failures)


def test_kkt_certificate_rejects_hardcoded_prior_gradient_zero() -> None:
    certificate = _valid_kkt_certificate()
    certificate["raw_evidence"]["prior_jacobian"] = [[1.0, 0.0]]
    certificate["raw_evidence"]["prior_residual"] = [0.5]
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("prior_gradient_inf_norm" in failure and "recompute" in failure for failure in failures)


def test_kkt_certificate_rejects_synthetic_one_e_minus_15_task_gradient() -> None:
    certificate = _valid_kkt_certificate()
    certificate["task_gradient_inf_norm"] = 1e-15
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("task_gradient_inf_norm" in failure and "recompute" in failure for failure in failures)


def test_kkt_certificate_requires_raw_j_e_q_bounds() -> None:
    certificate = _valid_kkt_certificate()
    certificate.pop("raw_evidence")
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("raw_evidence J/e/q/bounds" in failure for failure in failures)


def test_kkt_certificate_rejects_bool_mismatch_after_independent_recompute() -> None:
    certificate = _valid_kkt_certificate()
    certificate["raw_evidence"]["task_jacobian"] = [[1.0, 0.0]]
    certificate["raw_evidence"]["task_residual"] = [1.0]
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("certified=True but independent recompute is False" in failure for failure in failures)
    assert any("stationarity_inf_norm" in failure and "recompute" in failure for failure in failures)


def test_seed_consensus_rejects_same_residual_different_task_space_endpoint() -> None:
    certificate = _valid_kkt_certificate()
    certificate["seed_consistency"]["seed_results"] = [
        {"accepted": True, "task_residual_norm": 0.2, "task_space_endpoint": [0.0, 0.0]},
        {"accepted": True, "task_residual_norm": 0.2, "task_space_endpoint": [1.0, 0.0]},
    ]
    report = _report(task_payload=_task(normalized_residual=0.2, kkt_certificate=certificate))

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("task-space endpoints diverge" in failure for failure in failures)


def test_continuation_history_must_reach_alpha_one() -> None:
    task = _task(normalized_residual=0.2, kkt_certificate=_valid_kkt_certificate())
    task["continuation_history"] = [{"accepted": True, "alpha_end": 0.5}]
    report = _report(task_payload=task)

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("continuation final alpha < 1" in failure for failure in failures)


def test_rank_zero_nonzero_demand_must_be_preserved_as_unreachable() -> None:
    task = _task(normalized_residual=0.0, status="rank_zero")
    task["desired"] = [0.3, 0.0, 0.0]
    task["projected"] = [0.0, 0.0, 0.0]
    report = _report(task_payload=task)

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("rank-zero nonzero demand" in failure for failure in failures)


def test_target_geometry_failure_cannot_be_packaged_as_capability_pass() -> None:
    report = _report(task_payload=_task(normalized_residual=0.0), status="passed")
    report["canonical_target_validation"]["failures"] = ["source torso geometry is inconsistent"]

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("target geometry failure packaged as pass" in failure for failure in failures)


def test_stress_motion_does_not_mask_ordinary_failure() -> None:
    report = _report(task_payload=_task(normalized_residual=0.0), status="passed")
    report["canonical_projection_reports"]["motions"]["extreme_but_valid_joint_limit_stress"] = {
        "tasks": {"left_hand": _task(normalized_residual=0.9, motion="stress", task="left_hand")}
    }
    report["canonical_projection_reports"]["motions"]["crossed_body_reach"] = {
        "tasks": {"left_hand": _task(normalized_residual=0.21, motion="ordinary", task="left_hand")}
    }

    failures = audit_projection_reports({"fixture_humanoid": report})

    assert any("ordinary residual over threshold packaged as pass" in failure for failure in failures)
    assert not any(
        "extreme_but_valid_joint_limit_stress.left_hand" in failure
        and "missing schema-v2 capability_certificate" in failure
        for failure in failures
    )


def test_stress_only_limited_must_not_change_profile_status() -> None:
    report = _report(task_payload=_task(normalized_residual=0.0), status="capability_limited_passed")
    report["canonical_projection_reports"]["motions"] = {
        "neutral": {"tasks": {"left_hand": _task(normalized_residual=0.0, motion="neutral", task="left_hand")}},
        "extreme_but_valid_joint_limit_stress": {
            "tasks": {"left_hand": _task(normalized_residual=0.9, motion="stress", task="left_hand")}
        },
    }

    failures = audit_motion_status_policy({"fixture_humanoid": report})

    assert any("stress-only limited evidence changed status" in failure for failure in failures)


def test_negative_controls_are_not_humanoid_profile_passes() -> None:
    summary = {"negative_control_passed": 1, "profile_passed": 1}
    reports = {
        "bolt_urdf": {
            "status": "passed",
            "manifest_entry": {
                "id": "bolt_urdf",
                "expected_capability": "negative_control",
                "robot_class": "biped_no_arms",
            },
            "capability_status": "full_humanoid_ready",
            "canonical_projection_reports": {"motions": {}},
        }
    }

    failures = audit_negative_controls(summary, reports)

    assert any("negative control status" in failure for failure in failures)
    assert any("humanoid capability payload" in failure for failure in failures)
    assert any("profile_passed" in failure for failure in failures)


def test_deterministic_rerun_must_compare_all_44_models() -> None:
    deterministic = {
        "status": "passed",
        "totals": {
            "model_count": EXPECTED_MODEL_COUNT,
            "compared_count": EXPECTED_MODEL_COUNT - 1,
            "matched_count": EXPECTED_MODEL_COUNT - 1,
            "mismatch_count": 0,
            "skipped_non_pass_count": 1,
        },
        "models": {
            f"model_{index}": {"compared": index < EXPECTED_MODEL_COUNT - 1}
            for index in range(EXPECTED_MODEL_COUNT)
        },
    }

    failures = audit_deterministic_rerun(deterministic, expected_model_count=EXPECTED_MODEL_COUNT)

    assert any("compared_count" in failure for failure in failures)
    assert any("uncompared deterministic models" in failure for failure in failures)


def test_deterministic_rerun_must_compare_required_evidence_surfaces() -> None:
    deterministic = {
        "status": "passed",
        "totals": {
            "model_count": 1,
            "compared_count": 1,
            "matched_count": 1,
            "mismatch_count": 0,
            "skipped_non_pass_count": 0,
            "source_unavailable_count": 0,
            "rerun_failed_count": 0,
        },
        "models": {"fixture": {"compared": True, "comparisons": {"status": {"matched": True}}}},
    }

    failures = audit_deterministic_rerun(deterministic, expected_model_count=1)

    assert any("deterministic comparison missing canonical_projection_residuals" in failure for failure in failures)


def test_baseline_truth_rejects_fabricated_before_status_and_pass_downgrade() -> None:
    before_after = {
        "baseline_commit": EXPECTED_BASELINE_COMMIT,
        "baseline_artifact_root": "artifacts/retargeting_v3_step2_assets44",
        "row_count": 2,
        "rows": {
            "full_model": {"before_status": "partial_passed", "after_status": "capability_limited_passed"},
            "downgraded_model": {"before_status": "passed", "after_status": "algorithm_failed"},
        },
    }
    baseline_summary = {
        "baseline_commit": EXPECTED_BASELINE_COMMIT,
        "row_count": 2,
        "rows": {
            "full_model": {"status": "passed"},
            "downgraded_model": {"status": "passed"},
        },
    }

    failures = audit_baseline_truth(before_after, baseline_summary)

    assert any("fabricated before status" in failure for failure in failures)
    assert any("baseline passed downgraded" in failure for failure in failures)


def test_before_after_acceptance_rejects_failed_transition_and_final_counts() -> None:
    before_after = {
        "transition_validation": {"status": "failed"},
        "final_count_validation": {"status": "failed"},
        "allowed_transitions": ["passed->passed"],
        "status_transition_counts": {"passed->capability_limited_passed": 1},
    }
    summary = {"before_after": {"status_transition_counts": {"passed->passed": 1}}}

    failures = audit_before_after_acceptance(before_after, summary)

    assert any("transition_validation.status must be passed" in failure for failure in failures)
    assert any("final_count_validation.status must be passed" in failure for failure in failures)
    assert any("illegal transitions" in failure for failure in failures)
    assert any("summary.before_after.status_transition_counts differs" in failure for failure in failures)


def test_required_artifact_files_reject_missing_pytest_junit_lfs_state(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "per_robot").mkdir()
    (artifact_dir / "per_task").mkdir()
    (artifact_dir / "failures").mkdir()

    failures = audit_required_artifact_files(artifact_dir)

    assert any("lfs_state.json" in failure for failure in failures)
    assert any("test_results/pytest.txt" in failure for failure in failures)
    assert any("test_results/junit.xml" in failure for failure in failures)


def test_acceptance_ledger_rejects_stale_step22_audit() -> None:
    ledger = {
        "status": "passed",
        "command": "python scripts/audit_retargeting_v3_assets44.py --artifact-dir artifacts/retargeting_v3_step2_assets44",
    }

    failures = audit_acceptance_ledger(ledger)

    assert any("stale" in failure for failure in failures)
    assert any("assets44" in failure for failure in failures)


def test_workflow_config_rejects_wrong_branch_and_artifact_root(tmp_path: Path) -> None:
    workflow = tmp_path / "retargeting_v3_capability.yml"
    workflow.write_text(
        """
name: Retargeting V3 Capability Acceptance
on:
  push:
    branches:
      - retargeting-v3-step2-capability
env:
  RETARGETING_V3_CAPABILITY_ARTIFACTS: artifacts/retargeting_v3_step2_assets44
jobs:
  capability-red-team:
    steps:
      - run: python scripts/audit_retargeting_v3_capability.py --artifact-dir "$RETARGETING_V3_ARTIFACTS"
""".lstrip()
    )

    failures = audit_workflow_config(workflow)

    assert any("push branches" in failure for failure in failures)
    assert any("assets44" in failure for failure in failures)
    assert any("required job" in failure for failure in failures)


def test_provenance_rejects_unresolvable_source_commit(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "environment.json").write_text(
        json.dumps(
            {
                "git_status_short": "",
                "source_code_commit": "deadbeef",
                "artifact_commit": "feedface",
                "source_code_commit_remote_resolvable": True,
                "source_code_commit_is_artifact_commit_ancestor": True,
                "source_worktree_clean_before_run": True,
                "source_worktree_clean_after_run": True,
                "core_diff_after_source_commit": [],
            }
        )
    )
    (artifact_dir / "source_inventory.json").write_text("{}")

    def fake_git(args: list[str]):
        if args[:2] == ["cat-file", "-e"] and "deadbeef" in args[2]:
            return _completed(args, returncode=1)
        return _completed(args)

    failures = audit_clean_provenance(artifact_dir, repo_root=tmp_path, git_runner=fake_git)

    assert any("source_code_commit is not locally resolvable" in failure for failure in failures)


def test_provenance_rejects_source_to_artifact_core_drift(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "environment.json").write_text(
        json.dumps(
            {
                "git_status_short": "",
                "source_code_commit": "a" * 40,
                "artifact_commit": "b" * 40,
                "source_code_commit_remote_resolvable": True,
                "source_code_commit_is_artifact_commit_ancestor": True,
                "source_worktree_clean_before_run": True,
                "source_worktree_clean_after_run": True,
                "core_diff_after_source_commit": [],
            }
        )
    )
    (artifact_dir / "source_inventory.json").write_text("{}")

    def fake_git(args: list[str]):
        if args[:2] == ["diff", "--name-only"]:
            return _completed(args, stdout="soma_retargeter/robotics/v3/profile.py\n")
        return _completed(args)

    failures = audit_clean_provenance(artifact_dir, repo_root=tmp_path, git_runner=fake_git)

    assert any("core drift after source commit" in failure for failure in failures)


def test_agent_f_handoff_rejects_stale_fail_text(tmp_path: Path, monkeypatch) -> None:
    handoff = tmp_path / "capability_agent_f_red_team.md"
    handoff.write_text(
        "\n".join(
            [
                "# Capability Agent F Red Team Handoff",
                "final_head: current",
                "Live capability audit: **FAIL**",
                "Do not accept Step 2.3 capability yet.",
            ]
        )
    )

    def fake_run(*args, **kwargs):
        return _completed(["rev-parse"], stdout="current\n")

    monkeypatch.setattr("scripts.audit_retargeting_v3_capability.subprocess.run", fake_run)

    failures = audit_agent_f_handoff(handoff, tmp_path)

    assert any("stale FAIL" in failure for failure in failures)


def test_agent_f_blocked_handoff_does_not_require_self_referential_head(tmp_path: Path, monkeypatch) -> None:
    handoff = tmp_path / "capability_agent_f_red_team.md"
    handoff.write_text(
        "\n".join(
            [
                "# Capability Agent F Red Team Handoff",
                "verdict = BLOCKED",
                "final_head: pending-source-commit",
                "remaining blockers > 0",
            ]
        )
    )

    def fake_run(*args, **kwargs):
        return _completed(["rev-parse"], stdout="current\n")

    monkeypatch.setattr("scripts.audit_retargeting_v3_capability.subprocess.run", fake_run)

    failures = audit_agent_f_handoff(handoff, tmp_path)

    assert not any("final_head is stale" in failure for failure in failures)


def test_agent_f_pass_handoff_accepts_concrete_metadata_without_current_head_self_reference(
    tmp_path: Path, monkeypatch
) -> None:
    handoff = tmp_path / "capability_agent_f_red_team.md"
    handoff.write_text(
        "\n".join(
            [
                "# Capability Agent F Red Team Handoff",
                "verdict = PASS",
                "- final_head: `" + "a" * 40 + "`",
                "- source_code_commit: `" + "b" * 40 + "`",
                "- artifact_commit: `" + "c" * 40 + "`",
                "- workflow_run_id: 123456789",
                "- workflow name: Retargeting V3 Capability Acceptance",
                "- head SHA: `" + "a" * 40 + "`",
                "- conclusion=success",
                "- job conclusions: success",
                "- pytest summary: 362 passed, 10 skipped",
                "- live audit command: PYTHONPATH=. python scripts/audit_retargeting_v3_capability.py ...",
                "- live audit PASS",
                "- LFS fsck PASS",
                "- remaining blockers = 0",
            ]
        )
    )

    def fake_run(*args, **kwargs):
        return _completed(["rev-parse"], stdout=("d" * 40) + "\n")

    monkeypatch.setattr("scripts.audit_retargeting_v3_capability.subprocess.run", fake_run)

    assert audit_agent_f_handoff(handoff, tmp_path) == []


def test_agent_f_pass_handoff_rejects_placeholder_metadata(tmp_path: Path) -> None:
    handoff = tmp_path / "capability_agent_f_red_team.md"
    handoff.write_text(
        "\n".join(
            [
                "# Capability Agent F Red Team Handoff",
                "verdict = PASS",
                "- final_head: pending-source-commit",
                "- source_code_commit: not final integrated evidence",
                "- artifact_commit: not final integrated evidence",
                "- workflow_run_id: not available",
                "- pytest summary: pending",
                "- live audit command: pending",
                "- live audit PASS",
                "- LFS fsck PASS",
                "- remaining blockers = 0",
            ]
        )
    )

    failures = audit_agent_f_handoff(handoff, tmp_path)

    assert any("final_head must be a concrete" in failure for failure in failures)
    assert any("source_code_commit must be a concrete" in failure for failure in failures)
    assert any("artifact_commit must be a concrete" in failure for failure in failures)
    assert any("workflow_run_id must be concrete" in failure for failure in failures)
    assert any("stale blocker marker" in failure for failure in failures)


def _completed(args: list[str], *, returncode: int = 0, stdout: str = ""):
    import subprocess

    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr="")


def test_projection_audit_accepts_rank_zero_zero_demand_evidence() -> None:
    task = _task(normalized_residual=0.0, status="rank_zero")
    task["desired"] = [0.0, 0.0, 0.0]
    task["projected"] = [0.0, 0.0, 0.0]
    task["demand_residual"] = 0.0
    task["unreachable_demand"] = False
    task["rank_zero_reason"] = "no_active_coordinates_zero_demand"
    report = _report(task_payload=task)

    assert audit_projection_reports({"fixture_humanoid": report}) == []


def test_live_assets44_deterministic_fixture_shape_is_documented() -> None:
    path = Path("artifacts/retargeting_v3_step2_assets44/deterministic_rerun.json")
    deterministic = json.loads(path.read_text())

    assert deterministic["totals"]["model_count"] == EXPECTED_MODEL_COUNT
