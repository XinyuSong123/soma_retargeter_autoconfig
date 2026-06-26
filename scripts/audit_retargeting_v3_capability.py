#!/usr/bin/env python3
"""Red-team audit for Step 2.3 capability acceptance artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


EXPECTED_MODEL_COUNT = 44
EXPECTED_BASELINE_COMMIT = "5ad5a001c445c525d4c8bbaf6339dec5c5c2c719"
EXPECTED_CAPABILITY_BRANCH = "retargeting-v3-step2-capability-acceptance-hardening"
CAPABILITY_ARTIFACT_ROOT = "artifacts/retargeting_v3_step2_capability"
EXACT_PROJECTION_THRESHOLDS = {
    "neutral": 1e-3,
    "foot": 0.06,
    "hand": 0.12,
    "torso": 0.08,
    "default": 0.05,
}
STRESS_MOTION = "extreme_but_valid_joint_limit_stress"
PASS_STATUSES = {"passed", "partial_passed", "negative_control_passed"}
RANK_ZERO_STATUSES = {"rank_zero", "unreachable/rank_zero"}
LIMITED_CAPABILITY_CERTIFICATE_CLASSES = {
    "capability_limited_rank",
    "capability_limited_joint_limits",
    "capability_limited_mixed",
}
PRIOR_CANCELLATION_RATIO_MAX = 0.25
MESH_OR_TEXTURE_SUFFIXES = {
    ".stl",
    ".dae",
    ".obj",
    ".ply",
    ".mesh",
    ".glb",
    ".gltf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tga",
    ".mtl",
}
CORE_CAPABILITY_FILES = (
    "soma_retargeter/robotics/v3/chain_projection.py",
    "soma_retargeter/robotics/v3/canonical_projection.py",
    "soma_retargeter/robotics/v3/profile.py",
    "soma_retargeter/robotics/v3/target_builder.py",
    "soma_retargeter/robotics/v3/rest_frames.py",
    "soma_retargeter/robotics/v3/reachability.py",
    "soma_retargeter/robotics/v3/numerical_jacobian.py",
)
CORE_DRIFT_PATHS = (
    "soma_retargeter",
    "tests",
    "scripts",
    ".github",
)
REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "lfs_state.json",
    "commands.txt",
    "acceptance_ledger.json",
    "baseline_summary.json",
    "baseline_failure_ledger.json",
    "before_after.json",
    "certificate_thresholds.json",
    "summary.json",
    "capability_matrix.json",
    "target_geometry_matrix.json",
    "deterministic_rerun.json",
    "cross_format.json",
    "deferred_snapshots.json",
    "source_inventory.json",
    "load_matrix.json",
    "semantic_matrix.json",
    "validation_checks.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/pytest_summary.json",
)
REQUIRED_DETERMINISTIC_COMPARISONS = (
    "status",
    "canonical_projection_residuals",
    "task_certificate_summary",
    "deterministic_hash",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=CAPABILITY_ARTIFACT_ROOT)
    parser.add_argument("--numerical-artifact-dir", default="artifacts/retargeting_v3_step2_numerical")
    parser.add_argument("--manifest", default="assets/robot_zoo/robot_zoo_manifest.json")
    parser.add_argument("--lock", default="assets/robot_zoo/robot_zoo_lock.json")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--skip-lfs-fsck", action="store_true")
    parser.add_argument("--write-report")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    failures = audit(
        artifact_dir=repo_root / args.artifact_dir,
        numerical_artifact_dir=repo_root / args.numerical_artifact_dir,
        manifest_path=repo_root / args.manifest,
        lock_path=repo_root / args.lock,
        repo_root=repo_root,
        run_lfs_fsck=not args.skip_lfs_fsck,
    )
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "failure_count": len(failures),
        "failures": failures,
    }
    if args.write_report:
        output = Path(args.write_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failures:
        print("Step 2.3 capability red-team audit FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Step 2.3 capability red-team audit PASS")
    return 0


def audit(
    *,
    artifact_dir: Path,
    numerical_artifact_dir: Path,
    manifest_path: Path,
    lock_path: Path,
    repo_root: Path,
    run_lfs_fsck: bool = True,
) -> list[str]:
    failures: list[str] = []
    if not artifact_dir.exists():
        return [f"artifact directory missing: {artifact_dir}"]

    reports = _read_reports(artifact_dir / "per_robot")
    summary = _read_json_if_exists(artifact_dir / "summary.json")
    deterministic = _read_json_if_exists(artifact_dir / "deterministic_rerun.json")
    before_after = _read_json_if_exists(artifact_dir / "before_after.json")
    baseline_summary = _read_json_if_exists(artifact_dir / "baseline_summary.json")
    acceptance_ledger = _read_json_if_exists(artifact_dir / "acceptance_ledger.json")
    threshold_payload = _read_json_if_exists(numerical_artifact_dir / "threshold_calibration.json")

    failures.extend(audit_required_artifact_files(artifact_dir))
    failures.extend(audit_threshold_calibration(threshold_payload, check_live_function=False))
    failures.extend(audit_profile_threshold_source(repo_root))
    failures.extend(audit_baseline_truth(before_after, baseline_summary))
    failures.extend(audit_before_after_acceptance(before_after, summary))
    failures.extend(audit_projection_reports(reports))
    failures.extend(audit_motion_status_policy(reports))
    failures.extend(audit_negative_controls(summary, reports))
    failures.extend(audit_deterministic_rerun(deterministic, expected_model_count=EXPECTED_MODEL_COUNT))
    failures.extend(audit_no_robot_id_special_cases(repo_root, manifest_path))
    failures.extend(audit_torso_axis_retained(repo_root, reports))
    failures.extend(audit_lfs_policy(repo_root, lock_path=lock_path, run_fsck=run_lfs_fsck))
    failures.extend(audit_clean_provenance(artifact_dir, repo_root=repo_root))
    failures.extend(audit_workflow_config(repo_root / ".github/workflows/retargeting_v3_capability.yml"))
    failures.extend(audit_acceptance_ledger(acceptance_ledger))
    failures.extend(audit_agent_f_handoff(repo_root / "docs/retargeting_v3/subagents/capability_agent_f_red_team.md", repo_root))
    return failures


def audit_required_artifact_files(artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACT_FILES:
        path = artifact_dir / rel
        if not path.exists():
            failures.append(f"required artifact missing: {rel}")
    for rel in ("per_robot", "per_task", "failures"):
        path = artifact_dir / rel
        if not path.is_dir():
            failures.append(f"required artifact directory missing: {rel}")
            continue
        if not any(path.glob("*.json")):
            failures.append(f"required artifact directory has no json files: {rel}")
    return failures


def audit_baseline_truth(before_after: dict[str, Any], baseline_summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not baseline_summary:
        return ["baseline_summary.json missing or empty"]
    baseline_commit = baseline_summary.get("baseline_commit") or baseline_summary.get("source_code_commit")
    if baseline_commit != EXPECTED_BASELINE_COMMIT:
        failures.append(
            "baseline_summary baseline_commit must be "
            f"{EXPECTED_BASELINE_COMMIT}, got {baseline_commit!r}"
        )
    if before_after.get("baseline_commit") != EXPECTED_BASELINE_COMMIT:
        failures.append(
            "before_after.baseline_commit must be "
            f"{EXPECTED_BASELINE_COMMIT}, got {before_after.get('baseline_commit')!r}"
        )
    if not before_after.get("baseline_artifact_root"):
        failures.append("before_after.baseline_artifact_root missing")
    expected_summary_command = (
        f"git show {EXPECTED_BASELINE_COMMIT}:"
        "artifacts/retargeting_v3_step2_assets44/summary.json"
    )
    if before_after.get("baseline_source_command") != expected_summary_command:
        failures.append(
            "before_after.baseline_source_command must record the pinned git-object summary command"
        )
    rows = before_after.get("rows")
    baseline_rows = baseline_summary.get("rows")
    if not isinstance(baseline_rows, dict):
        baseline_rows = baseline_summary.get("reports")
    if not isinstance(rows, dict) or not rows:
        failures.append("before_after.rows missing or empty")
        rows = {}
    if not isinstance(baseline_rows, dict) or not baseline_rows:
        failures.append("baseline_summary.rows missing or empty")
        baseline_rows = {}
    if before_after.get("row_count") != len(rows):
        failures.append(f"before_after.row_count must equal rows length {len(rows)}, got {before_after.get('row_count')!r}")
    baseline_row_count = baseline_summary.get("row_count", baseline_summary.get("model_count"))
    if baseline_row_count not in (None, len(baseline_rows)):
        failures.append(
            "baseline_summary.row_count must equal rows length "
            f"{len(baseline_rows)}, got {baseline_row_count!r}"
        )
    for model_id, row in sorted(rows.items()):
        if not isinstance(row, dict):
            failures.append(f"{model_id}: before_after row must be an object")
            continue
        baseline_row = baseline_rows.get(model_id)
        if not isinstance(baseline_row, dict):
            failures.append(f"{model_id}: missing true baseline row")
            continue
        true_status = baseline_row.get("status") or baseline_row.get("baseline_status")
        before_status = row.get("before_status") or row.get("baseline_status")
        after_status = row.get("after_status") or row.get("current_status")
        if before_status != true_status:
            failures.append(
                f"{model_id}: fabricated before status {before_status!r}; "
                f"true baseline status is {true_status!r}"
            )
        expected_row_command = (
            f"git show {EXPECTED_BASELINE_COMMIT}:"
            f"artifacts/retargeting_v3_step2_assets44/per_robot/{model_id}.json"
        )
        if row.get("baseline_source_command") != expected_row_command:
            failures.append(f"{model_id}: before_after row missing pinned per-robot source command")
        if row.get("baseline_summary_source_command") != expected_summary_command:
            failures.append(f"{model_id}: before_after row missing pinned summary source command")
        if true_status == "passed" and after_status != "passed":
            failures.append(f"{model_id}: baseline passed downgraded to {after_status!r}")
    return failures


def audit_before_after_acceptance(before_after: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    transition_validation = before_after.get("transition_validation")
    if not isinstance(transition_validation, dict):
        failures.append("before_after.transition_validation missing")
    elif transition_validation.get("status") != "passed":
        failures.append(
            "before_after.transition_validation.status must be passed, "
            f"got {transition_validation.get('status')!r}"
        )
    final_count_validation = before_after.get("final_count_validation")
    if not isinstance(final_count_validation, dict):
        failures.append("before_after.final_count_validation missing")
    elif final_count_validation.get("status") != "passed":
        failures.append(
            "before_after.final_count_validation.status must be passed, "
            f"got {final_count_validation.get('status')!r}"
        )
    observed_transitions = set((before_after.get("status_transition_counts") or {}).keys())
    allowed_transitions = set(str(item) for item in before_after.get("allowed_transitions") or [])
    if not allowed_transitions:
        failures.append("before_after.allowed_transitions missing")
    else:
        illegal = sorted(observed_transitions - allowed_transitions)
        if illegal:
            failures.append("before_after.status_transition_counts contains illegal transitions: " + ", ".join(illegal))
    embedded_before_after = summary.get("before_after") if isinstance(summary, dict) else None
    if isinstance(embedded_before_after, dict):
        embedded_transitions = embedded_before_after.get("status_transition_counts")
        if embedded_transitions is not None and embedded_transitions != before_after.get("status_transition_counts"):
            failures.append("summary.before_after.status_transition_counts differs from before_after.json")
    return failures


def audit_threshold_calibration(payload: dict[str, Any], *, check_live_function: bool = False) -> list[str]:
    failures: list[str] = []
    projection = payload.get("projection_quality")
    if not isinstance(projection, dict):
        return ["threshold_calibration.projection_quality missing"]
    expected_fields = {
        "neutral_position_abs_m": EXACT_PROJECTION_THRESHOLDS["neutral"],
        "foot_rho_p": EXACT_PROJECTION_THRESHOLDS["foot"],
        "hand_rho_p": EXACT_PROJECTION_THRESHOLDS["hand"],
        "torso_rho_r": EXACT_PROJECTION_THRESHOLDS["torso"],
    }
    for field, expected in expected_fields.items():
        observed = projection.get(field)
        if observed != expected:
            failures.append(f"threshold {field} must remain {expected!r}, got {observed!r}")
    for path, value in _walk_json(payload):
        lowered = str(path[-1]).lower() if path else ""
        if any(token in lowered for token in ("override", "per_robot", "special_case", "robot_id")):
            failures.append(f"threshold calibration contains non-global key {'.'.join(path)}={value!r}")
    if check_live_function:
        failures.extend(audit_profile_threshold_source(Path(".")))
    return failures


def audit_profile_threshold_source(repo_root: Path) -> list[str]:
    path = repo_root / "soma_retargeter/robotics/v3/capability_status.py"
    if not path.exists():
        return [f"profile threshold source missing: {path}"]
    text = path.read_text(errors="ignore")
    expected_snippets = {
        "neutral": 'if motion_name == "neutral":\n        return 1e-3',
        "foot": 'if "foot" in task_name:\n        return 0.06',
        "hand": 'if "hand" in task_name:\n        return 0.12',
        "torso": 'if "torso" in task_name:\n        return 0.08',
        "default": "return 0.05",
    }
    failures: list[str] = []
    for label, snippet in expected_snippets.items():
        if snippet not in text:
            failures.append(f"capability_status.py projection threshold literal changed or missing for {label}")
    return failures


def audit_projection_reports(reports: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for model_id, report in sorted(reports.items()):
        status = str(report.get("status", ""))
        failures.extend(_audit_target_geometry_status(model_id, report, status))
        canonical = report.get("canonical_projection_reports", {})
        motions = canonical.get("motions", {}) if isinstance(canonical, dict) else {}
        if not isinstance(motions, dict):
            continue
        for motion_name, motion in sorted(motions.items()):
            tasks = motion.get("tasks", {}) if isinstance(motion, dict) else {}
            if not isinstance(tasks, dict):
                continue
            for task_name, payload in sorted(tasks.items()):
                if not isinstance(payload, dict):
                    continue
                label = f"{model_id}.{motion_name}.{task_name}"
                task_status = str(payload.get("status", ""))
                if task_status in RANK_ZERO_STATUSES:
                    failures.extend(_audit_rank_zero_payload(label, task_status, payload))
                    continue
                normalized = payload.get("normalized_residual")
                if normalized is None:
                    failures.append(f"{label}: normalized_residual missing")
                    continue
                try:
                    normalized_float = float(normalized)
                except (TypeError, ValueError):
                    failures.append(f"{label}: normalized_residual nonnumeric: {normalized!r}")
                    continue
                if not math.isfinite(normalized_float):
                    failures.append(f"{label}: normalized_residual nonfinite: {normalized!r}")
                    continue
                if motion_name == STRESS_MOTION:
                    continue
                threshold = _projection_threshold(task_name, motion_name)
                if normalized_float <= threshold:
                    continue
                if status in PASS_STATUSES:
                    failures.append(
                        f"{label}: ordinary residual over threshold packaged as pass "
                        f"({normalized_float:.6g} > {threshold:.6g})"
                    )
                legacy_certificate = payload.get("kkt_certificate")
                if isinstance(legacy_certificate, dict):
                    failures.extend(_audit_kkt_certificate(label, payload, legacy_certificate))
                    continue
                capability_certificate = payload.get("capability_certificate")
                if not isinstance(capability_certificate, dict) or capability_certificate.get("schema_version") != 2:
                    failures.append(
                        f"{label}: residual over threshold missing schema-v2 capability_certificate "
                        f"({normalized_float:.6g} > {threshold:.6g})"
                    )
                    continue
                failures.extend(_audit_capability_certificate_v2(label, payload, capability_certificate))
    return failures


def audit_motion_status_policy(reports: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for model_id, report in sorted(reports.items()):
        entry = report.get("manifest_entry", {})
        if entry.get("expected_capability") != "positive" or entry.get("robot_class") != "humanoid":
            continue
        status = str(report.get("status", ""))
        ordinary_limited = False
        stress_limited = False
        canonical = report.get("canonical_projection_reports", {})
        motions = canonical.get("motions", {}) if isinstance(canonical, dict) else {}
        if not isinstance(motions, dict):
            continue
        for motion_name, motion in motions.items():
            tasks = motion.get("tasks", {}) if isinstance(motion, dict) else {}
            if not isinstance(tasks, dict):
                continue
            for task_name, payload in tasks.items():
                if not isinstance(payload, dict):
                    continue
                normalized = _float_or_none(payload.get("normalized_residual"))
                if normalized is None:
                    continue
                over_threshold = normalized > _projection_threshold(str(task_name), str(motion_name))
                if not over_threshold:
                    continue
                if motion_name == STRESS_MOTION:
                    stress_limited = True
                else:
                    ordinary_limited = True
        if stress_limited and not ordinary_limited and status != "passed":
            failures.append(f"{model_id}: stress-only limited evidence changed status to {status!r}")
        if status == "capability_limited_passed" and not ordinary_limited:
            failures.append(f"{model_id}: capability_limited_passed without ordinary limited evidence")
    return failures


def audit_negative_controls(summary: dict[str, Any], reports: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    negative_count = 0
    negative_passed = 0
    positive_humanoid_passed = 0
    for model_id, report in sorted(reports.items()):
        entry = report.get("manifest_entry", {})
        expected = entry.get("expected_capability")
        robot_class = entry.get("robot_class")
        status = report.get("status")
        if expected == "positive" and robot_class == "humanoid" and status == "passed":
            positive_humanoid_passed += 1
        if expected != "negative_control":
            continue
        negative_count += 1
        if status == "negative_control_passed":
            negative_passed += 1
        else:
            failures.append(f"{model_id}: negative control status must be negative_control_passed, got {status!r}")
        if robot_class == "humanoid":
            failures.append(f"{model_id}: negative control must not be robot_class=humanoid")
        if report.get("capability_status") or report.get("canonical_projection_reports"):
            failures.append(f"{model_id}: negative control contains humanoid capability payload")
    if summary.get("negative_control_passed") != negative_passed:
        failures.append(
            "summary.negative_control_passed must equal negative-control pass count "
            f"{negative_passed}, got {summary.get('negative_control_passed')!r}"
        )
    if summary.get("profile_passed") != positive_humanoid_passed:
        failures.append(
            "summary.profile_passed must count only positive humanoid passed profiles "
            f"{positive_humanoid_passed}, got {summary.get('profile_passed')!r}"
        )
    if negative_count == 0:
        failures.append("no negative controls found in per_robot reports")
    return failures


def audit_deterministic_rerun(
    deterministic: dict[str, Any],
    *,
    expected_model_count: int = EXPECTED_MODEL_COUNT,
) -> list[str]:
    failures: list[str] = []
    totals = deterministic.get("totals", {}) if isinstance(deterministic, dict) else {}
    if deterministic.get("status") != "passed":
        failures.append(f"deterministic_rerun.status must be passed, got {deterministic.get('status')!r}")
    expected_totals = {
        "model_count": expected_model_count,
        "compared_count": expected_model_count,
        "matched_count": expected_model_count,
        "mismatch_count": 0,
        "skipped_non_pass_count": 0,
        "source_unavailable_count": 0,
        "rerun_failed_count": 0,
    }
    for key, expected in expected_totals.items():
        observed = totals.get(key)
        if observed != expected:
            failures.append(f"deterministic_rerun.totals.{key} must be {expected}, got {observed!r}")
    models = deterministic.get("models", {}) if isinstance(deterministic, dict) else {}
    if not isinstance(models, dict):
        failures.append("deterministic_rerun.models must be an object")
        return failures
    if len(models) != expected_model_count:
        failures.append(f"deterministic_rerun.models must cover {expected_model_count} models, got {len(models)}")
    uncompared = sorted(model_id for model_id, row in models.items() if not row.get("compared"))
    if uncompared:
        failures.append(f"uncompared deterministic models: {', '.join(uncompared[:12])}")
    for model_id, row in sorted(models.items()):
        if not isinstance(row, dict):
            failures.append(f"{model_id}: deterministic row must be an object")
            continue
        if row.get("compared") is not True:
            continue
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, dict):
            failures.append(f"{model_id}: deterministic comparisons missing")
            continue
        for key in REQUIRED_DETERMINISTIC_COMPARISONS:
            comparison = comparisons.get(key)
            if not isinstance(comparison, dict):
                failures.append(f"{model_id}: deterministic comparison missing {key}")
                continue
            if comparison.get("matched") is not True:
                failures.append(f"{model_id}: deterministic comparison {key} did not match")
    return failures


def audit_no_robot_id_special_cases(repo_root: Path, manifest_path: Path) -> list[str]:
    failures: list[str] = []
    if not manifest_path.exists():
        return [f"manifest missing for robot-ID special-case scan: {manifest_path}"]
    manifest = _read_json(manifest_path)
    model_ids = [entry["id"] for entry in manifest.get("models", []) if isinstance(entry, dict) and entry.get("id")]
    for rel in CORE_CAPABILITY_FILES:
        path = repo_root / rel
        if not path.exists():
            failures.append(f"capability source file missing: {rel}")
            continue
        text = path.read_text(errors="ignore")
        hits = [model_id for model_id in model_ids if model_id in text]
        if hits:
            failures.append(f"{rel}: robot-ID special cases present: {', '.join(sorted(hits))}")
    return failures


def audit_torso_axis_retained(repo_root: Path, reports: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    rest_frames = repo_root / "soma_retargeter/robotics/v3/rest_frames.py"
    if not rest_frames.exists():
        return [f"missing rest frame source: {rest_frames}"]
    text = rest_frames.read_text(errors="ignore")
    for token in ('"torso_axis"', '"Hips"', '"Chest"'):
        if token not in text:
            failures.append(f"rest_frames.py no longer retains compatible torso-axis candidate {token}")
    torso_axis_evidence = []
    for model_id, report in reports.items():
        sources = report.get("rest_calibration", {}).get("edge_frame_sources", {})
        for chain, source in sources.items() if isinstance(sources, dict) else ():
            if chain in {"left_arm", "right_arm"} and "torso_axis" in str(source):
                torso_axis_evidence.append(f"{model_id}.{chain}")
    if not torso_axis_evidence:
        failures.append("no per-robot rest_calibration arm evidence retained torso_axis")
    return failures


def audit_lfs_policy(
    repo_root: Path,
    *,
    lock_path: Path,
    run_fsck: bool = True,
    lfs_paths: list[str] | None = None,
) -> list[str]:
    failures: list[str] = []
    if run_fsck:
        result = subprocess.run(
            ["git", "lfs", "fsck"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            failures.append("git lfs fsck failed: " + result.stdout.strip())
    paths = lfs_paths if lfs_paths is not None else _git_lfs_paths(repo_root)
    for rel in paths:
        path = repo_root / rel
        if not path.exists():
            failures.append(f"LFS tracked file missing from working tree: {rel}")
            continue
        if _is_lfs_pointer(path):
            failures.append(f"LFS pointer-only validation risk: {rel}")
    snapshots = repo_root / "assets/robot_zoo/snapshots"
    if snapshots.exists():
        for path in snapshots.rglob("*"):
            if path.is_dir() and path.name == ".git":
                failures.append(f"upstream VCS directory vendored under snapshots: {_display(path, repo_root)}")
            if path.is_file() and path.suffix.lower() in MESH_OR_TEXTURE_SUFFIXES:
                failures.append(f"mesh/texture leakage under robot_zoo snapshots: {_display(path, repo_root)}")
    lock = _read_json_if_exists(lock_path)
    for model_id, row in (lock.get("entries") or {}).items():
        if row.get("snapshot_status") != "fetch_only":
            continue
        fetch_snapshot_dir = snapshots / model_id
        if fetch_snapshot_dir.exists():
            failures.append(f"fetch-only model has vendored snapshot directory: {_display(fetch_snapshot_dir, repo_root)}")
        prefix = f"assets/robot_zoo/snapshots/{model_id}/"
        leaked_lfs = [rel for rel in paths if rel.startswith(prefix)]
        if leaked_lfs:
            failures.append(f"fetch-only model has LFS-tracked vendored payload: {model_id}")
    return failures


def audit_clean_provenance(
    artifact_dir: Path,
    *,
    repo_root: Path | None = None,
    git_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> list[str]:
    failures: list[str] = []
    repo_root = repo_root or Path(".")
    environment = _read_json_if_exists(artifact_dir / "environment.json")
    if environment.get("git_status_short") != "":
        failures.append("environment.git_status_short must be clean for accepted artifacts")
    source_commit = environment.get("source_code_commit")
    if not source_commit:
        failures.append("environment.source_code_commit missing")
    if environment.get("source_code_commit_remote_resolvable") is not True:
        failures.append("environment.source_code_commit_remote_resolvable must be true")
    if environment.get("source_code_commit_is_artifact_commit_ancestor") is not True:
        failures.append("environment.source_code_commit_is_artifact_commit_ancestor must be true")
    if environment.get("source_worktree_clean_before_run") is not True:
        failures.append("environment.source_worktree_clean_before_run must be true")
    if environment.get("source_worktree_clean_after_run") is not True:
        failures.append("environment.source_worktree_clean_after_run must be true")
    if environment.get("core_diff_after_source_commit") != []:
        failures.append("environment.core_diff_after_source_commit must be []")
    artifact_commit = environment.get("artifact_commit") or environment.get("git_head")
    if source_commit:
        result = _run_git(repo_root, ["cat-file", "-e", f"{source_commit}^{{commit}}"], git_runner)
        if result.returncode != 0:
            failures.append(f"environment.source_code_commit is not locally resolvable: {source_commit}")
    if artifact_commit:
        result = _run_git(repo_root, ["cat-file", "-e", f"{artifact_commit}^{{commit}}"], git_runner)
        if result.returncode != 0:
            failures.append(f"environment.artifact commit is not locally resolvable: {artifact_commit}")
    if source_commit and artifact_commit:
        result = _run_git(repo_root, ["merge-base", "--is-ancestor", str(source_commit), str(artifact_commit)], git_runner)
        if result.returncode != 0:
            failures.append("environment.source_code_commit is not an ancestor of artifact_commit")
        diff_result = _run_git(
            repo_root,
            ["diff", "--name-only", f"{source_commit}..{artifact_commit}", "--", *CORE_DRIFT_PATHS],
            git_runner,
        )
        if diff_result.returncode != 0:
            failures.append("git diff source..artifact core drift check failed")
        else:
            drift = [line for line in diff_result.stdout.splitlines() if line.strip()]
            if drift:
                failures.append("core drift after source commit: " + ", ".join(drift[:12]))
    source_inventory = _read_json_if_exists(artifact_dir / "source_inventory.json")
    for model_id, row in source_inventory.items():
        path = row.get("path") if isinstance(row, dict) else None
        if isinstance(path, str) and _is_forbidden_absolute_path(path):
            failures.append(f"{model_id}: source inventory path leaks local absolute path {path!r}")
    failures.extend(_audit_no_absolute_paths(artifact_dir))
    return failures


def audit_workflow_config(workflow_path: Path) -> list[str]:
    if not workflow_path.exists():
        return [f"workflow missing: {workflow_path}"]
    text = workflow_path.read_text()
    failures: list[str] = []
    if EXPECTED_CAPABILITY_BRANCH not in text:
        failures.append(f"workflow push branches must include {EXPECTED_CAPABILITY_BRANCH}")
    if "retargeting-v3-step2-capability\n" in text:
        failures.append("workflow still references stale capability push branch")
    if f"RETARGETING_V3_CAPABILITY_ARTIFACTS: {CAPABILITY_ARTIFACT_ROOT}" not in text:
        failures.append(f"workflow RETARGETING_V3_CAPABILITY_ARTIFACTS must be {CAPABILITY_ARTIFACT_ROOT}")
    if "retargeting_v3_step2_assets44" in text:
        failures.append("workflow still references assets44 artifact root")
    if "python scripts/audit_retargeting_v3_capability.py" not in text:
        failures.append("workflow missing live capability audit command")
    if '--artifact-dir "$RETARGETING_V3_CAPABILITY_ARTIFACTS"' not in text:
        failures.append("workflow audit command must use capability artifact root")
    if "--lock assets/robot_zoo/robot_zoo_lock.json" not in text:
        failures.append("workflow audit command must pass robot zoo lock")
    if "git lfs pull" not in text or "git lfs fsck" not in text:
        failures.append("workflow must run git lfs pull and git lfs fsck")
    for job in ("capability-synthetic-tests", "capability-artifact-live-audit", "lfs-snapshot-smoke"):
        if re.search(rf"^\s{{2}}{re.escape(job)}:", text, flags=re.MULTILINE) is None:
            failures.append(f"workflow missing required job {job}")
    return failures


def audit_acceptance_ledger(ledger: dict[str, Any]) -> list[str]:
    if not ledger:
        return ["acceptance_ledger.json missing or empty"]
    command_parts = [str(ledger.get("command", ""))]
    commands = ledger.get("commands")
    if isinstance(commands, list):
        command_parts.extend(str(item) for item in commands)
    command = "\n".join(command_parts)
    failures: list[str] = []
    if "scripts/audit_retargeting_v3_capability.py" not in command:
        failures.append("acceptance_ledger is stale: does not run capability audit")
    if "audit_retargeting_v3_assets44.py" in command or "audit_retargeting_v3_step2.py" in command:
        failures.append("acceptance_ledger is stale Step2.2/Step2 audit evidence")
    if "retargeting_v3_step2_assets44" in command:
        failures.append("acceptance_ledger references assets44 artifact root")
    return failures


def audit_agent_f_handoff(handoff_path: Path, repo_root: Path) -> list[str]:
    if not handoff_path.exists():
        return [f"Agent F handoff missing: {handoff_path}"]
    text = handoff_path.read_text(errors="ignore")
    failures: list[str] = []
    if "Do not accept Step 2.3 capability yet" in text or "Live capability audit: **FAIL**" in text:
        failures.append("Agent F handoff is stale FAIL text")
    has_pass_verdict = re.search(r"verdict\s*=\s*PASS", text, flags=re.IGNORECASE) is not None
    has_blocked_verdict = re.search(r"verdict\s*=\s*BLOCKED", text, flags=re.IGNORECASE) is not None
    if has_pass_verdict and has_blocked_verdict:
        failures.append("Agent F handoff must not contain both PASS and BLOCKED verdicts")
    if not has_pass_verdict and not has_blocked_verdict:
        failures.append("Agent F handoff must state verdict = PASS or verdict = BLOCKED")
    if has_pass_verdict:
        required = (
            "final_head",
            "source_code_commit",
            "artifact_commit",
            "workflow_run_id",
            "pytest summary",
            "live audit command",
            "live audit PASS",
            "LFS fsck PASS",
            "remaining blockers = 0",
        )
        for token in required:
            if token not in text:
                failures.append(f"Agent F PASS handoff missing {token}")
        for field in ("final_head", "source_code_commit", "artifact_commit"):
            value = _extract_handoff_field(text, field)
            if not value or not re.fullmatch(r"[0-9a-f]{40}", value):
                failures.append(f"Agent F PASS handoff {field} must be a concrete 40-character git SHA")
        workflow_run_id = _extract_handoff_field(text, "workflow_run_id")
        if not workflow_run_id or re.search(r"\b(pending|not available|n/a|unknown)\b", workflow_run_id, flags=re.IGNORECASE):
            failures.append("Agent F PASS handoff workflow_run_id must be concrete")
        stale_blocker_markers = (
            "remaining blockers > 0",
            "remaining blockers: > 0",
            "known blockers include",
            "still expected to fail",
            "not final integrated evidence",
        )
        lowered = text.lower()
        for marker in stale_blocker_markers:
            if marker in lowered:
                failures.append(f"Agent F PASS handoff contains stale blocker marker: {marker}")
    return failures


def _extract_handoff_field(text: str, field: str) -> str | None:
    match = re.search(
        rf"^\s*[-*]?\s*{re.escape(field)}\s*:\s*`?([^`\n]+?)`?\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    return match.group(1).strip()


def _run_git(
    repo_root: Path,
    args: list[str],
    git_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None,
) -> subprocess.CompletedProcess[str]:
    if git_runner is not None:
        return git_runner(args)
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _audit_target_geometry_status(model_id: str, report: dict[str, Any], status: str) -> list[str]:
    failures: list[str] = []
    validation = report.get("canonical_target_validation", {})
    target_failures = validation.get("failures", []) if isinstance(validation, dict) else []
    target_failures = list(target_failures or [])
    failure_text = "\n".join(str(item) for item in target_failures + list(report.get("failures") or []))
    has_target_geometry_failure = bool(target_failures) or "canonical target validation failed" in failure_text
    if has_target_geometry_failure and status in PASS_STATUSES:
        failures.append(f"{model_id}: target geometry failure packaged as pass status {status!r}")
    return failures


def _audit_rank_zero_payload(label: str, task_status: str, payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in ("demand_residual", "unreachable_demand", "rank_zero_reason"):
        if field not in payload:
            failures.append(f"{label}: rank-zero projection missing {field}")
    demand = _float_or_none(payload.get("demand_residual"))
    residual = _float_or_none(payload.get("residual"))
    vector_demand = _desired_projected_distance(payload)
    nonzero_demand = any(value is not None and value > 1e-10 for value in (demand, residual, vector_demand))
    if nonzero_demand:
        if task_status != "unreachable/rank_zero":
            failures.append(f"{label}: rank-zero nonzero demand must be status unreachable/rank_zero")
        if demand is None or demand <= 1e-10:
            failures.append(f"{label}: rank-zero nonzero demand not preserved in demand_residual")
        if payload.get("unreachable_demand") is not True:
            failures.append(f"{label}: rank-zero nonzero demand must set unreachable_demand=true")
        if payload.get("rank_zero_reason") != "no_active_coordinates_nonzero_demand":
            failures.append(f"{label}: rank-zero nonzero demand has wrong rank_zero_reason")
    return failures


def _audit_capability_certificate_v2(
    label: str,
    payload: dict[str, Any],
    certificate: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if certificate.get("schema_version") != 2:
        failures.append(f"{label}: capability_certificate schema_version must be 2")
    certificate_class = str(certificate.get("certificate_class", ""))
    if certificate_class not in LIMITED_CAPABILITY_CERTIFICATE_CLASSES:
        failures.append(
            f"{label}: over-threshold residual requires limited capability certificate, "
            f"got {certificate_class!r}"
        )
    if certificate.get("passed") is not True:
        failures.append(f"{label}: capability_certificate.passed must be true for accepted limited evidence")
    gates = certificate.get("gates")
    if not isinstance(gates, dict):
        failures.append(f"{label}: capability_certificate.gates must be an object")
    else:
        for gate_name in (
            "projected_gradient_kkt",
            "seed_consensus",
            "residual_explained",
            "continuation",
            "joint_limits",
            "numerical",
        ):
            if gates.get(gate_name) is not True:
                failures.append(f"{label}: capability_certificate gate {gate_name} must be true")
    if certificate.get("exact_threshold_passed") is True:
        failures.append(f"{label}: over-threshold limited certificate has exact_threshold_passed=true")

    kkt_certificate, adapter_failures = _schema_v2_kkt_certificate(label, payload, certificate)
    failures.extend(adapter_failures)
    if kkt_certificate is not None:
        failures.extend(_audit_kkt_certificate(label, payload, kkt_certificate))
    return failures


def _schema_v2_kkt_certificate(
    label: str,
    payload: dict[str, Any],
    certificate: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    kkt = certificate.get("kkt")
    if not isinstance(kkt, dict):
        return None, [f"{label}: capability_certificate.kkt must be an object"]
    audit = certificate.get("audit_evidence")
    if not isinstance(audit, dict):
        return None, [f"{label}: capability_certificate.audit_evidence must be an object"]
    gates = certificate.get("gates", {})
    if isinstance(gates, dict) and gates.get("projected_gradient_kkt") is not bool(kkt.get("satisfied", False)):
        failures.append(f"{label}: capability_certificate kkt.satisfied disagrees with projected_gradient_kkt gate")
    for field in ("primal_feasible", "dual_feasible", "complementarity_passed", "satisfied"):
        if kkt.get(field) is not True:
            failures.append(f"{label}: capability_certificate.kkt.{field} must be true")
    raw, raw_failures = _schema_v2_raw_kkt_evidence(label, certificate)
    failures.extend(raw_failures)
    seed = _schema_v2_seed_consistency(certificate, payload)
    legacy_shape = {
        "certified": bool(kkt.get("satisfied", False)),
        "stationarity_inf_norm": kkt.get("stationarity_inf_norm"),
        "stationarity_tolerance": kkt.get("stationarity_tolerance", kkt.get("tolerance")),
        "complementarity_inf_norm": kkt.get("complementarity_inf_norm"),
        "complementarity_tolerance": kkt.get("complementarity_tolerance", kkt.get("stationarity_tolerance", kkt.get("tolerance"))),
        "primal_feasible": kkt.get("primal_feasible"),
        "dual_feasible": kkt.get("dual_feasible"),
        "task_gradient_inf_norm": kkt.get("task_gradient_inf_norm"),
        "prior_gradient_inf_norm": kkt.get("prior_gradient_inf_norm"),
        "prior_cancellation_ratio": kkt.get("prior_cancellation_ratio"),
        "raw_evidence": raw,
        "seed_consistency": seed,
    }
    return legacy_shape, failures


def _schema_v2_raw_kkt_evidence(label: str, certificate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    kkt = certificate.get("kkt", {})
    solver_raw = kkt.get("raw", {}) if isinstance(kkt, dict) else {}
    raw_evidence = solver_raw.get("raw_evidence") if isinstance(solver_raw, dict) else None
    if isinstance(raw_evidence, dict):
        raw = {
            "q": raw_evidence.get("q"),
            "lower_bounds": raw_evidence.get("lower_bounds", raw_evidence.get("lower")),
            "upper_bounds": raw_evidence.get("upper_bounds", raw_evidence.get("upper")),
            "task_jacobian": raw_evidence.get("task_jacobian", raw_evidence.get("J")),
            "task_residual": raw_evidence.get("task_residual", raw_evidence.get("e")),
            "prior_jacobian": raw_evidence.get("prior_jacobian", []),
            "prior_residual": raw_evidence.get("prior_residual", []),
            "active_bound_tolerance": kkt.get("active_bound_tolerance"),
        }
        return raw, []

    audit = certificate.get("audit_evidence")
    if not isinstance(audit, dict):
        return {}, [f"{label}: capability_certificate.audit_evidence must be an object"]
    scale = _float_or_none(audit.get("normalization_scale")) or 1.0
    jacobian = _matrix(audit.get("relevant_task_jacobian"))
    if jacobian is not None:
        jacobian = jacobian / max(scale, np.finfo(float).tiny)
    prior_gradient = _vector(audit.get("prior_gradient", []))
    q = _vector(audit.get("q_active", []))
    if prior_gradient is None or q is None:
        prior_jacobian: list[list[float]] = []
        prior_residual: list[float] = []
    elif prior_gradient.size == q.size:
        prior_jacobian = np.eye(q.size, dtype=float).tolist()
        prior_residual = prior_gradient.tolist()
    else:
        prior_jacobian = []
        prior_residual = []
    raw = {
        "q": audit.get("q_active"),
        "lower_bounds": audit.get("lower_bounds"),
        "upper_bounds": audit.get("upper_bounds"),
        "task_jacobian": None if jacobian is None else jacobian.tolist(),
        "task_residual": audit.get("normalized_residual_vector"),
        "prior_jacobian": prior_jacobian,
        "prior_residual": prior_residual,
        "active_bound_tolerance": kkt.get("active_bound_tolerance") if isinstance(kkt, dict) else None,
    }
    return raw, []


def _schema_v2_seed_consistency(certificate: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    seed = certificate.get("seed_consensus", {})
    seed = seed if isinstance(seed, dict) else {}
    rows = seed.get("seed_results")
    if not isinstance(rows, list):
        rows = payload.get("seed_results", [])
    normalized_rows: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                normalized_rows.append(row)
                continue
            normalized = dict(row)
            if "task_space_endpoint" not in normalized and "final_task_vector" in normalized:
                normalized["task_space_endpoint"] = normalized.get("final_task_vector")
            if "task_residual_norm" not in normalized and "normalized_residual" in normalized:
                normalized["task_residual_norm"] = normalized.get("normalized_residual")
            normalized_rows.append(normalized)
    passed = seed.get("passed")
    return {
        "checked": bool(seed.get("checked", bool(normalized_rows))),
        "status": "consistent" if passed is True else "inconsistent_rejected",
        "tolerance": seed.get("task_space_tolerance", seed.get("tolerance", 1e-7)),
        "seed_results": normalized_rows,
    }


def _audit_kkt_certificate(label: str, payload: dict[str, Any], certificate: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required = (
        "certified",
        "stationarity_inf_norm",
        "stationarity_tolerance",
        "complementarity_inf_norm",
        "complementarity_tolerance",
        "primal_feasible",
        "dual_feasible",
        "task_gradient_inf_norm",
        "prior_gradient_inf_norm",
        "prior_cancellation_ratio",
        "seed_consistency",
    )
    for field in required:
        if field not in certificate:
            failures.append(f"{label}: kkt_certificate missing {field}")
    recomputed = _recompute_kkt_from_raw_evidence(label, certificate)
    failures.extend(recomputed.failures)
    if not recomputed.failures:
        failures.extend(_compare_recomputed_kkt(label, certificate, recomputed))
    failures.extend(_audit_seed_task_space_consensus(label, payload, certificate))
    failures.extend(_audit_continuation_completion(label, payload))
    seed = certificate.get("seed_consistency")
    if not isinstance(seed, dict):
        failures.append(f"{label}: kkt_certificate.seed_consistency must be an object")
    else:
        if seed.get("checked") is not True:
            failures.append(f"{label}: kkt_certificate.seed_consistency.checked must be true")
        if seed.get("status") not in {"consistent", "inconsistent_rejected"}:
            failures.append(f"{label}: kkt_certificate.seed_consistency status must reject local-minimum inconsistency")
    return failures


class RecomputedKKT:
    def __init__(
        self,
        *,
        failures: list[str],
        primal_feasible: bool = False,
        dual_feasible: bool = False,
        certified: bool = False,
        task_gradient_inf_norm: float = math.nan,
        prior_gradient_inf_norm: float = math.nan,
        prior_cancellation_ratio: float = math.nan,
        stationarity_inf_norm: float = math.nan,
        complementarity_inf_norm: float = math.nan,
    ) -> None:
        self.failures = failures
        self.primal_feasible = primal_feasible
        self.dual_feasible = dual_feasible
        self.certified = certified
        self.task_gradient_inf_norm = task_gradient_inf_norm
        self.prior_gradient_inf_norm = prior_gradient_inf_norm
        self.prior_cancellation_ratio = prior_cancellation_ratio
        self.stationarity_inf_norm = stationarity_inf_norm
        self.complementarity_inf_norm = complementarity_inf_norm


def _recompute_kkt_from_raw_evidence(label: str, certificate: dict[str, Any]) -> RecomputedKKT:
    raw = certificate.get("raw_evidence")
    if not isinstance(raw, dict):
        return RecomputedKKT(failures=[f"{label}: kkt_certificate missing raw_evidence J/e/q/bounds"])
    failures: list[str] = []
    q = _vector(raw.get("q"))
    lower = _vector(raw.get("lower_bounds", raw.get("lower")))
    upper = _vector(raw.get("upper_bounds", raw.get("upper")))
    task_jacobian = _matrix(raw.get("task_jacobian", raw.get("J")))
    task_residual = _vector(raw.get("task_residual", raw.get("e")))
    prior_jacobian = _matrix(raw.get("prior_jacobian", []))
    prior_residual = _vector(raw.get("prior_residual", []))
    for field, value in (
        ("q", q),
        ("lower_bounds", lower),
        ("upper_bounds", upper),
        ("task_jacobian", task_jacobian),
        ("task_residual", task_residual),
    ):
        if value is None:
            failures.append(f"{label}: raw_evidence.{field} missing, nonnumeric, or nonfinite")
    if prior_jacobian is None or prior_residual is None:
        failures.append(f"{label}: raw_evidence prior_jacobian/prior_residual missing, nonnumeric, or nonfinite")
    if failures:
        return RecomputedKKT(failures=failures)
    assert q is not None
    assert lower is not None
    assert upper is not None
    assert task_jacobian is not None
    assert task_residual is not None
    assert prior_jacobian is not None
    assert prior_residual is not None
    n = q.size
    if lower.size != n or upper.size != n:
        failures.append(f"{label}: raw_evidence q/lower/upper lengths differ")
    if task_jacobian.ndim != 2 or task_jacobian.shape[1] != n or task_jacobian.shape[0] != task_residual.size:
        failures.append(f"{label}: raw_evidence task_jacobian shape is incompatible with q/task_residual")
    if prior_jacobian.size == 0 and prior_residual.size == 0:
        prior_jacobian = np.zeros((0, n), dtype=float)
        prior_residual = np.zeros((0,), dtype=float)
    elif prior_jacobian.ndim != 2 or prior_jacobian.shape[1] != n or prior_jacobian.shape[0] != prior_residual.size:
        failures.append(f"{label}: raw_evidence prior_jacobian shape is incompatible with q/prior_residual")
    if failures:
        return RecomputedKKT(failures=failures)
    tolerance = _float_or_none(certificate.get("stationarity_tolerance")) or 0.0
    bound_tolerance = _float_or_none(raw.get("active_bound_tolerance")) or max(tolerance, 1e-10)
    task_gradient = task_jacobian.T @ task_residual
    prior_gradient = prior_jacobian.T @ prior_residual
    total_gradient = task_gradient + prior_gradient
    task_norm = _inf_norm(task_gradient)
    prior_norm = _inf_norm(prior_gradient)
    if task_norm <= tolerance and prior_norm <= tolerance:
        cancellation_ratio = 0.0
    else:
        cancellation_ratio = prior_norm / max(task_norm, tolerance)
    primal_feasible = bool(np.all(q >= lower - bound_tolerance) and np.all(q <= upper + bound_tolerance))
    lower_active = q <= lower + bound_tolerance
    upper_active = q >= upper - bound_tolerance
    fixed = lower_active & upper_active
    free = ~(lower_active | upper_active)
    stationarity_terms = []
    if np.any(free):
        stationarity_terms.extend(np.abs(total_gradient[free]).tolist())
    if np.any(lower_active & ~fixed):
        stationarity_terms.extend(np.maximum(0.0, -total_gradient[lower_active & ~fixed]).tolist())
    if np.any(upper_active & ~fixed):
        stationarity_terms.extend(np.maximum(0.0, total_gradient[upper_active & ~fixed]).tolist())
    if np.any(fixed):
        stationarity_terms.extend(np.abs(total_gradient[fixed]).tolist())
    stationarity = max(stationarity_terms, default=0.0)
    dual_feasible = bool(stationarity <= tolerance)
    complementarity = _recompute_complementarity(raw, total_gradient, lower_active, upper_active, fixed)
    complementarity_tolerance = _float_or_none(certificate.get("complementarity_tolerance")) or tolerance
    certified = bool(primal_feasible and dual_feasible and complementarity <= complementarity_tolerance)
    return RecomputedKKT(
        failures=[],
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        certified=certified,
        task_gradient_inf_norm=task_norm,
        prior_gradient_inf_norm=prior_norm,
        prior_cancellation_ratio=cancellation_ratio,
        stationarity_inf_norm=stationarity,
        complementarity_inf_norm=complementarity,
    )


def _compare_recomputed_kkt(label: str, certificate: dict[str, Any], recomputed: RecomputedKKT) -> list[str]:
    failures: list[str] = []
    for field in ("primal_feasible", "dual_feasible", "certified"):
        if certificate.get(field) is not getattr(recomputed, field):
            failures.append(
                f"{label}: kkt_certificate.{field}={certificate.get(field)!r} "
                f"but independent recompute is {getattr(recomputed, field)!r}"
            )
    for field in (
        "stationarity_inf_norm",
        "complementarity_inf_norm",
        "task_gradient_inf_norm",
        "prior_gradient_inf_norm",
        "prior_cancellation_ratio",
    ):
        observed = _float_or_none(certificate.get(field))
        expected = getattr(recomputed, field)
        if observed is None or not _close(observed, expected):
            failures.append(f"{label}: kkt_certificate.{field}={observed!r} but recompute is {expected:.12g}")
    cancellation_ratio = _float_or_none(certificate.get("prior_cancellation_ratio"))
    if cancellation_ratio is None or cancellation_ratio > PRIOR_CANCELLATION_RATIO_MAX:
        failures.append(
            f"{label}: kkt_certificate.prior_cancellation_ratio must be <= "
            f"{PRIOR_CANCELLATION_RATIO_MAX}, got {cancellation_ratio!r}"
        )
    stationarity_tolerance = _float_or_none(certificate.get("stationarity_tolerance"))
    complementarity_tolerance = _float_or_none(certificate.get("complementarity_tolerance"))
    if stationarity_tolerance is None or recomputed.stationarity_inf_norm > stationarity_tolerance:
        failures.append(
            f"{label}: recomputed stationarity_inf_norm={recomputed.stationarity_inf_norm:.12g} "
            f"exceeds stationarity_tolerance={stationarity_tolerance!r}"
        )
    if complementarity_tolerance is None or recomputed.complementarity_inf_norm > complementarity_tolerance:
        failures.append(
            f"{label}: recomputed complementarity_inf_norm={recomputed.complementarity_inf_norm:.12g} "
            f"exceeds complementarity_tolerance={complementarity_tolerance!r}"
        )
    return failures


def _audit_seed_task_space_consensus(
    label: str,
    payload: dict[str, Any],
    certificate: dict[str, Any],
) -> list[str]:
    seed = certificate.get("seed_consistency", {})
    if not isinstance(seed, dict) or seed.get("checked") is not True:
        return []
    tolerance = _float_or_none(seed.get("tolerance")) or 1e-7
    seed_results = seed.get("seed_results")
    if not isinstance(seed_results, list):
        capability = payload.get("capability_certificate", {})
        capability_seed = capability.get("seed_consensus", {}) if isinstance(capability, dict) else {}
        seed_results = capability_seed.get("seed_results") if isinstance(capability_seed, dict) else None
    if not isinstance(seed_results, list):
        seed_results = payload.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) <= 1:
        return [f"{label}: seed_consistency checked without multi-seed raw task-space endpoints"]
    endpoints = []
    residuals = []
    failures: list[str] = []
    for index, row in enumerate(seed_results):
        if not isinstance(row, dict):
            failures.append(f"{label}: seed_results[{index}] must be an object")
            continue
        if row.get("accepted") is False:
            continue
        endpoint = _vector(row.get("final_task_vector", row.get("task_space_endpoint", row.get("endpoint", row.get("projected")))))
        residual = _float_or_none(row.get("normalized_residual", row.get("task_residual_norm", row.get("residual_norm"))))
        if endpoint is None:
            failures.append(f"{label}: seed_results[{index}] missing task-space endpoint")
        else:
            endpoints.append(endpoint)
        if residual is not None:
            residuals.append(residual)
    if failures:
        return failures
    if len(endpoints) <= 1:
        return [f"{label}: seed_consistency lacks at least two accepted task-space endpoints"]
    dimensions = {endpoint.size for endpoint in endpoints}
    if len(dimensions) != 1:
        return [f"{label}: seed task-space endpoint dimensions differ"]
    max_endpoint_delta = 0.0
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            max_endpoint_delta = max(max_endpoint_delta, _inf_norm(endpoints[left] - endpoints[right]))
    residual_spread = max(residuals, default=0.0) - min(residuals, default=0.0)
    if residual_spread <= tolerance and max_endpoint_delta > tolerance:
        return [
            f"{label}: seed residuals agree within {tolerance:g} but task-space endpoints diverge "
            f"by {max_endpoint_delta:.12g}"
        ]
    return []


def _audit_continuation_completion(label: str, payload: dict[str, Any]) -> list[str]:
    history = payload.get("continuation_history")
    if history in (None, []):
        return []
    if not isinstance(history, list):
        return [f"{label}: continuation_history must be a list"]
    accepted_alphas = []
    for index, step in enumerate(history):
        if not isinstance(step, dict):
            return [f"{label}: continuation_history[{index}] must be an object"]
        if step.get("accepted") is False:
            continue
        alpha = _float_or_none(step.get("alpha_end"))
        if alpha is None:
            return [f"{label}: continuation_history[{index}].alpha_end missing or nonfinite"]
        accepted_alphas.append(alpha)
    if not accepted_alphas:
        return [f"{label}: continuation_history has no accepted alpha_end"]
    if max(accepted_alphas) < 1.0 - 1e-12:
        return [f"{label}: continuation final alpha < 1 ({max(accepted_alphas):.12g})"]
    return []


def _recompute_complementarity(
    raw: dict[str, Any],
    total_gradient: np.ndarray,
    lower_active: np.ndarray,
    upper_active: np.ndarray,
    fixed: np.ndarray,
) -> float:
    multipliers = raw.get("multipliers")
    if not isinstance(multipliers, list):
        return 0.0
    values = []
    for row in multipliers:
        if not isinstance(row, dict):
            continue
        index = row.get("coordinate_index", row.get("index"))
        value = _float_or_none(row.get("value", row.get("multiplier")))
        side = str(row.get("side", row.get("bound", ""))).lower()
        if not isinstance(index, int) or value is None or index < 0 or index >= total_gradient.size:
            continue
        if side.startswith("lower") and not lower_active[index]:
            values.append(abs(value))
        elif side.startswith("upper") and not upper_active[index]:
            values.append(abs(value))
        elif fixed[index]:
            values.append(abs(value))
    return max(values, default=0.0)


def _vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        return None
    return array


def _matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return np.zeros((0, 0), dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        return None
    return array


def _inf_norm(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.max(np.abs(value)))


def _close(observed: float, expected: float) -> bool:
    if expected == 0.0:
        return abs(observed) <= 1e-18
    return math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-9)


def _projection_threshold(task_name: str, motion_name: str) -> float:
    if motion_name == "neutral":
        return EXACT_PROJECTION_THRESHOLDS["neutral"]
    if "foot" in task_name:
        return EXACT_PROJECTION_THRESHOLDS["foot"]
    if "hand" in task_name:
        return EXACT_PROJECTION_THRESHOLDS["hand"]
    if "torso" in task_name:
        return EXACT_PROJECTION_THRESHOLDS["torso"]
    return EXACT_PROJECTION_THRESHOLDS["default"]


def _git_lfs_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "lfs", "ls-files", "--long"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return []
    paths = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            paths.append(parts[-1])
    return paths


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        head = path.read_bytes()[:256]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1\n")


def _desired_projected_distance(payload: dict[str, Any]) -> float | None:
    desired = payload.get("desired")
    projected = payload.get("projected")
    if not isinstance(desired, list) or not isinstance(projected, list) or len(desired) != len(projected):
        return None
    try:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(desired, projected)))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _walk_json(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = (*path, str(index))
            yield child_path, child
            yield from _walk_json(child, child_path)


def _audit_no_absolute_paths(artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    for path in artifact_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() == ".xml":
            continue
        try:
            text = path.read_text(errors="ignore")
        except UnicodeDecodeError:
            continue
        if _is_forbidden_absolute_path(text):
            failures.append(f"artifact file contains local absolute path: {path}")
    return failures


def _is_forbidden_absolute_path(text: str) -> bool:
    forbidden = ("/mnt/", "/home/", "/Users/", "/private/var/", "/tmp/")
    return any(token in text for token in forbidden)


def _read_reports(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        return {}
    return {path.stem: _read_json(path) for path in sorted(directory.glob("*.json"))}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _display(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
