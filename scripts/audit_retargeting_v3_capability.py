#!/usr/bin/env python3
"""Red-team audit for Step 2.3 capability acceptance artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_MODEL_COUNT = 44
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/retargeting_v3_step2_assets44")
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
    threshold_payload = _read_json_if_exists(numerical_artifact_dir / "threshold_calibration.json")

    failures.extend(audit_threshold_calibration(threshold_payload, check_live_function=False))
    failures.extend(audit_profile_threshold_source(repo_root))
    failures.extend(audit_projection_reports(reports))
    failures.extend(audit_negative_controls(summary, reports))
    failures.extend(audit_deterministic_rerun(deterministic, expected_model_count=EXPECTED_MODEL_COUNT))
    failures.extend(audit_no_robot_id_special_cases(repo_root, manifest_path))
    failures.extend(audit_torso_axis_retained(repo_root, reports))
    failures.extend(audit_lfs_policy(repo_root, lock_path=lock_path, run_fsck=run_lfs_fsck))
    failures.extend(audit_clean_provenance(artifact_dir))
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
    path = repo_root / "soma_retargeter/robotics/v3/profile.py"
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
            failures.append(f"profile.py projection threshold literal changed or missing for {label}")
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
                certificate = payload.get("kkt_certificate")
                if not isinstance(certificate, dict):
                    failures.append(
                        f"{label}: residual over threshold missing kkt_certificate "
                        f"({normalized_float:.6g} > {threshold:.6g})"
                    )
                else:
                    failures.extend(_audit_kkt_certificate(label, certificate))
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


def audit_clean_provenance(artifact_dir: Path) -> list[str]:
    failures: list[str] = []
    environment = _read_json_if_exists(artifact_dir / "environment.json")
    if environment.get("git_status_short") != "":
        failures.append("environment.git_status_short must be clean for accepted artifacts")
    if not environment.get("source_code_commit"):
        failures.append("environment.source_code_commit missing")
    if environment.get("source_code_commit_is_artifact_commit_ancestor") is not True:
        failures.append("environment.source_code_commit_is_artifact_commit_ancestor must be true")
    source_inventory = _read_json_if_exists(artifact_dir / "source_inventory.json")
    for model_id, row in source_inventory.items():
        path = row.get("path") if isinstance(row, dict) else None
        if isinstance(path, str) and _is_forbidden_absolute_path(path):
            failures.append(f"{model_id}: source inventory path leaks local absolute path {path!r}")
    failures.extend(_audit_no_absolute_paths(artifact_dir))
    return failures


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


def _audit_kkt_certificate(label: str, certificate: dict[str, Any]) -> list[str]:
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
    if certificate.get("certified") is not True:
        failures.append(f"{label}: kkt_certificate.certified must be true")
    for field, tolerance_field in (
        ("stationarity_inf_norm", "stationarity_tolerance"),
        ("complementarity_inf_norm", "complementarity_tolerance"),
    ):
        value = _float_or_none(certificate.get(field))
        tolerance = _float_or_none(certificate.get(tolerance_field))
        if value is None or tolerance is None or value > tolerance:
            failures.append(f"{label}: kkt_certificate {field}={value!r} exceeds {tolerance_field}={tolerance!r}")
    if certificate.get("primal_feasible") is not True:
        failures.append(f"{label}: kkt_certificate.primal_feasible must be true")
    if certificate.get("dual_feasible") is not True:
        failures.append(f"{label}: kkt_certificate.dual_feasible must be true")
    task_gradient = _float_or_none(certificate.get("task_gradient_inf_norm"))
    if task_gradient is None or task_gradient <= 0.0:
        failures.append(f"{label}: kkt_certificate.task_gradient_inf_norm must be positive")
    cancellation_ratio = _float_or_none(certificate.get("prior_cancellation_ratio"))
    if cancellation_ratio is None or cancellation_ratio > PRIOR_CANCELLATION_RATIO_MAX:
        failures.append(
            f"{label}: kkt_certificate.prior_cancellation_ratio must be <= "
            f"{PRIOR_CANCELLATION_RATIO_MAX}, got {cancellation_ratio!r}"
        )
    seed = certificate.get("seed_consistency")
    if not isinstance(seed, dict):
        failures.append(f"{label}: kkt_certificate.seed_consistency must be an object")
    else:
        if seed.get("checked") is not True:
            failures.append(f"{label}: kkt_certificate.seed_consistency.checked must be true")
        if seed.get("status") not in {"consistent", "inconsistent_rejected"}:
            failures.append(f"{label}: kkt_certificate.seed_consistency status must reject local-minimum inconsistency")
    return failures


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
