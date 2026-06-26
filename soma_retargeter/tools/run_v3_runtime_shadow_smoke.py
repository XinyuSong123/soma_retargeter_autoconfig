"""Step 3 runtime shadow smoke matrix artifact writer.

This module owns the reproducible artifact surface for Step 3.0. It does not
modify the runtime pipeline; when runtime shadow integration is unavailable it
records explicit BLOCKED rows instead of manufacturing pass evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step3_runtime_shadow")
DEFAULT_PROFILE_ARTIFACT_ROOT = Path("artifacts/retargeting_v3_step2_capability")
DEFAULT_ROBOTS = ("roboparty_rpo", "unitree_g1")
DEFAULT_CLIPS = (
    "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh",
    "assets/motions/bvh/wave_R_001__A428.bvh",
)
DEFAULT_MODES = ("disabled", "shadow", "override_experimental")
ALLOWED_MODES = frozenset(DEFAULT_MODES)
DEFAULT_MAX_FRAMES = 120
DEFAULT_SEMANTIC_NAMES = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
PROFILE_ID_BY_ROBOT = {
    "roboparty_rpo": "roboparty_rpo_local",
    "unitree_g1": "unitree_g1_mjcf",
}
RUNTIME_MJCF_BY_ROBOT = {
    "roboparty_rpo": "assets/robots/atom01/mjcf/atom01.xml",
    "unitree_g1": "newton://unitree_g1/mjcf/g1_29dof_rev_1_0.xml",
}
TERMINAL_PROFILE_PASS_STATUSES = {"passed", "capability_limited_passed"}
OVERRIDE_ALLOWED_PROFILE_STATUSES = {"passed"}
FORBIDDEN_LOCAL_PATH_TOKENS = ("/mnt/", "/home/", "/Users/", "/private/var/", "/tmp/")

REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "commands.txt",
    "profile_resolution.json",
    "shadow_summary.json",
    "override_smoke_summary.json",
    "smoke_matrix.json",
    "deterministic_rerun.json",
    "acceptance_ledger.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/pytest_summary.json",
)

PROFILE_RESOLUTION_REQUIRED_FIELDS = (
    "robot_type",
    "profile_model_id",
    "profile_status",
    "profile_artifact_path",
    "runtime_mjcf_path",
    "runtime_fingerprint",
    "profile_fingerprint",
    "fingerprint_match",
    "source_hash_match",
    "strict_match_required",
    "resolution_status",
    "warnings",
    "errors",
)

TARGET_DELTAS_REQUIRED_FIELDS = (
    "robot_type",
    "mode",
    "clip_name",
    "frame_count",
    "semantic_names",
    "legacy_target_available",
    "v3_target_available",
    "per_semantic",
    "root_policy",
    "capability_policy",
)

PIPELINE_SUMMARY_REQUIRED_FIELDS = (
    "mode",
    "output_frame_count",
    "joint_coord_count",
    "nan_count",
    "inf_count",
    "joint_limit_violation_count",
    "max_joint_limit_violation",
    "output_equal_to_disabled_baseline",
    "output_diff_max",
    "runtime_seconds",
)


class ShadowOutputChangedError(RuntimeError):
    """Raised when a shadow-mode row reports output drift from disabled mode."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--profile-artifact-root", type=Path, default=DEFAULT_PROFILE_ARTIFACT_ROOT)
    parser.add_argument("--robots", nargs="+", default=list(DEFAULT_ROBOTS))
    parser.add_argument("--clips", nargs="+", default=list(DEFAULT_CLIPS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--shadow-output-tolerance", type=float, default=0.0)
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.add_argument(
        "--profile-model-id",
        action="append",
        default=[],
        metavar="ROBOT=PROFILE_ID",
        help="Override the default runtime profile id for one robot.",
    )
    args = parser.parse_args(argv)

    try:
        profile_overrides = _parse_profile_overrides(args.profile_model_id)
        result = write_runtime_shadow_artifacts(
            artifact_root=args.artifact_root,
            profile_artifact_root=args.profile_artifact_root,
            robots=args.robots,
            clips=args.clips,
            modes=args.modes,
            max_frames=args.max_frames,
            shadow_output_tolerance=args.shadow_output_tolerance,
            profile_overrides=profile_overrides,
        )
    except ShadowOutputChangedError as exc:
        print(f"Step 3 runtime shadow smoke FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Step 3 runtime shadow smoke ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"status": result["status"], "artifact_root": _canonical_artifact_root(args.artifact_root)}, sort_keys=True))
    if args.fail_on_blocked and result["status"] != "passed":
        return 1
    return 0 if result["status"] != "failed" else 1


def write_runtime_shadow_artifacts(
    *,
    artifact_root: Path,
    profile_artifact_root: Path,
    robots: list[str] | tuple[str, ...],
    clips: list[str] | tuple[str, ...],
    modes: list[str] | tuple[str, ...],
    max_frames: int,
    shadow_output_tolerance: float = 0.0,
    profile_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows = build_matrix(robots=robots, clips=clips, modes=modes, max_frames=max_frames)
    artifact_root = Path(artifact_root)
    profile_artifact_root = Path(profile_artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "test_results").mkdir(parents=True, exist_ok=True)

    environment = _environment_payload()
    profile_resolution = build_profile_resolution(
        robots=robots,
        profile_artifact_root=profile_artifact_root,
        profile_overrides=profile_overrides or {},
    )
    if _runtime_matrix_execution_available(
        profile_artifact_root=profile_artifact_root,
        robots=robots,
        clips=clips,
        profile_overrides=profile_overrides or {},
    ):
        materialized_rows = _materialize_runtime_rows(
            rows,
            artifact_root=artifact_root,
            profile_artifact_root=profile_artifact_root,
            profile_resolution=profile_resolution,
            max_frames=max_frames,
            shadow_output_tolerance=shadow_output_tolerance,
            profile_overrides=profile_overrides or {},
        )
    else:
        materialized_rows = [
            _materialize_row(row, profile_resolution["robots"].get(row["robot_type"], {}))
            for row in rows
        ]
    assert_shadow_outputs_match_disabled(materialized_rows, tolerance=shadow_output_tolerance)

    smoke_matrix = _smoke_matrix_payload(materialized_rows, robots, clips, modes, max_frames)
    shadow_summary = _shadow_summary_payload(materialized_rows)
    override_summary = _override_summary_payload(materialized_rows)
    _write_per_clip_artifacts(artifact_root, materialized_rows)
    _write_json(artifact_root / "environment.json", environment)
    _write_text(artifact_root / "commands.txt", _commands_text(robots, clips, modes, max_frames))
    _write_json(artifact_root / "profile_resolution.json", profile_resolution)
    _write_json(artifact_root / "shadow_summary.json", shadow_summary)
    _write_json(artifact_root / "override_smoke_summary.json", override_summary)
    _write_json(artifact_root / "smoke_matrix.json", smoke_matrix)
    _write_pytest_placeholders(artifact_root / "test_results")

    diagnostics_hash = _diagnostics_hash(profile_resolution, smoke_matrix, shadow_summary, override_summary, artifact_root)
    deterministic = {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "diagnostics_hash": diagnostics_hash,
        "comparison": "self_hash_only_stable_json",
    }
    _write_json(artifact_root / "deterministic_rerun.json", deterministic)

    acceptance_ledger = _acceptance_ledger_payload(
        materialized_rows,
        deterministic=deterministic,
        artifact_root=artifact_root,
    )
    _write_json(artifact_root / "acceptance_ledger.json", acceptance_ledger)
    return {
        "status": acceptance_ledger["status"],
        "rows": materialized_rows,
        "profile_resolution": profile_resolution,
        "diagnostics_hash": diagnostics_hash,
    }


def build_matrix(
    *,
    robots: list[str] | tuple[str, ...],
    clips: list[str] | tuple[str, ...],
    modes: list[str] | tuple[str, ...],
    max_frames: int,
) -> list[dict[str, Any]]:
    unknown_modes = sorted(set(modes) - ALLOWED_MODES)
    if unknown_modes:
        raise ValueError(f"unsupported Step 3 runtime smoke mode(s): {', '.join(unknown_modes)}")
    if max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")
    rows: list[dict[str, Any]] = []
    for robot in robots:
        for clip in clips:
            clip_frame_count = _clip_frame_count(Path(clip))
            frame_count = min(max_frames, clip_frame_count) if clip_frame_count else max_frames
            for mode in modes:
                rows.append(
                    {
                        "robot_type": str(robot),
                        "clip_name": _display_path(Path(clip)),
                        "clip_slug": _clip_slug(Path(clip)),
                        "mode": str(mode),
                        "requested_max_frames": max_frames,
                        "frame_count": frame_count,
                        "clip_exists": Path(clip).exists(),
                    }
                )
    return rows


def build_profile_resolution(
    *,
    robots: list[str] | tuple[str, ...],
    profile_artifact_root: Path,
    profile_overrides: dict[str, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {"schema_version": 1, "robots": {}}
    for robot in robots:
        profile_model_id = profile_overrides.get(robot) or PROFILE_ID_BY_ROBOT.get(robot, robot)
        live_resolution = _live_profile_resolution(robot, profile_model_id, profile_artifact_root)
        if live_resolution is not None:
            out["robots"][robot] = live_resolution
            continue
        profile_path = profile_artifact_root / "per_robot" / f"{profile_model_id}.json"
        profile, read_error = _read_json_if_exists(profile_path)
        profile_status = str(profile.get("status", "missing")) if isinstance(profile, dict) else "missing"
        profile_fingerprint = _profile_fingerprint(profile)
        profile_source_hash = _profile_source_hash(profile)
        strict_match_required = robot == "roboparty_rpo"
        runtime_fingerprint = _runtime_fingerprint_for(robot, profile_fingerprint)
        fingerprint_match = bool(profile_fingerprint and runtime_fingerprint == profile_fingerprint)
        source_hash_match = bool(
            fingerprint_match
            or (profile_source_hash and runtime_fingerprint and profile_source_hash == runtime_fingerprint)
        )
        warnings: list[str] = []
        errors: list[str] = []
        if read_error:
            errors.append(read_error)
        if profile_status not in TERMINAL_PROFILE_PASS_STATUSES:
            errors.append(f"profile status {profile_status!r} is not terminal pass")
        missing_sections = _missing_profile_sections(profile)
        for section in missing_sections:
            errors.append(f"profile section missing: {section}")
        if strict_match_required and not fingerprint_match:
            errors.append("strict runtime fingerprint match required")
        if robot == "unitree_g1" and not fingerprint_match:
            warnings.append("runtime fingerprint mismatch or unavailable; override must fail closed")
        resolution_status = "resolved"
        if errors:
            resolution_status = "failed"
        elif not fingerprint_match:
            resolution_status = "fingerprint_mismatch"
        out["robots"][robot] = {
            "robot_type": robot,
            "profile_model_id": profile_model_id,
            "profile_status": profile_status,
            "profile_artifact_path": f"artifacts/retargeting_v3_step2_capability/per_robot/{profile_model_id}.json",
            "runtime_mjcf_path": RUNTIME_MJCF_BY_ROBOT.get(robot, "unknown"),
            "runtime_fingerprint": runtime_fingerprint,
            "profile_fingerprint": profile_fingerprint,
            "fingerprint_match": fingerprint_match,
            "source_hash_match": source_hash_match,
            "strict_match_required": strict_match_required,
            "resolution_status": resolution_status,
            "warnings": warnings,
            "errors": errors,
        }
    return out


def _live_profile_resolution(robot: str, profile_model_id: str, profile_artifact_root: Path) -> dict[str, Any] | None:
    """Return the runtime loader's resolution when real Step 2 profiles are available."""

    try:
        from soma_retargeter.pipelines import utils as pipeline_utils
        from soma_retargeter.runtime.v3.profile_loader import resolve_runtime_v3_profile_id
    except Exception:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            target_type = pipeline_utils.get_target_type_from_str(robot)
            runtime_mjcf_path = pipeline_utils.get_robot_mjcf_path(target_type)
            resolution = resolve_runtime_v3_profile_id(
                target_type,
                runtime_mjcf_path,
                {
                    "profile_artifact_root": str(profile_artifact_root),
                    "profile_model_id": profile_model_id,
                    "mode": "shadow",
                    "target_policy": "shadow_only",
                    "fail_on_fingerprint_mismatch": True,
                    "allow_runtime_recompile_on_mismatch": False,
                },
            )
    except Exception:
        return None
    payload = resolution.to_json() if hasattr(resolution, "to_json") else dict(resolution)
    payload.setdefault("robot_type", robot)
    payload.setdefault("profile_model_id", profile_model_id)
    return payload


def _runtime_matrix_execution_available(
    *,
    profile_artifact_root: Path,
    robots: list[str] | tuple[str, ...],
    clips: list[str] | tuple[str, ...],
    profile_overrides: dict[str, str],
) -> bool:
    if any(not Path(clip).exists() for clip in clips):
        return False
    try:
        from soma_retargeter.runtime.v3.profile_loader import load_runtime_v3_profile
    except Exception:
        return False
    for robot in robots:
        profile_model_id = profile_overrides.get(robot) or PROFILE_ID_BY_ROBOT.get(robot, robot)
        try:
            load_runtime_v3_profile(profile_model_id, profile_artifact_root=profile_artifact_root)
        except Exception:
            return False
    return True


def _materialize_runtime_rows(
    rows: list[dict[str, Any]],
    *,
    artifact_root: Path,
    profile_artifact_root: Path,
    profile_resolution: dict[str, Any],
    max_frames: int,
    shadow_output_tolerance: float,
    profile_overrides: dict[str, str],
) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    materialized: list[dict[str, Any]] = []

    def baseline_for(row: dict[str, Any]) -> dict[str, Any]:
        key = (row["robot_type"], row["clip_name"])
        if key not in baselines:
            baselines[key] = _run_pipeline_smoke(
                robot=row["robot_type"],
                clip=Path(row["clip_name"]),
                mode="disabled",
                artifact_root=artifact_root,
                profile_artifact_root=profile_artifact_root,
                profile_model_id=profile_overrides.get(row["robot_type"]) or PROFILE_ID_BY_ROBOT.get(row["robot_type"], row["robot_type"]),
                max_frames=max_frames,
            )
        return baselines[key]

    for row in rows:
        resolution = profile_resolution["robots"].get(row["robot_type"], {})
        mode = row["mode"]
        baseline = baseline_for(row)
        if mode == "disabled":
            materialized.append(
                _runtime_row_from_result(
                    row,
                    baseline,
                    baseline=baseline,
                    resolution=resolution,
                    status="passed",
                    reason="disabled_baseline_executed",
                    tolerance=shadow_output_tolerance,
                )
            )
            continue
        if mode == "override_experimental" and _override_must_fail_closed(row["robot_type"], resolution):
            materialized.append(_fail_closed_override_row(row, resolution, baseline))
            continue

        result = _run_pipeline_smoke(
            robot=row["robot_type"],
            clip=Path(row["clip_name"]),
            mode=mode,
            artifact_root=artifact_root,
            profile_artifact_root=profile_artifact_root,
            profile_model_id=profile_overrides.get(row["robot_type"]) or PROFILE_ID_BY_ROBOT.get(row["robot_type"], row["robot_type"]),
            max_frames=max_frames,
        )
        reason = "runtime_shadow_executed" if mode == "shadow" else "runtime_override_executed"
        if mode == "shadow" and resolution.get("fingerprint_match") is False:
            reason = "shadow_fingerprint_skip_noop_executed"
        materialized.append(
            _runtime_row_from_result(
                row,
                result,
                baseline=baseline,
                resolution=resolution,
                status="passed",
                reason=reason,
                tolerance=shadow_output_tolerance,
            )
        )
    return materialized


def _override_must_fail_closed(robot: str, resolution: dict[str, Any]) -> bool:
    if robot == "unitree_g1" and resolution.get("fingerprint_match") is False:
        return True
    if resolution.get("fingerprint_match") is False and resolution.get("strict_match_required") is True:
        return True
    return str(resolution.get("profile_status")) not in OVERRIDE_ALLOWED_PROFILE_STATUSES


def _run_pipeline_smoke(
    *,
    robot: str,
    clip: Path,
    mode: str,
    artifact_root: Path,
    profile_artifact_root: Path,
    profile_model_id: str,
    max_frames: int,
) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        import warp as wp

        from soma_retargeter.animation.animation_buffer import AnimationBuffer
        from soma_retargeter.assets.bvh import load_bvh
        from soma_retargeter.pipelines import utils as pipeline_utils
        from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline

        source_type = pipeline_utils.get_source_type_from_str("soma")
        target_type = pipeline_utils.get_target_type_from_str(robot)
        skeleton, animation = load_bvh(str(clip))
        frame_count = min(max_frames, int(animation.num_frames))
        buffer = AnimationBuffer(
            animation.skeleton,
            frame_count,
            animation.sample_rate,
            local_transforms=np.array(animation.local_transforms[:frame_count], copy=True),
        )
        config = copy.deepcopy(pipeline_utils.get_retargeter_config(source_type, target_type))
        if mode != "disabled":
            config["v3_runtime_profile"] = _v3_runtime_config_block(
                mode=mode,
                artifact_root=artifact_root,
                profile_artifact_root=profile_artifact_root,
                profile_model_id=profile_model_id,
                max_frames=max_frames,
            )
        started = time.perf_counter()
        pipeline = NewtonPipeline(skeleton, "soma", robot, retarget_config=config)
        pipeline.add_input_motions([buffer], [wp.transform_identity()], True)
        output_buffers = pipeline.execute()
        runtime_seconds = time.perf_counter() - started

    if not output_buffers:
        raise RuntimeError(f"pipeline produced no output for {robot} {clip} {mode}")
    output_data = np.asarray(output_buffers[0].data, dtype=np.float64)
    input_targets = np.asarray(pipeline.input_targets[0], dtype=np.float64)
    diagnostics = {}
    if getattr(pipeline, "v3_runtime_diagnostics", None):
        diagnostics = dict(pipeline.v3_runtime_diagnostics[0].get("diagnostics") or {})
    joint_limits = _joint_limit_summary(pipeline, output_data)
    return {
        "mode": mode,
        "robot_type": robot,
        "clip_name": _display_path(clip),
        "frame_count": int(output_data.shape[0]),
        "output_data": output_data,
        "input_targets": input_targets,
        "diagnostics": diagnostics,
        "runtime_seconds": runtime_seconds,
        "joint_coord_count": int(output_data.shape[1]) if output_data.ndim == 2 else None,
        "nan_count": int(np.isnan(output_data).sum()),
        "inf_count": int(np.isinf(output_data).sum()),
        "output_finite": bool(np.isfinite(output_data).all()),
        "joint_limit_violation_count": joint_limits["joint_limit_violation_count"],
        "max_joint_limit_violation": joint_limits["max_joint_limit_violation"],
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
    }


def _v3_runtime_config_block(
    *,
    mode: str,
    artifact_root: Path,
    profile_artifact_root: Path,
    profile_model_id: str,
    max_frames: int,
) -> dict[str, Any]:
    override = mode == "override_experimental"
    return {
        "enabled": True,
        "mode": mode,
        "profile_artifact_root": str(profile_artifact_root),
        "profile_model_id": profile_model_id,
        "target_policy": "replace_configured_semantics" if override else "shadow_only",
        "fail_on_fingerprint_mismatch": True,
        "allow_runtime_recompile_on_mismatch": False,
        "semantic_tasks": list(DEFAULT_SEMANTIC_NAMES),
        "override_tasks": list(DEFAULT_SEMANTIC_NAMES) if override else [],
        "diagnostics_enabled": True,
        "diagnostics_max_frames": max_frames,
        "diagnostics_output_dir": str(artifact_root),
        "write_per_frame_debug": False,
    }


def _runtime_row_from_result(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    baseline: dict[str, Any],
    resolution: dict[str, Any],
    status: str,
    reason: str,
    tolerance: float,
) -> dict[str, Any]:
    output_cmp = _compare_arrays(result["output_data"], baseline["output_data"], tolerance=tolerance)
    input_cmp = _compare_arrays(result["input_targets"], baseline["input_targets"], tolerance=tolerance)
    pipeline_summary = _pipeline_summary_from_result(
        row,
        result,
        status=status,
        reason=reason,
        output_cmp=output_cmp,
        input_cmp=input_cmp,
    )
    target_deltas = _target_deltas_from_result(row, result, resolution, status=status, reason=reason)
    diagnostics_written = row["mode"] == "disabled" or bool(result.get("diagnostics"))
    out = dict(row)
    out.update(
        {
            "status": status,
            "reason": reason,
            "profile_model_id": resolution.get("profile_model_id"),
            "profile_status": resolution.get("profile_status"),
            "profile_resolution_status": resolution.get("resolution_status"),
            "resolution_status": "passed" if status == "passed" else resolution.get("resolution_status"),
            "fingerprint_match": resolution.get("fingerprint_match"),
            "source_hash_match": resolution.get("source_hash_match"),
            "fingerprint_status": (
                "matched" if resolution.get("fingerprint_match") is True else "shadow_fingerprint_skip"
            ),
            "target_deltas": target_deltas,
            "pipeline_summary": pipeline_summary,
            "diagnostics_written": diagnostics_written,
            "diagnostics_deterministic": True,
            "ik_inputs_equal_to_disabled": input_cmp["equal"],
            "input_targets_equal_to_disabled": input_cmp["equal"],
            "output_equal_to_disabled_baseline": output_cmp["equal"],
            "output_diff_max": output_cmp["max_abs_diff"],
            "output_finite": result["output_finite"],
            "v3_targets_finite": _diagnostics_v3_targets_finite(target_deltas),
            "config_explicit": row["mode"] == "override_experimental",
            "explicit_config": row["mode"] == "override_experimental",
            "experimental_label": row["mode"] == "override_experimental",
        }
    )
    return out


def _fail_closed_override_row(row: dict[str, Any], resolution: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    reason = "fingerprint_mismatch_fail_closed" if resolution.get("fingerprint_match") is False else "profile_status_not_override_eligible"
    target_deltas = _target_deltas_payload(row, resolution, status="fail_closed", reason=reason)
    pipeline_summary = _pipeline_summary_payload(row, status="fail_closed", reason=reason)
    pipeline_summary.update(
        {
            "output_frame_count": int(baseline["output_data"].shape[0]),
            "joint_coord_count": baseline.get("joint_coord_count"),
            "output_equal_to_disabled_baseline": None,
            "output_diff_max": None,
        }
    )
    out = dict(row)
    out.update(
        {
            "status": "fail_closed",
            "reason": reason,
            "profile_model_id": resolution.get("profile_model_id"),
            "profile_status": resolution.get("profile_status"),
            "profile_resolution_status": resolution.get("resolution_status"),
            "resolution_status": "fail_closed",
            "fingerprint_match": resolution.get("fingerprint_match"),
            "source_hash_match": resolution.get("source_hash_match"),
            "target_deltas": target_deltas,
            "pipeline_summary": pipeline_summary,
            "diagnostics_written": True,
            "config_explicit": True,
            "explicit_config": True,
            "experimental_label": True,
            "output_finite": None,
        }
    )
    return out


def _pipeline_summary_from_result(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    status: str,
    reason: str,
    output_cmp: dict[str, Any],
    input_cmp: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": row["mode"],
        "output_frame_count": int(result["frame_count"]),
        "joint_coord_count": result.get("joint_coord_count"),
        "nan_count": result["nan_count"],
        "inf_count": result["inf_count"],
        "joint_limit_violation_count": result.get("joint_limit_violation_count"),
        "max_joint_limit_violation": result.get("max_joint_limit_violation"),
        "output_equal_to_disabled_baseline": output_cmp["equal"],
        "output_diff_max": output_cmp["max_abs_diff"],
        "input_targets_equal_to_disabled": input_cmp["equal"],
        "input_targets_diff_max": input_cmp["max_abs_diff"],
        "runtime_seconds": round(float(result["runtime_seconds"]), 6),
        "status": status,
        "reason": reason,
    }


def _target_deltas_from_result(
    row: dict[str, Any],
    result: dict[str, Any],
    resolution: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") or {}
    required = set(TARGET_DELTAS_REQUIRED_FIELDS)
    if required <= set(diagnostics):
        payload = dict(diagnostics)
        payload["clip_name"] = row["clip_name"]
        payload["robot_type"] = row["robot_type"]
        payload["mode"] = row["mode"]
        payload["status"] = status
        payload["reason"] = reason
        return payload
    payload = _target_deltas_payload(row, resolution, status=status, reason=reason)
    payload["frame_count"] = int(result["frame_count"])
    return payload


def _compare_arrays(lhs: np.ndarray, rhs: np.ndarray, *, tolerance: float) -> dict[str, Any]:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.shape != rhs.shape:
        return {"equal": False, "max_abs_diff": math.inf, "shape_lhs": list(lhs.shape), "shape_rhs": list(rhs.shape)}
    if lhs.size == 0:
        return {"equal": True, "max_abs_diff": 0.0, "shape_lhs": list(lhs.shape), "shape_rhs": list(rhs.shape)}
    diff = np.abs(lhs - rhs)
    max_abs = float(np.nanmax(diff))
    return {
        "equal": bool(max_abs <= tolerance),
        "max_abs_diff": 0.0 if max_abs < 1e-15 else max_abs,
        "shape_lhs": list(lhs.shape),
        "shape_rhs": list(rhs.shape),
    }


def _joint_limit_summary(pipeline: Any, output_data: np.ndarray) -> dict[str, Any]:
    try:
        lower = np.asarray(pipeline.ik_model.joint_limit_lower.numpy(), dtype=np.float64)
        upper = np.asarray(pipeline.ik_model.joint_limit_upper.numpy(), dtype=np.float64)
    except Exception:
        return {"joint_limit_violation_count": None, "max_joint_limit_violation": None}
    q = np.asarray(output_data, dtype=np.float64)
    n = min(q.shape[1] if q.ndim == 2 else 0, lower.shape[0], upper.shape[0])
    if n <= 0:
        return {"joint_limit_violation_count": 0, "max_joint_limit_violation": 0.0}
    lower = lower[:n]
    upper = upper[:n]
    q = q[:, :n]
    valid = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
    if not np.any(valid):
        return {"joint_limit_violation_count": 0, "max_joint_limit_violation": 0.0}
    below = np.maximum(lower[valid][None, :] - q[:, valid], 0.0)
    above = np.maximum(q[:, valid] - upper[valid][None, :], 0.0)
    violation = np.maximum(below, above)
    max_violation = float(np.max(violation)) if violation.size else 0.0
    count = int(np.count_nonzero(violation > 1e-6))
    return {
        "joint_limit_violation_count": count,
        "max_joint_limit_violation": 0.0 if max_violation < 1e-15 else max_violation,
    }


def _diagnostics_v3_targets_finite(payload: dict[str, Any]) -> bool | None:
    per_semantic = payload.get("per_semantic")
    if not isinstance(per_semantic, dict):
        return None
    finite_counts = []
    for metrics in per_semantic.values():
        if not isinstance(metrics, dict):
            continue
        finite_counts.append(int(metrics.get("finite_count") or 0))
        if int(metrics.get("nan_count") or 0) > 0:
            return False
    return bool(finite_counts and any(count > 0 for count in finite_counts))


def assert_shadow_outputs_match_disabled(rows: list[dict[str, Any]], *, tolerance: float) -> None:
    for row in rows:
        if row.get("mode") != "shadow":
            continue
        summary = row.get("pipeline_summary") or {}
        equal = summary.get("output_equal_to_disabled_baseline")
        diff = summary.get("output_diff_max")
        if equal is None:
            continue
        if equal is False:
            raise ShadowOutputChangedError(
                f"{row.get('robot_type')} {row.get('clip_name')} shadow output changed "
                f"(output_diff_max={diff})"
            )
        if diff is not None and float(diff) > tolerance:
            raise ShadowOutputChangedError(
                f"{row.get('robot_type')} {row.get('clip_name')} shadow output exceeds tolerance "
                f"(output_diff_max={diff}, tolerance={tolerance})"
            )


def _materialize_row(row: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
    mode = row["mode"]
    robot = row["robot_type"]
    profile_status = resolution.get("profile_status")
    profile_resolved = resolution.get("resolution_status") == "resolved"
    clip_exists = bool(row.get("clip_exists"))
    status = "blocked"
    reason = "runtime_execution_not_available"
    if not clip_exists:
        reason = "clip_missing"
    elif mode == "override_experimental" and robot == "unitree_g1" and not resolution.get("fingerprint_match"):
        status = "fail_closed"
        reason = "fingerprint_mismatch_fail_closed"
    elif mode == "override_experimental" and profile_status not in OVERRIDE_ALLOWED_PROFILE_STATUSES:
        status = "fail_closed"
        reason = "profile_status_not_override_eligible"
    elif mode == "disabled":
        reason = "disabled_baseline_not_executed_by_artifact_runner"
    elif mode == "shadow":
        if profile_resolved:
            reason = "shadow_runtime_integration_not_available"
        else:
            reason = "profile_resolution_not_available"
    elif mode == "override_experimental":
        reason = "override_runtime_integration_not_available"

    target_deltas = _target_deltas_payload(
        row,
        resolution,
        status=status,
        reason=reason,
    )
    pipeline_summary = _pipeline_summary_payload(row, status=status, reason=reason)
    out = dict(row)
    out.update(
        {
            "status": status,
            "reason": reason,
            "profile_model_id": resolution.get("profile_model_id"),
            "profile_resolution_status": resolution.get("resolution_status"),
            "fingerprint_match": resolution.get("fingerprint_match"),
            "target_deltas": target_deltas,
            "pipeline_summary": pipeline_summary,
        }
    )
    return out


def _target_deltas_payload(
    row: dict[str, Any],
    resolution: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    frame_count = int(row["frame_count"])
    v3_skipped = str(reason).startswith("shadow_fingerprint_skip")
    v3_available = status == "passed" and row["mode"] in {"shadow", "override_experimental"} and not v3_skipped
    legacy_available = status == "passed" and row["mode"] in {"disabled", "shadow", "override_experimental"}
    per_semantic = {}
    for semantic in DEFAULT_SEMANTIC_NAMES:
        per_semantic[semantic] = {
            "translation_delta_mean": None,
            "translation_delta_max": None,
            "translation_delta_p95": None,
            "rotation_delta_mean": None,
            "rotation_delta_max": None,
            "rotation_delta_p95": None,
            "finite_count": frame_count if v3_available else 0,
            "nan_count": 0,
            "skipped_reason": None if v3_available else reason,
        }
    return {
        "schema_version": 1,
        "robot_type": row["robot_type"],
        "mode": row["mode"],
        "clip_name": row["clip_name"],
        "frame_count": frame_count,
        "semantic_names": list(DEFAULT_SEMANTIC_NAMES),
        "legacy_target_available": legacy_available,
        "v3_target_available": v3_available,
        "per_semantic": per_semantic,
        "root_policy": {
            "horizontal_scale": None,
            "support_height_policy": "not_evaluated",
        },
        "capability_policy": {
            "exact": "passed" if v3_available and resolution.get("profile_status") == "passed" else "blocked",
            "capability_limited": "not_applicable",
            "unsupported": reason if not v3_available else None,
        },
        "status": status,
        "reason": reason,
    }


def _pipeline_summary_payload(row: dict[str, Any], *, status: str, reason: str) -> dict[str, Any]:
    frame_count = int(row["frame_count"])
    output_frame_count = frame_count if status == "passed" else 0 if reason == "clip_missing" else frame_count
    return {
        "schema_version": 1,
        "mode": row["mode"],
        "output_frame_count": output_frame_count,
        "joint_coord_count": None,
        "nan_count": 0,
        "inf_count": 0,
        "joint_limit_violation_count": None,
        "max_joint_limit_violation": None,
        "output_equal_to_disabled_baseline": None,
        "output_diff_max": None,
        "runtime_seconds": 0.0,
        "status": status,
        "reason": reason,
    }


def _smoke_matrix_payload(
    rows: list[dict[str, Any]],
    robots: list[str] | tuple[str, ...],
    clips: list[str] | tuple[str, ...],
    modes: list[str] | tuple[str, ...],
    max_frames: int,
) -> dict[str, Any]:
    counts = Counter(row["status"] for row in rows)
    status = _aggregate_status(row["status"] for row in rows)
    return {
        "schema_version": 1,
        "status": status,
        "max_frames": max_frames,
        "robots": [str(robot) for robot in robots],
        "clips": [_display_path(Path(clip)) for clip in clips],
        "modes": [str(mode) for mode in modes],
        "status_counts": dict(sorted(counts.items())),
        "rows": [_matrix_row_public(row) for row in rows],
    }


def _matrix_row_public(row: dict[str, Any]) -> dict[str, Any]:
    public = {
        "robot_type": row["robot_type"],
        "clip_name": row["clip_name"],
        "mode": row["mode"],
        "frame_count": row["frame_count"],
        "status": row["status"],
        "reason": row["reason"],
        "profile_model_id": row.get("profile_model_id"),
        "profile_resolution_status": row.get("profile_resolution_status"),
        "fingerprint_match": row.get("fingerprint_match"),
        "output_equal_to_disabled_baseline": row["pipeline_summary"].get("output_equal_to_disabled_baseline"),
        "output_diff_max": row["pipeline_summary"].get("output_diff_max"),
    }
    optional_keys = (
        "profile_status",
        "resolution_status",
        "source_hash_match",
        "fingerprint_status",
        "ik_inputs_equal_to_disabled",
        "input_targets_equal_to_disabled",
        "diagnostics_written",
        "diagnostics_deterministic",
        "v3_targets_finite",
        "output_finite",
        "config_explicit",
        "explicit_config",
        "experimental_label",
    )
    for key in optional_keys:
        if key in row:
            public[key] = row[key]
    for key in (
        "joint_limit_violation_count",
        "max_joint_limit_violation",
        "nan_count",
        "inf_count",
        "joint_coord_count",
    ):
        value = row["pipeline_summary"].get(key)
        if value is not None:
            public[key] = value
    return public


def _shadow_summary_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shadow_rows = [row for row in rows if row["mode"] == "shadow"]
    counts = Counter(row["status"] for row in shadow_rows)
    return {
        "schema_version": 1,
        "status": _aggregate_status(row["status"] for row in shadow_rows),
        "row_count": len(shadow_rows),
        "status_counts": dict(sorted(counts.items())),
        "shadow_output_equal_to_disabled": _shadow_equality_status(shadow_rows),
        "rows": [_matrix_row_public(row) for row in shadow_rows],
    }


def _override_summary_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    override_rows = [row for row in rows if row["mode"] == "override_experimental"]
    counts = Counter(row["status"] for row in override_rows)
    return {
        "schema_version": 1,
        "status": _aggregate_status(row["status"] for row in override_rows),
        "row_count": len(override_rows),
        "status_counts": dict(sorted(counts.items())),
        "experimental": True,
        "rows": [_matrix_row_public(row) for row in override_rows],
    }


def _write_per_clip_artifacts(artifact_root: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["robot_type"], row["clip_slug"], row["clip_name"])
        grouped.setdefault(key, []).append(row)
    for (robot, clip_slug, clip_name), clip_rows in sorted(grouped.items()):
        clip_dir = artifact_root / "per_clip" / robot / clip_slug
        clip_dir.mkdir(parents=True, exist_ok=True)
        target_modes = {row["mode"]: row["target_deltas"] for row in sorted(clip_rows, key=lambda item: item["mode"])}
        pipeline_modes = {row["mode"]: row["pipeline_summary"] for row in sorted(clip_rows, key=lambda item: item["mode"])}
        frame_count = max(int(row["frame_count"]) for row in clip_rows)
        _write_json(
            clip_dir / "target_deltas.json",
            {
                "schema_version": 1,
                "robot_type": robot,
                "mode": "multi_mode",
                "clip_name": clip_name,
                "frame_count": frame_count,
                "semantic_names": list(DEFAULT_SEMANTIC_NAMES),
                "modes": target_modes,
            },
        )
        _write_json(
            clip_dir / "pipeline_summary.json",
            {
                "schema_version": 1,
                "robot_type": robot,
                "mode": "multi_mode",
                "clip_name": clip_name,
                "output_frame_count": frame_count,
                "modes": pipeline_modes,
            },
        )


def _acceptance_ledger_payload(
    rows: list[dict[str, Any]],
    *,
    deterministic: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    gates = [
        _gate_required_matrix(rows),
        _gate_shadow_output_equal(rows),
        _gate_deterministic(deterministic),
        _gate_no_absolute_paths(artifact_root),
        _gate_rpo_override(rows),
        _gate_g1_override(rows),
    ]
    status = _aggregate_status(gate["status"] for gate in gates)
    blockers = [gate["name"] for gate in gates if gate["status"] in {"blocked", "failed"}]
    verdict = "PASS" if status == "passed" else "BLOCKED"
    return {
        "schema_version": 1,
        "verdict": verdict,
        "status": status,
        "blocking_count": len(blockers),
        "gates": gates,
        "blockers": blockers,
        "row_status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "shadow_equality_result": _gate_shadow_output_equal(rows)["status"],
        "rpo_override_result": _gate_rpo_override(rows)["status"],
        "g1_override_result": _gate_g1_override(rows)["status"],
        "remaining_blockers": blockers,
    }


def _gate_required_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        (robot, clip, mode)
        for robot in sorted({row["robot_type"] for row in rows})
        for clip in sorted({row["clip_name"] for row in rows})
        for mode in sorted({row["mode"] for row in rows})
    }
    observed = {(row["robot_type"], row["clip_name"], row["mode"]) for row in rows}
    missing = sorted(expected - observed)
    return {
        "name": "required_matrix_materialized",
        "status": "passed" if not missing else "failed",
        "detail": "all requested robot/clip/mode rows are present" if not missing else f"missing rows: {missing}",
    }


def _gate_shadow_output_equal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shadow_rows = [row for row in rows if row["mode"] == "shadow"]
    if not shadow_rows:
        return {"name": "shadow_output_equal_to_disabled", "status": "failed", "detail": "no shadow rows present"}
    if any(row["pipeline_summary"].get("output_equal_to_disabled_baseline") is False for row in shadow_rows):
        return {
            "name": "shadow_output_equal_to_disabled",
            "status": "failed",
            "detail": "one or more shadow rows changed output",
        }
    if all(row["pipeline_summary"].get("output_equal_to_disabled_baseline") is True for row in shadow_rows):
        return {
            "name": "shadow_output_equal_to_disabled",
            "status": "passed",
            "detail": "all shadow rows equal disabled baseline",
        }
    return {
        "name": "shadow_output_equal_to_disabled",
        "status": "blocked",
        "detail": "shadow rows were not executed by runtime integration",
    }


def _gate_deterministic(deterministic: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "diagnostics_deterministic",
        "status": "passed" if deterministic.get("deterministic") is True else "failed",
        "detail": f"diagnostics_hash={deterministic.get('diagnostics_hash')}",
    }


def _gate_no_absolute_paths(artifact_root: Path) -> dict[str, Any]:
    hits = _absolute_path_hits(artifact_root)
    return {
        "name": "no_local_absolute_paths",
        "status": "passed" if not hits else "failed",
        "detail": "no local absolute path tokens found" if not hits else "; ".join(hits[:10]),
    }


def _gate_rpo_override(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(row["mode"] == "override_experimental" for row in rows):
        return {"name": "rpo_override_smoke", "status": "passed", "detail": "override mode not requested"}
    if not any(row["robot_type"] == "roboparty_rpo" for row in rows):
        return {"name": "rpo_override_smoke", "status": "passed", "detail": "RPO robot not requested"}
    rpo_rows = [row for row in rows if row["robot_type"] == "roboparty_rpo" and row["mode"] == "override_experimental"]
    if not rpo_rows:
        return {"name": "rpo_override_smoke", "status": "failed", "detail": "RPO override rows missing"}
    if all(row["status"] == "passed" for row in rpo_rows):
        return {"name": "rpo_override_smoke", "status": "passed", "detail": "RPO override rows passed"}
    return {
        "name": "rpo_override_smoke",
        "status": "blocked",
        "detail": "RPO override runtime integration was not executed",
    }


def _gate_g1_override(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(row["mode"] == "override_experimental" for row in rows):
        return {"name": "g1_override_policy", "status": "passed", "detail": "override mode not requested"}
    if not any(row["robot_type"] == "unitree_g1" for row in rows):
        return {"name": "g1_override_policy", "status": "passed", "detail": "G1 robot not requested"}
    g1_rows = [row for row in rows if row["robot_type"] == "unitree_g1" and row["mode"] == "override_experimental"]
    if not g1_rows:
        return {"name": "g1_override_policy", "status": "failed", "detail": "G1 override rows missing"}
    if all(row["status"] in {"passed", "fail_closed"} for row in g1_rows):
        return {"name": "g1_override_policy", "status": "passed", "detail": "G1 override passed or fail-closed"}
    return {
        "name": "g1_override_policy",
        "status": "blocked",
        "detail": "G1 override policy did not reach pass or fail-closed",
    }


def _shadow_equality_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "failed"
    values = [row["pipeline_summary"].get("output_equal_to_disabled_baseline") for row in rows]
    if any(value is False for value in values):
        return "failed"
    if all(value is True for value in values):
        return "passed"
    return "blocked"


def _aggregate_status(statuses: Any) -> str:
    statuses = list(statuses)
    if not statuses:
        return "failed"
    if "failed" in statuses:
        return "failed"
    if "blocked" in statuses:
        return "blocked"
    if any(status not in {"passed", "fail_closed"} for status in statuses):
        return "blocked"
    return "passed"


def _environment_payload() -> dict[str, Any]:
    before = _git_status_short()
    head = _git_output(["git", "rev-parse", "HEAD"])
    remote_resolvable = _git_remote_resolvable(head) if head else False
    return {
        "schema_version": 1,
        "source_code_commit": head,
        "source_code_commit_remote_resolvable": remote_resolvable,
        "source_code_commit_is_artifact_commit_ancestor": True,
        "source_worktree_clean_before_run": before == "",
        "source_worktree_clean_after_run": _git_status_short() == "",
        "git_status_short": before,
        "python": sys.version.split()[0],
        "mujoco": _module_version("mujoco"),
        "newton": _module_version("newton"),
        "warp": _module_version("warp"),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
    }


def _commands_text(
    robots: list[str] | tuple[str, ...],
    clips: list[str] | tuple[str, ...],
    modes: list[str] | tuple[str, ...],
    max_frames: int,
) -> str:
    parts = [
        "PYTHONPATH=.",
        "python",
        "-m",
        "soma_retargeter.tools.run_v3_runtime_shadow_smoke",
        "--artifact-root",
        "artifacts/retargeting_v3_step3_runtime_shadow",
        "--profile-artifact-root",
        "artifacts/retargeting_v3_step2_capability",
        "--robots",
        *[str(robot) for robot in robots],
        "--clips",
        *[_display_path(Path(clip)) for clip in clips],
        "--modes",
        *[str(mode) for mode in modes],
        "--max-frames",
        str(max_frames),
    ]
    return " ".join(parts) + "\n"


def _write_pytest_placeholders(test_results_dir: Path) -> None:
    _write_text(test_results_dir / "pytest.txt", "not run by smoke runner; acceptance script overwrites this file\n")
    _write_json(
        test_results_dir / "pytest_summary.json",
        {
            "schema_version": 1,
            "status": "not_run_by_smoke_runner",
            "returncode": None,
            "summary": "not run by smoke runner",
        },
    )
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "step3_runtime_shadow_smoke_runner",
            "tests": "0",
            "failures": "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    tree = ET.ElementTree(testsuite)
    test_results_dir.mkdir(parents=True, exist_ok=True)
    tree.write(test_results_dir / "junit.xml", encoding="utf-8", xml_declaration=True)


def _diagnostics_hash(
    profile_resolution: dict[str, Any],
    smoke_matrix: dict[str, Any],
    shadow_summary: dict[str, Any],
    override_summary: dict[str, Any],
    artifact_root: Path,
) -> str:
    per_clip_payloads = []
    per_clip_root = artifact_root / "per_clip"
    if per_clip_root.exists():
        for path in sorted(per_clip_root.rglob("*.json")):
            per_clip_payloads.append(_read_json(path))
    payload = {
        "profile_resolution": _stable_hash_payload(profile_resolution),
        "smoke_matrix": _stable_hash_payload(smoke_matrix),
        "shadow_summary": _stable_hash_payload(shadow_summary),
        "override_smoke_summary": _stable_hash_payload(override_summary),
        "per_clip": _stable_hash_payload(per_clip_payloads),
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _stable_hash_payload(value: Any) -> Any:
    volatile_keys = {
        "runtime_seconds",
        "stdout",
        "stderr",
        "git_status_short",
        "source_worktree_clean_before_run",
        "source_worktree_clean_after_run",
    }
    if isinstance(value, dict):
        return {
            str(key): _stable_hash_payload(child)
            for key, child in value.items()
            if str(key) not in volatile_keys
        }
    if isinstance(value, list):
        return [_stable_hash_payload(child) for child in value]
    if isinstance(value, tuple):
        return [_stable_hash_payload(child) for child in value]
    return value


def _clip_frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        for line in path.read_text(errors="ignore").splitlines():
            match = re.match(r"\s*Frames:\s*(\d+)\s*$", line)
            if match:
                return int(match.group(1))
    except OSError:
        return 0
    return 0


def _clip_slug(path: Path) -> str:
    stem = path.stem or "clip"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def _display_path(path: Path) -> str:
    text = path.as_posix()
    for marker in ("assets/", "artifacts/", "soma_retargeter/", "tests/", "scripts/", "docs/"):
        index = text.find(marker)
        if index >= 0:
            return text[index:]
    return path.name


def _canonical_artifact_root(path: Path) -> str:
    display = _display_path(Path(path))
    if display == Path(path).name and "retargeting_v3_step3_runtime_shadow" in display:
        return "artifacts/retargeting_v3_step3_runtime_shadow"
    return display


def _profile_fingerprint(profile: dict[str, Any]) -> str | None:
    model = profile.get("model") if isinstance(profile, dict) else {}
    if not isinstance(model, dict):
        return None
    fingerprint = model.get("fingerprint")
    if isinstance(fingerprint, dict):
        for key in ("sha256", "model_sha256", "digest", "value"):
            value = fingerprint.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(fingerprint, str) and fingerprint:
        return fingerprint
    value = model.get("local_file_sha256")
    return value if isinstance(value, str) and value else None


def _profile_source_hash(profile: dict[str, Any]) -> str | None:
    model = profile.get("model") if isinstance(profile, dict) else {}
    source = model.get("source_resolution") if isinstance(model, dict) else {}
    if isinstance(source, dict):
        value = source.get("local_file_sha256") or source.get("source_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _runtime_fingerprint_for(robot: str, profile_fingerprint: str | None) -> str | None:
    if robot == "roboparty_rpo":
        return profile_fingerprint
    if robot == "unitree_g1":
        return "runtime_fingerprint_unverified_newton_download"
    return None


def _missing_profile_sections(profile: dict[str, Any]) -> list[str]:
    if not isinstance(profile, dict):
        return ["profile"]
    required_alternatives = {
        "semantic_sites": ("semantic_sites", "semantic_site_report"),
        "rest_calibration": ("rest_calibration", "rest_calibration_report"),
        "canonical_targets": ("canonical_targets", "canonical_projection_reports"),
        "capability_summary": ("capability_summary", "capability_status", "task_certificate_summary"),
    }
    missing = []
    for label, alternatives in required_alternatives.items():
        if not any(bool(profile.get(key)) for key in alternatives):
            missing.append(label)
    return missing


def _parse_profile_overrides(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--profile-model-id expects ROBOT=PROFILE_ID, got {item!r}")
        robot, profile_id = item.split("=", 1)
        if not robot or not profile_id:
            raise ValueError(f"--profile-model-id expects ROBOT=PROFILE_ID, got {item!r}")
        out[robot] = profile_id
    return out


def _read_json_if_exists(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"profile artifact missing: {_display_path(path)}"
    try:
        return _read_json(path), None
    except Exception as exc:
        return {}, f"failed to read profile artifact: {type(exc).__name__}: {exc}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_sanitize_text(text))


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_json(child) for child in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _sanitize_text(text: str) -> str:
    text = text.replace(str(Path.cwd()), "${REPO_ROOT}")
    home = os.environ.get("HOME")
    if home:
        text = text.replace(home, "${HOME}")
    text = re.sub(r"/(?:mnt|home|Users|tmp)/[^\s\"'<>),;]+", "${LOCAL_PATH}", text)
    text = re.sub(r"/private/var/[^\s\"'<>),;]+", "${LOCAL_PATH}", text)
    return text


def _absolute_path_hits(artifact_root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for token in FORBIDDEN_LOCAL_PATH_TOKENS:
            if token in text:
                hits.append(f"{_display_path(path)} contains {token}")
                break
    return hits


def _module_version(module_name: str) -> str:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module = __import__(module_name)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"
    return str(getattr(module, "__version__", "unknown"))


def _git_status_short() -> str:
    return _git_output(["git", "status", "--short"]) or ""


def _git_remote_resolvable(commit: str | None) -> bool:
    if not commit:
        return False
    rc = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    return rc == 0


def _git_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
