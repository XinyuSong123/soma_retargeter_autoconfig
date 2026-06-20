# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.robotics.reachability import (
    project_relative_rotation_quat_xyzw,
    project_vector,
    quat_xyzw_to_rotation_vector,
)
from soma_retargeter.robot_registry_parser import (
    build_runtime_retargeter_config,
    ensure_compiled_retarget_profile,
    ensure_generated_scaler_config,
    get_robot_profile,
    get_robot_model_height,
    get_profile_path,
    get_registered_robot_names,
    make_config_reference,
    resolve_robot_name,
    validate_compiled_retarget_profile,
)


METRIC_NAMES = (
    "task_residual_by_type_priority",
    "joint_limit_margin",
    "foot_slide",
    "penetration",
    "root_tilt",
    "torso_reachable_residual",
    "torso_unreachable_residual",
    "hand_position_rmse",
    "hand_reachable_position_rmse",
    "foot_position_rmse",
    "foot_reachable_position_rmse",
    "velocity_p95",
    "acceleration_p95",
    "root_velocity_p95",
    "root_acceleration_p95",
    "solver_iterations",
    "runtime_seconds",
    "fallback_counts",
    "confidence",
    "warnings",
)

_RUNTIME_METRIC_NAMES = (
    "task_residual_by_type_priority",
    "joint_limit_margin",
    "foot_slide",
    "penetration",
    "root_tilt",
    "torso_reachable_residual",
    "torso_unreachable_residual",
    "hand_position_rmse",
    "hand_reachable_position_rmse",
    "foot_position_rmse",
    "foot_reachable_position_rmse",
    "velocity_p95",
    "acceleration_p95",
    "root_velocity_p95",
    "root_acceleration_p95",
    "runtime_seconds",
    "fallback_counts",
)

_DEFAULT_COVERAGE_TARGETS = (
    "roboparty_rpo",
    "unitree_g1",
    "unitree_g1_23dof",
    "unitree_g1_29dof",
    "e3_v2",
    "oli",
)

_BENCHMARK_GATE_SPECS = (
    ("penetration", "not_increase", 1.0e-5, "m"),
    ("root_tilt", "not_increase", 0.0, "rad"),
    ("torso_unreachable_residual", "not_increase", 0.0, "rad"),
    ("hand_position_rmse", "relative_not_worse", 0.05, "m"),
    ("foot_position_rmse", "relative_not_worse", 0.05, "m"),
    ("velocity_p95", "not_increase", 0.0, "actuated_joint_coord_per_s"),
    ("acceleration_p95", "not_increase", 0.0, "actuated_joint_coord_per_s2"),
    ("runtime_seconds.motion_runtime", "relative_not_worse", 0.25, "s"),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=io_utils.get_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "")) or None


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            "numpy": np.__version__,
            "warp": _package_version("warp"),
            "newton": _package_version("newton"),
        },
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_commit": _git_commit(),
        "seed": int(args.seed),
        "cli_args": vars(args),
        "asset_fingerprint": {
            "motions": [str(Path(path)) for path in args.motions],
        },
    }


def _path_payload(path: Path | None) -> dict[str, Any]:
    return {
        "path": str(path) if path is not None else None,
        "exists": bool(path is not None and path.exists()),
    }


def _compiled_profile_payload(path: Path | None) -> dict[str, Any]:
    payload = _path_payload(path)
    payload.update(
        {
            "schema_version": None,
            "compiler_version": None,
            "robot_fingerprint": None,
            "diagnostics": [],
            "warnings": [],
        }
    )
    if path is None or not path.exists():
        return payload
    try:
        profile = io_utils.load_json(path)
    except Exception as exc:
        payload["diagnostics"] = [{"code": "compiled_profile_read_failed", "message": str(exc)}]
        return payload
    payload["schema_version"] = profile.get("schema_version")
    payload["compiler_version"] = profile.get("compiler_version")
    payload["robot_fingerprint"] = profile.get("robot_fingerprint")
    payload["diagnostics"] = validate_compiled_retarget_profile(profile)
    warnings = profile.get("warnings", [])
    payload["warnings"] = warnings if isinstance(warnings, list) else []
    return payload


def _missing_robot_onboarding_report(robot_name: str) -> dict[str, Any]:
    config_dir = f"soma_retargeter/configs/{robot_name}"
    return {
        "status": "missing_assets_or_registration",
        "required_files": {
            "mjcf_or_xml": f"assets/robots/{robot_name}/mjcf/<robot>.xml",
            "urdf_optional": f"assets/robots/{robot_name}/urdf/<robot>.urdf",
            "retargeter_config": f"{config_dir}/soma_to_{robot_name}_retargeter_config.json",
        },
        "params_py_entries": [
            f"ROBOT_XML_DICT[{robot_name!r}]",
            f"RETARGETER_CONFIG_DICT[{robot_name!r}]",
            f"ROBOT_URDF_DICT[{robot_name!r}] optional",
        ],
        "minimal_semantic_ik_map_template": {
            "Hips": "<pelvis_or_base_body>",
            "Chest": "<torso_body>",
            "LeftHand": "<left_hand_or_forearm_distal_body>",
            "RightHand": "<right_hand_or_forearm_distal_body>",
            "LeftFoot": "<left_foot_or_ankle_body>",
            "RightFoot": "<right_foot_or_ankle_body>",
        },
        "next_commands": [
            f"python -m soma_retargeter.tools.autoconfigure_robot --robot {robot_name} --validate-only",
            f"python -m soma_retargeter.tools.autoconfigure_robot --robot {robot_name} --force",
            f"python -m soma_retargeter.tools.benchmark_retargeting --robots {robot_name} --motions assets/motions/bvh --compare legacy v2 --output artifacts/retargeting_v2",
        ],
        "notes": [
            "Do not fabricate robot assets or download unlicensed files.",
            "If only a higher-DoF model is available, use an explicit alias or a small locked-DoF fixture for capability coverage.",
            "Pose-pair files are optional bounded refinement inputs after the compiled profile validates.",
        ],
    }


def _coverage_entry(robot_name: str) -> dict[str, Any]:
    resolved = resolve_robot_name(robot_name)
    profile = get_robot_profile(resolved)
    registered = profile is not None
    mjcf_path = get_profile_path(resolved, "mjcf_path") if registered else None
    urdf_path = get_profile_path(resolved, "urdf_path") if registered else None
    retargeter_path = get_profile_path(resolved, "retargeter_config") if registered else None
    compiled_path = get_profile_path(resolved, "compiled_retarget_profile") if registered else None
    compiled_payload = _compiled_profile_payload(compiled_path)

    blockers: list[str] = []
    if not registered:
        blockers.append("missing_registration")
    else:
        if not (mjcf_path and mjcf_path.exists()):
            blockers.append("missing_mjcf_path")
        if not (retargeter_path and retargeter_path.exists()):
            blockers.append("missing_retargeter_config")
        if not compiled_payload["exists"]:
            blockers.append("missing_compiled_profile")
        if compiled_payload["diagnostics"]:
            blockers.append("invalid_compiled_profile")
        warning_codes = {str(item.get("code")) for item in compiled_payload["warnings"] if isinstance(item, dict)}
        if compiled_payload["robot_fingerprint"] == "missing-mjcf" or "missing_mjcf_path" in warning_codes:
            blockers.append("incomplete_morphology")

    status = "ready" if not blockers else ("missing_registration" if not registered else "registered_incomplete")
    entry = {
        "requested_name": robot_name,
        "resolved_name": resolved,
        "status": status,
        "blockers": sorted(set(blockers)),
        "registered": registered,
        "paths": {
            "mjcf_path": _path_payload(mjcf_path),
            "urdf_path": _path_payload(urdf_path),
            "retargeter_config": _path_payload(retargeter_path),
        },
        "compiled_profile": compiled_payload,
    }
    if registered and retargeter_path is not None and retargeter_path.exists():
        try:
            raw_config = json.loads(retargeter_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_config = {}
        if isinstance(raw_config, dict) and raw_config.get("synthetic_fixture"):
            entry["asset_kind"] = "synthetic_fixture"
            entry["asset_note"] = str(raw_config.get("synthetic_fixture_reason", "synthetic capability fixture"))
    if not registered:
        entry["onboarding_report"] = _missing_robot_onboarding_report(robot_name)
    return entry


def build_registry_coverage_report(targets: tuple[str, ...] = _DEFAULT_COVERAGE_TARGETS) -> dict[str, Any]:
    entries = [_coverage_entry(target) for target in targets]
    status_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "targets": list(targets),
        "registered_robots": get_registered_robot_names(),
        "status_counts": status_counts,
        "robots": entries,
    }


def _task_summary(profile: dict[str, Any]) -> dict[str, Any]:
    by_type_priority: dict[str, int] = {}
    enabled = 0
    disabled = 0
    for task in profile.get("tasks", []):
        key = f"{task.get('task_type', 'unknown')}:p{task.get('priority', 'unknown')}"
        by_type_priority[key] = by_type_priority.get(key, 0) + 1
        if task.get("enabled"):
            enabled += 1
        else:
            disabled += 1
    return {
        "by_type_priority": by_type_priority,
        "enabled": enabled,
        "disabled": disabled,
    }


def _chain_summary(profile: dict[str, Any]) -> dict[str, Any]:
    chains = profile.get("chains", {})
    total_lengths = []
    rotational_ranks = {}
    translational_ranks = {}
    for name, chain in chains.items():
        total_lengths.append(float(chain.get("total_length", 0.0)))
        rotational_ranks[name] = int(chain.get("rotational_rank", 0))
        translational_ranks[name] = int(chain.get("translational_rank", 0))
    finite_lengths = [value for value in total_lengths if np.isfinite(value)]
    return {
        "count": len(chains),
        "min_total_length": min(finite_lengths) if finite_lengths else None,
        "max_total_length": max(finite_lengths) if finite_lengths else None,
        "rotational_ranks": rotational_ranks,
        "translational_ranks": translational_ranks,
    }


def _collision_summary(profile: dict[str, Any]) -> dict[str, Any]:
    collision = profile.get("collision", {})
    proxies = collision.get("proxies", []) if isinstance(collision, dict) else []
    pairs = collision.get("pairs", []) if isinstance(collision, dict) else []
    return {
        "enabled": bool(collision.get("enabled", False)) if isinstance(collision, dict) else False,
        "margin": collision.get("margin") if isinstance(collision, dict) else None,
        "proxy_count": len(proxies) if isinstance(proxies, list) else 0,
        "pair_count": len(pairs) if isinstance(pairs, list) else 0,
        "runtime_barrier": collision.get("runtime_barrier") if isinstance(collision, dict) else None,
    }


def _root_ground_summary(profile: dict[str, Any]) -> dict[str, Any]:
    root_motion = profile.get("rest_frame_alignment", {}).get("root_motion", {})
    if not isinstance(root_motion, dict):
        return {"status": "missing"}
    keys = (
        "source",
        "horizontal_scale",
        "robot_leg_length_m",
        "source_leg_length_m",
        "robot_nominal_pelvis_height_m",
        "vertical_height_source",
        "ground_height_m",
        "ground_height_source",
        "confidence",
    )
    return {key: root_motion.get(key) for key in keys}


def _not_run_metrics(task_summary: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    return {
        "task_residual_by_type_priority": {
            "status": "not_run",
            "reason": "motion runtime benchmark was not requested",
            "compiled_task_counts": task_summary["by_type_priority"],
        },
        "joint_limit_margin": {"status": "not_run"},
        "foot_slide": {"status": "not_run"},
        "penetration": {"status": "not_run"},
        "root_tilt": {"status": "not_run"},
        "torso_reachable_residual": {"status": "not_run"},
        "torso_unreachable_residual": {"status": "not_run"},
        "hand_position_rmse": {"status": "not_run"},
        "foot_position_rmse": {"status": "not_run"},
        "velocity_p95": {"status": "not_run"},
        "acceleration_p95": {"status": "not_run"},
        "solver_iterations": {"status": "not_run"},
        "runtime_seconds": {"compile_profile": elapsed_s},
        "fallback_counts": {"status": "not_run"},
        "confidence": 0.0,
        "warnings": 0,
    }


def _merge_runtime_metrics(metrics: dict[str, Any], runtime: dict[str, Any] | None) -> None:
    if runtime is None:
        return
    for name in ("task_residual_by_type_priority", "torso_reachable_residual", "torso_unreachable_residual"):
        if metrics.get(name, {}).get("status") == "not_run":
            payload = dict(metrics[name])
            payload["status"] = "unavailable"
            payload["reason"] = "runtime rollout ran, but no enabled profile task residuals were available"
            metrics[name] = payload
    for name, payload in runtime.get("metrics", {}).items():
        metrics[name] = payload
    runtime_seconds = dict(metrics.get("runtime_seconds", {}))
    runtime_seconds.update(runtime.get("runtime_seconds", {}))
    metrics["runtime_seconds"] = runtime_seconds


def _profile_metrics(task_summary: dict[str, Any], confidence: float, warning_count: int, elapsed_s: float, runtime: dict[str, Any] | None) -> dict[str, Any]:
    metrics = _not_run_metrics(task_summary, elapsed_s)
    metrics["confidence"] = confidence
    metrics["warnings"] = warning_count
    _merge_runtime_metrics(metrics, runtime)
    return metrics


def summarize_profile(
    robot: str,
    profile_path: Path,
    elapsed_s: float,
    runtime: dict[str, Any] | None = None,
    compare_runtimes: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    profile = io_utils.load_json(profile_path)
    diagnostics = validate_compiled_retarget_profile(profile)
    warnings = list(profile.get("warnings", [])) + diagnostics
    task_summary = _task_summary(profile)
    chain_summary = _chain_summary(profile)
    collision_summary = _collision_summary(profile)
    root_ground_summary = _root_ground_summary(profile)
    confidence = float(profile.get("confidence", 0.0))
    metrics = _profile_metrics(task_summary, confidence, len(warnings), elapsed_s, runtime)
    compare_results = {}
    for compare_mode, compare_runtime in (compare_runtimes or {}).items():
        compare_results[compare_mode] = {
            "status": compare_runtime.get("status", "not_run") if isinstance(compare_runtime, dict) else "not_run",
            "motion_benchmark": compare_runtime,
            "metrics": _profile_metrics(task_summary, confidence, len(warnings), elapsed_s, compare_runtime),
        }
    return {
        "robot": robot,
        "status": "ok" if not diagnostics else "diagnostics",
        "profile_path": str(profile_path),
        "profile_schema_version": profile.get("schema_version"),
        "compiler_version": profile.get("compiler_version"),
        "quaternion_order": profile.get("quaternion_order"),
        "robot_fingerprint": profile.get("robot_fingerprint"),
        "source_skeleton_fingerprint": profile.get("source_skeleton_fingerprint"),
        "confidence": float(profile.get("confidence", 0.0)),
        "warnings": warnings,
        "task_summary": task_summary,
        "chain_summary": chain_summary,
        "collision_summary": collision_summary,
        "root_ground_summary": root_ground_summary,
        "motion_benchmark": runtime,
        "compare_results": compare_results,
        "metrics": metrics,
    }


def _failure_payload(robot: str, command: str, exc: BaseException) -> dict[str, Any]:
    profile = get_robot_profile(robot)
    return {
        "robot": robot,
        "status": "failed",
        "command": command,
        "exception": type(exc).__name__,
        "message": str(exc),
        "stack": traceback.format_exc(),
        "semantic_profile": profile or {},
        "task_compilation_warnings": [],
        "frame_index": None,
        "solver_residual": None,
    }


def _resolve_motion_paths(raw_paths: list[str], max_motions: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = io_utils.get_repo_root() / path
        if path.is_dir():
            paths.extend(sorted(p for p in path.rglob("*.bvh") if p.is_file()))
        elif path.is_file() and path.suffix.lower() == ".bvh":
            paths.append(path)
    deduped = list(dict.fromkeys(paths))
    if max_motions is not None and max_motions > 0:
        return deduped[:max_motions]
    return deduped


def _normalize_legacy_ik_map(raw_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # Keep this local to the benchmark so legacy comparison does not affect the
    # v2 runtime registry path.
    from soma_retargeter.robot_registry_parser import _normalize_ik_mapping_entry

    out: dict[str, dict[str, Any]] = {}
    raw_ik_map = raw_config.get("ik_map", {})
    if not isinstance(raw_ik_map, dict):
        return out
    for joint_name, entry in raw_ik_map.items():
        normalized = _normalize_ik_mapping_entry(str(joint_name), entry)
        if normalized is not None:
            out[str(joint_name)] = normalized
    return out


def _legacy_runtime_retargeter_config(robot: str) -> dict[str, Any]:
    from soma_retargeter.robot_registry_parser import _build_contact_aware_foot_ik_from_virtual_anchors

    raw_path = get_profile_path(robot, "retargeter_config")
    if raw_path is None:
        raise FileNotFoundError(f"Retargeter config is not registered for robot {robot!r}")
    raw_config = io_utils.load_json(raw_path)
    scaler_path = ensure_generated_scaler_config(robot)
    if scaler_path is None:
        raise FileNotFoundError(f"Scaler config cannot be generated for robot {robot!r}")
    config = {
        "initialization_pose": "soma/soma_zero_frame0.bvh",
        "num_initialization_frames": 10,
        "num_stabilization_frames": 5,
        "human_robot_scaler_config": make_config_reference(scaler_path),
        "model_height": get_robot_model_height(robot),
        "ik_iterations": 24,
        "joint_limit_weight": 10.0,
        "smooth_joint_filter_weight": 5.5,
        "collision_weight": 0.0,
        "enable_post_processing": False,
        "ik_map": _normalize_legacy_ik_map(raw_config),
    }
    for key in (
        "ik_iterations",
        "joint_limit_weight",
        "smooth_joint_filter_weight",
        "temporal_velocity_weight",
        "temporal_acceleration_weight",
        "collision_weight",
        "enable_post_processing",
        "feet_stabilizer_config",
        "smooth_joint_filter_objective_body_masks",
        "output_default_pose_blend_frames",
        "output_default_pose_blend_bodies",
        "enable_virtual_foot_grounding",
        "virtual_foot_grounding_smooth_window",
        "contact_aware_foot_ik",
        "ground_barrier",
        "model_height",
        "human_robot_scaler_config",
    ):
        if key in raw_config:
            config[key] = raw_config[key]
    if isinstance(config.get("contact_aware_foot_ik"), dict):
        contact_cfg = dict(config["contact_aware_foot_ik"])
        if contact_cfg.get("enabled", False) and "anchor_offsets" not in contact_cfg:
            auto_contact_cfg = _build_contact_aware_foot_ik_from_virtual_anchors(raw_config)
            if auto_contact_cfg is not None:
                contact_cfg.setdefault("contact_source", auto_contact_cfg.get("contact_source", "auto"))
                contact_cfg["anchor_offsets"] = auto_contact_cfg["anchor_offsets"]
                config["contact_aware_foot_ik"] = contact_cfg
    return config


def _runtime_retargeter_config(robot: str, compare_mode: str) -> dict[str, Any] | None:
    compare_mode = compare_mode.lower()
    if compare_mode == "legacy":
        return _legacy_runtime_retargeter_config(robot)
    if compare_mode == "v2":
        return None
    if compare_mode == "v2_no_pole":
        config = _build_v2_runtime_retargeter_config(robot)
        config["pole_vector_tasks"] = []
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode == "v2_pos_projected":
        profile_path = get_profile_path(robot, "compiled_retarget_profile")
        if profile_path is None or not profile_path.exists():
            raise FileNotFoundError(f"Compiled retarget profile is not registered for robot {robot!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        profile = io_utils.load_json(profile_path)
        chains = profile.get("chains", {})
        for semantic, mapping in config.get("ik_map", {}).items():
            if not isinstance(mapping, dict):
                continue
            chain = chains.get(semantic)
            if not isinstance(chain, dict):
                continue
            try:
                rank = int(chain.get("translational_rank", 3))
            except (TypeError, ValueError):
                rank = 3
            if rank <= 0:
                mapping["t_weight"] = 0.0
                mapping["v2_position_disabled_reason"] = "benchmark experiment: rank-0 translational basis"
                continue
            if rank >= 3:
                continue
            basis = chain.get("translational_basis")
            if isinstance(basis, list) and len(basis) == 3:
                mapping["v2_position_basis"] = basis
                mapping["v2_position_weight_source"] = "benchmark experiment: projected to translational basis"
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode.startswith("v2_pole_keep_"):
        config = _build_v2_runtime_retargeter_config(robot)
        tokens = [token for token in compare_mode.removeprefix("v2_pole_keep_").replace("+", "_").split("_") if token]
        if not tokens:
            raise ValueError(f"Invalid pole keep selector in compare mode {compare_mode!r}")
        kept_tasks = []
        for task in config.get("pole_vector_tasks", []):
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("name", "")).lower()
            target_site = str(task.get("target_site", "")).lower()
            reference_site = str(task.get("reference_site", "")).lower()
            selector_text = f"{task_name} {target_site} {reference_site}"
            keep = False
            for token in tokens:
                if token == "arm":
                    keep = keep or "arm" in selector_text or "forearm" in selector_text or "hand" in selector_text
                elif token == "leg":
                    keep = keep or "leg" in selector_text or "shin" in selector_text or "foot" in selector_text
                elif token == "hand":
                    keep = keep or "forearm_pole_vector" in task_name or "hand" in target_site
                elif token == "foot":
                    keep = keep or "shin_pole_vector" in task_name or "foot" in target_site
                elif token == "proximal":
                    keep = keep or (
                        ("arm_pole_vector" in task_name and "forearm" not in task_name)
                        or ("leg_pole_vector" in task_name and "shin" not in task_name)
                    )
                elif token == "distal":
                    keep = keep or "forearm_pole_vector" in task_name or "shin_pole_vector" in task_name
                else:
                    keep = keep or token in selector_text
            if keep:
                task = dict(task)
                task["selection_reason"] = f"benchmark experiment: kept by pole selector {','.join(tokens)}"
                kept_tasks.append(task)
        config["pole_vector_tasks"] = kept_tasks
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode.startswith("v2_direction_analytic_"):
        config = _build_v2_runtime_retargeter_config(robot)
        tokens = [
            token
            for token in compare_mode.removeprefix("v2_direction_analytic_").replace("+", "_").split("_")
            if token
        ]
        if not tokens:
            raise ValueError(f"Invalid direction analytic selector in compare mode {compare_mode!r}")
        for task in config.get("direction_tasks", []):
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("name", "")).lower()
            target_site = str(task.get("target_site", "")).lower()
            reference_site = str(task.get("reference_site", "")).lower()
            selector_text = f"{task_name} {target_site} {reference_site}"
            enable = False
            for token in tokens:
                if token == "all":
                    enable = True
                elif token == "arm":
                    enable = enable or "arm" in selector_text or "forearm" in selector_text or "hand" in selector_text
                elif token == "leg":
                    enable = enable or "leg" in selector_text or "shin" in selector_text or "foot" in selector_text
                elif token == "hand":
                    enable = enable or "arm" in selector_text or "forearm" in selector_text or "hand" in selector_text
                elif token == "foot":
                    enable = enable or "leg" in selector_text or "shin" in selector_text or "foot" in selector_text
                elif token == "proximal":
                    enable = enable or (
                        ("arm_direction" in task_name and "forearm" not in task_name)
                        or ("leg_direction" in task_name and "shin" not in task_name)
                    )
                elif token == "distal":
                    enable = enable or "forearm_direction" in task_name or "shin_direction" in task_name
                else:
                    enable = enable or token in selector_text
            if enable:
                task["analytic_jacobian"] = True
                task["jacobian_schedule_reason"] = f"benchmark experiment: force analytic direction by selector {','.join(tokens)}"
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode.startswith("v2_iter"):
        try:
            ik_iterations = int(compare_mode.removeprefix("v2_iter"))
        except ValueError as exc:
            raise ValueError(f"Invalid IK iteration count in compare mode {compare_mode!r}") from exc
        if ik_iterations <= 0:
            raise ValueError(f"Invalid IK iteration count in compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        config["ik_iterations"] = ik_iterations
        config["benchmark_compare_mode"] = compare_mode
        return config
    pole_iter_match = re.fullmatch(r"v2_pole_analytic_iter([0-9]+)", compare_mode)
    if pole_iter_match:
        ik_iterations = int(pole_iter_match.group(1))
        if ik_iterations <= 0:
            raise ValueError(f"Invalid IK iteration count in compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        _force_analytic_pole_tasks(config, 1.0)
        config["ik_iterations"] = ik_iterations
        config["benchmark_compare_mode"] = compare_mode
        return config
    combined_match = re.fullmatch(r"v2_pole_analytic_w([0-9]+(?:\.[0-9]+)?)_hand_w([0-9]+(?:\.[0-9]+)?)", compare_mode)
    if combined_match:
        pole_analytic_weight_scale = float(combined_match.group(1))
        hand_weight = float(combined_match.group(2))
        if pole_analytic_weight_scale < 0.0 or hand_weight < 0.0:
            raise ValueError(f"Invalid combined compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        _force_analytic_pole_tasks(config, pole_analytic_weight_scale)
        _override_position_weights(config, ("LeftHand", "RightHand"), hand_weight, "hand")
        config["benchmark_compare_mode"] = compare_mode
        return config
    hand_hips_match = re.fullmatch(r"v2_hand_w([0-9]+(?:\.[0-9]+)?)_hips_r([0-9]+(?:\.[0-9]+)?)", compare_mode)
    if hand_hips_match:
        hand_weight = float(hand_hips_match.group(1))
        hips_rotation_weight = float(hand_hips_match.group(2))
        if hand_weight < 0.0 or hips_rotation_weight < 0.0:
            raise ValueError(f"Invalid hand/hips compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        _override_position_weights(config, ("LeftHand", "RightHand"), hand_weight, "hand")
        _override_rotation_weight(config, "Hips", hips_rotation_weight, "hips")
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode.startswith("v2_hand_w"):
        try:
            hand_weight = float(compare_mode.removeprefix("v2_hand_w"))
        except ValueError as exc:
            raise ValueError(f"Invalid hand weight in compare mode {compare_mode!r}") from exc
        if hand_weight < 0.0:
            raise ValueError(f"Invalid hand weight in compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        _override_position_weights(config, ("LeftHand", "RightHand"), hand_weight, "hand")
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode.startswith("v2_foot_w"):
        try:
            foot_weight = float(compare_mode.removeprefix("v2_foot_w"))
        except ValueError as exc:
            raise ValueError(f"Invalid foot weight in compare mode {compare_mode!r}") from exc
        if foot_weight < 0.0:
            raise ValueError(f"Invalid foot weight in compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        _override_position_weights(config, ("LeftFoot", "RightFoot"), foot_weight, "foot")
        config["benchmark_compare_mode"] = compare_mode
        return config
    if compare_mode.startswith("v2_hips_r"):
        try:
            hips_rotation_weight = float(compare_mode.removeprefix("v2_hips_r"))
        except ValueError as exc:
            raise ValueError(f"Invalid hips rotation weight in compare mode {compare_mode!r}") from exc
        if hips_rotation_weight < 0.0:
            raise ValueError(f"Invalid hips rotation weight in compare mode {compare_mode!r}")
        config = _build_v2_runtime_retargeter_config(robot)
        _override_rotation_weight(config, "Hips", hips_rotation_weight, "hips")
        config["benchmark_compare_mode"] = compare_mode
        return config
    pole_analytic_weight_scale = 1.0
    if compare_mode.startswith("v2_pole_analytic_w"):
        try:
            pole_analytic_weight_scale = float(compare_mode.removeprefix("v2_pole_analytic_w"))
        except ValueError as exc:
            raise ValueError(f"Invalid pole analytic weight scale in compare mode {compare_mode!r}") from exc
        compare_mode = "v2_pole_analytic"
    if compare_mode == "v2_pole_analytic":
        config = _build_v2_runtime_retargeter_config(robot)
        _force_analytic_pole_tasks(config, pole_analytic_weight_scale)
        config["benchmark_compare_mode"] = compare_mode if pole_analytic_weight_scale == 1.0 else f"{compare_mode}_w{pole_analytic_weight_scale:g}"
        return config
    if compare_mode == "v2_pole_tangent_analytic":
        config = _build_v2_runtime_retargeter_config(robot)
        _force_analytic_pole_tasks(config, 1.0, residual_mode="tangent2")
        config["benchmark_compare_mode"] = compare_mode
        return config
    raise ValueError(
        f"Unsupported compare mode {compare_mode!r}; expected 'legacy', 'v2', 'v2_no_pole', 'v2_pos_projected', 'v2_pole_keep_<selector>', 'v2_direction_analytic_<selector>', 'v2_iter<N>', 'v2_pole_analytic_iter<N>', 'v2_hand_w<weight>', 'v2_foot_w<weight>', 'v2_hips_r<weight>', 'v2_hand_w<weight>_hips_r<weight>', 'v2_pole_analytic', 'v2_pole_analytic_w<scale>', 'v2_pole_analytic_w<scale>_hand_w<weight>', or 'v2_pole_tangent_analytic'."
    )


def _build_v2_runtime_retargeter_config(robot: str) -> dict[str, Any]:
    raw_path = get_profile_path(robot, "retargeter_config")
    if raw_path is None:
        raise FileNotFoundError(f"Retargeter config is not registered for robot {robot!r}")
    return build_runtime_retargeter_config(robot, io_utils.load_json(raw_path))


def _force_analytic_pole_tasks(config: dict[str, Any], weight_scale: float, residual_mode: str | None = None) -> None:
    for task in config.get("pole_vector_tasks", []):
        if isinstance(task, dict):
            task["analytic_jacobian"] = True
            task["weight"] = float(task.get("weight", 0.0)) * weight_scale
            task["normalized_weight"] = float(task.get("normalized_weight", 0.0)) * weight_scale
            task["jacobian_schedule_reason"] = "benchmark experiment: force analytic pole-vector Jacobian"
            if residual_mode is not None:
                task["residual_mode"] = residual_mode
                task["residual_mode_reason"] = "benchmark experiment: tangent-space pole-vector residual"


def _override_position_weights(config: dict[str, Any], semantics: tuple[str, ...], weight: float, label: str) -> None:
    for semantic in semantics:
        if semantic in config.get("ik_map", {}):
            config["ik_map"][semantic]["t_weight"] = weight
            config["ik_map"][semantic]["v2_position_weight_source"] = (
                f"benchmark experiment: override {label} position weight"
            )


def _override_rotation_weight(config: dict[str, Any], semantic: str, weight: float, label: str) -> None:
    if semantic in config.get("ik_map", {}):
        config["ik_map"][semantic]["r_weight"] = weight
        config["ik_map"][semantic]["v2_rotation_weight_source"] = (
            f"benchmark experiment: override {label} rotation weight"
        )


def _metric_payload(value: float | int | None, **extra: Any) -> dict[str, Any]:
    if value is None or not np.isfinite(float(value)):
        return {"status": "unavailable", **extra}
    return {"status": "ok", "value": float(value), **extra}


def _percentile_metric(values: np.ndarray, percentile: float = 95.0) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _quat_rotate_xyzw(quat_xyzw: np.ndarray, point: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat = quat / norm
    q_vec = quat[0:3]
    point = np.asarray(point, dtype=np.float64)
    t = 2.0 * np.cross(q_vec, point)
    return point + quat[3] * t + np.cross(q_vec, t)


def _transform_point_xyzw(transform_values: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    transform_values = np.asarray(transform_values, dtype=np.float64)
    return transform_values[0:3] + _quat_rotate_xyzw(transform_values[3:7], local_point)


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat = quat / norm
    return -quat if quat[3] < 0.0 else quat


def _quat_mul_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = _normalize_quat_xyzw(lhs)
    rx, ry, rz, rw = _normalize_quat_xyzw(rhs)
    return _normalize_quat_xyzw(
        np.asarray(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
    )


def _quat_inverse_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(quat_xyzw)
    return np.asarray([-x, -y, -z, w], dtype=np.float64)


def _quat_error_rotvec_xyzw(actual_xyzw: np.ndarray, target_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw_to_rotation_vector(_quat_mul_xyzw(actual_xyzw, _quat_inverse_xyzw(target_xyzw)))


def _direction_between_points(reference: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-8:
        return None
    return delta / length


def _pole_normal_between_points(reference: np.ndarray, middle: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    a = np.asarray(middle, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64) - np.asarray(middle, dtype=np.float64)
    normal = np.cross(a, b)
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-8:
        return None
    return normal / length


def _joint_limit_margin(model: Any, frames: np.ndarray) -> float | None:
    if frames.size == 0:
        return None
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()
    lower = model.joint_limit_lower.numpy().astype(np.float64)
    upper = model.joint_limit_upper.numpy().astype(np.float64)
    margins: list[float] = []
    for joint_idx in range(model.joint_count):
        coord0 = int(q_start[joint_idx])
        dof0 = int(qd_start[joint_idx])
        lin, ang = joint_dof_dim[joint_idx]
        for k in range(int(lin + ang)):
            coord_idx = coord0 + k
            dof_idx = dof0 + k
            if coord_idx < 7 or coord_idx >= frames.shape[1] or dof_idx >= len(lower):
                continue
            span = float(upper[dof_idx] - lower[dof_idx])
            if not np.isfinite(span) or span <= 1.0e-8 or span >= 1.0e8:
                continue
            values = frames[:, coord_idx].astype(np.float64)
            margins.extend(np.minimum(values - lower[dof_idx], upper[dof_idx] - values) / span)
    return float(np.min(margins)) if margins else None


def _root_tilt_p95(frames: np.ndarray) -> float | None:
    if frames.size == 0 or frames.shape[1] < 7:
        return None
    tilts = []
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    for row in frames:
        root_up = _quat_rotate_xyzw(row[3:7], world_up)
        cos_angle = float(np.clip(np.dot(root_up, world_up) / max(np.linalg.norm(root_up), 1.0e-12), -1.0, 1.0))
        tilts.append(float(np.arccos(cos_angle)))
    return _percentile_metric(np.asarray(tilts, dtype=np.float64))


def _trajectory_velocity_metrics(
    frames: np.ndarray,
    sample_rate: float,
    *,
    coord_slice: slice | None = None,
) -> tuple[float | None, float | None]:
    if frames.shape[0] < 2:
        return None, None
    if coord_slice is not None:
        frames = frames[:, coord_slice]
    if frames.shape[1] == 0:
        return None, None
    dt = 1.0 / max(float(sample_rate), 1.0e-6)
    velocities = np.diff(frames.astype(np.float64), axis=0) / dt
    velocity_norms = np.linalg.norm(velocities, axis=1)
    if frames.shape[0] < 3:
        return _percentile_metric(velocity_norms), None
    accelerations = np.diff(velocities, axis=0) / dt
    acceleration_norms = np.linalg.norm(accelerations, axis=1)
    return _percentile_metric(velocity_norms), _percentile_metric(acceleration_norms)


def _semantic_body_indices(profile: dict[str, Any], pipeline: Any, semantics: tuple[str, ...]) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    labels = {
        str(label).split("/")[-1]: index
        for index, label in enumerate(getattr(pipeline.robot_builder, "body_label", []))
    }
    sites = profile.get("semantic_sites", {})
    out: dict[str, tuple[int, np.ndarray]] = {}
    for semantic in semantics:
        site = sites.get(semantic)
        if not isinstance(site, dict):
            continue
        body_name = site.get("body_name")
        if body_name not in labels:
            continue
        out[semantic] = (
            labels[body_name],
            np.asarray(site.get("local_position", [0.0, 0.0, 0.0]), dtype=np.float64),
            _normalize_quat_xyzw(np.asarray(site.get("local_rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)),
        )
    return out


def _body_site_pose_trajectories(pipeline: Any, profile: dict[str, Any], frames: np.ndarray, semantics: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    import newton

    sites = _semantic_body_indices(profile, pipeline, semantics)
    if not sites or frames.size == 0:
        return {}
    model = pipeline.ik_model
    state = model.state()
    position_trajectories = {semantic: [] for semantic in sites}
    rotation_trajectories = {semantic: [] for semantic in sites}
    for row in frames:
        wp.copy(model.joint_q, wp.array(row.astype(np.float32), dtype=wp.float32))
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        for semantic, (body_idx, local_position, local_rotation) in sites.items():
            position_trajectories[semantic].append(_transform_point_xyzw(body_q[body_idx], local_position))
            rotation_trajectories[semantic].append(_quat_mul_xyzw(body_q[body_idx][3:7], local_rotation))
    return {
        semantic: {
            "position": np.asarray(position_trajectories[semantic], dtype=np.float64),
            "rotation": np.asarray(rotation_trajectories[semantic], dtype=np.float64),
        }
        for semantic in sites
    }


def _tracking_rmse(targets: np.ndarray, actual: np.ndarray) -> float | None:
    count = min(len(targets), len(actual))
    if count == 0:
        return None
    residuals = np.asarray(actual[:count], dtype=np.float64) - np.asarray(targets[:count], dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def _tracking_projection_basis(profile: dict[str, Any], semantic: str) -> np.ndarray | None:
    chain = profile.get("chains", {}).get(semantic)
    if not isinstance(chain, dict):
        return None
    basis = chain.get("translational_basis")
    if not isinstance(basis, list):
        return None
    arr = np.asarray(basis, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 3:
        return None
    if arr.shape[1] == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _tracking_residual_stats(targets: np.ndarray, actual: np.ndarray, projection_basis: np.ndarray | None = None) -> dict[str, Any] | None:
    count = min(len(targets), len(actual))
    if count == 0:
        return None
    residuals = np.asarray(actual[:count], dtype=np.float64) - np.asarray(targets[:count], dtype=np.float64)
    finite = np.all(np.isfinite(residuals), axis=1)
    if not np.any(finite):
        return None
    residuals = residuals[finite]
    projection_rank = None
    if projection_basis is not None:
        basis = np.asarray(projection_basis, dtype=np.float64)
        projection_rank = int(basis.shape[1]) if basis.ndim == 2 and basis.shape[0] == 3 else None
        if projection_rank == 0:
            residuals = np.zeros_like(residuals)
        elif projection_rank is not None:
            residuals = (residuals @ basis) @ basis.T
    axis_rmse = np.sqrt(np.mean(residuals * residuals, axis=0))
    payload = {
        "rmse": float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))),
        "axis_rmse": [float(value) for value in axis_rmse],
        "mean_error": [float(value) for value in np.mean(residuals, axis=0)],
        "p95_abs_error": [float(value) for value in np.percentile(np.abs(residuals), 95.0, axis=0)],
        "count": int(len(residuals)),
    }
    if projection_rank is not None:
        payload["projection_rank"] = projection_rank
    return payload


def _weighted_tracking_payload(stats: list[dict[str, Any]], *, unit: str = "m") -> dict[str, Any]:
    if not stats:
        return _metric_payload(None, unit=unit, statistic="rmse")
    weights = np.asarray([max(int(item.get("count", 0)), 0) for item in stats], dtype=np.float64)
    if weights.sum() <= 0.0:
        weights = np.ones(len(stats), dtype=np.float64)
    values = np.asarray([float(item["rmse"]) for item in stats], dtype=np.float64)
    axis_rmse = np.asarray([item["axis_rmse"] for item in stats], dtype=np.float64)
    mean_error = np.asarray([item["mean_error"] for item in stats], dtype=np.float64)
    p95_abs_error = np.asarray([item["p95_abs_error"] for item in stats], dtype=np.float64)
    return {
        **_metric_payload(float(np.average(values, weights=weights)), unit=unit, statistic="rmse"),
        "axis_rmse": [float(value) for value in np.average(axis_rmse, axis=0, weights=weights)],
        "mean_error": [float(value) for value in np.average(mean_error, axis=0, weights=weights)],
        "p95_abs_error": [float(value) for value in np.average(p95_abs_error, axis=0, weights=weights)],
        "axis_order": ["x", "y", "z"],
        "sample_count": int(np.sum(weights)),
    }


def _tracking_metric_payload(stats_by_semantic: dict[str, dict[str, Any]], *, unit: str = "m") -> dict[str, Any]:
    if not stats_by_semantic:
        return _metric_payload(None, unit=unit, statistic="rmse")
    payload = _weighted_tracking_payload(list(stats_by_semantic.values()), unit=unit)
    payload["by_semantic"] = {
        semantic: {
            **_metric_payload(float(stats["rmse"]), unit=unit, statistic="rmse"),
            "axis_rmse": list(stats["axis_rmse"]),
            "mean_error": list(stats["mean_error"]),
            "p95_abs_error": list(stats["p95_abs_error"]),
            "axis_order": ["x", "y", "z"],
            "sample_count": int(stats.get("count", 0)),
        }
        for semantic, stats in sorted(stats_by_semantic.items())
    }
    projection_ranks = sorted(
        {
            int(stats["projection_rank"])
            for stats in stats_by_semantic.values()
            if "projection_rank" in stats
        }
    )
    if projection_ranks:
        payload["projection_ranks"] = projection_ranks
        for semantic, stats in stats_by_semantic.items():
            if "projection_rank" in stats and semantic in payload["by_semantic"]:
                payload["by_semantic"][semantic]["projection_rank"] = int(stats["projection_rank"])
    return payload


def _aligned_runtime_target_frames(pipeline: Any, motion_index: int) -> np.ndarray | None:
    input_targets = getattr(pipeline, "input_targets", [])
    if motion_index >= len(input_targets):
        return None
    target_frames = np.asarray(input_targets[motion_index], dtype=np.float64)
    leading_frames = int(getattr(pipeline, "num_initialization_frames", 0) or 0) + int(getattr(pipeline, "num_stabilization_frames", 0) or 0)
    if leading_frames <= 0:
        return target_frames
    return target_frames[min(leading_frames, len(target_frames)) :]


def _series_payload(values: list[float], *, unit: str, statistic: str = "p95", **extra: Any) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable", **extra}
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"status": "unavailable", **extra}
    return {
        "status": "ok",
        "value": float(np.percentile(arr, 95.0)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
        "unit": unit,
        "statistic": statistic,
        **extra,
    }


def _profile_runtime_residual_metrics(
    profile: dict[str, Any],
    pipeline: Any,
    motion_index: int,
    semantic_pose: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    mapped_joints = list(getattr(pipeline, "mapped_joints", []))
    target_frames = _aligned_runtime_target_frames(pipeline, motion_index)
    if target_frames is None:
        return {}

    def target_position(semantic: str) -> np.ndarray | None:
        if semantic not in mapped_joints:
            return None
        return target_frames[:, mapped_joints.index(semantic), 0:3]

    def target_rotation(semantic: str) -> np.ndarray | None:
        if semantic not in mapped_joints:
            return None
        return target_frames[:, mapped_joints.index(semantic), 3:7]

    groups: dict[str, list[float]] = {}
    torso_reachable: list[float] = []
    torso_unreachable: list[float] = []
    torso_source_unreachable: list[float] = []

    for task in profile.get("tasks", []):
        if not isinstance(task, dict) or not task.get("enabled", False):
            continue
        task_type = str(task.get("task_type", "unknown"))
        priority = task.get("priority", "unknown")
        group_key = f"{task_type}:p{priority}"
        target_site = task.get("target_site") or task.get("source_semantic")
        reference_site = task.get("reference_site")
        source_semantic = task.get("source_semantic") or target_site
        if not isinstance(target_site, str):
            continue
        residuals: list[float] = []

        if task_type == "position":
            targets = target_position(target_site)
            actual = semantic_pose.get(target_site, {}).get("position")
            if targets is not None and actual is not None:
                count = min(len(targets), len(actual))
                scale = max(float(task.get("characteristic_length", 1.0) or 1.0), 1.0e-6)
                residuals = (np.linalg.norm(actual[:count] - targets[:count], axis=1) / scale).tolist()

        elif task_type == "projected_relative_rotation":
            targets = target_rotation(target_site)
            actual = semantic_pose.get(target_site, {}).get("rotation")
            raw_basis = task.get("rotation_mask_or_basis")
            basis = np.asarray(raw_basis, dtype=np.float64) if raw_basis is not None else None
            if targets is not None and actual is not None:
                count = min(len(targets), len(actual))
                for frame_idx in range(count):
                    target_quat = _normalize_quat_xyzw(targets[frame_idx])
                    actual_quat = _normalize_quat_xyzw(actual[frame_idx])
                    projected_target = (
                        project_relative_rotation_quat_xyzw(target_quat, basis)
                        if basis is not None and basis.ndim == 2 and basis.shape[0] == 3 and basis.shape[1] > 0
                        else target_quat
                    )
                    error_rotvec = _quat_error_rotvec_xyzw(actual_quat, projected_target)
                    if basis is not None and basis.ndim == 2 and basis.shape[0] == 3 and basis.shape[1] > 0:
                        reachable_error = project_vector(error_rotvec, basis)
                        residuals.append(float(np.linalg.norm(reachable_error)))
                        if target_site == "Chest":
                            actual_rotvec = quat_xyzw_to_rotation_vector(actual_quat)
                            target_rotvec = quat_xyzw_to_rotation_vector(target_quat)
                            torso_reachable.append(float(np.linalg.norm(reachable_error)))
                            torso_unreachable.append(float(np.linalg.norm(actual_rotvec - project_vector(actual_rotvec, basis))))
                            torso_source_unreachable.append(float(np.linalg.norm(target_rotvec - project_vector(target_rotvec, basis))))
                    else:
                        residuals.append(float(np.linalg.norm(error_rotvec)))

        elif task_type == "direction" and isinstance(reference_site, str):
            target_ref = target_position(reference_site)
            target_tip = target_position(target_site)
            actual_ref = semantic_pose.get(reference_site, {}).get("position")
            actual_tip = semantic_pose.get(target_site, {}).get("position")
            if target_ref is not None and target_tip is not None and actual_ref is not None and actual_tip is not None:
                count = min(len(target_ref), len(target_tip), len(actual_ref), len(actual_tip))
                for frame_idx in range(count):
                    target_dir = _direction_between_points(target_ref[frame_idx], target_tip[frame_idx])
                    actual_dir = _direction_between_points(actual_ref[frame_idx], actual_tip[frame_idx])
                    if target_dir is not None and actual_dir is not None:
                        residuals.append(float(np.linalg.norm(actual_dir - target_dir)))

        elif task_type == "pole_vector" and isinstance(reference_site, str) and isinstance(source_semantic, str):
            target_ref = target_position(reference_site)
            target_middle = target_position(source_semantic)
            target_tip = target_position(target_site)
            actual_ref = semantic_pose.get(reference_site, {}).get("position")
            actual_middle = semantic_pose.get(source_semantic, {}).get("position")
            actual_tip = semantic_pose.get(target_site, {}).get("position")
            if all(item is not None for item in (target_ref, target_middle, target_tip, actual_ref, actual_middle, actual_tip)):
                count = min(len(target_ref), len(target_middle), len(target_tip), len(actual_ref), len(actual_middle), len(actual_tip))
                for frame_idx in range(count):
                    target_normal = _pole_normal_between_points(target_ref[frame_idx], target_middle[frame_idx], target_tip[frame_idx])
                    actual_normal = _pole_normal_between_points(actual_ref[frame_idx], actual_middle[frame_idx], actual_tip[frame_idx])
                    if target_normal is not None and actual_normal is not None:
                        residuals.append(float(np.linalg.norm(actual_normal - target_normal)))

        if residuals:
            groups.setdefault(group_key, []).extend(residuals)

    group_payloads = {
        key: _series_payload(values, unit="normalized_residual" if ":p" in key and key.startswith("position") else "residual")
        for key, values in sorted(groups.items())
    }
    task_values = [value for values in groups.values() for value in values]
    metrics = {
        "task_residual_by_type_priority": {
            **_series_payload(task_values, unit="residual"),
            "groups": group_payloads,
        }
    }
    if torso_reachable:
        metrics["torso_reachable_residual"] = _series_payload(torso_reachable, unit="rad", source_unreachable_p95=float(np.percentile(torso_source_unreachable, 95.0)) if torso_source_unreachable else None)
    if torso_unreachable:
        metrics["torso_unreachable_residual"] = _series_payload(torso_unreachable, unit="rad", meaning="actual_unreachable_leakage")
    return metrics


def _runtime_metrics_for_buffer(profile: dict[str, Any], pipeline: Any, motion_index: int, buffer: Any) -> dict[str, Any]:
    frames = np.asarray(buffer.data, dtype=np.float32)
    sample_rate = float(buffer.sample_rate)
    velocity_p95, acceleration_p95 = _trajectory_velocity_metrics(frames, sample_rate, coord_slice=slice(7, None))
    root_velocity_p95, root_acceleration_p95 = _trajectory_velocity_metrics(frames, sample_rate, coord_slice=slice(0, 7))
    metrics = {
        "joint_limit_margin": _metric_payload(_joint_limit_margin(pipeline.ik_model, frames), unit="normalized_margin"),
        "root_tilt": _metric_payload(_root_tilt_p95(frames), unit="rad", statistic="p95"),
        "velocity_p95": _metric_payload(velocity_p95, unit="actuated_joint_coord_per_s", statistic="p95"),
        "acceleration_p95": _metric_payload(acceleration_p95, unit="actuated_joint_coord_per_s2", statistic="p95"),
        "root_velocity_p95": _metric_payload(root_velocity_p95, unit="root_coord_per_s", statistic="p95"),
        "root_acceleration_p95": _metric_payload(root_acceleration_p95, unit="root_coord_per_s2", statistic="p95"),
    }

    ground_height = profile.get("rest_frame_alignment", {}).get("root_motion", {}).get("ground_height_m", 0.0)
    try:
        ground_height = float(ground_height)
    except (TypeError, ValueError):
        ground_height = 0.0
    semantics = tuple(dict.fromkeys(["LeftFoot", "RightFoot", "LeftHand", "RightHand", *getattr(pipeline, "mapped_joints", [])]))
    semantic_pose = _body_site_pose_trajectories(pipeline, profile, frames, semantics)
    foot_points = [semantic_pose[name]["position"] for name in ("LeftFoot", "RightFoot") if name in semantic_pose]
    if foot_points:
        all_foot = np.concatenate(foot_points, axis=0)
        penetration = float(max(0.0, ground_height - float(np.min(all_foot[:, 2]))))
        slide_speeds = []
        dt = 1.0 / max(sample_rate, 1.0e-6)
        for points in foot_points:
            if len(points) < 2:
                continue
            near_ground = points[:-1, 2] <= ground_height + 0.03
            speeds = np.linalg.norm(np.diff(points[:, 0:2], axis=0), axis=1) / dt
            slide_speeds.extend(speeds[near_ground].tolist())
        metrics["penetration"] = _metric_payload(penetration, unit="m", statistic="max")
        metrics["foot_slide"] = _metric_payload(_percentile_metric(np.asarray(slide_speeds)), unit="m_per_s", statistic="p95")
    else:
        metrics["penetration"] = {"status": "unavailable", "reason": "semantic foot sites unavailable"}
        metrics["foot_slide"] = {"status": "unavailable", "reason": "semantic foot sites unavailable"}

    mapped_joints = list(getattr(pipeline, "mapped_joints", []))
    target_frames = _aligned_runtime_target_frames(pipeline, motion_index)
    if target_frames is not None:
        for metric_name, reachable_metric_name, semantics in (
            ("hand_position_rmse", "hand_reachable_position_rmse", ("LeftHand", "RightHand")),
            ("foot_position_rmse", "foot_reachable_position_rmse", ("LeftFoot", "RightFoot")),
        ):
            stats_by_semantic = {}
            reachable_stats_by_semantic = {}
            for semantic in semantics:
                if semantic not in semantic_pose or semantic not in mapped_joints:
                    continue
                target_idx = mapped_joints.index(semantic)
                stat = _tracking_residual_stats(target_frames[:, target_idx, 0:3], semantic_pose[semantic]["position"])
                if stat is not None:
                    stats_by_semantic[semantic] = stat
                projection_basis = _tracking_projection_basis(profile, semantic)
                if projection_basis is not None:
                    projected_stat = _tracking_residual_stats(
                        target_frames[:, target_idx, 0:3],
                        semantic_pose[semantic]["position"],
                        projection_basis,
                    )
                    if projected_stat is not None:
                        reachable_stats_by_semantic[semantic] = projected_stat
            metrics[metric_name] = _tracking_metric_payload(stats_by_semantic, unit="m")
            if reachable_stats_by_semantic:
                metrics[reachable_metric_name] = _tracking_metric_payload(reachable_stats_by_semantic, unit="m")
    metrics.update(_profile_runtime_residual_metrics(profile, pipeline, motion_index, semantic_pose))
    return metrics


def _aggregate_motion_metrics(motion_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for metric_name in _RUNTIME_METRIC_NAMES:
        values = []
        seen_payloads = []
        for motion in motion_payloads:
            payload = motion.get("metrics", {}).get(metric_name)
            if isinstance(payload, dict):
                seen_payloads.append(payload)
            if isinstance(payload, dict) and payload.get("status") == "ok" and "value" in payload:
                values.append(float(payload["value"]))
        if values:
            aggregated[metric_name] = {
                "status": "ok",
                "value": float(np.mean(values)),
                "motion_count": len(values),
                "aggregation": "mean",
            }
            tracking_payloads = [
                payload
                for payload in seen_payloads
                if payload.get("status") == "ok"
                and "axis_rmse" in payload
                and "mean_error" in payload
                and "p95_abs_error" in payload
            ]
            if tracking_payloads:
                weights = np.asarray([max(int(payload.get("sample_count", 0)), 0) for payload in tracking_payloads], dtype=np.float64)
                if weights.sum() <= 0.0:
                    weights = np.ones(len(tracking_payloads), dtype=np.float64)
                for key in ("axis_rmse", "mean_error", "p95_abs_error"):
                    arr = np.asarray([payload[key] for payload in tracking_payloads], dtype=np.float64)
                    aggregated[metric_name][key] = [float(value) for value in np.average(arr, axis=0, weights=weights)]
                aggregated[metric_name]["axis_order"] = list(tracking_payloads[0].get("axis_order", ["x", "y", "z"]))
                aggregated[metric_name]["sample_count"] = int(np.sum(weights))
                projection_ranks = sorted(
                    {
                        int(rank)
                        for payload in tracking_payloads
                        for rank in payload.get("projection_ranks", [])
                    }
                )
                if projection_ranks:
                    aggregated[metric_name]["projection_ranks"] = projection_ranks
                semantic_names = sorted(
                    {
                        str(semantic)
                        for payload in tracking_payloads
                        for semantic in (payload.get("by_semantic", {}) if isinstance(payload.get("by_semantic"), dict) else {})
                    }
                )
                by_semantic = {}
                for semantic in semantic_names:
                    semantic_payloads = [
                        payload["by_semantic"][semantic]
                        for payload in tracking_payloads
                        if isinstance(payload.get("by_semantic"), dict)
                        and isinstance(payload["by_semantic"].get(semantic), dict)
                        and payload["by_semantic"][semantic].get("status") == "ok"
                    ]
                    if not semantic_payloads:
                        continue
                    semantic_weights = np.asarray(
                        [max(int(payload.get("sample_count", 0)), 0) for payload in semantic_payloads],
                        dtype=np.float64,
                    )
                    if semantic_weights.sum() <= 0.0:
                        semantic_weights = np.ones(len(semantic_payloads), dtype=np.float64)
                    by_semantic[semantic] = {
                        "status": "ok",
                        "value": float(
                            np.average(
                                np.asarray([float(payload["value"]) for payload in semantic_payloads], dtype=np.float64),
                                weights=semantic_weights,
                            )
                        ),
                        "unit": tracking_payloads[0].get("unit", ""),
                        "statistic": "rmse",
                        "axis_order": list(semantic_payloads[0].get("axis_order", ["x", "y", "z"])),
                        "sample_count": int(np.sum(semantic_weights)),
                    }
                    for key in ("axis_rmse", "mean_error", "p95_abs_error"):
                        arr = np.asarray([payload[key] for payload in semantic_payloads], dtype=np.float64)
                        by_semantic[semantic][key] = [
                            float(value) for value in np.average(arr, axis=0, weights=semantic_weights)
                        ]
                    semantic_projection_ranks = sorted(
                        {
                            int(payload["projection_rank"])
                            for payload in semantic_payloads
                            if "projection_rank" in payload
                        }
                    )
                    if semantic_projection_ranks:
                        by_semantic[semantic]["projection_rank"] = semantic_projection_ranks[0]
                if by_semantic:
                    aggregated[metric_name]["by_semantic"] = by_semantic
        elif seen_payloads:
            reasons = sorted({str(payload.get("reason", payload.get("status", "unavailable"))) for payload in seen_payloads})
            aggregated[metric_name] = {
                "status": "unavailable",
                "motion_count": len(seen_payloads),
                "reason": "; ".join(reasons),
            }
    return aggregated


def _metric_value(metrics: dict[str, Any], metric_name: str) -> float | None:
    if metric_name == "runtime_seconds.motion_runtime":
        value = metrics.get("runtime_seconds", {}).get("motion_runtime")
    else:
        payload = metrics.get(metric_name)
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            return None
        value = payload.get("value")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _gate_entry(
    metric_name: str,
    rule: str,
    tolerance: float,
    unit: str,
    legacy_metrics: dict[str, Any],
    v2_metrics: dict[str, Any],
) -> dict[str, Any]:
    legacy_value = _metric_value(legacy_metrics, metric_name)
    v2_value = _metric_value(v2_metrics, metric_name)
    entry = {
        "metric": metric_name,
        "rule": rule,
        "tolerance": tolerance,
        "unit": unit,
        "legacy_value": legacy_value,
        "v2_value": v2_value,
    }
    if legacy_value is None or v2_value is None:
        return {**entry, "status": "unavailable", "reason": "legacy or v2 metric is unavailable"}

    if rule == "not_increase":
        allowed = legacy_value + tolerance
        passed = v2_value <= allowed
    elif rule == "relative_not_worse":
        allowed = legacy_value * (1.0 + tolerance)
        passed = v2_value <= allowed
    else:
        return {**entry, "status": "unavailable", "reason": f"unsupported gate rule {rule!r}"}
    return {
        **entry,
        "allowed_max": allowed,
        "status": "passed" if passed else "failed",
        "delta": v2_value - legacy_value,
    }


def build_benchmark_gate_report(results: list[dict[str, Any]], *, strict: bool = False) -> dict[str, Any]:
    robot_reports = []
    status_counts: dict[str, int] = {}
    for result in results:
        compare_results = result.get("compare_results", {})
        if not isinstance(compare_results, dict) or "legacy" not in compare_results or "v2" not in compare_results:
            robot_report = {
                "robot": result.get("robot"),
                "status": "unavailable",
                "reason": "legacy and v2 compare results are required",
                "gates": [],
            }
        else:
            legacy_metrics = compare_results["legacy"].get("metrics", {})
            v2_metrics = compare_results["v2"].get("metrics", {})
            gates = [
                _gate_entry(metric_name, rule, tolerance, unit, legacy_metrics, v2_metrics)
                for metric_name, rule, tolerance, unit in _BENCHMARK_GATE_SPECS
            ]
            failed = sum(1 for gate in gates if gate["status"] == "failed")
            unavailable = sum(1 for gate in gates if gate["status"] == "unavailable")
            passed = sum(1 for gate in gates if gate["status"] == "passed")
            robot_report = {
                "robot": result.get("robot"),
                "status": "failed" if failed else ("unavailable" if unavailable == len(gates) else "passed"),
                "failed_count": failed,
                "unavailable_count": unavailable,
                "passed_count": passed,
                "gates": gates,
            }
        status = str(robot_report["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        robot_reports.append(robot_report)

    overall_status = "failed" if status_counts.get("failed", 0) else ("unavailable" if not status_counts.get("passed", 0) and status_counts.get("unavailable", 0) else "passed")
    return {
        "schema_version": 1,
        "strict": bool(strict),
        "status": overall_status,
        "status_counts": status_counts,
        "robots": robot_reports,
    }


def write_gate_failure_payloads(
    output_dir: Path,
    gate_report: dict[str, Any],
    results: list[dict[str, Any]],
    command: str,
) -> list[dict[str, Any]]:
    """Persist structured failure summaries for report-only gate failures."""

    results_by_robot = {str(result.get("robot")): result for result in results if result.get("robot") is not None}
    written = []
    for robot_report in gate_report.get("robots", []):
        if not isinstance(robot_report, dict) or robot_report.get("status") != "failed":
            continue
        robot = str(robot_report.get("robot"))
        result = results_by_robot.get(robot, {})
        failed_gates = [gate for gate in robot_report.get("gates", []) if isinstance(gate, dict) and gate.get("status") == "failed"]
        unavailable_gates = [
            gate for gate in robot_report.get("gates", []) if isinstance(gate, dict) and gate.get("status") == "unavailable"
        ]
        payload = {
            "schema_version": 1,
            "status": "failed",
            "failure_type": "benchmark_gate",
            "robot": robot,
            "command": command,
            "reproduction_command": command,
            "per_robot_artifact": f"per_robot/{robot}.json",
            "profile_path": result.get("profile_path"),
            "profile_schema_version": result.get("profile_schema_version"),
            "compiler_version": result.get("compiler_version"),
            "robot_fingerprint": result.get("robot_fingerprint"),
            "warnings": result.get("warnings", []),
            "task_summary": result.get("task_summary", {}),
            "chain_summary": result.get("chain_summary", {}),
            "root_ground_summary": result.get("root_ground_summary", {}),
            "failed_gates": failed_gates,
            "unavailable_gates": unavailable_gates,
            "compare_metrics": {
                mode: payload.get("metrics", {})
                for mode, payload in (result.get("compare_results") or {}).items()
                if isinstance(payload, dict)
            },
            "compare_runtime_seconds": {
                mode: (payload.get("motion_benchmark") or {}).get("runtime_seconds", {})
                for mode, payload in (result.get("compare_results") or {}).items()
                if isinstance(payload, dict)
            },
            "compare_solver_objectives": {
                mode: (payload.get("motion_benchmark") or {}).get("solver_objectives", {})
                for mode, payload in (result.get("compare_results") or {}).items()
                if isinstance(payload, dict)
            },
        }
        _write_json(output_dir / "failures" / f"{robot}_gates.json", payload)
        written.append({"robot": robot, "path": f"failures/{robot}_gates.json", "failed_count": len(failed_gates)})
    return written


def _select_motion_window(animation: Any, max_frames: int) -> tuple[int, int, str]:
    frame_count = int(getattr(animation, "num_frames", 0))
    if max_frames <= 0 or frame_count <= max_frames:
        return 0, frame_count, "full_clip"
    transforms = np.asarray(animation.local_transforms, dtype=np.float64)
    if transforms.ndim < 2 or len(transforms) != frame_count:
        return 0, max_frames, "prefix_fallback"
    flattened = transforms.reshape(frame_count, -1)
    frame_motion = np.linalg.norm(np.diff(flattened, axis=0), axis=1)
    if frame_motion.size == 0 or not np.any(np.isfinite(frame_motion)):
        return 0, max_frames, "prefix_fallback"
    frame_motion = np.nan_to_num(frame_motion, nan=0.0, posinf=0.0, neginf=0.0)
    window_edges = max_frames - 1
    if window_edges <= 0:
        start = int(np.argmax(frame_motion))
        return min(start, frame_count - max_frames), max_frames, "max_motion_window"
    window_scores = np.convolve(frame_motion, np.ones(window_edges, dtype=np.float64), mode="valid")
    start = int(np.argmax(window_scores))
    start = min(max(start, 0), frame_count - max_frames)
    return start, max_frames, "max_motion_window"


def _run_runtime_benchmark(
    robot: str,
    profile_path: Path,
    motion_paths: list[Path],
    max_frames: int,
    compare_mode: str = "v2",
) -> dict[str, Any] | None:
    if not motion_paths:
        return None
    import soma_retargeter.assets.bvh as bvh_utils
    from soma_retargeter.animation.animation_buffer import AnimationBuffer
    from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline

    profile = io_utils.load_json(profile_path)
    first_skeleton = None
    animations = []
    used_paths = []
    load_started = time.perf_counter()
    for path in motion_paths:
        skeleton, animation = bvh_utils.load_bvh(str(path), first_skeleton)
        if first_skeleton is None:
            first_skeleton = skeleton
        source_start, source_count, frame_selection = _select_motion_window(animation, max_frames)
        if max_frames > 0 and animation.num_frames > max_frames:
            animation = AnimationBuffer(
                animation.skeleton,
                source_count,
                animation.sample_rate,
                np.array(animation.local_transforms[source_start:source_start + source_count], copy=True),
            )
            animation.benchmark_source_frame_start = source_start
            animation.benchmark_frame_selection = frame_selection
        else:
            animation.benchmark_source_frame_start = source_start
            animation.benchmark_frame_selection = frame_selection
        animations.append(animation)
        used_paths.append(path)
    load_elapsed = time.perf_counter() - load_started
    if first_skeleton is None or not animations:
        return None

    construct_started = time.perf_counter()
    pipeline = NewtonPipeline(first_skeleton, "soma", robot, retarget_config=_runtime_retargeter_config(robot, compare_mode))
    construct_elapsed = time.perf_counter() - construct_started
    setup_started = time.perf_counter()
    pipeline.add_input_motions(animations, [wp.transform_identity()] * len(animations), True)
    setup_elapsed = time.perf_counter() - setup_started
    solve_started = time.perf_counter()
    output_buffers = pipeline.execute()
    solve_elapsed = time.perf_counter() - solve_started
    elapsed = setup_elapsed + solve_elapsed
    metrics_started = time.perf_counter()
    motion_payloads = []
    for idx, buffer in enumerate(output_buffers or []):
        motion_payloads.append(
            {
                "motion": str(used_paths[idx]),
                "frames": int(buffer.num_frames),
                "sample_rate": float(buffer.sample_rate),
                "source_frame_start": int(getattr(animations[idx], "benchmark_source_frame_start", 0)),
                "frame_selection": str(getattr(animations[idx], "benchmark_frame_selection", "unknown")),
                "metrics": _runtime_metrics_for_buffer(profile, pipeline, idx, buffer),
            }
        )
    metrics_elapsed = time.perf_counter() - metrics_started
    output_frame_count = int(sum(int(getattr(buffer, "num_frames", 0)) for buffer in output_buffers or []))
    return {
        "status": "ok",
        "compare_mode": compare_mode,
        "motions": motion_payloads,
        "runtime_seconds": {
            "bvh_load_runtime": load_elapsed,
            "pipeline_construct_runtime": construct_elapsed,
            "target_setup_runtime": setup_elapsed,
            "solve_runtime": solve_elapsed,
            "metric_runtime": metrics_elapsed,
            "motion_runtime": elapsed,
            "motion_count": len(motion_payloads),
            "output_frame_count": output_frame_count,
            "solve_fps": (output_frame_count / solve_elapsed) if solve_elapsed > 0.0 else None,
        },
        "priority_guard": getattr(pipeline, "priority_guard_report", {}),
        "solver_objectives": getattr(pipeline, "ik_objective_summary", {}),
        "contact_scores": getattr(pipeline, "contact_score_summary", []),
        "metrics": {
            **_aggregate_motion_metrics(motion_payloads),
            "fallback_counts": {
                "status": "ok",
                "pole_vector": getattr(getattr(pipeline, "pole_vector_fallback_counts", np.asarray([], dtype=np.int64)), "tolist", lambda: [])(),
            },
            "solver_iterations": {
                "status": "ok",
                "value": float(getattr(pipeline, "ik_iterations", 0)),
            },
        },
    }


def _runtime_failure_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "exception": type(exc).__name__,
        "message": str(exc),
        "stack": traceback.format_exc(),
        "runtime_seconds": {},
        "metrics": {name: {"status": "failed", "reason": str(exc)} for name in _RUNTIME_METRIC_NAMES},
    }


def run_robot(robot_arg: str, output_dir: Path, force: bool, command: str, args: argparse.Namespace) -> dict[str, Any]:
    robot = resolve_robot_name(robot_arg)
    started = time.perf_counter()
    try:
        profile_path = ensure_compiled_retarget_profile(robot, force=force)
        if profile_path is None:
            raise ValueError(f"No compiled profile path available for robot {robot!r}")
        elapsed = time.perf_counter() - started
        compare_runtimes: dict[str, dict[str, Any] | None] = {}
        motion_paths = _resolve_motion_paths(args.motions, args.max_motions)
        if motion_paths:
            for compare_mode in args.compare:
                try:
                    compare_runtimes[compare_mode] = _run_runtime_benchmark(robot, profile_path, motion_paths, args.max_frames, compare_mode)
                except Exception as exc:
                    compare_runtimes[compare_mode] = _runtime_failure_payload(exc)
        primary_runtime = compare_runtimes.get("v2") or next((payload for payload in compare_runtimes.values() if payload is not None), None)
        result = summarize_profile(robot, profile_path, elapsed, runtime=primary_runtime, compare_runtimes=compare_runtimes)
        _write_json(output_dir / "per_robot" / f"{robot}.json", result)
        return result
    except Exception as exc:
        payload = _failure_payload(robot, command, exc)
        _write_json(output_dir / "failures" / f"{robot}.json", payload)
        return payload


def _write_frames_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["robot", "compare_mode", "motion", "frame", "metric", "value", "status"],
        )
        writer.writeheader()
        for result in results:
            compare_results = result.get("compare_results", {})
            for compare_mode in result.get("compare_modes", []):
                compare_metrics = result.get("metrics", {})
                if isinstance(compare_results, dict) and compare_mode in compare_results:
                    compare_metrics = compare_results[compare_mode].get("metrics", compare_metrics)
                for metric in METRIC_NAMES:
                    metric_payload = compare_metrics.get(metric, {})
                    metric_status = metric_payload.get("status", result.get("status")) if isinstance(metric_payload, dict) else result.get("status")
                    metric_value = metric_payload.get("value", "") if isinstance(metric_payload, dict) else ""
                    writer.writerow(
                        {
                            "robot": result.get("robot"),
                            "compare_mode": compare_mode,
                            "motion": "",
                            "frame": "",
                            "metric": metric,
                            "value": metric_value,
                            "status": metric_status,
                        }
                    )
                compare_payload = compare_results.get(compare_mode, {}) if isinstance(compare_results, dict) else {}
                motion_benchmark = compare_payload.get("motion_benchmark", {})
                if not isinstance(motion_benchmark, dict):
                    continue
                for motion_idx, motion in enumerate(motion_benchmark.get("motions", [])):
                    for metric, metric_payload in motion.get("metrics", {}).items():
                        writer.writerow(
                            {
                                "robot": result.get("robot"),
                                "compare_mode": compare_mode,
                                "motion": motion.get("motion", ""),
                                "frame": motion_idx,
                                "metric": metric,
                                "value": metric_payload.get("value", "") if isinstance(metric_payload, dict) else "",
                                "status": metric_payload.get("status", result.get("status")) if isinstance(metric_payload, dict) else result.get("status"),
                            }
                        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible SOMA retargeting v2 benchmark artifact generation.")
    parser.add_argument("--robots", nargs="+", required=True)
    parser.add_argument("--motions", nargs="+", default=[])
    parser.add_argument("--max-motions", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--compare",
        nargs="+",
        default=["legacy", "v2"],
        help="Compare modes: legacy, v2, v2_no_pole, v2_pos_projected, v2_pole_keep_<selector>, v2_direction_analytic_<selector>, v2_iter<N>, v2_pole_analytic_iter<N>, v2_hand_w<weight>, v2_foot_w<weight>, v2_hips_r<weight>, v2_hand_w<weight>_hips_r<weight>, v2_pole_analytic, v2_pole_analytic_w<scale>, v2_pole_analytic_w<scale>_hand_w<weight>, or v2_pole_tangent_analytic.",
    )
    parser.add_argument("--output", default="artifacts/retargeting_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-gates", action="store_true", help="Return exit code 4 when benchmark gates fail.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output)
    command = "python -m soma_retargeter.tools.benchmark_retargeting " + " ".join(sys.argv[1:] if argv is None else argv)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "commands.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(output_dir / "environment.json", collect_environment(args))
    resolved_motion_paths = _resolve_motion_paths(args.motions, args.max_motions)
    registry_coverage = build_registry_coverage_report()
    _write_json(output_dir / "registry_coverage.json", registry_coverage)

    results = []
    for robot in args.robots:
        result = run_robot(robot, output_dir, args.force, command, args)
        result["compare_modes"] = list(args.compare)
        results.append(result)
    benchmark_gates = build_benchmark_gate_report(results, strict=args.strict_gates)
    _write_json(output_dir / "benchmark_gates.json", benchmark_gates)
    gate_failure_artifacts = write_gate_failure_payloads(output_dir, benchmark_gates, results, command)

    summary = {
        "schema_version": 1,
        "status": "ok" if all(result.get("status") in {"ok", "diagnostics"} for result in results) else "failed",
        "benchmark_gate_status": benchmark_gates["status"],
        "gate_failure_artifacts": gate_failure_artifacts,
        "robots": [result.get("robot") for result in results],
        "compare_modes": list(args.compare),
        "motions": [str(Path(path)) for path in args.motions],
        "resolved_motions": [str(path) for path in resolved_motion_paths],
        "registry_coverage": registry_coverage,
        "benchmark_gates": benchmark_gates,
        "metric_names": list(METRIC_NAMES),
        "results": results,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    _write_frames_csv(output_dir / "benchmark_frames.csv", results)
    if args.strict_gates and benchmark_gates["status"] == "failed":
        return 4
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
