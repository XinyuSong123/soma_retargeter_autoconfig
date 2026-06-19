# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.robot_registry_parser import (
    ensure_compiled_retarget_profile,
    get_robot_profile,
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
    "foot_position_rmse",
    "velocity_p95",
    "acceleration_p95",
    "solver_iterations",
    "runtime_seconds",
    "fallback_counts",
    "confidence",
    "warnings",
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


def summarize_profile(robot: str, profile_path: Path, elapsed_s: float) -> dict[str, Any]:
    profile = io_utils.load_json(profile_path)
    diagnostics = validate_compiled_retarget_profile(profile)
    warnings = list(profile.get("warnings", [])) + diagnostics
    task_summary = _task_summary(profile)
    chain_summary = _chain_summary(profile)
    collision_summary = _collision_summary(profile)
    root_ground_summary = _root_ground_summary(profile)
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
        "metrics": {
            "task_residual_by_type_priority": {
                "status": "not_run",
                "reason": "motion runtime benchmark is not implemented yet",
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
            "confidence": float(profile.get("confidence", 0.0)),
            "warnings": len(warnings),
        },
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


def run_robot(robot_arg: str, output_dir: Path, force: bool, command: str) -> dict[str, Any]:
    robot = resolve_robot_name(robot_arg)
    started = time.perf_counter()
    try:
        profile_path = ensure_compiled_retarget_profile(robot, force=force)
        if profile_path is None:
            raise ValueError(f"No compiled profile path available for robot {robot!r}")
        elapsed = time.perf_counter() - started
        result = summarize_profile(robot, profile_path, elapsed)
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
            fieldnames=["robot", "compare_mode", "frame", "metric", "value", "status"],
        )
        writer.writeheader()
        for result in results:
            for compare_mode in result.get("compare_modes", []):
                for metric in METRIC_NAMES:
                    metric_payload = result.get("metrics", {}).get(metric, {})
                    metric_status = metric_payload.get("status", result.get("status")) if isinstance(metric_payload, dict) else result.get("status")
                    writer.writerow(
                        {
                            "robot": result.get("robot"),
                            "compare_mode": compare_mode,
                            "frame": "",
                            "metric": metric,
                            "value": "",
                            "status": metric_status,
                        }
                    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible SOMA retargeting v2 benchmark artifact generation.")
    parser.add_argument("--robots", nargs="+", required=True)
    parser.add_argument("--motions", nargs="+", default=[])
    parser.add_argument("--compare", nargs="+", default=["legacy", "v2"])
    parser.add_argument("--output", default="artifacts/retargeting_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output)
    command = "python -m soma_retargeter.tools.benchmark_retargeting " + " ".join(sys.argv[1:] if argv is None else argv)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "commands.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(output_dir / "environment.json", collect_environment(args))

    results = []
    for robot in args.robots:
        result = run_robot(robot, output_dir, args.force, command)
        result["compare_modes"] = list(args.compare)
        results.append(result)

    summary = {
        "schema_version": 1,
        "status": "ok" if all(result.get("status") in {"ok", "diagnostics"} for result in results) else "failed",
        "robots": [result.get("robot") for result in results],
        "compare_modes": list(args.compare),
        "motions": [str(Path(path)) for path in args.motions],
        "metric_names": list(METRIC_NAMES),
        "results": results,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    _write_frames_csv(output_dir / "benchmark_frames.csv", results)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
