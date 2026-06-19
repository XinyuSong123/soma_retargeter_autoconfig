# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.robot_registry_parser import get_robot_mjcf_path, get_robot_profile, resolve_robot_name
from soma_retargeter.robotics.morphology import analyze_mjcf_morphology
from soma_retargeter.robotics.retarget_profile import write_profile_json
from soma_retargeter.robotics.task_compiler import compile_retarget_profile


def _load_raw_config(path: str | None) -> dict:
    if not path:
        return {"ik_map": {}}
    return io_utils.load_json(path)


def _default_output_path(robot_name: str, raw_config_path: str | None) -> Path:
    filename = f"{robot_name}_compiled_retarget_profile_v2.json"
    if raw_config_path:
        return Path(raw_config_path).with_name(filename)
    return Path(filename)


def _default_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.autoconfig_report.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compiled_summary(compiled: Any | None) -> dict[str, Any]:
    if compiled is None:
        return {}
    tasks = getattr(compiled, "tasks", [])
    semantic_sites = getattr(compiled, "semantic_sites", {})
    chains = getattr(compiled, "chains", {})
    return {
        "schema_version": getattr(compiled, "schema_version", None),
        "compiler_version": getattr(compiled, "compiler_version", None),
        "robot_fingerprint": getattr(compiled, "robot_fingerprint", None),
        "source_config_hash": getattr(compiled, "source_config_hash", None),
        "confidence": getattr(compiled, "confidence", None),
        "warnings": getattr(compiled, "warnings", []),
        "task_count": len(tasks) if hasattr(tasks, "__len__") else None,
        "semantic_site_count": len(semantic_sites) if hasattr(semantic_sites, "__len__") else None,
        "chain_count": len(chains) if hasattr(chains, "__len__") else None,
    }


def _build_report(
    *,
    args: argparse.Namespace,
    requested_robot: str,
    robot_name: str,
    output: Path,
    report_path: Path,
    raw_config_path: str | None,
    mjcf_path: Path | str | None,
    compiled: Any | None,
    wrote_profile: bool,
    benchmark_args: list[str] | None = None,
    benchmark_return_code: int | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = "error" if error else "ok"
    if benchmark_return_code == 4:
        status = "benchmark_gate_failed"
    return {
        "schema_version": 1,
        "status": status,
        "requested_robot": requested_robot,
        "robot": robot_name,
        "seed": int(args.seed),
        "strict": bool(args.strict),
        "dry_run": bool(args.dry_run),
        "validate_only": bool(args.validate_only),
        "force": bool(args.force),
        "output": str(output),
        "report_path": str(report_path),
        "wrote_profile": bool(wrote_profile),
        "source_config_path": raw_config_path,
        "mjcf_path": str(mjcf_path) if mjcf_path is not None else None,
        "compiled_profile": _compiled_summary(compiled),
        "benchmark": {
            "requested": bool(args.benchmark),
            "args": benchmark_args,
            "return_code": benchmark_return_code,
            "strict_gates": bool(args.strict),
        },
        "error": error,
        "cli_args": vars(args),
    }


def _run_benchmark(robot_name: str, *, seed: int, strict: bool) -> tuple[int, list[str]]:
    from soma_retargeter.tools import benchmark_retargeting

    benchmark_args = [
        "--robots",
        robot_name,
        "--motions",
        "assets/motions/bvh",
        "--compare",
        "legacy",
        "v2",
        "--output",
        "artifacts/retargeting_v2",
        "--seed",
        str(int(seed)),
    ]
    if strict:
        benchmark_args.append("--strict-gates")
    return benchmark_retargeting.main(benchmark_args), benchmark_args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile a deterministic SOMA retargeting v2 profile.")
    parser.add_argument("--robot", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    robot_name = resolve_robot_name(args.robot)
    profile = get_robot_profile(robot_name)
    output = Path(args.output) if args.output else _default_output_path(robot_name, profile.get("retargeter_config") if profile else None)
    report_path = _default_report_path(output)
    if profile is None:
        error = {"error": "unknown_robot", "robot": robot_name}
        if args.write_report:
            _write_json(
                report_path,
                _build_report(
                    args=args,
                    requested_robot=args.robot,
                    robot_name=robot_name,
                    output=output,
                    report_path=report_path,
                    raw_config_path=None,
                    mjcf_path=None,
                    compiled=None,
                    wrote_profile=False,
                    error=error,
                ),
            )
        print(json.dumps(error, sort_keys=True))
        return 3

    raw_config_path = profile.get("retargeter_config")
    raw_config = _load_raw_config(raw_config_path)
    mjcf_path = get_robot_mjcf_path(robot_name)
    morphology = analyze_mjcf_morphology(mjcf_path)
    compiled = compile_retarget_profile(
        robot_name=robot_name,
        raw_config=raw_config,
        morphology=morphology,
        source_config_path=raw_config_path,
    )

    needs_confirmation = any(w.get("code") == "low_semantic_confidence" for w in compiled.warnings)
    invalid = any(w.get("code") in {"mjcf_not_found", "mjcf_parse_error", "invalid_ik_map"} for w in compiled.warnings)
    wrote_profile = False
    if not args.validate_only and not args.dry_run and not invalid and not (args.strict and needs_confirmation):
        if output.exists() and not args.force:
            error = {"error": "output_exists", "path": str(output)}
            if args.write_report:
                _write_json(
                    report_path,
                    _build_report(
                        args=args,
                        requested_robot=args.robot,
                        robot_name=robot_name,
                        output=output,
                        report_path=report_path,
                        raw_config_path=raw_config_path,
                        mjcf_path=mjcf_path,
                        compiled=compiled,
                        wrote_profile=False,
                        error=error,
                    ),
                )
            print(json.dumps(error, sort_keys=True))
            return 3
        write_profile_json(compiled, output)
        wrote_profile = True

    benchmark_return_code = None
    benchmark_args = None
    if not invalid and not (args.strict and needs_confirmation) and args.benchmark:
        benchmark_return_code, benchmark_args = _run_benchmark(robot_name, seed=args.seed, strict=args.strict)

    if args.write_report:
        _write_json(
            report_path,
            _build_report(
                args=args,
                requested_robot=args.robot,
                robot_name=robot_name,
                output=output,
                report_path=report_path,
                raw_config_path=raw_config_path,
                mjcf_path=mjcf_path,
                compiled=compiled,
                wrote_profile=wrote_profile,
                benchmark_args=benchmark_args,
                benchmark_return_code=benchmark_return_code,
            ),
        )

    print(
        json.dumps(
            {
                "robot": robot_name,
                "output": str(output),
                "report": str(report_path) if args.write_report else None,
                "confidence": compiled.confidence,
                "warnings": compiled.warnings,
                "benchmark_return_code": benchmark_return_code,
            },
            sort_keys=True,
        )
    )
    if invalid:
        return 3
    if benchmark_return_code == 4:
        return 4
    if benchmark_return_code not in (None, 0):
        return benchmark_return_code
    if needs_confirmation:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
