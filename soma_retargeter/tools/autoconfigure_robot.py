# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.robot_registry_parser import get_robot_profile, resolve_robot_name
from soma_retargeter.robotics.morphology import analyze_mjcf_morphology
from soma_retargeter.robotics.retarget_profile import write_profile_json
from soma_retargeter.robotics.task_compiler import compile_retarget_profile


def _load_raw_config(path: str | None) -> dict:
    if not path:
        return {"ik_map": {}}
    return io_utils.load_json(path)


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
    if profile is None:
        print(json.dumps({"error": "unknown_robot", "robot": robot_name}, sort_keys=True))
        return 3

    raw_config_path = profile.get("retargeter_config")
    raw_config = _load_raw_config(raw_config_path)
    morphology = analyze_mjcf_morphology(profile.get("mjcf_path"))
    compiled = compile_retarget_profile(
        robot_name=robot_name,
        raw_config=raw_config,
        morphology=morphology,
        source_config_path=raw_config_path,
    )

    needs_confirmation = any(w.get("code") == "low_semantic_confidence" for w in compiled.warnings)
    invalid = any(w.get("code") in {"mjcf_not_found", "mjcf_parse_error", "invalid_ik_map"} for w in compiled.warnings)
    if args.strict and (needs_confirmation or invalid):
        print(json.dumps({"warnings": compiled.warnings}, sort_keys=True))
        return 3 if invalid else 2

    output = Path(args.output) if args.output else Path(raw_config_path).with_name(f"{robot_name}_compiled_retarget_profile_v2.json")
    if not args.validate_only and not args.dry_run:
        if output.exists() and not args.force:
            print(json.dumps({"error": "output_exists", "path": str(output)}, sort_keys=True))
            return 3
        write_profile_json(compiled, output)

    print(json.dumps({"robot": robot_name, "output": str(output), "confidence": compiled.confidence, "warnings": compiled.warnings}, sort_keys=True))
    if invalid:
        return 3
    if needs_confirmation:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
