"""CLI for compiling one KinematicProfileV3 JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from soma_retargeter.robotics.v3.model_adapter import NewtonRuntimeModelAdapter
from soma_retargeter.robotics.v3.profile import compile_kinematic_profile_v3, write_profile
from soma_retargeter.robotics.v3.robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    display_path,
    load_robot_zoo_manifest,
    reproduction_compile_command,
    resolve_robot_source,
)
from soma_retargeter.robotics.v3.semantic_sites import default_rpo_semantic_map, load_semantic_map
from soma_retargeter.robotics.v3.semantic_sites import infer_semantic_map_from_body_names
from soma_retargeter.robotics.v3.validation import DEFAULT_LOW_DISCREPANCY_COUNT, augment_validation_report_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--manifest", default=str(DEFAULT_ROBOT_ZOO_MANIFEST_PATH))
    parser.add_argument("--allow-source-fetch", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-format", default=None)
    parser.add_argument("--backend", choices=("mujoco", "newton"), default="mujoco")
    parser.add_argument("--semantic-map", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-discrepancy-count", type=int, default=DEFAULT_LOW_DISCREPANCY_COUNT)
    args = parser.parse_args()
    if not args.model and not args.robot_id:
        parser.error("one of --model or --robot-id is required")
    model = args.model
    model_id = args.model_id
    model_format = args.model_format
    if args.robot_id:
        manifest = load_robot_zoo_manifest(args.manifest)
        entry = manifest.model_by_id.get(args.robot_id)
        if entry is None:
            raise SystemExit(f"Robot Zoo manifest entry {args.robot_id!r} is not present")
        resolved = resolve_robot_source(entry, allow_fetch=args.allow_source_fetch)
        if not resolved.available:
            raise SystemExit(f"Robot Zoo source unavailable for {entry.id}: {resolved.reason}")
        model = str(resolved.path)
        model_id = entry.id
        model_format = entry.model_format
    if args.semantic_map:
        semantic_map = load_semantic_map(args.semantic_map)
    elif args.robot_id == "roboparty_rpo_local":
        semantic_map = default_rpo_semantic_map()
    elif args.robot_id:
        adapter = NewtonRuntimeModelAdapter(model, model_format=model_format)
        try:
            semantic_map = infer_semantic_map_from_body_names(adapter)
        finally:
            adapter.close()
    else:
        semantic_map = default_rpo_semantic_map()
    if args.robot_id:
        reproduction_command = reproduction_compile_command(
            args.robot_id,
            manifest_path=args.manifest,
            output_path=args.output,
            low_discrepancy_count=args.low_discrepancy_count,
            backend=args.backend,
        )
    else:
        command_args = [
            "python",
            "-m",
            "soma_retargeter.tools.compile_kinematic_profile_v3",
            "--backend",
            args.backend,
            "--model",
            display_path(Path(model)),
        ]
        if model_id:
            command_args.extend(["--model-id", model_id])
        if model_format:
            command_args.extend(["--model-format", model_format])
        if args.semantic_map:
            command_args.extend(["--semantic-map", display_path(Path(args.semantic_map))])
        command_args.extend(
            [
            "--output",
            display_path(Path(args.output)),
            "--low-discrepancy-count",
            str(args.low_discrepancy_count),
            ]
        )
        reproduction_command = shlex.join(command_args)
    profile = compile_kinematic_profile_v3(
        model,
        semantic_map,
        model_id=model_id,
        model_format=model_format,
        backend=args.backend,
        low_discrepancy_count=args.low_discrepancy_count,
        reproduction_command=reproduction_command,
    )
    if args.semantic_map or args.robot_id:
        payload = augment_validation_report_metadata(profile.to_json(), semantic_map_path=args.semantic_map, manifest_path=args.manifest)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        write_profile(profile, args.output)
    print(json.dumps({"output": str(Path(args.output)), "failures": profile.failures}, sort_keys=True))


if __name__ == "__main__":
    main()
