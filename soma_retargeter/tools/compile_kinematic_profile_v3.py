"""CLI for compiling one KinematicProfileV3 JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from soma_retargeter.robotics.v3.profile import compile_kinematic_profile_v3, write_profile
from soma_retargeter.robotics.v3.semantic_sites import default_rpo_semantic_map, load_semantic_map
from soma_retargeter.robotics.v3.validation import DEFAULT_LOW_DISCREPANCY_COUNT, augment_validation_report_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-format", default=None)
    parser.add_argument("--backend", choices=("mujoco", "newton"), default="mujoco")
    parser.add_argument("--semantic-map", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-discrepancy-count", type=int, default=DEFAULT_LOW_DISCREPANCY_COUNT)
    args = parser.parse_args()
    if args.semantic_map:
        semantic_map = load_semantic_map(args.semantic_map)
    else:
        semantic_map = default_rpo_semantic_map()
    command_args = [
        "python",
        "-m",
        "soma_retargeter.tools.compile_kinematic_profile_v3",
        "--backend",
        args.backend,
        "--model",
        args.model,
    ]
    if args.model_id:
        command_args.extend(["--model-id", args.model_id])
    if args.model_format:
        command_args.extend(["--model-format", args.model_format])
    if args.semantic_map:
        command_args.extend(["--semantic-map", args.semantic_map])
    command_args.extend(
        [
        "--output",
        args.output,
        "--low-discrepancy-count",
        str(args.low_discrepancy_count),
        ]
    )
    reproduction_command = shlex.join(command_args)
    profile = compile_kinematic_profile_v3(
        args.model,
        semantic_map,
        model_id=args.model_id,
        model_format=args.model_format,
        backend=args.backend,
        low_discrepancy_count=args.low_discrepancy_count,
        reproduction_command=reproduction_command,
    )
    if args.semantic_map:
        payload = augment_validation_report_metadata(profile.to_json(), semantic_map_path=args.semantic_map)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        write_profile(profile, args.output)
    print(json.dumps({"output": str(Path(args.output)), "failures": profile.failures}, sort_keys=True))


if __name__ == "__main__":
    main()
