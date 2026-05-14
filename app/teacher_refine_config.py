# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated CLI for the legacy teacher-guided refinement path."""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soma_retargeter.robot_registry_parser as robot_registry_parser
from soma_retargeter.teacher_refinement import refine_registered_robot_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated legacy teacher-guided retargeter config refinement. "
            "Normal configs now generate contact-aware virtual sole anchors automatically."
        )
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=robot_registry_parser.get_active_robot_name(),
        help="Registered robot name or alias from params.py.",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="unitree_g1",
        help="Deprecated metadata-only teacher robot name.",
    )
    parser.add_argument(
        "--retargeter-config",
        type=str,
        default=None,
        help="Retargeter config path. Defaults to params.py registration.",
    )
    parser.add_argument(
        "--capability-profile",
        type=str,
        default=None,
        help="Optional robot_capability.json/yaml path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the final chosen config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute reports in memory without writing debug files or config changes.",
    )
    parser.add_argument(
        "--force-accept",
        action="store_true",
        help="Write the refined config even if acceptance gates fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refine_registered_robot_config(
        args.robot,
        teacher=args.teacher,
        retargeter_config_path=args.retargeter_config,
        capability_profile_path=args.capability_profile,
        output_path=args.output,
        force_accept=args.force_accept,
        write=not args.dry_run,
    )
    print(f"[INFO] Robot: {result['robot']}")
    print(f"[INFO] Template: {result['template']}")
    print(f"[INFO] Decision: {result['decision']}")
    print(f"[INFO] Reason: {result['reason']}")
    if not args.dry_run:
        print(f"[INFO] Config path: {result['output_path']}")


if __name__ == "__main__":
    main()
