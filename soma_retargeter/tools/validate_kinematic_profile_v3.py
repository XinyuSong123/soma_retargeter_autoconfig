"""CLI for writing Step-2 validation artifacts."""

from __future__ import annotations

import argparse
import json

from soma_retargeter.robotics.v3.robot_zoo import DEFAULT_ROBOT_ZOO_MANIFEST_PATH
from soma_retargeter.robotics.v3.validation import DEFAULT_LOW_DISCREPANCY_COUNT, write_validation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_ROBOT_ZOO_MANIFEST_PATH))
    parser.add_argument("--output-dir", default="artifacts/retargeting_v3_step2")
    parser.add_argument("--low-discrepancy-count", type=int, default=DEFAULT_LOW_DISCREPANCY_COUNT)
    parser.add_argument("--deterministic-rerun", action="store_true")
    parser.add_argument("--allow-source-fetch", action="store_true")
    args = parser.parse_args()
    summary = write_validation_artifacts(
        args.output_dir,
        manifest_path=args.manifest,
        low_discrepancy_count=args.low_discrepancy_count,
        deterministic_rerun=args.deterministic_rerun,
        allow_source_fetch=args.allow_source_fetch,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
