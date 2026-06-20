"""CLI for writing Step-2 validation artifacts."""

from __future__ import annotations

import argparse
import json

from soma_retargeter.robotics.v3.validation import DEFAULT_LOW_DISCREPANCY_COUNT, write_validation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/retargeting_v3_step2")
    parser.add_argument("--low-discrepancy-count", type=int, default=DEFAULT_LOW_DISCREPANCY_COUNT)
    args = parser.parse_args()
    summary = write_validation_artifacts(args.output_dir, low_discrepancy_count=args.low_discrepancy_count)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
