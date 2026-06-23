#!/usr/bin/env python3
"""Extract the Step 2.3 capability baseline failure ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soma_retargeter.robotics.v3.failure_analysis import (
    DEFAULT_BASELINE_LEDGER_PATH,
    DEFAULT_FAILURE_REPORT_DIR,
    build_baseline_failure_ledger,
    write_baseline_failure_ledger,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-dir", default=str(DEFAULT_FAILURE_REPORT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_BASELINE_LEDGER_PATH))
    parser.add_argument("--check", action="store_true", help="fail if the output file differs from source reports")
    args = parser.parse_args(argv)

    failure_dir = Path(args.failure_dir)
    output = Path(args.output)
    ledger = build_baseline_failure_ledger(failure_dir)

    if args.check:
        if not output.exists():
            print(f"Step 2.3 failure ledger check FAILED: missing {output}")
            return 1
        existing = json.loads(output.read_text())
        if existing != ledger:
            print(f"Step 2.3 failure ledger check FAILED: {output} differs from source reports")
            return 1
        print(
            "Step 2.3 failure ledger check PASS: "
            f"{ledger['counts']['failed_rows']} failures across {ledger['counts']['robots']} robots"
        )
        return 0

    write_baseline_failure_ledger(output, failure_dir=failure_dir)
    print(
        "Wrote Step 2.3 failure ledger: "
        f"{output} ({ledger['counts']['failed_rows']} failures across {ledger['counts']['robots']} robots)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
