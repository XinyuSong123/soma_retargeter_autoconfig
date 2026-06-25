#!/usr/bin/env python3
"""Build true Step 2.3 capability before/after artifacts from the pinned baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soma_retargeter.robotics.v3.failure_analysis import (
    DEFAULT_BASELINE_ARTIFACT_ROOT,
    DEFAULT_BASELINE_COMMIT,
    DEFAULT_BASELINE_SUMMARY_PATH,
    DEFAULT_BEFORE_AFTER_PATH,
    DEFAULT_CAPABILITY_SUMMARY_PATH,
    build_capability_before_after,
    build_true_baseline_summary,
    write_capability_before_after,
    write_true_baseline_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit", default=DEFAULT_BASELINE_COMMIT)
    parser.add_argument("--artifact-root", default=str(DEFAULT_BASELINE_ARTIFACT_ROOT))
    parser.add_argument("--current-summary", default=str(DEFAULT_CAPABILITY_SUMMARY_PATH))
    parser.add_argument("--baseline-output", default=str(DEFAULT_BASELINE_SUMMARY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_BEFORE_AFTER_PATH))
    parser.add_argument("--check", action="store_true", help="fail if committed artifacts differ from generated truth")
    parser.add_argument(
        "--strict-transitions",
        action="store_true",
        help="return nonzero when the generated transition matrix contains illegal baseline transitions",
    )
    args = parser.parse_args(argv)

    baseline_output = Path(args.baseline_output)
    output = Path(args.output)
    baseline = build_true_baseline_summary(
        source_commit=args.baseline_commit,
        artifact_root=args.artifact_root,
    )
    before_after = build_capability_before_after(
        current_summary_path=args.current_summary,
        source_commit=args.baseline_commit,
        artifact_root=args.artifact_root,
    )

    if args.check:
        failures: list[str] = []
        if not baseline_output.exists():
            failures.append(f"missing {baseline_output}")
        elif json.loads(baseline_output.read_text()) != baseline:
            failures.append(f"{baseline_output} differs from pinned baseline")
        if not output.exists():
            failures.append(f"missing {output}")
        elif json.loads(output.read_text()) != before_after:
            failures.append(f"{output} differs from true before/after matrix")
        if failures:
            print("Step 2.3 true baseline check FAILED:")
            for failure in failures:
                print(f"- {failure}")
            return 1
        print(
            "Step 2.3 true baseline check PASS: "
            f"{before_after['row_count']} rows, "
            f"transition_validation={before_after['transition_validation']['status']}, "
            f"final_count_validation={before_after['final_count_validation']['status']}"
        )
        if args.strict_transitions and (
            before_after["transition_validation"]["status"] != "passed"
            or before_after["final_count_validation"]["status"] != "passed"
        ):
            return 2
        return 0

    write_true_baseline_summary(
        baseline_output,
        source_commit=args.baseline_commit,
        artifact_root=args.artifact_root,
    )
    write_capability_before_after(
        output,
        current_summary_path=args.current_summary,
        source_commit=args.baseline_commit,
        artifact_root=args.artifact_root,
    )
    print(
        "Wrote Step 2.3 true baseline artifacts: "
        f"{baseline_output}, {output}; "
        f"transition_validation={before_after['transition_validation']['status']}, "
        f"final_count_validation={before_after['final_count_validation']['status']}"
    )
    if args.strict_transitions and (
        before_after["transition_validation"]["status"] != "passed"
        or before_after["final_count_validation"]["status"] != "passed"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
