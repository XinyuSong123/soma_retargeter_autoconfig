"""Compile Step 3.1 runtime-local V3 profiles for selected fleet rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soma_retargeter.runtime.v3.fleet_inventory import load_fleet_runtime_cases
from soma_retargeter.runtime.v3.runtime_local_profile import close_runtime_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/retargeting_v3_step3_runtime_quality"))
    parser.add_argument("--step2-profile-root", type=Path, default=Path("artifacts/retargeting_v3_step2_capability"))
    parser.add_argument("--lock", type=Path, default=Path("assets/robot_zoo/robot_zoo_lock.json"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/robot_zoo/robot_zoo_manifest.json"))
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--low-discrepancy-count", type=int, default=8)
    args = parser.parse_args(argv)

    cases = load_fleet_runtime_cases(
        lock_path=args.lock,
        manifest_path=args.manifest,
        step2_profile_root=args.step2_profile_root,
    )
    selected = set(args.model_id)
    if selected:
        cases = [case for case in cases if case.model_id in selected]
    closures = [
        close_runtime_profile(
            case,
            artifact_root=args.artifact_root,
            low_discrepancy_count=args.low_discrepancy_count,
        ).to_json()
        for case in cases
    ]
    print(json.dumps({"row_count": len(closures), "rows": closures}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
