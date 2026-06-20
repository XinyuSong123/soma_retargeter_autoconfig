"""CLI for same-source URDF to canonical MJCF conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.model_conversion import (
    compare_runtime_models,
    convert_urdf_to_canonical_mjcf,
    write_conversion_report,
)
from soma_retargeter.robotics.v3.semantic_sites import load_semantic_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source URDF path.")
    parser.add_argument("--output", required=True, help="Canonical MJCF output path.")
    parser.add_argument("--report", default=None, help="Optional conversion/equivalence report JSON.")
    parser.add_argument("--semantic-map", default=None, help="Optional semantic map for neutral FK equivalence.")
    parser.add_argument("--compare", action="store_true", help="Run strict same-source comparison after conversion.")
    args = parser.parse_args()

    report = convert_urdf_to_canonical_mjcf(args.input, args.output)
    if args.compare:
        left = MuJoCoRuntimeModelAdapter(args.input, model_format="urdf")
        right = MuJoCoRuntimeModelAdapter(args.output, model_format="xml")
        semantic_map = load_semantic_map(args.semantic_map) if args.semantic_map else None
        report["strict_equivalence"] = compare_runtime_models(left, right, semantic_map=semantic_map)
        left.close()
        right.close()

    if args.report:
        write_conversion_report(report, args.report)
    print(json.dumps({"output": str(Path(args.output)), "report": args.report, "sha256": report["output_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
