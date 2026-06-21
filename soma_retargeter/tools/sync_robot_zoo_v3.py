"""Robot Zoo source resolution and sync-plan CLI for Step-2 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soma_retargeter.robotics.v3.robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    display_path,
    load_robot_zoo_manifest,
    resolve_robot_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_ROBOT_ZOO_MANIFEST_PATH))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-fetch", action="store_true")
    args = parser.parse_args()

    manifest = load_robot_zoo_manifest(args.manifest)
    inventory = {
        "schema_version": 1,
        "manifest": {
            "path": display_path(manifest.path),
            "sha256": manifest.sha256,
            "model_count": len(manifest.entries),
        },
        "dry_run": bool(args.dry_run),
        "entries": {},
    }
    for entry in manifest.entries:
        resolved = resolve_robot_source(entry, cache_root=args.cache_root, allow_fetch=args.allow_fetch)
        inventory["entries"][entry.id] = resolved.to_json(
            manifest_path=manifest.path,
            manifest_sha256=manifest.sha256,
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, sort_keys=True))


if __name__ == "__main__":
    main()
