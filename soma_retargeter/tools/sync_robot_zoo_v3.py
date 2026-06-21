"""Fetch and verify fixed Robot Zoo sources without vendoring them into Git.

Sources are materialized by the pinned ``robot_descriptions==2.0.0`` registry.
The committed ``assets/robot_zoo/source_lock.json`` independently records the
repositories that were unavailable in the previous full run, so version or
registry drift becomes a hard error rather than a silent source substitution.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from soma_retargeter.robotics.v3.robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    display_path,
    load_robot_zoo_manifest,
    resolve_robot_source,
    sha256_file,
)


DEFAULT_SOURCE_LOCK = Path("assets/robot_zoo/source_lock.json")
DEFAULT_OUTPUT = Path("artifacts/retargeting_v3_assets/source_sync.json")
PINNED_ROBOT_DESCRIPTIONS_VERSION = "2.0.0"


def _cache_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "robot_descriptions").resolve()


def _load_registry():
    try:
        version = importlib.metadata.version("robot_descriptions")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "robot_descriptions is required; install the project robot-zoo extra"
        ) from exc
    if version != PINNED_ROBOT_DESCRIPTIONS_VERSION:
        raise RuntimeError(
            f"robot_descriptions=={PINNED_ROBOT_DESCRIPTIONS_VERSION} is required, got {version}"
        )
    from robot_descriptions._cache import clone_to_cache
    from robot_descriptions._repositories import REPOSITORIES

    return version, clone_to_cache, REPOSITORIES


def _verify_source_lock(source_lock: dict, registry: dict) -> list[str]:
    errors: list[str] = []
    for name, expected in sorted(source_lock.get("repositories", {}).items()):
        actual = registry.get(name)
        if actual is None:
            errors.append(f"{name}: missing from robot_descriptions registry")
            continue
        comparisons = {
            "url": (str(actual.url), str(expected.get("url"))),
            "ref": (str(actual.commit), str(expected.get("ref"))),
            "cache_path": (str(actual.cache_path), str(expected.get("cache_path"))),
        }
        for field, (actual_value, expected_value) in comparisons.items():
            if actual_value.rstrip("/") != expected_value.rstrip("/"):
                errors.append(
                    f"{name}: {field} drift: registry={actual_value!r}, lock={expected_value!r}"
                )
    return errors


def _selected_entries(manifest, *, include_fetch_only: bool):
    for entry in manifest.entries:
        if entry.required:
            yield entry
        elif include_fetch_only and entry.redistribution == "fetch_only":
            yield entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_ROBOT_ZOO_MANIFEST_PATH))
    parser.add_argument("--source-lock", default=str(DEFAULT_SOURCE_LOCK))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--include-fetch-only",
        action="store_true",
        help="also fetch optional GPL/CC/NASA sources into the external cache",
    )
    parser.add_argument(
        "--allow-fetch",
        action="store_true",
        help="compatibility flag; fetching is the default unless --verify-only/--dry-run",
    )
    args = parser.parse_args()

    manifest = load_robot_zoo_manifest(args.manifest)
    source_lock_path = Path(args.source_lock)
    source_lock = json.loads(source_lock_path.read_text())
    cache_root = _cache_root(args.cache_root)
    os.environ["ROBOT_DESCRIPTIONS_CACHE"] = str(cache_root)

    version, clone_to_cache, registry = _load_registry()
    lock_errors = _verify_source_lock(source_lock, registry)
    should_fetch = not args.dry_run and not args.verify_only

    repository_actions: dict[str, dict] = {}
    if should_fetch:
        # Direct Menagerie entries do not import a robot_descriptions module, so
        # materialize the pinned aggregator explicitly before normal resolution.
        menagerie_path = Path(clone_to_cache("mujoco_menagerie"))
        repository_actions["mujoco_menagerie"] = {
            "status": "fetched",
            "path": display_path(menagerie_path),
            "ref": str(registry["mujoco_menagerie"].commit),
        }

    entries: dict[str, dict] = {}
    required_failures: list[str] = []
    for entry in _selected_entries(manifest, include_fetch_only=bool(args.include_fetch_only)):
        resolved = resolve_robot_source(
            entry,
            cache_root=cache_root,
            allow_fetch=should_fetch,
        )
        payload = resolved.to_json(
            manifest_path=manifest.path,
            manifest_sha256=manifest.sha256,
        )
        entries[entry.id] = payload
        if entry.required and not resolved.available:
            required_failures.append(entry.id)

    inventory = {
        "schema_version": 2,
        "status": "passed" if not lock_errors and not required_failures else "blocked",
        "robot_descriptions_version": version,
        "manifest": {
            "path": display_path(manifest.path),
            "sha256": manifest.sha256,
            "model_count": len(manifest.entries),
        },
        "source_lock": {
            "path": display_path(source_lock_path),
            "sha256": sha256_file(source_lock_path),
            "registry_errors": lock_errors,
        },
        "cache_root": "${ROBOT_DESCRIPTIONS_CACHE}",
        "dry_run": bool(args.dry_run),
        "verify_only": bool(args.verify_only),
        "include_fetch_only": bool(args.include_fetch_only),
        "repository_actions": repository_actions,
        "entries": entries,
        "required_failures": sorted(required_failures),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, sort_keys=True))

    if lock_errors:
        print("source-lock drift:\n- " + "\n- ".join(lock_errors), file=sys.stderr)
    if required_failures:
        print(
            "required Robot Zoo sources remain unavailable: " + ", ".join(required_failures),
            file=sys.stderr,
        )
    if lock_errors or required_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
