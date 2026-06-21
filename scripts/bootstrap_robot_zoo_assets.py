#!/usr/bin/env python3
"""Fetch and verify the public Robot Zoo sources at their pinned revisions.

This command only uses the public sources declared by
``assets/robot_zoo/robot_zoo_manifest.json`` and
``assets/robot_zoo/source_lock.json``. It never scans unrelated local model
directories and never commits fetched assets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
DEFAULT_LOCK = Path("assets/robot_zoo/source_lock.json")
DEFAULT_CACHE = Path("assets/robot_zoo/cache")
DEFAULT_OUTPUT = Path("assets/robot_zoo/resolved_source_inventory.json")


def _run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display(path: Path, cache_root: Path) -> str:
    path = path.resolve()
    try:
        return "${ROBOT_ZOO_CACHE}/" + str(path.relative_to(cache_root.resolve()))
    except ValueError:
        pass
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return "${LOCAL_SOURCE_PATH}/" + path.name


def _require_dependency_versions(lock: dict[str, Any]) -> dict[str, str]:
    expected = lock["dependencies"]
    versions: dict[str, str] = {}
    for distribution in ("robot_descriptions", "mujoco", "GitPython"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"missing asset dependency {distribution!r}; run "
                "python -m pip install -e '.[robot-zoo]'"
            ) from exc
    required_robot_descriptions = expected["robot_descriptions"].removeprefix("==")
    if versions["robot_descriptions"] != required_robot_descriptions:
        raise RuntimeError(
            "robot_descriptions version mismatch: "
            f"expected {required_robot_descriptions}, got {versions['robot_descriptions']}"
        )
    try:
        collada = importlib.import_module("collada")
    except Exception as exc:
        raise RuntimeError(
            "pycollada is required by public URDFs that reference DAE meshes; "
            "run python -m pip install -e '.[robot-zoo]'"
        ) from exc
    versions["pycollada"] = str(getattr(collada, "__version__", "installed"))
    return versions


def _validate_public_git_url(url: str) -> None:
    if not url.startswith("https://github.com/") or not url.endswith(".git"):
        raise ValueError(f"only pinned public GitHub repositories are allowed: {url!r}")


def _ensure_repository(
    *,
    url: str,
    revision: str,
    destination: Path,
    verify_only: bool,
) -> dict[str, Any]:
    _validate_public_git_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        if verify_only:
            return {
                "status": "source_unavailable",
                "reason": f"repository is not cached: {destination}",
            }
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"refusing to clone into non-empty non-git directory: {destination}")
        _run(["git", "clone", "--no-checkout", url, str(destination)])
    if not verify_only:
        _run(["git", "fetch", "--tags", "--force", "origin", revision], cwd=destination)
        _run(["git", "checkout", "--detach", revision], cwd=destination)
        _run(["git", "reset", "--hard", revision], cwd=destination)
        _run(["git", "clean", "-ffd"], cwd=destination)
    head = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    expected = _run(["git", "rev-parse", f"{revision}^{{commit}}"], cwd=destination)
    if head != expected:
        raise RuntimeError(
            f"repository revision mismatch for {destination}: expected {expected}, got {head}"
        )
    dirty = _run(["git", "status", "--porcelain"], cwd=destination)
    if dirty:
        raise RuntimeError(f"public source cache is dirty: {destination}")
    return {
        "status": "available",
        "repository": url,
        "revision": revision,
        "head": head,
        "cache_path": str(destination),
    }


def _robot_descriptions_source(
    entry: dict[str, Any],
    *,
    cache_root: Path,
    verify_only: bool,
) -> dict[str, Any]:
    description_name = entry.get("description_name")
    if not description_name:
        return {"status": "source_unavailable", "reason": "missing description_name"}
    env = dict(os.environ)
    descriptions_cache = cache_root / "robot_descriptions"
    descriptions_cache.mkdir(parents=True, exist_ok=True)
    env["ROBOT_DESCRIPTIONS_CACHE"] = str(descriptions_cache)
    os.environ["ROBOT_DESCRIPTIONS_CACHE"] = str(descriptions_cache)
    if not verify_only:
        _run(
            [sys.executable, "-m", "robot_descriptions", "pull", str(description_name)],
            env=env,
        )
    module_name = f"robot_descriptions.{description_name}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "status": "source_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "module": module_name,
        }
    attr_candidates = ["MJCF_PATH"] if entry["format"] == "mjcf" else ["URDF_PATH", "XACRO_PATH"]
    path = None
    path_attr = None
    for attr in attr_candidates:
        value = getattr(module, attr, None)
        if value and Path(value).exists():
            path = Path(value)
            path_attr = attr
            break
    if path is None:
        return {
            "status": "source_unavailable",
            "reason": f"{module_name} exposes no existing {'/'.join(attr_candidates)}",
            "module": module_name,
        }
    repository_path = getattr(module, "REPOSITORY_PATH", None)
    repository_head = None
    expected_revision = None
    if repository_path and (Path(repository_path) / ".git").exists():
        repository_path = Path(repository_path)
        repository_head = _run(["git", "rev-parse", "HEAD"], cwd=repository_path)
        try:
            repositories = importlib.import_module("robot_descriptions._repositories").REPOSITORIES
            for metadata in repositories.values():
                if Path(metadata.cache_path).name == repository_path.name:
                    expected_revision = _run(
                        ["git", "rev-parse", f"{metadata.commit}^{{commit}}"],
                        cwd=repository_path,
                    )
                    if repository_head != expected_revision:
                        raise RuntimeError(
                            f"{description_name} repository is not at the package-pinned revision: "
                            f"expected {expected_revision}, got {repository_head}"
                        )
                    break
        except AttributeError:
            pass
    return {
        "status": "available",
        "module": module_name,
        "path_attr": path_attr,
        "path": str(path),
        "sha256": _sha256(path),
        "repository_path": str(repository_path) if repository_path else None,
        "repository_head": repository_head,
        "expected_revision": expected_revision,
    }


def _direct_source(
    entry: dict[str, Any],
    *,
    lock: dict[str, Any],
    cache_root: Path,
    verify_only: bool,
) -> dict[str, Any]:
    family = entry["source_family"]
    if family == "local":
        path = Path(entry.get("local_path", "assets/robots/atom01/mjcf/atom01.xml"))
        if not path.exists():
            return {"status": "source_unavailable", "reason": f"missing local model: {path}"}
        return {"status": "available", "path": str(path), "sha256": _sha256(path)}
    if family == "pinned_git":
        provider = {
            "repository": entry["repository_url"],
            "commit": entry["repository_ref"],
            "cache_subdirectory": entry["repository_cache_path"],
        }
        relative_path = entry["repository_path"]
    elif family == "mujoco_menagerie":
        provider = lock["providers"]["mujoco_menagerie"]
        directory = str(entry["notes"]).split("Menagerie directory:", 1)[1].strip().split()[0]
        relative_path = None
    else:
        return {"status": "source_unavailable", "reason": f"unsupported direct family {family!r}"}
    repository_dir = cache_root / provider["cache_subdirectory"]
    repo_result = _ensure_repository(
        url=provider["repository"],
        revision=provider["commit"],
        destination=repository_dir,
        verify_only=verify_only,
    )
    if repo_result["status"] != "available":
        return repo_result
    if relative_path is not None:
        candidates = [repository_dir / relative_path]
    else:
        model_dir = repository_dir / directory
        candidates = [
            model_dir / f"{directory}.xml",
            model_dir / "model.xml",
            model_dir / "scene.xml",
        ]
        if model_dir.exists():
            candidates.extend(
                path for path in sorted(model_dir.glob("*.xml")) if not path.name.startswith("scene")
            )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {
            **repo_result,
            "status": "source_unavailable",
            "reason": "model file is absent at the pinned revision",
            "candidates": [str(candidate) for candidate in candidates],
        }
    return {
        **repo_result,
        "path": str(path),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--required-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    lock = json.loads(args.lock.read_text())
    if manifest.get("schema_version") != 2:
        raise RuntimeError("asset bootstrap requires Robot Zoo manifest schema_version=2")
    versions = _require_dependency_versions(lock)
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOT_ZOO_CACHE"] = str(cache_root)

    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for entry in manifest["models"]:
        if args.required_only and not entry.get("required", False):
            results[entry["id"]] = {
                "status": "skipped_optional",
                "reason": "--required-only",
            }
            continue
        try:
            if entry["source_family"] == "robot_descriptions":
                result = _robot_descriptions_source(
                    entry,
                    cache_root=cache_root,
                    verify_only=args.verify_only,
                )
            else:
                result = _direct_source(
                    entry,
                    lock=lock,
                    cache_root=cache_root,
                    verify_only=args.verify_only,
                )
        except Exception as exc:
            result = {
                "status": "source_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if result.get("path"):
            result["path"] = _display(Path(result["path"]), cache_root)
        if result.get("repository_path"):
            result["repository_path"] = _display(Path(result["repository_path"]), cache_root)
        results[entry["id"]] = result
        if entry.get("required", False) and result["status"] != "available":
            failures.append(f"{entry['id']}: {result.get('reason', result['status'])}")

    inventory = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "source_lock": str(args.lock),
        "cache_root": "${ROBOT_ZOO_CACHE}",
        "verify_only": bool(args.verify_only),
        "versions": versions,
        "model_count": len(manifest["models"]),
        "available_count": sum(result["status"] == "available" for result in results.values()),
        "required_failure_count": len(failures),
        "required_failures": failures,
        "entries": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps(inventory, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
