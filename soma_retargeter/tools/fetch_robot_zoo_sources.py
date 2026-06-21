"""Fetch fixed-ref public Robot Zoo sources into a local cache.

The lock file is data-only. This tool never imports or executes fetched code.
It verifies each selected model with the Git blob SHA recorded in the lock and
writes a local resolved manifest containing concrete ``source_path`` values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile


DEFAULT_LOCK = Path("assets/robot_zoo/source_lock.json")
DEFAULT_MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
DEFAULT_OVERLAY = Path("assets/robot_zoo/manifest_overlay_assets.json")
GITHUB_URL = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$"
)
SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def load_sources(path: Path) -> dict[str, dict]:
    payload = read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported source-lock schema")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source lock contains no sources")
    checked: dict[str, dict] = {}
    for model_id, raw in sources.items():
        source = dict(raw)
        repository = str(source.get("repository", ""))
        ref = str(source.get("ref", ""))
        cache_key = str(source.get("cache_key", ""))
        if source.get("transport") != "git":
            raise ValueError(f"{model_id}: transport must be git")
        if not GITHUB_URL.fullmatch(repository):
            raise ValueError(f"{model_id}: unsafe repository URL")
        if not SAFE_REF.fullmatch(ref) or ref.startswith("-"):
            raise ValueError(f"{model_id}: unsafe ref")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_key):
            raise ValueError(f"{model_id}: unsafe cache_key")
        source["model_path"] = safe_relative(
            str(source.get("model_path", "")), "model_path"
        )
        source["sparse_paths"] = [
            safe_relative(str(item), "sparse_path")
            for item in source.get("sparse_paths", [])
        ]
        if not re.fullmatch(
            r"[0-9a-f]{40}", str(source.get("github_blob_sha", ""))
        ):
            raise ValueError(f"{model_id}: invalid github_blob_sha")
        checked[model_id] = source
    return checked


def git(repo: Path, *args: str, timeout: int = 900) -> str:
    env = os.environ.copy()
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_LFS_SKIP_SMUDGE="1",
    )
    result = subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=never",
            "-C",
            str(repo),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result.stdout.strip()


def ref_token(ref: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)[:40]
        or hashlib.sha256(ref.encode()).hexdigest()[:16]
    )


def checkout_group(
    group: list[tuple[str, dict]],
    cache_root: Path,
    fetch: bool,
    dry_run: bool,
) -> tuple[Path | None, str | None, str]:
    first = group[0][1]
    target = (
        cache_root
        / "sources"
        / first["cache_key"]
        / ref_token(first["ref"])
    )
    sparse = sorted(
        {
            path
            for _, source in group
            for path in source.get("sparse_paths", [])
        }
    )
    if dry_run:
        return target, None, ""
    if not target.exists():
        if not fetch:
            return None, None, "checkout absent; rerun with --fetch"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(
            tempfile.mkdtemp(
                prefix=f".{first['cache_key']}-", dir=target.parent
            )
        )
        try:
            git(temp, "init")
            git(temp, "remote", "add", "origin", first["repository"])
            git(temp, "sparse-checkout", "init", "--no-cone")
            if sparse:
                git(temp, "sparse-checkout", "set", "--no-cone", *sparse)
            git(
                temp,
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                first["ref"],
            )
            git(temp, "checkout", "--detach", "FETCH_HEAD")
            os.replace(temp, target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
    else:
        if not (target / ".git").exists():
            raise RuntimeError(f"not a Git checkout: {target}")
        origin = git(target, "remote", "get-url", "origin")
        if origin.rstrip("/") != first["repository"].rstrip("/"):
            raise RuntimeError(f"origin mismatch in {target}")
        if sparse:
            git(target, "sparse-checkout", "set", "--no-cone", *sparse)
        if fetch:
            git(
                target,
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                first["ref"],
            )
            git(target, "checkout", "--detach", "FETCH_HEAD")
    return target, git(target, "rev-parse", "HEAD"), ""


def verify(
    model_id: str,
    source: dict,
    checkout: Path,
    head: str | None,
) -> dict:
    root = checkout.resolve()
    model = (checkout / source["model_path"]).resolve()
    if root not in model.parents or not model.is_file():
        return {
            "status": "failed",
            "reason": "declared model path is absent or escapes checkout",
        }
    blob_sha = git(checkout, "hash-object", source["model_path"])
    expected = source["github_blob_sha"]
    status = "available" if blob_sha == expected else "failed"
    return {
        "status": status,
        "reason": ""
        if status == "available"
        else f"Git blob SHA mismatch: expected {expected}, got {blob_sha}",
        "repository": source["repository"],
        "ref": source["ref"],
        "checkout_commit": head,
        "checkout_path": str(checkout),
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "git_blob_sha": blob_sha,
    }


def merge_manifest(
    manifest_path: Path,
    overlay_path: Path,
    results: dict[str, dict],
) -> dict:
    manifest = read_json(manifest_path)
    models = [dict(entry) for entry in manifest.get("models", [])]
    by_id = {str(entry["id"]): entry for entry in models}
    if overlay_path.exists():
        for addition in read_json(overlay_path).get("additions", []):
            model_id = str(addition["id"])
            if model_id in by_id:
                raise ValueError(f"overlay duplicates model id {model_id}")
            entry = dict(addition)
            models.append(entry)
            by_id[model_id] = entry
    for model_id, result in results.items():
        if model_id not in by_id:
            continue
        by_id[model_id]["source_lock_id"] = model_id
        if result.get("status") == "available":
            by_id[model_id]["source_path"] = result["model_path"]
    manifest["models"] = models
    manifest["resolved_source_lock"] = str(DEFAULT_LOCK)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ROBOT_ZOO_CACHE",
                "~/.cache/soma_retargeter/robot_zoo",
            )
        ).expanduser(),
    )
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-required", action="store_true")
    parser.add_argument("--output-inventory", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args(argv)

    sources = load_sources(args.source_lock)
    selected = {
        model_id: source
        for model_id, source in sources.items()
        if bool(source.get("required")) or args.include_optional
    }
    groups: dict[tuple[str, str, str], list[tuple[str, dict]]] = {}
    for model_id, source in selected.items():
        key = (
            source["repository"],
            source["ref"],
            source["cache_key"],
        )
        groups.setdefault(key, []).append((model_id, source))

    cache_root = args.cache_root.expanduser().resolve()
    results: dict[str, dict] = {}
    for group in groups.values():
        try:
            checkout, head, reason = checkout_group(
                group, cache_root, args.fetch, args.dry_run
            )
        except Exception as exc:
            for model_id, source in group:
                results[model_id] = {
                    "status": "failed",
                    "repository": source["repository"],
                    "ref": source["ref"],
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            continue
        for model_id, source in group:
            if args.dry_run:
                assert checkout is not None
                results[model_id] = {
                    "status": "planned",
                    "repository": source["repository"],
                    "ref": source["ref"],
                    "checkout_path": str(checkout),
                    "model_path": str(checkout / source["model_path"]),
                    "git_blob_sha": source["github_blob_sha"],
                    "reason": "",
                }
            elif checkout is None:
                results[model_id] = {
                    "status": "unavailable",
                    "repository": source["repository"],
                    "ref": source["ref"],
                    "reason": reason,
                }
            else:
                results[model_id] = verify(
                    model_id, source, checkout, head
                )

    counts: dict[str, int] = {}
    for result in results.values():
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1

    inventory = {
        "schema_version": 1,
        "source_lock": str(args.source_lock),
        "source_lock_sha256": sha256_file(args.source_lock),
        "cache_root": str(cache_root),
        "fetch": bool(args.fetch),
        "include_optional": bool(args.include_optional),
        "dry_run": bool(args.dry_run),
        "status_counts": counts,
        "entries": dict(sorted(results.items())),
    }
    inventory_path = (
        args.output_inventory or cache_root / "source_inventory.json"
    )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )

    resolved = merge_manifest(args.manifest, args.overlay, results)
    resolved_path = (
        args.output_manifest or cache_root / "resolved_manifest.json"
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    )

    required_failures = sorted(
        model_id
        for model_id, source in sources.items()
        if bool(source.get("required"))
        and results.get(model_id, {}).get("status") != "available"
    )
    print(
        json.dumps(
            {
                "inventory": str(inventory_path),
                "resolved_manifest": str(resolved_path),
                "status_counts": counts,
                "required_failures": required_failures,
            },
            sort_keys=True,
        )
    )
    if args.strict_required and required_failures and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
