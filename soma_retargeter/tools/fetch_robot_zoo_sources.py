"""Fetch pinned public Robot Zoo sources without executing upstream code.

This tool consumes ``assets/robot_zoo/source_lock.json``. It performs fixed-ref
Git checkouts into a user cache, verifies the selected model file against the
Git blob SHA recorded in the lock, and writes a local resolved manifest whose
``source_path`` fields point to the checked-out files.

No fetched repository is imported or executed. Fetch-only assets remain in the
cache and are never copied into this repository.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Iterable


DEFAULT_MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
DEFAULT_OVERLAY = Path("assets/robot_zoo/manifest_overlay_assets.json")
DEFAULT_SOURCE_LOCK = Path("assets/robot_zoo/source_lock.json")
_GITHUB_HTTPS_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


@dataclass(frozen=True)
class SourceSpec:
    model_id: str
    repository: str
    ref: str
    cache_key: str
    model_path: str
    sparse_paths: tuple[str, ...]
    license: str
    redistribution: str
    required: bool
    github_blob_sha: str


@dataclass(frozen=True)
class SourceResult:
    model_id: str
    status: str
    repository: str
    ref: str
    checkout_commit: str | None
    checkout_path: str | None
    model_path: str | None
    model_sha256: str | None
    git_blob_sha: str | None
    reason: str

    def to_json(self) -> dict:
        return self.__dict__.copy()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe {field}: {value!r}")
    return path.as_posix()


def _validate_spec(model_id: str, payload: dict) -> SourceSpec:
    repository = str(payload.get("repository", ""))
    ref = str(payload.get("ref", ""))
    cache_key = str(payload.get("cache_key", ""))
    model_path = _safe_relative_path(str(payload.get("model_path", "")), field="model_path")
    sparse_paths = tuple(
        _safe_relative_path(str(item), field="sparse_path") for item in payload.get("sparse_paths", [])
    )
    if payload.get("transport") != "git":
        raise ValueError(f"{model_id}: only git transport is supported")
    if not _GITHUB_HTTPS_RE.fullmatch(repository):
        raise ValueError(f"{model_id}: repository must be a GitHub HTTPS URL, got {repository!r}")
    if not _SAFE_REF_RE.fullmatch(ref) or ref.startswith("-"):
        raise ValueError(f"{model_id}: unsafe or empty ref {ref!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_key):
        raise ValueError(f"{model_id}: unsafe cache_key {cache_key!r}")
    blob_sha = str(payload.get("github_blob_sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", blob_sha):
        raise ValueError(f"{model_id}: github_blob_sha must be a 40-character Git blob SHA")
    return SourceSpec(
        model_id=model_id,
        repository=repository,
        ref=ref,
        cache_key=cache_key,
        model_path=model_path,
        sparse_paths=sparse_paths or (str(PurePosixPath(model_path).parent),),
        license=str(payload.get("license", "unknown")),
        redistribution=str(payload.get("redistribution", "fetch_only")),
        required=bool(payload.get("required", False)),
        github_blob_sha=blob_sha,
    )


def load_specs(lock_path: Path) -> dict[str, SourceSpec]:
    payload = _read_json(lock_path)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported source lock schema: {payload.get('schema_version')!r}")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("source lock has no sources")
    return {model_id: _validate_spec(model_id, raw) for model_id, raw in raw_sources.items()}


def _ref_token(ref: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)
    return clean[:40] or hashlib.sha256(ref.encode()).hexdigest()[:16]


def _git(repo_dir: Path, *args: str, timeout: int = 900) -> str:
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LFS_SKIP_SMUDGE": "1",
        }
    )
    completed = subprocess.run(
        ["git", "-c", "protocol.file.allow=never", "-C", str(repo_dir), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )
    return completed.stdout.strip()


def _prepare_empty_repository(path: Path, repository: str, sparse_paths: Iterable[str]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    _git(path, "init")
    _git(path, "remote", "add", "origin", repository)
    _git(path, "sparse-checkout", "init", "--no-cone")
    _git(path, "sparse-checkout", "set", "--no-cone", *sorted(set(sparse_paths)))


def _checkout_group(
    specs: list[SourceSpec],
    *,
    cache_root: Path,
    fetch: bool,
    dry_run: bool,
) -> tuple[Path | None, str | None, str]:
    first = specs[0]
    sparse_paths = sorted({path for spec in specs for path in spec.sparse_paths})
    checkout_dir = cache_root / "sources" / first.cache_key / _ref_token(first.ref)

    if dry_run:
        return checkout_dir, None, "dry_run"

    if not checkout_dir.exists():
        if not fetch:
            return None, None, "checkout is absent and --fetch was not supplied"
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_parent = checkout_dir.parent
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{first.cache_key}-", dir=temp_parent))
        try:
            _prepare_empty_repository(temp_dir, first.repository, sparse_paths)
            _git(temp_dir, "fetch", "--depth", "1", "--filter=blob:none", "origin", first.ref)
            _git(temp_dir, "checkout", "--detach", "FETCH_HEAD")
            os.replace(temp_dir, checkout_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    else:
        if not (checkout_dir / ".git").exists():
            raise RuntimeError(f"cache path exists but is not a Git checkout: {checkout_dir}")
        origin = _git(checkout_dir, "remote", "get-url", "origin")
        if origin.rstrip("/") != first.repository.rstrip("/"):
            raise RuntimeError(
                f"cache origin mismatch for {first.cache_key}: expected {first.repository!r}, got {origin!r}"
            )
        _git(checkout_dir, "sparse-checkout", "set", "--no-cone", *sparse_paths)
        if fetch:
            _git(checkout_dir, "fetch", "--depth", "1", "--filter=blob:none", "origin", first.ref)
            _git(checkout_dir, "checkout", "--detach", "FETCH_HEAD")

    head = _git(checkout_dir, "rev-parse", "HEAD")
    return checkout_dir, head, ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(spec: SourceSpec, checkout_dir: Path, checkout_commit: str | None) -> SourceResult:
    model = (checkout_dir / spec.model_path).resolve()
    checkout_resolved = checkout_dir.resolve()
    if checkout_resolved not in model.parents:
        return SourceResult(
            spec.model_id,
            "failed",
            spec.repository,
            spec.ref,
            checkout_commit,
            str(checkout_dir),
            None,
            None,
            None,
            "model path escapes checkout",
        )
    if not model.is_file():
        return SourceResult(
            spec.model_id,
            "failed",
            spec.repository,
            spec.ref,
            checkout_commit,
            str(checkout_dir),
            str(model),
            None,
            None,
            "pinned checkout does not contain the declared model path",
        )
    blob_sha = _git(checkout_dir, "hash-object", spec.model_path)
    if blob_sha != spec.github_blob_sha:
        return SourceResult(
            spec.model_id,
            "failed",
            spec.repository,
            spec.ref,
            checkout_commit,
            str(checkout_dir),
            str(model),
            _sha256_file(model),
            blob_sha,
            f"Git blob SHA mismatch: expected {spec.github_blob_sha}, got {blob_sha}",
        )
    return SourceResult(
        spec.model_id,
        "available",
        spec.repository,
        spec.ref,
        checkout_commit,
        str(checkout_dir),
        str(model),
        _sha256_file(model),
        blob_sha,
        "",
    )


def fetch_sources(
    specs: dict[str, SourceSpec],
    *,
    cache_root: Path,
    fetch: bool,
    include_optional: bool,
    dry_run: bool,
) -> dict[str, SourceResult]:
    selected = {
        model_id: spec
        for model_id, spec in specs.items()
        if spec.required or include_optional
    }
    groups: dict[tuple[str, str, str], list[SourceSpec]] = {}
    for spec in selected.values():
        groups.setdefault((spec.repository, spec.ref, spec.cache_key), []).append(spec)

    results: dict[str, SourceResult] = {}
    for group_specs in groups.values():
        try:
            checkout, head, reason = _checkout_group(
                group_specs,
                cache_root=cache_root,
                fetch=fetch,
                dry_run=dry_run,
            )
        except Exception as exc:
            for spec in group_specs:
                results[spec.model_id] = SourceResult(
                    spec.model_id,
                    "failed",
                    spec.repository,
                    spec.ref,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
            continue
        for spec in group_specs:
            if dry_run:
                results[spec.model_id] = SourceResult(
                    spec.model_id,
                    "planned",
                    spec.repository,
                    spec.ref,
                    None,
                    str(checkout) if checkout else None,
                    str(checkout / spec.model_path) if checkout else None,
                    None,
                    spec.github_blob_sha,
                    "",
                )
            elif checkout is None:
                results[spec.model_id] = SourceResult(
                    spec.model_id,
                    "unavailable",
                    spec.repository,
                    spec.ref,
                    None,
                    None,
                    None,
                    None,
                    None,
                    reason,
                )
            else:
                results[spec.model_id] = _verify_source(spec, checkout, head)
    return results


def _merge_manifest(base_path: Path, overlay_path: Path, results: dict[str, SourceResult]) -> dict:
    manifest = _read_json(base_path)
    models = [dict(entry) for entry in manifest.get("models", [])]
    by_id = {str(entry["id"]): entry for entry in models}
    if overlay_path.exists():
        overlay = _read_json(overlay_path)
        for addition in overlay.get("additions", []):
            model_id = str(addition["id"])
            if model_id in by_id:
                raise ValueError(f"overlay duplicates manifest id {model_id!r}")
            entry = dict(addition)
            models.append(entry)
            by_id[model_id] = entry
    for model_id, result in results.items():
        entry = by_id.get(model_id)
        if entry is None:
            continue
        entry["source_lock_id"] = model_id
        if result.status == "available" and result.model_path:
            entry["source_path"] = result.model_path
    manifest["models"] = models
    manifest["resolved_source_lock"] = str(DEFAULT_SOURCE_LOCK)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("ROBOT_ZOO_CACHE", "~/.cache/soma_retargeter/robot_zoo")).expanduser(),
    )
    parser.add_argument("--fetch", action="store_true", help="Fetch missing or refresh existing fixed-ref checkouts")
    parser.add_argument("--include-optional", action="store_true", help="Also fetch GPL/CC/LGPL/NASA fetch-only sources")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-required", action="store_true")
    parser.add_argument("--output-inventory", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    specs = load_specs(args.source_lock)
    cache_root = args.cache_root.resolve()
    results = fetch_sources(
        specs,
        cache_root=cache_root,
        fetch=args.fetch,
        include_optional=args.include_optional,
        dry_run=args.dry_run,
    )

    inventory = {
        "schema_version": 1,
        "source_lock": str(args.source_lock),
        "source_lock_sha256": _sha256_file(args.source_lock),
        "cache_root": str(cache_root),
        "fetch": bool(args.fetch),
        "include_optional": bool(args.include_optional),
        "dry_run": bool(args.dry_run),
        "status_counts": {},
        "entries": {model_id: result.to_json() for model_id, result in sorted(results.items())},
    }
    for result in results.values():
        inventory["status_counts"][result.status] = inventory["status_counts"].get(result.status, 0) + 1

    output_inventory = args.output_inventory or cache_root / "source_inventory.json"
    output_inventory.parent.mkdir(parents=True, exist_ok=True)
    output_inventory.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    resolved_manifest = _merge_manifest(args.manifest, args.overlay, results)
    output_manifest = args.output_manifest or cache_root / "resolved_manifest.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(resolved_manifest, indent=2, sort_keys=True) + "\n")

    required_failures = [
        spec.model_id
        for spec in specs.values()
        if spec.required and results.get(spec.model_id, SourceResult(spec.model_id, "unavailable", spec.repository, spec.ref, None, None, None, None, None, "not selected")).status != "available"
    ]
    print(json.dumps({
        "inventory": str(output_inventory),
        "resolved_manifest": str(output_manifest),
        "status_counts": inventory["status_counts"],
        "required_failures": required_failures,
    }, sort_keys=True))
    if args.strict_required and required_failures and not args.dry_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
