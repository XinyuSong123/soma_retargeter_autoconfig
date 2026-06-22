"""Create deterministic, license-aware Robot Zoo kinematic snapshots.

This tool only processes entries declared in ``robot_zoo_manifest.json``. Full
upstream repositories live in an external cache. Only entries explicitly marked
``kinematic_snapshot`` are eligible for vendoring; fetch-only entries are
recorded in the lock file but never copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import mujoco

from soma_retargeter.robotics.v3.robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    load_robot_zoo_manifest,
    resolve_robot_source,
    sha256_file,
)


GENERATOR_VERSION = "robot-zoo-snapshot-v1"
DEFAULT_OUTPUT_ROOT = Path("assets/robot_zoo/snapshots")
DEFAULT_LOCK_PATH = Path("assets/robot_zoo/robot_zoo_lock.json")
FORBIDDEN_LICENSE_MARKERS = (
    "gpl",
    "lgpl",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nd",
    "nasa",
    "non-commercial",
)
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt", "NOTICE", "NOTICE.txt")
MESH_SUFFIXES = (".stl", ".dae", ".obj", ".ply", ".glb", ".gltf", ".fbx", ".3ds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_ROBOT_ZOO_MANIFEST_PATH))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--lock-output", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_root = Path(args.output_root)
    lock_path = Path(args.lock_output)
    manifest = load_robot_zoo_manifest(args.manifest)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    max_snapshot_bytes = int(manifest.payload.get("policies", {}).get("max_committed_snapshot_bytes_per_robot", 2_000_000))
    lock_entries: dict[str, dict] = {}
    blockers: list[str] = []

    for entry in manifest.entries:
        resolved = resolve_robot_source(entry, cache_root=cache_root, allow_fetch=True)
        record = {
            "id": entry.id,
            "description_name": entry.description_name,
            "format": entry.model_format,
            "license": entry.raw.get("license"),
            "redistribution": entry.redistribution,
            "required": entry.required,
            "source_family": entry.source_family,
            "resolution_status": resolved.status,
            "snapshot_status": "not_applicable",
        }
        if not resolved.available or resolved.path is None:
            record["reason"] = resolved.reason
            lock_entries[entry.id] = record
            if entry.required:
                blockers.append(f"{entry.id}: source unavailable: {resolved.reason}")
            continue

        source_path = resolved.path.resolve()
        source_meta = _source_metadata(source_path)
        record.update(source_meta)
        record["source_sha256"] = sha256_file(source_path)

        if entry.redistribution in {"fetch_only", "excluded"}:
            record["snapshot_status"] = "fetch_only"
            record["reason"] = "manifest redistribution policy forbids vendoring"
            lock_entries[entry.id] = record
            continue
        if entry.redistribution == "local_existing":
            record["snapshot_status"] = "local_existing"
            record["reason"] = "project-local model already exists in the repository"
            lock_entries[entry.id] = record
            continue
        if entry.redistribution != "kinematic_snapshot":
            record["snapshot_status"] = "policy_blocked"
            record["reason"] = f"unsupported redistribution policy: {entry.redistribution}"
            lock_entries[entry.id] = record
            blockers.append(f"{entry.id}: unsupported redistribution policy {entry.redistribution}")
            continue

        license_name = str(entry.raw.get("license", ""))
        if _license_is_forbidden(license_name):
            record["snapshot_status"] = "license_blocked"
            record["reason"] = f"license is not eligible for an Apache-repository snapshot: {license_name}"
            lock_entries[entry.id] = record
            blockers.append(f"{entry.id}: manifest marks non-permissive license as kinematic_snapshot")
            continue

        snapshot_dir = output_root / entry.id
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True)

        try:
            if entry.model_format == "urdf":
                model_file = snapshot_dir / "model.urdf"
                _write_urdf_snapshot(source_path, model_file)
            elif entry.model_format == "mjcf":
                model_file = snapshot_dir / "model.xml"
                _write_mjcf_snapshot(source_path, model_file)
            else:
                raise ValueError(f"unsupported model format: {entry.model_format}")

            license_files = _copy_license_files(source_path, snapshot_dir / "LICENSES")
            if not license_files:
                raise RuntimeError("no upstream LICENSE/COPYING/NOTICE file was found")

            snapshot_size = sum(path.stat().st_size for path in snapshot_dir.rglob("*") if path.is_file())
            if snapshot_size > max_snapshot_bytes:
                raise RuntimeError(f"snapshot size {snapshot_size} exceeds policy limit {max_snapshot_bytes}")

            source_payload = {
                "robot_id": entry.id,
                "description_name": entry.description_name,
                "upstream_repository": source_meta.get("upstream_repository"),
                "upstream_ref": source_meta.get("upstream_ref"),
                "source_file": source_meta.get("source_file"),
                "source_sha256": record["source_sha256"],
                "format": entry.model_format,
                "license": license_name,
                "license_files": license_files,
                "license_sha256": license_files[0]["sha256"],
                "redistribution": entry.redistribution,
                "generator_version": GENERATOR_VERSION,
                "snapshot_file": model_file.name,
                "snapshot_sha256": sha256_file(model_file),
                "snapshot_size_bytes": snapshot_size,
            }
            (snapshot_dir / "SOURCE.json").write_text(json.dumps(source_payload, indent=2, sort_keys=True) + "\n")
            _assert_snapshot_safe(snapshot_dir)
            record.update(
                {
                    "snapshot_status": "vendored",
                    "snapshot_path": str(snapshot_dir.relative_to(repo_root)),
                    "snapshot_file": model_file.name,
                    "snapshot_sha256": source_payload["snapshot_sha256"],
                    "snapshot_size_bytes": snapshot_size,
                    "license_files": license_files,
                }
            )
        except Exception as exc:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            record["snapshot_status"] = "snapshot_failed"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            blockers.append(f"{entry.id}: {record['reason']}")
        lock_entries[entry.id] = record

    lock_payload = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "manifest_path": str(Path(args.manifest)),
        "manifest_sha256": manifest.sha256,
        "cache_is_external": not _is_relative_to(cache_root, repo_root),
        "entries": lock_entries,
        "totals": _totals(lock_entries),
        "blockers": blockers,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"lock": str(lock_path), "totals": lock_payload["totals"], "blocker_count": len(blockers)}, sort_keys=True))
    if blockers and not args.allow_partial:
        raise SystemExit("snapshot generation has blockers; inspect robot_zoo_lock.json or rerun with --allow-partial for diagnostics")


def _write_urdf_snapshot(source_path: Path, output_path: Path) -> None:
    root = ET.parse(source_path).getroot()
    remove_tags = {"visual", "collision", "gazebo", "transmission", "ros2_control"}
    for parent in list(root.iter()):
        for child in list(parent):
            if _local_name(child.tag) in remove_tags:
                parent.remove(child)
    _indent(root)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    _assert_no_mesh_references(output_path)


def _write_mjcf_snapshot(source_path: Path, output_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(source_path))
    with tempfile.TemporaryDirectory(prefix="soma-mjcf-snapshot-") as temp_dir:
        canonical_path = Path(temp_dir) / "canonical.xml"
        mujoco.mj_saveLastXML(str(canonical_path), model)
        root = ET.parse(canonical_path).getroot()

    _add_compiled_inertials(root, model)
    remove_top_level = {"asset", "visual", "contact", "equality", "tendon", "actuator", "sensor", "keyframe"}
    for child in list(root):
        if _local_name(child.tag) in remove_top_level:
            root.remove(child)
    for parent in list(root.iter()):
        for child in list(parent):
            if _local_name(child.tag) in {"geom", "camera", "light"}:
                parent.remove(child)
    _indent(root)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    _assert_no_mesh_references(output_path)
    mujoco.MjModel.from_xml_path(str(output_path))


def _add_compiled_inertials(root: ET.Element, model: mujoco.MjModel) -> None:
    for body in root.iter():
        if _local_name(body.tag) != "body":
            continue
        name = body.attrib.get("name")
        if not name or any(_local_name(child.tag) == "inertial" for child in body):
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0 or float(model.body_mass[body_id]) <= 0.0:
            continue
        inertial = ET.Element(
            "inertial",
            {
                "pos": _numbers(model.body_ipos[body_id]),
                "quat": _numbers(model.body_iquat[body_id]),
                "mass": f"{float(model.body_mass[body_id]):.17g}",
                "diaginertia": _numbers(model.body_inertia[body_id]),
            },
        )
        body.insert(0, inertial)


def _source_metadata(source_path: Path) -> dict:
    try:
        repo_root = Path(_git(source_path.parent, "rev-parse", "--show-toplevel"))
        upstream_ref = _git(repo_root, "rev-parse", "HEAD")
        upstream_repository = _git(repo_root, "remote", "get-url", "origin")
        source_file = str(source_path.relative_to(repo_root))
    except Exception:
        repo_root = source_path.parent
        upstream_ref = None
        upstream_repository = None
        source_file = source_path.name
    return {
        "upstream_repository": upstream_repository,
        "upstream_ref": upstream_ref,
        "source_file": source_file,
        "source_repository_detected": upstream_repository is not None and upstream_ref is not None,
    }


def _copy_license_files(source_path: Path, output_dir: Path) -> list[dict]:
    candidates: list[Path] = []
    try:
        repo_root = Path(_git(source_path.parent, "rev-parse", "--show-toplevel"))
    except Exception:
        repo_root = source_path.parent
    search_roots = [repo_root, source_path.parent]
    for root in search_roots:
        for name in LICENSE_NAMES:
            candidate = root / name
            if candidate.is_file() and candidate.stat().st_size <= 1_000_000:
                candidates.append(candidate)
        if root.exists():
            for candidate in root.glob("LICENSE*"):
                if candidate.is_file() and candidate.stat().st_size <= 1_000_000:
                    candidates.append(candidate)
    unique = sorted({candidate.resolve() for candidate in candidates})
    if not unique:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(unique):
        name = source.name if index == 0 else f"{index:02d}_{source.name}"
        target = output_dir / name
        shutil.copyfile(source, target)
        records.append({"path": str(Path("LICENSES") / name), "sha256": sha256_file(target)})
    return records


def _assert_snapshot_safe(snapshot_dir: Path) -> None:
    denylist = [item.strip().lower() for item in os.environ.get("PRIVATE_ASSET_DENYLIST", "cxxx_190").split(",") if item.strip()]
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in MESH_SUFFIXES:
            raise RuntimeError(f"visual mesh was included in snapshot: {path.name}")
        text = path.read_text(errors="ignore").lower()
        for token in denylist:
            if token in text:
                raise RuntimeError(f"private-asset denylist token found: {token}")
        if re.search(r"/(?:home|mnt|Users|private/var)/", text):
            raise RuntimeError(f"local absolute path leaked into snapshot: {path.name}")


def _assert_no_mesh_references(path: Path) -> None:
    text = path.read_text(errors="replace").lower()
    if any(suffix in text for suffix in MESH_SUFFIXES):
        raise RuntimeError(f"snapshot retains a visual mesh reference: {path}")
    if "package://" in text:
        raise RuntimeError(f"snapshot retains a package URI: {path}")


def _license_is_forbidden(license_name: str) -> bool:
    lowered = license_name.lower()
    return any(marker in lowered for marker in FORBIDDEN_LICENSE_MARKERS)


def _totals(entries: dict[str, dict]) -> dict:
    totals: dict[str, int] = {"entries": len(entries)}
    for record in entries.values():
        key = str(record.get("snapshot_status", "unknown"))
        totals[key] = totals.get(key, 0) + 1
    totals["source_available"] = sum(1 for record in entries.values() if record.get("resolution_status") == "available")
    return totals


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _indent(root: ET.Element) -> None:
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover
        pass


if __name__ == "__main__":
    main()
