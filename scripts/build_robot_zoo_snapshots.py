#!/usr/bin/env python3
"""Build license-aware, mesh-free Robot Zoo kinematic snapshots and a lock file."""

from __future__ import annotations

import argparse
import importlib
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
MESH_SUFFIXES = (".stl", ".dae", ".obj", ".ply", ".glb", ".gltf", ".fbx", ".3ds")
FORBIDDEN_LICENSE_MARKERS = ("gpl", "lgpl", "cc-by-sa", "cc-by-nc", "cc-by-nd", "nasa", "non-commercial")
LICENSE_GLOBS = ("LICENSE*", "COPYING*", "NOTICE*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_ROBOT_ZOO_MANIFEST_PATH))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", default="assets/robot_zoo/snapshots")
    parser.add_argument("--lock-output", default="assets/robot_zoo/robot_zoo_lock.json")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_root = Path(args.output_root)
    lock_output = Path(args.lock_output)
    manifest = load_robot_zoo_manifest(args.manifest)
    max_bytes = int(manifest.payload.get("policies", {}).get("max_committed_snapshot_bytes_per_robot", 2_000_000))

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    blockers: list[str] = []
    for entry in manifest.entries:
        resolved = resolve_robot_source(entry, cache_root=cache_root, allow_fetch=True)
        row = {
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
            row["reason"] = resolved.reason
            entries[entry.id] = row
            if entry.required:
                blockers.append(f"{entry.id}: source unavailable: {resolved.reason}")
            continue

        source = resolved.path.resolve()
        roots = _description_roots(entry.description_name, source)
        source_meta = _source_metadata(source, roots)
        row.update(source_meta)
        row["source_sha256"] = sha256_file(source)

        if entry.redistribution in {"fetch_only", "excluded"}:
            row.update(snapshot_status="fetch_only", reason="manifest policy forbids vendoring")
            entries[entry.id] = row
            continue
        if entry.redistribution == "local_existing":
            row.update(snapshot_status="local_existing", reason="project-local source already exists")
            entries[entry.id] = row
            continue
        if entry.redistribution != "kinematic_snapshot":
            row.update(snapshot_status="policy_blocked", reason=f"unsupported policy {entry.redistribution}")
            entries[entry.id] = row
            blockers.append(f"{entry.id}: unsupported redistribution policy")
            continue
        if _forbidden_license(str(entry.raw.get("license", ""))):
            row.update(snapshot_status="license_blocked", reason="non-permissive license cannot be vendored")
            entries[entry.id] = row
            blockers.append(f"{entry.id}: non-permissive license marked kinematic_snapshot")
            continue
        if not source_meta["source_repository_detected"]:
            row.update(snapshot_status="snapshot_failed", reason="upstream repository/ref could not be determined")
            entries[entry.id] = row
            blockers.append(f"{entry.id}: upstream repository/ref unavailable")
            continue

        snapshot_dir = output_root / entry.id
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        snapshot_dir.mkdir(parents=True)
        try:
            model_file = snapshot_dir / ("model.urdf" if entry.model_format == "urdf" else "model.xml")
            if entry.model_format == "urdf":
                _write_urdf(source, model_file)
            elif entry.model_format == "mjcf":
                _write_mjcf(source, model_file)
            else:
                raise RuntimeError(f"unsupported model format {entry.model_format}")

            licenses = _copy_licenses(roots, snapshot_dir / "LICENSES")
            if not licenses:
                raise RuntimeError("no upstream LICENSE/COPYING/NOTICE file found")

            source_json = {
                "robot_id": entry.id,
                "description_name": entry.description_name,
                "upstream_repository": source_meta["upstream_repository"],
                "upstream_ref": source_meta["upstream_ref"],
                "source_file": source_meta["source_file"],
                "source_sha256": row["source_sha256"],
                "format": entry.model_format,
                "license": entry.raw.get("license"),
                "license_files": licenses,
                "license_sha256": licenses[0]["sha256"],
                "redistribution": entry.redistribution,
                "generator_version": GENERATOR_VERSION,
                "snapshot_file": model_file.name,
                "snapshot_sha256": sha256_file(model_file),
            }
            (snapshot_dir / "SOURCE.json").write_text(json.dumps(source_json, indent=2, sort_keys=True) + "\n")
            _assert_safe(snapshot_dir)
            size = sum(path.stat().st_size for path in snapshot_dir.rglob("*") if path.is_file())
            if size > max_bytes:
                raise RuntimeError(f"snapshot size {size} exceeds {max_bytes} byte policy limit")
            row.update(
                snapshot_status="vendored",
                snapshot_path=str(snapshot_dir.resolve().relative_to(repo_root)),
                snapshot_file=model_file.name,
                snapshot_sha256=source_json["snapshot_sha256"],
                snapshot_size_bytes=size,
                license_files=licenses,
            )
        except Exception as exc:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            row.update(snapshot_status="snapshot_failed", reason=f"{type(exc).__name__}: {exc}")
            blockers.append(f"{entry.id}: {row['reason']}")
        entries[entry.id] = row

    payload = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "manifest_path": str(Path(args.manifest)),
        "manifest_sha256": manifest.sha256,
        "cache_is_external": not _is_relative_to(cache_root, repo_root),
        "entries": entries,
        "totals": _totals(entries),
        "blockers": blockers,
    }
    lock_output.parent.mkdir(parents=True, exist_ok=True)
    lock_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"lock": str(lock_output), "totals": payload["totals"], "blocker_count": len(blockers)}, sort_keys=True))
    if blockers and not args.allow_partial:
        raise SystemExit("asset snapshot blockers remain; inspect robot_zoo_lock.json")


def _description_roots(description_name: str | None, source: Path) -> list[Path]:
    roots = [source.parent]
    if description_name:
        try:
            module = importlib.import_module(f"robot_descriptions.{description_name}")
            for attr in ("REPOSITORY_PATH", "PACKAGE_PATH"):
                value = getattr(module, attr, None)
                if value:
                    roots.append(Path(value).expanduser().resolve())
        except Exception:
            pass
    expanded = []
    for root in roots:
        expanded.append(root)
        git_root = _git_root(root)
        if git_root is not None:
            expanded.append(git_root)
    return sorted(set(path.resolve() for path in expanded if path.exists()))


def _source_metadata(source: Path, roots: list[Path]) -> dict:
    git_root = next((root for root in roots if (root / ".git").exists()), None)
    if git_root is None:
        git_root = _git_root(source.parent)
    if git_root is None:
        return {
            "upstream_repository": None,
            "upstream_ref": None,
            "source_file": source.name,
            "source_repository_detected": False,
        }
    try:
        return {
            "upstream_repository": _git(git_root, "remote", "get-url", "origin"),
            "upstream_ref": _git(git_root, "rev-parse", "HEAD"),
            "source_file": str(source.relative_to(git_root)),
            "source_repository_detected": True,
        }
    except Exception:
        return {
            "upstream_repository": None,
            "upstream_ref": None,
            "source_file": source.name,
            "source_repository_detected": False,
        }


def _write_urdf(source: Path, output: Path) -> None:
    root = ET.parse(source).getroot()
    remove = {"visual", "collision", "gazebo", "transmission", "ros2_control"}
    for parent in list(root.iter()):
        for child in list(parent):
            if _local(child.tag) in remove:
                parent.remove(child)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    _assert_no_mesh_refs(output)


def _write_mjcf(source: Path, output: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(source))
    with tempfile.TemporaryDirectory(prefix="soma-mjcf-") as temp:
        canonical = Path(temp) / "canonical.xml"
        mujoco.mj_saveLastXML(str(canonical), model)
        root = ET.parse(canonical).getroot()
    _inject_inertials(root, model)
    for child in list(root):
        if _local(child.tag) in {"asset", "visual", "contact", "equality", "tendon", "actuator", "sensor", "keyframe"}:
            root.remove(child)
    for parent in list(root.iter()):
        for child in list(parent):
            if _local(child.tag) in {"geom", "camera", "light"}:
                parent.remove(child)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    _assert_no_mesh_refs(output)
    mujoco.MjModel.from_xml_path(str(output))


def _inject_inertials(root: ET.Element, model: mujoco.MjModel) -> None:
    for body in root.iter():
        if _local(body.tag) != "body" or any(_local(child.tag) == "inertial" for child in body):
            continue
        name = body.attrib.get("name")
        if not name:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0 or float(model.body_mass[body_id]) <= 0.0:
            continue
        body.insert(
            0,
            ET.Element(
                "inertial",
                {
                    "pos": _numbers(model.body_ipos[body_id]),
                    "quat": _numbers(model.body_iquat[body_id]),
                    "mass": f"{float(model.body_mass[body_id]):.17g}",
                    "diaginertia": _numbers(model.body_inertia[body_id]),
                },
            ),
        )


def _copy_licenses(roots: list[Path], destination: Path) -> list[dict]:
    files = []
    for root in roots:
        for pattern in LICENSE_GLOBS:
            files.extend(path for path in root.glob(pattern) if path.is_file() and path.stat().st_size <= 1_000_000)
    unique = sorted(set(path.resolve() for path in files))
    if not unique:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(unique):
        filename = source.name if index == 0 else f"{index:02d}_{source.name}"
        target = destination / filename
        shutil.copyfile(source, target)
        records.append({"path": str(Path("LICENSES") / filename), "sha256": sha256_file(target)})
    return records


def _assert_no_mesh_refs(path: Path) -> None:
    text = path.read_text(errors="replace").lower()
    if "package://" in text or any(suffix in text for suffix in MESH_SUFFIXES):
        raise RuntimeError(f"snapshot retains mesh/package references: {path.name}")


def _assert_safe(root: Path) -> None:
    denylist = [token.strip().lower() for token in os.environ.get("PRIVATE_ASSET_DENYLIST", "cxxx_190").split(",") if token.strip()]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in MESH_SUFFIXES:
            raise RuntimeError(f"mesh file included: {path.name}")
        text = path.read_text(errors="ignore").lower()
        if any(token in text for token in denylist):
            raise RuntimeError(f"private-asset denylist match in {path.name}")
        if re.search(r"/(?:home|mnt|users|private/var)/", text):
            raise RuntimeError(f"absolute local path leaked into {path.name}")


def _forbidden_license(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in FORBIDDEN_LICENSE_MARKERS)


def _totals(entries: dict[str, dict]) -> dict:
    totals = {"entries": len(entries), "source_available": sum(row.get("resolution_status") == "available" for row in entries.values())}
    for row in entries.values():
        status = str(row.get("snapshot_status", "unknown"))
        totals[status] = totals.get(status, 0) + 1
    return totals


def _git_root(path: Path) -> Path | None:
    try:
        return Path(_git(path, "rev-parse", "--show-toplevel")).resolve()
    except Exception:
        return None


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
