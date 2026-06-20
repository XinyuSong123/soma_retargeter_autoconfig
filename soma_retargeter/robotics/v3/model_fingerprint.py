"""Deterministic runtime-model fingerprint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class FileDigest:
    path: str
    role: str
    sha256: str | None
    size_bytes: int | None
    exists: bool

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "exists": self.exists,
        }


def model_fingerprint_payload(
    model_path: str | Path,
    *,
    backend: str,
    model_format: str,
    loader_name: str,
    loader_version: str | None = None,
    conversion_settings: dict | None = None,
    loader_provenance: dict | None = None,
    compiled_summary: dict | None = None,
) -> dict:
    """Return a stable payload and digest for model/runtime truth.

    The payload deliberately records input provenance separately from compiled
    runtime summaries. Raw XML is not treated as FK truth; it is only an input
    whose includes and referenced assets must be auditable.
    """

    root = Path(model_path)
    files = _collect_files(root)
    payload = {
        "schema_version": 1,
        "model_path": _stable_path(root),
        "model_format": model_format,
        "backend": backend,
        "loader": {
            "name": loader_name,
            "version": loader_version or _package_version(loader_name),
        },
        "conversion_settings": _stable_json(conversion_settings or {}),
        "loader_provenance": _stable_json(loader_provenance or {}),
        "files": [file.to_json() for file in files],
        "compiled_summary": _stable_json(compiled_summary or {}),
    }
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(stable).hexdigest()
    return payload


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(model_path: Path) -> list[FileDigest]:
    out: list[FileDigest] = []
    seen: set[Path] = set()
    parsed_includes: set[Path] = set()

    def add(path: Path, role: str) -> None:
        resolved = path.resolve()
        key = resolved if path.exists() else path
        if key in seen:
            return
        seen.add(key)
        if path.exists() and path.is_file():
            out.append(
                FileDigest(
                    path=_stable_path(path),
                    role=role,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                    exists=True,
                )
            )
        else:
            out.append(FileDigest(path=_stable_path(path), role=role, sha256=None, size_bytes=None, exists=False))

    def visit_xml(path: Path, *, root_file: bool = False) -> None:
        key = path.resolve() if path.exists() else path
        if key in parsed_includes:
            return
        parsed_includes.add(key)
        add(path, "primary_model" if root_file else "resolved_include")
        for include in _xml_includes(path):
            visit_xml(include)
        for asset in _xml_referenced_assets(path):
            add(asset, "referenced_asset")

    visit_xml(model_path, root_file=True)
    return sorted(out, key=lambda item: (item.role, item.path))


def _xml_includes(path: Path) -> list[Path]:
    root = _parse_xml(path)
    if root is None:
        return []
    base = path.parent
    includes: list[Path] = []
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        filename = elem.attrib.get("file") or elem.attrib.get("filename")
        if tag == "include" and filename:
            includes.append(_resolve_reference(filename, base))
    return includes


def _xml_referenced_assets(path: Path) -> list[Path]:
    root = _parse_xml(path)
    if root is None:
        return []
    base = path.parent
    refs: list[Path] = []
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        filename = elem.attrib.get("file") or elem.attrib.get("filename")
        if not filename:
            continue
        if tag in {"mesh", "texture", "hfield"} or filename.startswith(("package://", "file://")):
            refs.append(_resolve_reference(filename, base))
    return refs


def _resolve_reference(filename: str, base_dir: Path) -> Path:
    if filename.startswith("file://"):
        return Path(filename.removeprefix("file://")).expanduser()
    if filename.startswith("package://"):
        rest = filename.removeprefix("package://")
        parts = Path(rest).parts
        if not parts:
            return base_dir / rest
        package_name = parts[0]
        rel = Path(*parts[1:]) if len(parts) > 1 else Path()
        for parent in [base_dir, *base_dir.parents]:
            if parent.name == package_name:
                return parent / rel
            candidate = parent / package_name / rel
            if candidate.exists():
                return candidate
        return base_dir.parent / rel
    ref = Path(filename)
    if ref.is_absolute():
        return ref
    return base_dir / ref


def _parse_xml(path: Path) -> ET.Element | None:
    try:
        return ET.fromstring(path.read_text())
    except (OSError, ET.ParseError, UnicodeDecodeError):
        return None


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _stable_json(payload: dict) -> dict:
    return json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _package_version(name: str) -> str | None:
    candidates = {
        "mujoco": "mujoco",
        "newton": "newton",
        "warp": "warp-lang",
    }
    package = candidates.get(name, name)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
