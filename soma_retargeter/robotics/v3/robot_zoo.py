"""Manifest-driven Robot Zoo resolution for Step-2 validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib
import json
import os


DEFAULT_ROBOT_ZOO_MANIFEST_PATH = Path("assets/robot_zoo/robot_zoo_manifest.json")
DEFAULT_RPO_MODEL_PATH = Path("assets/robots/atom01/mjcf/atom01.xml")


class RobotValidationStatus(str, Enum):
    PASSED = "passed"
    PARTIAL_PASSED = "partial_passed"
    NEGATIVE_CONTROL_PASSED = "negative_control_passed"
    ALGORITHM_FAILED = "algorithm_failed"
    SEMANTIC_FAILED = "semantic_failed"
    MODEL_LOAD_FAILED = "model_load_failed"
    SOURCE_UNAVAILABLE = "source_unavailable"
    LICENSE_BLOCKED = "license_blocked"


TERMINAL_PASS_STATUSES = {
    RobotValidationStatus.PASSED.value,
    RobotValidationStatus.PARTIAL_PASSED.value,
    RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value,
}


@dataclass(frozen=True)
class RobotZooEntry:
    raw: dict

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def description_name(self) -> str | None:
        value = self.raw.get("description_name")
        return str(value) if value else None

    @property
    def model_format(self) -> str:
        return str(self.raw["format"])

    @property
    def expected_capability(self) -> str:
        return str(self.raw["expected_capability"])

    @property
    def robot_class(self) -> str:
        return str(self.raw["robot_class"])

    @property
    def source_family(self) -> str:
        return str(self.raw["source_family"])

    @property
    def redistribution(self) -> str:
        return str(self.raw["redistribution"])

    @property
    def required(self) -> bool:
        return bool(self.raw.get("required", False))


@dataclass(frozen=True)
class ResolvedRobotSource:
    entry: RobotZooEntry
    status: str
    path: Path | None
    reason: str
    resolver: str
    module_name: str | None = None
    path_attr: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "available" and self.path is not None

    def to_json(self, *, manifest_path: Path, manifest_sha256: str) -> dict:
        payload = {
            "status": self.status,
            "reason": self.reason,
            "resolver": self.resolver,
            "path": display_path(self.path) if self.path else None,
            "module": self.module_name,
            "path_attr": self.path_attr,
            "manifest_path": display_path(manifest_path),
            "manifest_sha256": manifest_sha256,
            "entry": self.entry.raw,
        }
        if self.path:
            payload["local_file_sha256"] = sha256_file(self.path)
        return payload


@dataclass(frozen=True)
class RobotZooManifest:
    path: Path
    payload: dict
    sha256: str
    entries: tuple[RobotZooEntry, ...]

    @property
    def model_by_id(self) -> dict[str, RobotZooEntry]:
        return {entry.id: entry for entry in self.entries}

    def required_ids(self) -> list[str]:
        return [entry.id for entry in self.entries if entry.required]


def load_robot_zoo_manifest(path: str | Path = DEFAULT_ROBOT_ZOO_MANIFEST_PATH) -> RobotZooManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    entries = tuple(RobotZooEntry(raw=dict(entry)) for entry in payload.get("models", []))
    ids = [entry.id for entry in entries]
    if len(ids) != len(set(ids)):
        duplicates = sorted({rid for rid in ids if ids.count(rid) > 1})
        raise ValueError(f"duplicate Robot Zoo manifest ids: {', '.join(duplicates)}")
    return RobotZooManifest(
        path=manifest_path,
        payload=payload,
        sha256=sha256_file(manifest_path),
        entries=entries,
    )


def resolve_robot_source(
    entry: RobotZooEntry,
    *,
    cache_root: str | Path | None = None,
    allow_fetch: bool = False,
) -> ResolvedRobotSource:
    if entry.raw.get("license_blocked"):
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.LICENSE_BLOCKED.value,
            path=None,
            reason="manifest marks this source as license blocked",
            resolver="manifest_policy",
        )
    explicit = entry.raw.get("source_path")
    if explicit:
        path = Path(str(explicit))
        return _path_resolution(entry, path, resolver="manifest_source_path")
    if entry.source_family == "local":
        path = Path(str(entry.raw.get("local_path", DEFAULT_RPO_MODEL_PATH)))
        return _path_resolution(entry, path, resolver="local_workspace")
    if entry.description_name:
        return _resolve_robot_descriptions(entry, allow_fetch=allow_fetch)
    if entry.source_family == "mujoco_menagerie":
        return _resolve_menagerie_cache(entry, cache_root=cache_root)
    return ResolvedRobotSource(
        entry=entry,
        status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
        path=None,
        reason=f"no resolver is defined for source family {entry.source_family!r}",
        resolver="unsupported_source_family",
    )


def reproduction_compile_command(
    robot_id: str,
    *,
    manifest_path: str | Path = DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    output_path: str | Path,
    low_discrepancy_count: int,
    backend: str = "newton",
) -> str:
    parts = [
        "python",
        "-m",
        "soma_retargeter.tools.compile_kinematic_profile_v3",
        "--robot-id",
        robot_id,
        "--manifest",
        display_path(Path(manifest_path)),
        "--backend",
        backend,
        "--output",
        display_path(Path(output_path)),
        "--low-discrepancy-count",
        str(low_discrepancy_count),
    ]
    return " ".join(_shell_quote(part) for part in parts)


def reproduction_validate_command(
    *,
    manifest_path: str | Path = DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    output_dir: str | Path,
    low_discrepancy_count: int,
    deterministic_rerun: bool = False,
) -> str:
    parts = [
        "python",
        "-m",
        "soma_retargeter.tools.validate_kinematic_profile_v3",
        "--manifest",
        display_path(Path(manifest_path)),
        "--output-dir",
        display_path(Path(output_dir)),
        "--low-discrepancy-count",
        str(low_discrepancy_count),
    ]
    if deterministic_rerun:
        parts.append("--deterministic-rerun")
    return " ".join(_shell_quote(part) for part in parts)


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        cache_root = os.environ.get("ROBOT_ZOO_CACHE")
        if cache_root:
            try:
                return "${ROBOT_ZOO_CACHE}/" + str(path.resolve().relative_to(Path(cache_root).resolve()))
            except Exception:
                pass
        return str(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_resolution(entry: RobotZooEntry, path: Path, *, resolver: str) -> ResolvedRobotSource:
    if path.exists():
        return ResolvedRobotSource(entry=entry, status="available", path=path, reason="", resolver=resolver)
    return ResolvedRobotSource(
        entry=entry,
        status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
        path=None,
        reason=f"model path is unavailable: {display_path(path)}",
        resolver=resolver,
    )


def _resolve_robot_descriptions(entry: RobotZooEntry, *, allow_fetch: bool) -> ResolvedRobotSource:
    module_name = f"robot_descriptions.{entry.description_name}"
    path_attr = "MJCF_PATH" if entry.model_format == "mjcf" else "URDF_PATH"
    if not allow_fetch:
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            path=None,
            reason=(
                "robot_descriptions module import is disabled because imports may fetch upstream sources; "
                "rerun with explicit fetch/import enabled when cache/network use is intended"
            ),
            resolver="robot_descriptions_no_fetch",
            module_name=module_name,
            path_attr=path_attr,
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            path=None,
            reason=f"{type(exc).__name__}: {exc}",
            resolver="robot_descriptions",
            module_name=module_name,
            path_attr=path_attr,
        )
    path_value = getattr(module, path_attr, None)
    if not path_value:
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            path=None,
            reason=f"module does not expose {path_attr}",
            resolver="robot_descriptions",
            module_name=module_name,
            path_attr=path_attr,
        )
    path = Path(path_value)
    if not path.exists():
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            path=None,
            reason=f"{path_attr} does not exist: {display_path(path)}",
            resolver="robot_descriptions",
            module_name=module_name,
            path_attr=path_attr,
        )
    return ResolvedRobotSource(
        entry=entry,
        status="available",
        path=path,
        reason="",
        resolver="robot_descriptions",
        module_name=module_name,
        path_attr=path_attr,
    )


def _resolve_menagerie_cache(entry: RobotZooEntry, *, cache_root: str | Path | None) -> ResolvedRobotSource:
    root = Path(cache_root or os.environ.get("ROBOT_ZOO_CACHE", "assets/robot_zoo/cache"))
    directory = _menagerie_directory(entry)
    candidates = [
        root / "mujoco_menagerie" / directory / f"{directory}.xml",
        root / "mujoco_menagerie" / directory / "scene.xml",
        root / "mujoco_menagerie" / directory / "model.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return ResolvedRobotSource(
                entry=entry,
                status="available",
                path=candidate,
                reason="",
                resolver="mujoco_menagerie_cache",
            )
    return ResolvedRobotSource(
        entry=entry,
        status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
        path=None,
        reason=(
            f"fixed-ref menagerie source is not in cache for {directory!r}; "
            "set ROBOT_ZOO_CACHE or run the sync plan"
        ),
        resolver="mujoco_menagerie_cache",
    )


def _menagerie_directory(entry: RobotZooEntry) -> str:
    notes = str(entry.raw.get("notes", ""))
    prefix = "Menagerie directory:"
    if prefix in notes:
        return notes.split(prefix, 1)[1].strip().split()[0]
    return entry.id.removesuffix("_mjcf_direct")


@lru_cache(maxsize=1)
def allowed_status_values() -> tuple[str, ...]:
    return tuple(status.value for status in RobotValidationStatus)


def _shell_quote(value: str) -> str:
    import shlex

    if value.startswith("${") and "}" in value:
        return value
    return shlex.quote(value)
