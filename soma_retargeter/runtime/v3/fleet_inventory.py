"""Full-fleet Robot Zoo inventory for Step 3.1 runtime quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable

from soma_retargeter.robotics.v3.robot_zoo import load_robot_zoo_manifest, resolve_robot_source, sha256_file


DEFAULT_LOCK_PATH = Path("assets/robot_zoo/robot_zoo_lock.json")
DEFAULT_MANIFEST_PATH = Path("assets/robot_zoo/robot_zoo_manifest.json")
DEFAULT_STEP2_PROFILE_ROOT = Path("artifacts/retargeting_v3_step2_capability")

FULL_HUMANOID_PROFILE = "full_humanoid_profile"
PARTIAL_HUMANOID_PROFILE = "partial_humanoid_profile"
NEGATIVE_CONTROL = "negative_control"

EXPECTED_CATEGORY_COUNTS = {
    FULL_HUMANOID_PROFILE: 32,
    PARTIAL_HUMANOID_PROFILE: 3,
    NEGATIVE_CONTROL: 9,
}

FULL_PROFILE_STATUSES = {"passed", "capability_limited_passed"}
PARTIAL_PROFILE_STATUSES = {"partial_passed"}
NEGATIVE_PROFILE_STATUSES = {"negative_control_passed"}
TERMINAL_PROFILE_STATUSES = FULL_PROFILE_STATUSES | PARTIAL_PROFILE_STATUSES | NEGATIVE_PROFILE_STATUSES

_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


class FleetInventoryError(RuntimeError):
    """Raised when the Step 3.1 fleet scope cannot be derived safely."""


@dataclass(frozen=True)
class FleetRuntimeCase:
    model_id: str
    category: str
    runtime_source_path: Path
    model_format: str
    profile_path: Path
    profile_status: str
    expected_capability: str
    semantic_map_path: Path | None
    semantic_expectation_path: Path | None
    supported_semantics: tuple[str, ...]
    missing_required_semantics: tuple[str, ...]
    lock_entry: dict[str, Any] = field(repr=False)
    manifest_entry: dict[str, Any] = field(repr=False)
    profile: dict[str, Any] = field(repr=False)
    runtime_source_resolver: str = ""
    runtime_source_status: str = "available"
    runtime_source_sha256: str | None = None
    lfs_pointer: bool = False

    @property
    def is_full(self) -> bool:
        return self.category == FULL_HUMANOID_PROFILE

    @property
    def is_partial(self) -> bool:
        return self.category == PARTIAL_HUMANOID_PROFILE

    @property
    def is_negative(self) -> bool:
        return self.category == NEGATIVE_CONTROL

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "category": self.category,
            "profile_status": self.profile_status,
            "expected_capability": self.expected_capability,
            "model_format": self.model_format,
            "runtime_source_path": display_path(self.runtime_source_path),
            "runtime_source_status": self.runtime_source_status,
            "runtime_source_resolver": self.runtime_source_resolver,
            "runtime_source_sha256": self.runtime_source_sha256,
            "runtime_source_lfs_pointer": self.lfs_pointer,
            "profile_path": display_path(self.profile_path),
            "semantic_map_path": display_path(self.semantic_map_path),
            "semantic_expectation_path": display_path(self.semantic_expectation_path),
            "supported_semantics": list(self.supported_semantics),
            "missing_required_semantics": list(self.missing_required_semantics),
        }


def load_fleet_runtime_cases(
    *,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    step2_profile_root: str | Path = DEFAULT_STEP2_PROFILE_ROOT,
) -> list[FleetRuntimeCase]:
    """Derive the exact Step 3.1 full-fleet scope from lock/profile artifacts."""

    lock_path = Path(lock_path)
    manifest_path = Path(manifest_path)
    profile_root = Path(step2_profile_root)
    lock = _read_json(lock_path)
    summary = _read_json(profile_root / "summary.json")
    manifest = load_robot_zoo_manifest(manifest_path)

    deferred = set(str(v) for v in lock.get("scope_decision", {}).get("deferred_snapshot_ids", []))
    lock_entries = lock.get("entries")
    if not isinstance(lock_entries, dict):
        raise FleetInventoryError("robot_zoo_lock.json must contain an entries object")
    in_scope_ids = sorted(str(model_id) for model_id in lock_entries if str(model_id) not in deferred)

    expected_total = int(summary.get("in_scope_total", -1))
    if expected_total != len(in_scope_ids):
        raise FleetInventoryError(
            f"lock/profile in-scope mismatch: lock={len(in_scope_ids)} summary={expected_total}"
        )

    cases = [
        _build_case(
            model_id=model_id,
            lock_entry=dict(lock_entries[model_id]),
            manifest_entry=dict(manifest.model_by_id[model_id].raw),
            profile_root=profile_root,
            manifest_path=manifest_path,
        )
        for model_id in in_scope_ids
    ]
    validate_fleet_cases(cases, summary=summary)
    return cases


def validate_fleet_cases(cases: Iterable[FleetRuntimeCase], *, summary: dict[str, Any] | None = None) -> None:
    rows = list(cases)
    ids = [case.model_id for case in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
        raise FleetInventoryError(f"duplicate fleet model ids: {', '.join(duplicates)}")
    if len(rows) != 44:
        raise FleetInventoryError(f"Step 3.1 requires exactly 44 in-scope rows, got {len(rows)}")
    counts = category_counts(rows)
    if counts != EXPECTED_CATEGORY_COUNTS:
        raise FleetInventoryError(f"unexpected category counts: {counts}")
    missing_sources = [case.model_id for case in rows if case.runtime_source_status != "available"]
    if missing_sources:
        raise FleetInventoryError(f"runtime source unavailable for: {', '.join(missing_sources)}")
    lfs_pointers = [case.model_id for case in rows if case.lfs_pointer]
    if lfs_pointers:
        raise FleetInventoryError(f"runtime source is a Git LFS pointer for: {', '.join(lfs_pointers)}")
    bad_profiles = [case.model_id for case in rows if case.profile_status not in TERMINAL_PROFILE_STATUSES]
    if bad_profiles:
        raise FleetInventoryError(f"non-terminal Step 2 profile status for: {', '.join(bad_profiles)}")
    if summary is not None:
        status_counts = dict(summary.get("status_counts", {}))
        if int(status_counts.get("passed", 0)) != EXPECTED_CATEGORY_COUNTS[FULL_HUMANOID_PROFILE]:
            raise FleetInventoryError("Step 2 summary passed count does not match expected full humanoid total")
        if int(status_counts.get("partial_passed", 0)) != EXPECTED_CATEGORY_COUNTS[PARTIAL_HUMANOID_PROFILE]:
            raise FleetInventoryError("Step 2 summary partial count does not match expected partial total")
        if int(status_counts.get("negative_control_passed", 0)) != EXPECTED_CATEGORY_COUNTS[NEGATIVE_CONTROL]:
            raise FleetInventoryError("Step 2 summary negative count does not match expected negative total")


def category_counts(cases: Iterable[FleetRuntimeCase]) -> dict[str, int]:
    counts = {key: 0 for key in EXPECTED_CATEGORY_COUNTS}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return counts


def write_model_matrix(path: str | Path, cases: Iterable[FleetRuntimeCase]) -> dict[str, Any]:
    rows = [case.to_json() for case in cases]
    payload = {
        "schema_version": 1,
        "in_scope_total": len(rows),
        "category_counts": category_counts(cases),
        "rows": rows,
    }
    write_json(path, payload)
    return payload


def display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    resolved = p.resolve()
    roots = [
        ("WORKSPACE", Path.cwd()),
        ("ROBOT_DESCRIPTIONS_CACHE", _candidate_cache_root("ROBOT_DESCRIPTIONS_CACHE")),
        ("ROBOT_ZOO_CACHE", _candidate_cache_root("ROBOT_ZOO_CACHE")),
        ("HOME", Path.home()),
    ]
    for name, root in roots:
        if root is None:
            continue
        try:
            return f"${{{name}}}/" + resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return "${LOCAL_SOURCE_PATH}/" + p.name


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_payload_hash(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_case(
    *,
    model_id: str,
    lock_entry: dict[str, Any],
    manifest_entry: dict[str, Any],
    profile_root: Path,
    manifest_path: Path,
) -> FleetRuntimeCase:
    profile_path = profile_root / "per_robot" / f"{model_id}.json"
    profile = _read_json(profile_path)
    profile_status = str(profile.get("status", ""))
    category = _category_for_status(profile_status)
    runtime_path, resolver = _runtime_source_path(
        model_id=model_id,
        lock_entry=lock_entry,
        manifest_entry=manifest_entry,
        profile=profile,
        manifest_path=manifest_path,
    )
    if runtime_path is None:
        raise FleetInventoryError(f"unable to resolve runtime source path for {model_id}")
    if not runtime_path.exists():
        raise FleetInventoryError(f"runtime source does not exist for {model_id}: {display_path(runtime_path)}")
    if not runtime_path.is_file():
        raise FleetInventoryError(f"runtime source is not a file for {model_id}: {display_path(runtime_path)}")

    semantic_map = _semantic_map_path(model_id, profile, category)
    semantic_expectation = _semantic_expectation_path(model_id, profile, category)
    supported, missing = _semantic_support(profile, category)
    lfs_pointer = is_lfs_pointer(runtime_path)
    return FleetRuntimeCase(
        model_id=model_id,
        category=category,
        runtime_source_path=runtime_path,
        model_format=str(lock_entry.get("format") or manifest_entry.get("format") or profile.get("model", {}).get("format", "")),
        profile_path=profile_path,
        profile_status=profile_status,
        expected_capability=str(manifest_entry.get("expected_capability") or profile.get("manifest_entry", {}).get("expected_capability", "")),
        semantic_map_path=semantic_map,
        semantic_expectation_path=semantic_expectation,
        supported_semantics=tuple(supported),
        missing_required_semantics=tuple(missing),
        lock_entry=lock_entry,
        manifest_entry=manifest_entry,
        profile=profile,
        runtime_source_resolver=resolver,
        runtime_source_sha256=sha256_file(runtime_path),
        lfs_pointer=lfs_pointer,
    )


def _category_for_status(status: str) -> str:
    if status in FULL_PROFILE_STATUSES:
        return FULL_HUMANOID_PROFILE
    if status in PARTIAL_PROFILE_STATUSES:
        return PARTIAL_HUMANOID_PROFILE
    if status in NEGATIVE_PROFILE_STATUSES:
        return NEGATIVE_CONTROL
    raise FleetInventoryError(f"unsupported Step 2 profile status for runtime fleet: {status!r}")


def _runtime_source_path(
    *,
    model_id: str,
    lock_entry: dict[str, Any],
    manifest_entry: dict[str, Any],
    profile: dict[str, Any],
    manifest_path: Path,
) -> tuple[Path | None, str]:
    candidates: list[tuple[str, str | Path | None]] = []
    model = profile.get("model", {}) if isinstance(profile.get("model"), dict) else {}
    candidates.append(("profile.model.path", model.get("path")))
    source_resolution = model.get("source_resolution") if isinstance(model.get("source_resolution"), dict) else {}
    candidates.append(("profile.model.source_resolution.path", source_resolution.get("path")))
    if lock_entry.get("snapshot_path") and lock_entry.get("snapshot_file"):
        candidates.append(("lock.snapshot", Path(str(lock_entry["snapshot_path"])) / str(lock_entry["snapshot_file"])))
    if lock_entry.get("source_file") and lock_entry.get("snapshot_status") == "local_existing":
        candidates.append(("lock.local_source_file", lock_entry.get("source_file")))
    for resolver, value in candidates:
        path = _coerce_runtime_path(value)
        if path is not None and path.exists() and path.is_file():
            return path, resolver

    try:
        manifest = load_robot_zoo_manifest(manifest_path)
        resolved = resolve_robot_source(manifest.model_by_id[model_id], allow_fetch=False)
        if resolved.available and resolved.path is not None:
            return resolved.path, f"robot_zoo.{resolved.resolver}"
    except Exception:
        pass
    return None, "unresolved"


def _coerce_runtime_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text or text == "None":
        return None
    if text.startswith("${"):
        return _expand_placeholder_path(text)
    return Path(text)


def _expand_placeholder_path(text: str) -> Path | None:
    end = text.find("}")
    if not text.startswith("${") or end < 0:
        return None
    variable = text[2:end]
    suffix = text[end + 1 :].lstrip("/")
    for root in _placeholder_roots(variable):
        path = root / suffix
        if path.exists():
            return path
    return None


def _placeholder_roots(variable: str) -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get(variable)
    if env:
        roots.append(Path(env).expanduser())
    if variable == "ROBOT_DESCRIPTIONS_CACHE":
        roots.extend(
            [
                Path.home() / ".cache" / "robot_descriptions",
                Path.cwd().parent / "soma_robot_zoo_cache" / "robot_descriptions",
            ]
        )
    elif variable == "ROBOT_ZOO_CACHE":
        roots.extend([Path.home() / ".cache" / "robot_zoo", Path.cwd().parent / "soma_robot_zoo_cache"])
    elif variable == "LOCAL_SOURCE_PATH":
        roots.append(Path.cwd())
    return _dedupe_paths(roots)


def _candidate_cache_root(variable: str) -> Path | None:
    roots = _placeholder_roots(variable)
    return roots[0] if roots else None


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _semantic_map_path(model_id: str, profile: dict[str, Any], category: str) -> Path | None:
    if category != FULL_HUMANOID_PROFILE:
        return None
    resolution = profile.get("semantic_map_resolution")
    if isinstance(resolution, dict):
        value = resolution.get("path")
        path = _coerce_runtime_path(str(value)) if value else None
        if path and path.exists():
            return path
    fallback = Path("assets/robot_zoo/semantic_maps") / f"{model_id}.json"
    return fallback if fallback.exists() else None


def _semantic_expectation_path(model_id: str, profile: dict[str, Any], category: str) -> Path | None:
    if category != PARTIAL_HUMANOID_PROFILE:
        return None
    resolution = profile.get("semantic_map_resolution")
    if isinstance(resolution, dict):
        value = resolution.get("path")
        path = _coerce_runtime_path(str(value)) if value else None
        if path and path.exists():
            return path
    fallback = Path("assets/robot_zoo/semantic_expectations") / f"{model_id}.json"
    return fallback if fallback.exists() else None


def _semantic_support(profile: dict[str, Any], category: str) -> tuple[list[str], list[str]]:
    if category == FULL_HUMANOID_PROFILE:
        return ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"], []
    morphology = profile.get("morphology_classification")
    if isinstance(morphology, dict):
        supported = [str(v) for v in morphology.get("supported_semantics", [])]
        missing = [str(v) for v in morphology.get("missing_required_semantics", [])]
        return supported, missing
    return [], []


def is_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        prefix = p.read_bytes()[:256]
    except OSError:
        return False
    return prefix.decode("utf-8", errors="ignore").startswith(_LFS_POINTER_PREFIX)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FleetInventoryError(f"required JSON artifact is missing: {display_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FleetInventoryError(f"required JSON artifact is not an object: {display_path(path)}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, Path):
        return display_path(value)
    return value
