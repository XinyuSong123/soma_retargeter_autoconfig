"""Manifest-driven Robot Zoo resolution for Step-2 validation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib
import json
import os
import sys


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

    @property
    def semantic_map_path(self) -> str | None:
        for key in ("semantic_map_path", "verified_semantic_map", "semantic_map"):
            value = self.raw.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def to_json(self) -> dict:
        payload = dict(self.raw)
        for key in ("source_path", "local_path", "semantic_map_path", "verified_semantic_map", "semantic_map"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                payload[key] = display_path(Path(value))
        return payload


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
            "entry": self.entry.to_json(),
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
        return _resolve_robot_descriptions(entry, cache_root=cache_root, allow_fetch=allow_fetch)
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
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except Exception:
        cache_root = os.environ.get("ROBOT_ZOO_CACHE")
        if cache_root:
            try:
                return "${ROBOT_ZOO_CACHE}/" + str(resolved.relative_to(Path(cache_root).resolve()))
            except Exception:
                pass
        for variable, root in _display_cache_roots():
            try:
                return f"${{{variable}}}/" + str(resolved.relative_to(root.resolve()))
            except Exception:
                pass
        return "${LOCAL_SOURCE_PATH}/" + path.name


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


def _resolve_robot_descriptions(
    entry: RobotZooEntry,
    *,
    cache_root: str | Path | None,
    allow_fetch: bool,
) -> ResolvedRobotSource:
    module_name = f"robot_descriptions.{entry.description_name}"
    path_attr = "MJCF_PATH" if entry.model_format == "mjcf" else "URDF_PATH"
    if not allow_fetch:
        cached = _resolve_robot_descriptions_local_cache(entry, path_attr=path_attr, cache_root=cache_root)
        if cached.available:
            return cached
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            path=None,
            reason=cached.reason,
            resolver=cached.resolver,
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


def _resolve_robot_descriptions_local_cache(
    entry: RobotZooEntry,
    *,
    path_attr: str,
    cache_root: str | Path | None,
) -> ResolvedRobotSource:
    module_name = f"robot_descriptions.{entry.description_name}"
    result = _declared_robot_description_paths(entry.description_name, cache_root=cache_root)
    if result["status"] != "available":
        return ResolvedRobotSource(
            entry=entry,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            path=None,
            reason=(
                f"robot_descriptions module import is disabled; no local cache path could be inferred for "
                f"{module_name}: {result['reason']}"
            ),
            resolver="robot_descriptions_local_cache",
            module_name=module_name,
            path_attr=path_attr,
        )
    paths = result["paths"]
    path_candidates = result.get("path_candidates", {})
    attr_candidates = [path_attr]
    if path_attr == "URDF_PATH":
        attr_candidates.append("XACRO_PATH")
    expected_paths = []
    for attr in attr_candidates:
        candidates = path_candidates.get(attr) or ([paths[attr]] if attr in paths else [])
        for path in candidates:
            expected_paths.append(path)
            if path.exists():
                return ResolvedRobotSource(
                    entry=entry,
                    status="available",
                    path=path,
                    reason="",
                    resolver="robot_descriptions_local_cache",
                    module_name=module_name,
                    path_attr=attr,
                )
    if expected_paths:
        reason = "declared local cache files are unavailable: " + ", ".join(
            display_path(path) or str(path) for path in expected_paths
        )
    else:
        reason = f"module source does not declare {path_attr}"
    return ResolvedRobotSource(
        entry=entry,
        status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
        path=None,
        reason=reason,
        resolver="robot_descriptions_local_cache",
        module_name=module_name,
        path_attr=path_attr,
    )


def _resolve_menagerie_cache(entry: RobotZooEntry, *, cache_root: str | Path | None) -> ResolvedRobotSource:
    directory = _menagerie_directory(entry)
    candidates = _menagerie_model_candidates(directory, cache_root=cache_root)
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
            f"fixed-ref menagerie source is not in local cache for {directory!r}; "
            "checked ROBOT_ZOO_CACHE, ROBOT_DESCRIPTIONS_CACHE, and default cache roots"
        ),
        resolver="mujoco_menagerie_cache",
    )


def _menagerie_directory(entry: RobotZooEntry) -> str:
    notes = str(entry.raw.get("notes", ""))
    prefix = "Menagerie directory:"
    if prefix in notes:
        return notes.split(prefix, 1)[1].strip().split()[0]
    return entry.id.removesuffix("_mjcf_direct")


def _menagerie_model_candidates(directory: str, *, cache_root: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    for root in _menagerie_roots(cache_root):
        model_dir = root / directory
        candidates.extend(
            [
                model_dir / f"{directory}.xml",
                model_dir / "model.xml",
            ]
        )
        if model_dir.exists():
            xml_files = sorted(model_dir.glob("*.xml"))
            candidates.extend(path for path in xml_files if not path.name.startswith("scene"))
            candidates.extend(path for path in xml_files if path.name.startswith("scene"))
        else:
            candidates.append(model_dir / "scene.xml")
    return _dedupe_paths(candidates)


def _menagerie_roots(cache_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if cache_root is not None:
        root = Path(cache_root).expanduser()
        roots.extend([root / "mujoco_menagerie", root])
    if os.environ.get("ROBOT_ZOO_CACHE"):
        roots.append(Path(os.environ["ROBOT_ZOO_CACHE"]).expanduser() / "mujoco_menagerie")
    roots.append(Path("assets/robot_zoo/cache/mujoco_menagerie"))
    roots.extend(root / "mujoco_menagerie" for root in _robot_descriptions_cache_roots(cache_root=None))
    return _dedupe_paths(roots)


def _declared_robot_description_paths(description_name: str | None, *, cache_root: str | Path | None) -> dict:
    if not description_name:
        return {"status": "unavailable", "reason": "manifest entry has no description_name", "paths": {}}
    module_filename = f"{description_name}.py"
    module_files = [root / module_filename for root in _robot_descriptions_package_roots()]
    module_files = _dedupe_paths(module_files)
    existing_module_files = [path for path in module_files if path.exists()]
    if not existing_module_files:
        return {
            "status": "unavailable",
            "reason": f"{module_filename} was not found on sys.path",
            "paths": {},
            "path_candidates": {},
        }
    errors: list[str] = []
    first_paths: dict[str, Path] | None = None
    path_candidates: dict[str, list[Path]] = {}
    for module_file in existing_module_files:
        repository_cache_paths = _repository_cache_paths(module_file.parent)
        for descriptions_cache in _robot_descriptions_cache_roots(cache_root):
            try:
                paths = _evaluate_robot_description_module(module_file, descriptions_cache, repository_cache_paths)
            except Exception as exc:
                errors.append(f"{display_path(module_file) or module_file}: {type(exc).__name__}: {exc}")
                continue
            if first_paths is None:
                first_paths = paths
            for attr, path in paths.items():
                path_candidates.setdefault(attr, []).append(path)
    if first_paths is not None:
        for attr, candidates in list(path_candidates.items()):
            path_candidates[attr] = _dedupe_paths(candidates)
        return {
            "status": "available",
            "reason": "",
            "paths": first_paths,
            "path_candidates": path_candidates,
        }
    return {
        "status": "unavailable",
        "reason": "; ".join(errors) if errors else f"{module_filename} could not be evaluated",
        "paths": {},
        "path_candidates": {},
    }


def _evaluate_robot_description_module(
    module_file: Path,
    descriptions_cache: Path,
    repository_cache_paths: dict[str, str],
) -> dict[str, Path]:
    tree = ast.parse(module_file.read_text(), filename=str(module_file))
    env: dict[str, str] = {}
    exported_paths: dict[str, Path] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        if value_node is None:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            value = _evaluate_robot_description_expr(
                value_node,
                env=env,
                descriptions_cache=descriptions_cache,
                repository_cache_paths=repository_cache_paths,
            )
            if value is None:
                continue
            env[target.id] = value
            if target.id.endswith("_PATH"):
                exported_paths[target.id] = Path(value).expanduser()
    return exported_paths


def _evaluate_robot_description_expr(
    node: ast.AST,
    *,
    env: dict[str, str],
    descriptions_cache: Path,
    repository_cache_paths: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Call):
        if _is_path_join_call(node):
            parts = [
                _evaluate_robot_description_expr(
                    arg,
                    env=env,
                    descriptions_cache=descriptions_cache,
                    repository_cache_paths=repository_cache_paths,
                )
                for arg in node.args
            ]
            if any(part is None for part in parts):
                return None
            return str(Path(str(parts[0])).joinpath(*(str(part) for part in parts[1:])))
        if isinstance(node.func, ast.Name) and node.func.id == "_clone_to_cache" and node.args:
            repository_name = _literal_string(node.args[0])
            if repository_name is None:
                return None
            cache_path = repository_cache_paths.get(repository_name, repository_name)
            commit = os.environ.get("ROBOT_DESCRIPTION_COMMIT")
            if commit:
                return str(descriptions_cache / f"{cache_path}-{commit}" / cache_path)
            return str(descriptions_cache / cache_path)
    return None


def _is_path_join_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "join"
        and isinstance(func.value, ast.Name)
        and func.value.id == "_path"
    )


def _literal_string(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    return value if isinstance(value, str) else None


@lru_cache(maxsize=None)
def _repository_cache_paths(package_root: Path) -> dict[str, str]:
    repositories_file = package_root / "_repositories.py"
    if not repositories_file.exists():
        return {}
    tree = ast.parse(repositories_file.read_text(), filename=str(repositories_file))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "REPOSITORIES" for target in statement.targets):
            continue
        if not isinstance(statement.value, ast.Dict):
            continue
        paths: dict[str, str] = {}
        for key_node, value_node in zip(statement.value.keys, statement.value.values):
            key = _literal_string(key_node) if key_node else None
            if key is None or not isinstance(value_node, ast.Call):
                continue
            cache_path = None
            for keyword in value_node.keywords:
                if keyword.arg == "cache_path":
                    cache_path = _literal_string(keyword.value)
                    break
            if cache_path:
                paths[key] = cache_path
        return paths
    return {}


def _robot_descriptions_package_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("ROBOT_DESCRIPTIONS_PACKAGE_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    for search_path in sys.path:
        if not search_path:
            search_path = "."
        roots.append(Path(search_path).expanduser() / "robot_descriptions")
    return _dedupe_paths(roots)


def _robot_descriptions_cache_roots(cache_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if cache_root is not None:
        root = Path(cache_root).expanduser()
        roots.extend([root / "robot_descriptions", root])
    configured = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.append(Path.home() / ".cache" / "robot_descriptions")
    return _dedupe_paths(roots)


def _display_cache_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    descriptions_cache = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
    roots.append(
        (
            "ROBOT_DESCRIPTIONS_CACHE",
            Path(descriptions_cache).expanduser() if descriptions_cache else Path.home() / ".cache" / "robot_descriptions",
        )
    )
    newton_cache = os.environ.get("NEWTON_CACHE")
    roots.append(
        (
            "NEWTON_CACHE",
            Path(newton_cache).expanduser() if newton_cache else Path.home() / ".cache" / "newton",
        )
    )
    return roots


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


@lru_cache(maxsize=1)
def allowed_status_values() -> tuple[str, ...]:
    return tuple(status.value for status in RobotValidationStatus)


def _shell_quote(value: str) -> str:
    import shlex

    if value.startswith("${") and "}" in value:
        return value
    return shlex.quote(value)
