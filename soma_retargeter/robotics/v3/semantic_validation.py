"""Validation helpers for semantic-site evidence and morphology coverage."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from .robot_zoo import DEFAULT_ROBOT_ZOO_MANIFEST_PATH, DEFAULT_RPO_MODEL_PATH, display_path, load_robot_zoo_manifest


REQUIRED_HUMANOID_SEMANTICS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
LOWER_BODY_SEMANTICS = ("Hips", "Chest", "LeftFoot", "RightFoot")
DISTAL_SEMANTIC_SUFFIXES = ("Hand", "Foot", "Toe", "Heel")
STRUCTURAL_PARTIAL_COVERAGE_STATUSES = ("partial_blocked", "structurally_incomplete")
EXPLICIT_BLOCKER_COVERAGE_STATUSES = (*STRUCTURAL_PARTIAL_COVERAGE_STATUSES, "blocked")


@dataclass(frozen=True)
class SemanticValidationIssue:
    semantic_name: str
    code: str
    message: str

    def to_json(self) -> dict:
        return {
            "semantic_name": self.semantic_name,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class SemanticCoverageClassification:
    status: str
    missing_required: tuple[str, ...]
    available: tuple[str, ...]
    issues: tuple[SemanticValidationIssue, ...]

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "missing_required": list(self.missing_required),
            "available": list(self.available),
            "issues": [issue.to_json() for issue in self.issues],
        }


@dataclass(frozen=True)
class SemanticMapCoverageEntry:
    model_id: str
    required: bool
    redistribution: str
    source_status: str
    source_resolver: str
    source_path: str | None
    semantic_map_path: str
    semantic_map_status: str
    coverage_status: str
    expectation_path: str | None
    blocker_code: str | None
    supported_semantics: tuple[str, ...]
    missing_required_semantics: tuple[str, ...]
    blocker_evidence: tuple[str, ...]
    issues: tuple[SemanticValidationIssue, ...]

    @property
    def source_available(self) -> bool:
        return self.source_status == "available"

    @property
    def has_verified_map(self) -> bool:
        return self.coverage_status == "verified_map"

    @property
    def has_explicit_partial_blocker(self) -> bool:
        return self.coverage_status in STRUCTURAL_PARTIAL_COVERAGE_STATUSES

    @property
    def has_explicit_blocker(self) -> bool:
        return self.coverage_status in EXPLICIT_BLOCKER_COVERAGE_STATUSES

    def to_json(self) -> dict:
        return {
            "model_id": self.model_id,
            "required": self.required,
            "redistribution": self.redistribution,
            "source_status": self.source_status,
            "source_resolver": self.source_resolver,
            "source_path": self.source_path,
            "semantic_map_path": self.semantic_map_path,
            "semantic_map_status": self.semantic_map_status,
            "coverage_status": self.coverage_status,
            "expectation_path": self.expectation_path,
            "blocker_code": self.blocker_code,
            "supported_semantics": list(self.supported_semantics),
            "missing_required_semantics": list(self.missing_required_semantics),
            "blocker_evidence": list(self.blocker_evidence),
            "issues": [issue.to_json() for issue in self.issues],
        }


@dataclass(frozen=True)
class SemanticMapCoverageReport:
    entries: tuple[SemanticMapCoverageEntry, ...]

    @property
    def positive_humanoid_count(self) -> int:
        return len(self.entries)

    @property
    def available_count(self) -> int:
        return sum(entry.source_available for entry in self.entries)

    @property
    def unavailable_count(self) -> int:
        return self.positive_humanoid_count - self.available_count

    @property
    def verified_map_count(self) -> int:
        return sum(entry.has_verified_map for entry in self.entries)

    @property
    def partial_blocked_count(self) -> int:
        return sum(entry.has_explicit_partial_blocker for entry in self.entries)

    @property
    def blocked_count(self) -> int:
        return sum(entry.coverage_status == "blocked" for entry in self.entries)

    @property
    def coverage_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.coverage_status] = counts.get(entry.coverage_status, 0) + 1
        return counts

    @property
    def available_missing_verified_map_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.model_id
            for entry in self.entries
            if entry.source_available and entry.coverage_status == "missing"
        )

    @property
    def available_invalid_semantic_map_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.model_id
            for entry in self.entries
            if entry.source_available and entry.coverage_status == "invalid"
        )

    @property
    def available_partial_blocked_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.model_id
            for entry in self.entries
            if entry.source_available and entry.has_explicit_partial_blocker
        )

    @property
    def available_blocked_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.model_id
            for entry in self.entries
            if entry.source_available and entry.coverage_status == "blocked"
        )

    @property
    def available_structurally_incomplete_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.model_id
            for entry in self.entries
            if entry.source_available and entry.coverage_status == "structurally_incomplete"
        )

    @property
    def unavailable_missing_verified_map_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.model_id
            for entry in self.entries
            if not entry.source_available and entry.coverage_status == "missing"
        )

    def to_json(self) -> dict:
        return {
            "positive_humanoid_count": self.positive_humanoid_count,
            "available_count": self.available_count,
            "unavailable_count": self.unavailable_count,
            "verified_map_count": self.verified_map_count,
            "partial_blocked_count": self.partial_blocked_count,
            "blocked_count": self.blocked_count,
            "coverage_status_counts": self.coverage_status_counts,
            "available_missing_verified_map_ids": list(self.available_missing_verified_map_ids),
            "available_invalid_semantic_map_ids": list(self.available_invalid_semantic_map_ids),
            "available_partial_blocked_ids": list(self.available_partial_blocked_ids),
            "available_blocked_ids": list(self.available_blocked_ids),
            "available_structurally_incomplete_ids": list(self.available_structurally_incomplete_ids),
            "unavailable_missing_verified_map_ids": list(self.unavailable_missing_verified_map_ids),
            "entries": [entry.to_json() for entry in self.entries],
        }


@dataclass(frozen=True)
class SemanticExpectationCoverage:
    coverage_status: str
    expectation_path: Path | None = None
    blocker_code: str | None = None
    supported_semantics: tuple[str, ...] = ()
    missing_required_semantics: tuple[str, ...] = ()
    blocker_evidence: tuple[str, ...] = ()
    issues: tuple[SemanticValidationIssue, ...] = ()


def is_inferred_source(source: str) -> bool:
    return source.startswith("inferred") or "inferred" in source


def validate_no_fake_semantic_evidence(sites: dict) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    for semantic_name, site in sites.items():
        source = str(getattr(site, "source", ""))
        confidence = float(getattr(site, "confidence", 0.0))
        if source == "explicit_semantic_override":
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "legacy_explicit_override_source",
                    "semantic evidence must distinguish configured, verified, and inferred provenance",
                )
            )
        if is_inferred_source(source) and confidence >= 1.0:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "inferred_confidence_one",
                    "inferred semantic evidence must not be assigned confidence=1.0",
                )
            )
        if confidence > 0.99:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "confidence_above_cap",
                    "semantic confidence must be capped below certainty",
                )
            )
    return issues


def validate_nonzero_distal_sites(
    sites: dict,
    *,
    distal_suffixes: tuple[str, ...] = DISTAL_SEMANTIC_SUFFIXES,
    atol: float = 1e-9,
) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    for semantic_name, site in sites.items():
        if not semantic_name.endswith(distal_suffixes):
            continue
        pos = np.asarray(getattr(site, "local_position", np.zeros(3)), dtype=float)
        if float(np.linalg.norm(pos)) <= atol:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "distal_site_at_body_origin",
                    "distal hand/sole/toe/heel site lacks nonzero local geometry evidence",
                )
            )
    return issues


def validate_verified_semantic_map_payload(payload: dict) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    semantics = payload.get("semantics", {})
    if payload.get("verification_status") != "verified":
        issues.append(
            SemanticValidationIssue(
                str(payload.get("model_id", "")),
                "map_not_verified",
                "semantic map payload must explicitly declare verification_status='verified'",
            )
        )
    if not payload.get("model_fingerprint"):
        issues.append(
            SemanticValidationIssue(
                str(payload.get("model_id", "")),
                "missing_model_fingerprint",
                "verified semantic map must bind to a model fingerprint",
            )
        )
    for semantic_name, entry in semantics.items():
        if isinstance(entry, str):
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "legacy_string_entry",
                    "verified maps must store evidence/provenance entries, not bare body strings",
                )
            )
            continue
        evidence = entry.get("evidence", [])
        if not evidence:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "missing_evidence",
                    "semantic map entry must include evidence",
                )
            )
        confidence = float(entry.get("confidence", 0.0))
        if confidence > 0.99:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "confidence_above_cap",
                    "verified map confidence must be capped below certainty",
                )
            )
    return issues


def classify_semantic_coverage(
    sites: dict,
    *,
    expected_capability: str = "positive",
) -> SemanticCoverageClassification:
    available = tuple(sorted(sites))
    missing = tuple(name for name in REQUIRED_HUMANOID_SEMANTICS if name not in sites)
    issues: list[SemanticValidationIssue] = []

    if expected_capability == "negative_control":
        if not missing:
            issues.append(
                SemanticValidationIssue(
                    "Robot",
                    "negative_control_full_humanoid_semantics",
                    "negative control unexpectedly resolved a complete humanoid semantic set",
                )
            )
            status = "semantic_false_positive"
        else:
            status = "negative_control_passed"
        return SemanticCoverageClassification(status, missing, available, tuple(issues))

    if not missing:
        return SemanticCoverageClassification("full_humanoid_ready", missing, available, tuple(issues))

    lower_body_ready = all(name in sites for name in LOWER_BODY_SEMANTICS)
    only_upper_missing = set(missing) <= {"LeftHand", "RightHand"}
    if lower_body_ready and only_upper_missing:
        return SemanticCoverageClassification("partial_humanoid", missing, available, tuple(issues))

    return SemanticCoverageClassification("semantic_incomplete", missing, available, tuple(issues))


def compute_verified_semantic_map_coverage(
    *,
    manifest_path: str | Path = DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    semantic_maps_dir: str | Path = Path("assets/robot_zoo/semantic_maps"),
    semantic_expectations_dir: str | Path = Path("assets/robot_zoo/semantic_expectations"),
    robot_descriptions_cache: str | Path | None = None,
    robot_zoo_cache: str | Path | None = None,
) -> SemanticMapCoverageReport:
    """Report verified-map coverage without importing modules that may fetch assets."""

    manifest = load_robot_zoo_manifest(manifest_path)
    semantic_maps_root = Path(semantic_maps_dir)
    semantic_expectations_root = Path(semantic_expectations_dir)
    entries: list[SemanticMapCoverageEntry] = []
    for entry in manifest.entries:
        if entry.expected_capability != "positive" or entry.robot_class != "humanoid":
            continue
        source_path, source_status, source_resolver = _resolve_local_manifest_source(
            entry,
            robot_descriptions_cache=robot_descriptions_cache,
            robot_zoo_cache=robot_zoo_cache,
        )
        semantic_map_path = _coverage_semantic_map_path(
            entry,
            manifest_path=manifest.path,
            semantic_maps_dir=semantic_maps_root,
        )
        semantic_map_status, issues = _semantic_map_status(entry.id, semantic_map_path)
        expectation_path = semantic_expectations_root / f"{entry.id}.json"
        expectation_coverage = _expectation_coverage_for_missing_map(
            entry.id,
            semantic_map_status=semantic_map_status,
            expectation_path=expectation_path,
        )
        all_issues = tuple(issues) + expectation_coverage.issues
        entries.append(
            SemanticMapCoverageEntry(
                model_id=entry.id,
                required=entry.required,
                redistribution=entry.redistribution,
                source_status=source_status,
                source_resolver=source_resolver,
                source_path=display_path(source_path) if source_path else None,
                semantic_map_path=display_path(semantic_map_path) or str(semantic_map_path),
                semantic_map_status=semantic_map_status,
                coverage_status=expectation_coverage.coverage_status,
                expectation_path=(
                    display_path(expectation_coverage.expectation_path)
                    if expectation_coverage.expectation_path
                    else None
                ),
                blocker_code=expectation_coverage.blocker_code,
                supported_semantics=expectation_coverage.supported_semantics,
                missing_required_semantics=expectation_coverage.missing_required_semantics,
                blocker_evidence=expectation_coverage.blocker_evidence,
                issues=all_issues,
            )
        )
    return SemanticMapCoverageReport(tuple(entries))


def _coverage_semantic_map_path(entry, *, manifest_path: Path, semantic_maps_dir: Path) -> Path:
    if entry.semantic_map_path:
        configured = Path(entry.semantic_map_path)
        candidates = [configured]
        if not configured.is_absolute():
            candidates.append(manifest_path.parent / configured)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]
    return semantic_maps_dir / f"{entry.id}.json"


def _semantic_map_status(model_id: str, path: Path) -> tuple[str, list[SemanticValidationIssue]]:
    if not path.exists():
        return "missing", []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return "invalid", [
            SemanticValidationIssue(model_id, "invalid_json", f"semantic map is not valid JSON: {exc}")
        ]
    issues = validate_verified_semantic_map_payload(payload)
    if payload.get("model_id") != model_id:
        issues.append(
            SemanticValidationIssue(
                model_id,
                "model_id_mismatch",
                f"semantic map model_id={payload.get('model_id')!r} does not match manifest id",
            )
        )
    semantics = payload.get("semantics", {})
    missing = [name for name in REQUIRED_HUMANOID_SEMANTICS if name not in semantics]
    if missing:
        issues.append(
            SemanticValidationIssue(
                model_id,
                "missing_required_semantics",
                "verified map is missing required semantics: " + ", ".join(missing),
            )
        )
    return ("verified" if not issues else "invalid", issues)


def _expectation_coverage_for_missing_map(
    model_id: str,
    *,
    semantic_map_status: str,
    expectation_path: Path,
) -> SemanticExpectationCoverage:
    if semantic_map_status == "verified":
        return SemanticExpectationCoverage("verified_map")
    if semantic_map_status == "invalid":
        return SemanticExpectationCoverage("invalid")
    if not expectation_path.exists():
        return SemanticExpectationCoverage("missing")

    try:
        payload = json.loads(expectation_path.read_text())
    except json.JSONDecodeError as exc:
        return SemanticExpectationCoverage(
            "invalid",
            expectation_path=expectation_path,
            issues=(
                SemanticValidationIssue(
                    model_id,
                    "invalid_expectation_json",
                    f"semantic expectation is not valid JSON: {exc}",
                ),
            ),
        )

    coverage_status = str(payload.get("coverage_status", ""))
    verification_status = str(payload.get("verification_status", ""))
    blocker = payload.get("blocker", {})
    blocker_code = str(blocker.get("code", "")) if isinstance(blocker, dict) else ""
    is_structural_partial = (
        coverage_status in STRUCTURAL_PARTIAL_COVERAGE_STATUSES
        or verification_status in STRUCTURAL_PARTIAL_COVERAGE_STATUSES
        or blocker_code == "structurally_incomplete"
    )
    is_explicit_blocker = verification_status == "blocked" and isinstance(blocker, dict)
    if not is_structural_partial and not is_explicit_blocker:
        return SemanticExpectationCoverage("missing")

    supported = _string_tuple(payload.get("supported_semantics", ()))
    missing = _string_tuple(payload.get("missing_required_semantics", ()))
    evidence = _expectation_blocker_evidence(payload)
    issues: list[SemanticValidationIssue] = []
    if payload.get("model_id") != model_id:
        issues.append(
            SemanticValidationIssue(
                model_id,
                "expectation_model_id_mismatch",
                f"semantic expectation model_id={payload.get('model_id')!r} does not match manifest id",
            )
        )
    if is_structural_partial:
        if payload.get("expected_capability") != "partial_humanoid":
            issues.append(
                SemanticValidationIssue(
                    model_id,
                    "partial_blocker_expected_capability",
                    "structural partial blockers must declare expected_capability='partial_humanoid'",
                )
            )
        if not missing:
            issues.append(
                SemanticValidationIssue(
                    model_id,
                    "partial_blocker_missing_required_semantics",
                    "structural partial blockers must list the required semantics that cannot be verified",
                )
            )
        unsupported_missing = sorted(set(missing) - set(REQUIRED_HUMANOID_SEMANTICS))
        if unsupported_missing:
            issues.append(
                SemanticValidationIssue(
                    model_id,
                    "partial_blocker_unknown_missing_semantics",
                    "missing_required_semantics contains non-required names: " + ", ".join(unsupported_missing),
                )
            )
        if not {"LeftHand", "RightHand"} <= set(missing):
            issues.append(
                SemanticValidationIssue(
                    model_id,
                    "partial_blocker_hands_not_missing",
                    "structural partial blockers must explicitly account for missing distal hand semantics",
                )
            )
        overlap = sorted(set(supported) & set(missing))
        if overlap:
            issues.append(
                SemanticValidationIssue(
                    model_id,
                    "partial_blocker_supported_missing_overlap",
                    "supported_semantics overlaps missing_required_semantics: " + ", ".join(overlap),
                )
            )
    if not blocker_code:
        issues.append(
            SemanticValidationIssue(
                model_id,
                "blocker_missing_code",
                "semantic coverage blockers must include blocker.code",
            )
        )
    if not evidence:
        issues.append(
            SemanticValidationIssue(
                model_id,
                "blocker_missing_evidence",
                "semantic coverage blockers must include evidence from the inspected local model",
            )
        )
    if issues:
        return SemanticExpectationCoverage(
            "invalid",
            expectation_path=expectation_path,
            blocker_code=blocker_code or None,
            supported_semantics=supported,
            missing_required_semantics=missing,
            blocker_evidence=evidence,
            issues=tuple(issues),
        )

    resolved_status = "blocked"
    if coverage_status in STRUCTURAL_PARTIAL_COVERAGE_STATUSES:
        resolved_status = coverage_status
    if verification_status in STRUCTURAL_PARTIAL_COVERAGE_STATUSES:
        resolved_status = verification_status
    if is_structural_partial and blocker_code == "structurally_incomplete":
        resolved_status = "structurally_incomplete"
    return SemanticExpectationCoverage(
        resolved_status,
        expectation_path=expectation_path,
        blocker_code=blocker_code,
        supported_semantics=supported,
        missing_required_semantics=missing,
        blocker_evidence=evidence,
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))


def _expectation_blocker_evidence(payload: dict) -> tuple[str, ...]:
    evidence: list[str] = []
    blocker = payload.get("blocker", {})
    if isinstance(blocker, dict):
        evidence.extend(_string_tuple(blocker.get("evidence", ())))
        for key in ("observed_loader_error", "required_evidence", "source_path", "local_file_sha256"):
            value = blocker.get(key)
            if isinstance(value, str) and value:
                evidence.append(f"{key}:{value}")
        observed_bodies = _string_tuple(blocker.get("observed_compiled_bodies", ()))
        observed_sites = _string_tuple(blocker.get("observed_compiled_sites", ()))
        if observed_bodies:
            evidence.append("observed_compiled_bodies:" + ",".join(observed_bodies))
        if observed_sites:
            evidence.append("observed_compiled_sites:" + ",".join(observed_sites))
    evidence.extend(_string_tuple(payload.get("blockers", ())))
    return tuple(evidence)


def _resolve_local_manifest_source(
    entry,
    *,
    robot_descriptions_cache: str | Path | None,
    robot_zoo_cache: str | Path | None,
) -> tuple[Path | None, str, str]:
    explicit = entry.raw.get("source_path")
    if explicit:
        return _path_status(Path(str(explicit)), "manifest_source_path")
    if entry.source_family == "local":
        return _path_status(Path(str(entry.raw.get("local_path", DEFAULT_RPO_MODEL_PATH))), "local_workspace")
    if entry.description_name:
        return _resolve_robot_descriptions_source(
            entry.description_name,
            entry.model_format,
            robot_descriptions_cache=robot_descriptions_cache,
        )
    if entry.source_family == "mujoco_menagerie":
        return _resolve_menagerie_source(entry, robot_descriptions_cache=robot_descriptions_cache, robot_zoo_cache=robot_zoo_cache)
    return None, "source_unavailable", "unsupported_source_family"


def _path_status(path: Path, resolver: str) -> tuple[Path | None, str, str]:
    if path.exists():
        return path, "available", resolver
    return path, "source_unavailable", resolver


def _resolve_robot_descriptions_source(
    description_name: str,
    model_format: str,
    *,
    robot_descriptions_cache: str | Path | None,
) -> tuple[Path | None, str, str]:
    module_name = f"robot_descriptions.{description_name}"
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None, "source_unavailable", "robot_descriptions_module_missing"
    cache_root = _robot_descriptions_cache_root(robot_descriptions_cache)
    try:
        path = _evaluate_robot_description_path(Path(spec.origin), model_format, cache_root=cache_root)
    except Exception:
        return None, "source_unavailable", "robot_descriptions_static_resolution_failed"
    return _path_status(path, "robot_descriptions_static_cache")


def _robot_descriptions_cache_root(robot_descriptions_cache: str | Path | None) -> Path:
    if robot_descriptions_cache is not None:
        return Path(robot_descriptions_cache).expanduser()
    return Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")).expanduser()


def _evaluate_robot_description_path(module_path: Path, model_format: str, *, cache_root: Path) -> Path:
    path_attr = "MJCF_PATH" if model_format == "mjcf" else "URDF_PATH"
    env: dict[str, object] = {}
    tree = ast.parse(module_path.read_text())
    for statement in tree.body:
        target = None
        value = None
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target = statement.target.id
            value = statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            value = statement.value
        if target is None or value is None:
            continue
        try:
            env[target] = _evaluate_robot_description_expr(value, env, cache_root=cache_root)
        except Exception:
            continue
    path_value = env.get(path_attr)
    if not path_value:
        raise ValueError(f"{module_path} does not define {path_attr}")
    return Path(str(path_value))


def _evaluate_robot_description_expr(node: ast.AST, env: dict[str, object], *, cache_root: Path) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            return os.path.join(*(str(_evaluate_robot_description_expr(arg, env, cache_root=cache_root)) for arg in node.args))
        if isinstance(node.func, ast.Name) and node.func.id == "_getenv":
            key = str(_evaluate_robot_description_expr(node.args[0], env, cache_root=cache_root))
            default = _evaluate_robot_description_expr(node.args[1], env, cache_root=cache_root) if len(node.args) > 1 else None
            return os.environ.get(key, default)
        if isinstance(node.func, ast.Name) and node.func.id == "_clone_to_cache":
            repo_name = str(_evaluate_robot_description_expr(node.args[0], env, cache_root=cache_root))
            commit = None
            for keyword in node.keywords:
                if keyword.arg == "commit":
                    commit = _evaluate_robot_description_expr(keyword.value, env, cache_root=cache_root)
                    break
            return str(_robot_descriptions_cache_path(repo_name, commit=commit, cache_root=cache_root))
    raise ValueError(ast.dump(node))


def _robot_descriptions_cache_path(repo_name: str, *, commit: object, cache_root: Path) -> Path:
    try:
        from robot_descriptions._repositories import REPOSITORIES
    except Exception as exc:
        raise ValueError("robot_descriptions repository metadata is unavailable") from exc
    repository = REPOSITORIES[repo_name]
    cache_path = repository.cache_path
    if commit is not None:
        cache_path = f"{cache_path}-{commit}/{cache_path}"
    return cache_root / cache_path


def _resolve_menagerie_source(
    entry,
    *,
    robot_descriptions_cache: str | Path | None,
    robot_zoo_cache: str | Path | None,
) -> tuple[Path | None, str, str]:
    directory = _menagerie_directory_from_entry(entry)
    roots: list[Path] = []
    if robot_zoo_cache is not None:
        roots.append(Path(robot_zoo_cache).expanduser())
    elif os.environ.get("ROBOT_ZOO_CACHE"):
        roots.append(Path(os.environ["ROBOT_ZOO_CACHE"]).expanduser())
    roots.append(Path("assets/robot_zoo/cache"))
    roots.append(_robot_descriptions_cache_root(robot_descriptions_cache))
    for root in roots:
        directory_path = root / "mujoco_menagerie" / directory
        for candidate in _menagerie_xml_candidates(directory_path, directory):
            if candidate.exists():
                return candidate, "available", "mujoco_menagerie_local_cache"
    return None, "source_unavailable", "mujoco_menagerie_local_cache"


def _menagerie_directory_from_entry(entry) -> str:
    notes = str(entry.raw.get("notes", ""))
    prefix = "Menagerie directory:"
    if prefix in notes:
        return notes.split(prefix, 1)[1].strip().split()[0]
    return entry.id.removesuffix("_mjcf_direct")


def _menagerie_xml_candidates(directory_path: Path, directory: str) -> list[Path]:
    candidates = [
        directory_path / f"{directory}.xml",
        directory_path / "model.xml",
    ]
    if directory_path.exists():
        model_xmls = [
            path
            for path in directory_path.glob("*.xml")
            if not path.name.startswith("scene")
            and not path.stem.endswith(("_motor", "_position", "_velocity", "_mjx", "_pos"))
            and "actuator" not in path.stem
            and "keyframe" not in path.stem
        ]
        candidates.extend(sorted(model_xmls, key=lambda path: (len(path.name), path.name)))
    candidates.append(directory_path / "scene.xml")
    deduped: list[Path] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped
