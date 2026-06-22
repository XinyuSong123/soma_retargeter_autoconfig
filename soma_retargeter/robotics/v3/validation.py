"""Manifest-driven Step-2 offline validation artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import inspect
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .model_adapter import MuJoCoRuntimeModelAdapter
from .model_conversion import compare_runtime_models, convert_urdf_to_canonical_mjcf
from .model_adapter import NewtonRuntimeModelAdapter
from .profile import compile_kinematic_profile_v3
from .robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    RobotValidationStatus,
    RobotZooEntry,
    ResolvedRobotSource,
    TERMINAL_PASS_STATUSES,
    allowed_status_values,
    display_path,
    load_robot_zoo_manifest,
    model_load_failure_diagnostic,
    reproduction_compile_command,
    reproduction_validate_command,
    resolve_robot_source,
    sha256_file,
)
from .semantic_sites import load_semantic_map


DEFAULT_LOW_DISCREPANCY_COUNT = 32
ROBOT_ZOO_MANIFEST_PATH = DEFAULT_ROBOT_ZOO_MANIFEST_PATH
DEFAULT_ROBOT_ZOO_LOCK_PATH = Path("assets/robot_zoo/robot_zoo_lock.json")

# Compatibility exports for older tests/imports. The manifest remains authoritative.
try:
    _DEFAULT_MANIFEST = load_robot_zoo_manifest(ROBOT_ZOO_MANIFEST_PATH)
    REQUIRED_ARTIFACT_IDS = _DEFAULT_MANIFEST.required_ids()
    MANIFEST_MODEL_ID_BY_REPORT_ID = {entry.id: entry.id for entry in _DEFAULT_MANIFEST.entries}
except Exception:
    REQUIRED_ARTIFACT_IDS = []
    MANIFEST_MODEL_ID_BY_REPORT_ID = {}


_FAILURE_ARTIFACT_STATUSES = {
    RobotValidationStatus.MODEL_LOAD_FAILED.value,
    RobotValidationStatus.SEMANTIC_FAILED.value,
    RobotValidationStatus.ALGORITHM_FAILED.value,
}
_REQUIRED_REPRODUCIBILITY_ARTIFACTS = (
    "acceptance_ledger.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/coverage.json",
)
_LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w$])/(?:mnt|home|Users|tmp|var|private/var)/[^\s\"'<>),\]}`:]+"
)


def _load_assets44_lock(lock_path: str | Path | None) -> dict | None:
    if lock_path is None:
        return None
    path = Path(lock_path)
    payload = json.loads(path.read_text())
    payload["_path"] = path
    return payload


def _entries_for_validation(entries: tuple[RobotZooEntry, ...], *, lock: dict | None) -> tuple[RobotZooEntry, ...]:
    if lock is None:
        return entries
    rows = lock.get("entries", {})
    scope = lock.get("scope_decision", {})
    deferred = set(scope.get("deferred_snapshot_ids", ()))
    selected = []
    for entry in entries:
        row = rows.get(entry.id, {})
        if entry.id in deferred or row.get("snapshot_status") == "snapshot_failed":
            continue
        selected.append(entry)
    return tuple(selected)


def _resolve_entry_for_validation(
    entry: RobotZooEntry,
    *,
    lock: dict | None,
    allow_source_fetch: bool,
) -> ResolvedRobotSource:
    if lock is not None:
        row = lock.get("entries", {}).get(entry.id, {})
        snapshot_path = row.get("snapshot_path")
        if snapshot_path:
            path = Path(snapshot_path)
            if path.exists():
                return ResolvedRobotSource(
                    entry=entry,
                    status="available",
                    path=path / str(row.get("snapshot_file", "model.xml")),
                    reason="",
                    resolver="robot_zoo_committed_snapshot",
                )
        if row.get("snapshot_status") == "local_existing":
            return resolve_robot_source(entry, allow_fetch=allow_source_fetch)
        if row.get("snapshot_status") == "fetch_only":
            resolved = _resolve_fetch_only_from_lock(entry, row)
            if resolved.available:
                return resolved
    return resolve_robot_source(entry, allow_fetch=allow_source_fetch)


def _resolve_fetch_only_from_lock(entry: RobotZooEntry, row: dict) -> ResolvedRobotSource:
    source_file = str(row.get("source_file") or "")
    source_sha = str(row.get("source_sha256") or "")
    candidates: list[Path] = []
    for root in _external_cache_roots():
        if source_file:
            candidates.append(root / source_file)
            candidates.append(root / source_file.replace("-", "_"))
            candidates.append(root / source_file.replace("_", "-"))
        if source_file:
            name = Path(source_file).name
            candidates.extend(root.rglob(name) if root.exists() else [])
    for candidate in _dedupe_validation_paths(candidates):
        if not candidate.exists() or not candidate.is_file():
            continue
        if source_sha and sha256_file(candidate) != source_sha:
            continue
        return ResolvedRobotSource(
            entry=entry,
            status="available",
            path=candidate,
            reason="",
            resolver="robot_zoo_lock_fetch_only_cache",
        )
    return resolve_robot_source(entry, allow_fetch=False)


def _external_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for variable in ("ROBOT_DESCRIPTIONS_CACHE", "ROBOT_ZOO_CACHE"):
        value = os.environ.get(variable)
        if not value:
            continue
        root = Path(value).expanduser()
        roots.append(root)
        roots.append(root / "robot_descriptions")
    roots.append(Path.home() / ".cache" / "robot_descriptions")
    return _dedupe_validation_paths(roots)


def _dedupe_validation_paths(paths) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _scope_summary(lock: dict | None, *, entries: tuple[RobotZooEntry, ...]) -> dict:
    if lock is None:
        return {
            "mode": "manifest",
            "in_scope_total": len(entries),
        }
    scope = lock.get("scope_decision", {})
    totals = lock.get("totals", {})
    return {
        "mode": "assets44_lock",
        "lock_path": display_path(Path(lock.get("_path", DEFAULT_ROBOT_ZOO_LOCK_PATH))),
        "manifest_total": totals.get("entries"),
        "source_available": totals.get("source_available"),
        "vendored_snapshot_count": totals.get("vendored"),
        "fetch_only_cached_count": totals.get("fetch_only"),
        "project_local_count": totals.get("local_existing"),
        "deferred_snapshot_count": scope.get("deferred_snapshot_count"),
        "deferred_snapshot_ids": list(scope.get("deferred_snapshot_ids", ())),
        "in_scope_total": len(entries),
        "scope_decision": scope,
    }


def _load_matrix(full_reports: dict[str, dict]) -> dict:
    rows = {}
    for model_id, report in sorted(full_reports.items()):
        status = report.get("status")
        rows[model_id] = {
            "status": "failed" if status in {RobotValidationStatus.SOURCE_UNAVAILABLE.value, RobotValidationStatus.MODEL_LOAD_FAILED.value} else "passed",
            "validation_status": status,
            "source_status": report.get("model", {}).get("source_resolution", {}).get("status"),
            "resolver": report.get("model", {}).get("source_resolution", {}).get("resolver"),
            "path": report.get("model", {}).get("path"),
            "runtime_backend": report.get("runtime_adapter", {}).get("backend"),
        }
    return {"schema_version": 1, "rows": rows}


def _semantic_matrix(full_reports: dict[str, dict]) -> dict:
    rows = {}
    for model_id, report in sorted(full_reports.items()):
        status = report.get("status")
        entry = report.get("manifest_entry", {})
        if entry.get("expected_capability") == "negative_control":
            semantic_status = "negative_control_passed"
        elif status == RobotValidationStatus.SEMANTIC_FAILED.value:
            semantic_status = "failed"
        else:
            semantic_status = "passed"
        rows[model_id] = {
            "status": semantic_status,
            "validation_status": status,
            "semantic_map_resolution": report.get("semantic_map_resolution"),
            "semantic_map_artifact": report.get("semantic_map_artifact") or report.get("model", {}).get("semantic_map_artifact"),
            "morphology_classification": report.get("morphology_classification"),
        }
    return {"schema_version": 1, "rows": rows}


def _deferred_snapshots(lock: dict | None) -> dict:
    if lock is None:
        return {"schema_version": 1, "deferred_snapshot_count": 0, "deferred": {}}
    ids = lock.get("scope_decision", {}).get("deferred_snapshot_ids", ())
    entries = lock.get("entries", {})
    return {
        "schema_version": 1,
        "deferred_snapshot_count": len(ids),
        "deferred": {model_id: entries.get(model_id, {}) for model_id in ids},
    }


def write_validation_artifacts(
    output_dir: str | Path = "artifacts/retargeting_v3_step2",
    *,
    manifest_path: str | Path = ROBOT_ZOO_MANIFEST_PATH,
    lock_path: str | Path | None = None,
    include_missing_required_reports: bool = True,
    low_discrepancy_count: int = DEFAULT_LOW_DISCREPANCY_COUNT,
    deterministic_rerun: bool = False,
    allow_source_fetch: bool = False,
) -> dict:
    """Write one structured validation result for every Robot Zoo manifest entry.

    ``include_missing_required_reports`` is accepted for compatibility; missing
    required and optional entries are always materialized as explicit
    ``source_unavailable`` or ``model_load_failed`` results.
    """

    del include_missing_required_reports
    manifest = load_robot_zoo_manifest(manifest_path)
    lock = _load_assets44_lock(lock_path)
    entries = _entries_for_validation(manifest.entries, lock=lock)
    pre_generation_git = _git_snapshot()
    out = Path(output_dir)
    per_robot = out / "per_robot"
    failures_dir = out / "failures"
    semantic_maps_dir = out / "semantic_maps"
    test_results_dir = out / "test_results"
    for directory in (per_robot, failures_dir, semantic_maps_dir, test_results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_json_files(per_robot)
    _clear_json_files(failures_dir)
    _clear_json_files(semantic_maps_dir)
    _clear_matching_files(test_results_dir, ("*.json", "*.xml", "*.txt"))

    reports: dict[str, dict] = {}
    full_reports: dict[str, dict] = {}
    resolved_sources: dict[str, ResolvedRobotSource] = {}
    commands: list[str] = []
    source_inventory: dict[str, dict] = {}
    command_artifact_root = Path("${RETARGETING_V3_ARTIFACTS}")
    command_manifest_path = Path("${ROBOT_ZOO_MANIFEST}")
    for entry in entries:
        command = reproduction_compile_command(
            entry.id,
            manifest_path=command_manifest_path,
            output_path=command_artifact_root / "per_robot" / f"{entry.id}.json",
            low_discrepancy_count=low_discrepancy_count,
            backend="newton",
        )
        commands.append(command)
        resolved = _resolve_entry_for_validation(entry, lock=lock, allow_source_fetch=allow_source_fetch)
        resolved_sources[entry.id] = resolved
        source_inventory[entry.id] = _sanitize_artifact_payload(
            resolved.to_json(manifest_path=manifest.path, manifest_sha256=manifest.sha256)
        )
        report = _validate_entry(
            entry,
            resolved,
            manifest_path=manifest.path,
            manifest_sha256=manifest.sha256,
            semantic_maps_dir=semantic_maps_dir,
            low_discrepancy_count=low_discrepancy_count,
            reproduction_command=command,
        )
        report = _sanitize_artifact_payload(report)
        _assert_allowed_status(report["status"])
        _write_json(per_robot / f"{entry.id}.json", report)
        full_reports[entry.id] = report
        reports[entry.id] = _summary_entry(report)
        if report["status"] in _FAILURE_ARTIFACT_STATUSES:
            _write_json(failures_dir / f"{entry.id}.json", report)

    validation_checks = _run_validation_checks(
        out,
        manifest_path=manifest.path,
        full_reports=full_reports,
        resolved_sources=resolved_sources,
    )
    cross_format = _cross_format_report(reports, per_robot, validation_checks=validation_checks)
    deterministic = _deterministic_rerun_report(
        entries,
        full_reports,
        resolved_sources,
        manifest_path=manifest.path,
        manifest_sha256=manifest.sha256,
        low_discrepancy_count=low_discrepancy_count,
        deterministic_rerun=deterministic_rerun,
    )
    environment = _environment_report(manifest, git_snapshot=pre_generation_git)
    validation_command = reproduction_validate_command(
        manifest_path=command_manifest_path,
        output_dir=command_artifact_root,
        low_discrepancy_count=low_discrepancy_count,
        deterministic_rerun=deterministic_rerun,
    )
    commands.append(validation_command)
    commands.extend(_external_reproducibility_commands(command_artifact_root))
    (out / "commands.txt").write_text(_sanitize_artifact_string("\n".join(commands)) + "\n")
    _write_json(out / "environment.json", environment)
    _write_json(out / "scope.json", _scope_summary(lock, entries=entries))
    _write_json(out / "source_inventory.json", source_inventory)
    _write_json(out / "load_matrix.json", _load_matrix(full_reports))
    _write_json(out / "semantic_matrix.json", _semantic_matrix(full_reports))
    _write_json(out / "deferred_snapshots.json", _deferred_snapshots(lock))
    _write_json(out / "cross_format.json", cross_format)
    _write_json(out / "deterministic_rerun.json", deterministic)
    _write_json(out / "validation_checks.json", validation_checks)

    status_counts = Counter(item["status"] for item in reports.values())
    class_counts = Counter(item["robot_class"] for item in reports.values())
    capability_counts = Counter(item["expected_capability"] for item in reports.values())
    failure_artifact_status_counts = {
        status: status_counts[status]
        for status in sorted(_FAILURE_ARTIFACT_STATUSES)
        if status_counts[status]
    }
    summary = {
        "schema_version": 4,
        "manifest_total": 46 if lock is not None else len(manifest.entries),
        "in_scope_total": len(entries),
        "deferred_snapshot_count": (lock or {}).get("scope_decision", {}).get("deferred_snapshot_count", 0),
        "vendored_snapshot_count": (lock or {}).get("totals", {}).get("vendored", 0),
        "fetch_only_cached_count": (lock or {}).get("totals", {}).get("fetch_only", 0),
        "project_local_count": (lock or {}).get("totals", {}).get("local_existing", 0),
        "source_available_in_scope": len(entries) - status_counts[RobotValidationStatus.SOURCE_UNAVAILABLE.value],
        "load_passed_in_scope": len(entries)
        - status_counts[RobotValidationStatus.SOURCE_UNAVAILABLE.value]
        - status_counts[RobotValidationStatus.MODEL_LOAD_FAILED.value],
        "semantic_passed_in_scope": len(entries) - status_counts[RobotValidationStatus.SEMANTIC_FAILED.value],
        "profile_eligible": sum(
            1
            for item in reports.values()
            if item["expected_capability"] == "positive" and item["robot_class"] == "humanoid"
        ),
        "profile_passed": status_counts[RobotValidationStatus.PASSED.value],
        "partial_passed": status_counts[RobotValidationStatus.PARTIAL_PASSED.value],
        "negative_control_passed": status_counts[RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value],
        "algorithm_failed": status_counts[RobotValidationStatus.ALGORITHM_FAILED.value],
        "source_unavailable": status_counts[RobotValidationStatus.SOURCE_UNAVAILABLE.value],
        "model_load_failed": status_counts[RobotValidationStatus.MODEL_LOAD_FAILED.value],
        "semantic_failed": status_counts[RobotValidationStatus.SEMANTIC_FAILED.value],
        "deterministic_compared": deterministic.get("model_count", 0) if deterministic.get("status") == "passed" else 0,
        "deterministic_matched": deterministic.get("model_count", 0) if deterministic.get("status") == "passed" else 0,
        "manifest": {
            "path": display_path(manifest.path),
            "sha256": manifest.sha256,
            "model_count": len(entries),
            "allowed_statuses": list(allowed_status_values()),
        },
        "scope": _scope_summary(lock, entries=entries),
        "reports": reports,
        "status_counts": dict(sorted(status_counts.items())),
        "model_count_by_class": dict(sorted(class_counts.items())),
        "model_count_by_expected_capability": dict(sorted(capability_counts.items())),
        "algorithm_pass_count": sum(
            status_counts[key]
            for key in (
                RobotValidationStatus.PASSED.value,
                RobotValidationStatus.PARTIAL_PASSED.value,
                RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value,
            )
        ),
        "source_unavailable_count": status_counts[RobotValidationStatus.SOURCE_UNAVAILABLE.value],
        "license_blocked_count": status_counts[RobotValidationStatus.LICENSE_BLOCKED.value],
        "model_load_failed_count": status_counts[RobotValidationStatus.MODEL_LOAD_FAILED.value],
        "semantic_failed_count": status_counts[RobotValidationStatus.SEMANTIC_FAILED.value],
        "algorithm_failed_count": status_counts[RobotValidationStatus.ALGORITHM_FAILED.value],
        "failure_artifact_status_counts": failure_artifact_status_counts,
        "failure_artifacts_count": sum(failure_artifact_status_counts.values()),
        "cross_format": cross_format,
        "validation_checks": validation_checks,
        "deterministic_rerun": deterministic,
        "required_reproducibility_artifacts": _required_reproducibility_artifact_protocol(command_artifact_root),
        "notes": [
            "compiled is intentionally not a validation status",
            "source_unavailable is counted separately from algorithm pass/fail",
            "license_blocked is counted separately from failure artifacts",
            "failure artifacts contain only loader/compile, semantic, or algorithm failures",
        ],
    }
    summary = _sanitize_artifact_payload(summary)
    _write_json(out / "summary.json", summary)
    return summary


def augment_validation_report_metadata(
    report: dict,
    *,
    semantic_map_path: str | Path | None,
    manifest_path: str | Path = ROBOT_ZOO_MANIFEST_PATH,
) -> dict:
    """Add manifest/provenance metadata without changing compiler semantics."""

    manifest = load_robot_zoo_manifest(manifest_path)
    model = report.setdefault("model", {})
    model_id = str(model.get("id", ""))
    entry = manifest.model_by_id.get(model_id)
    if entry:
        resolved = resolve_robot_source(entry, allow_fetch=False)
        model["manifest"] = {
            "status": "available",
            "manifest_path": display_path(manifest.path),
            "manifest_sha256": manifest.sha256,
            "manifest_model_id": entry.id,
            "entry": entry.to_json(),
        }
        model["source_resolution"] = resolved.to_json(manifest_path=manifest.path, manifest_sha256=manifest.sha256)
    else:
        model["manifest"] = _unavailable(f"Robot Zoo manifest entry {model_id!r} is not present")
    path = Path(model["path"]) if model.get("path") else None
    model["path"] = display_path(path) if path else None
    model["local_file_sha256"] = sha256_file(path) if path and path.exists() else _unavailable("model path is unavailable")
    model["semantic_map_artifact"] = display_path(Path(semantic_map_path)) if semantic_map_path else _unavailable(
        "semantic map was not available before model resolution failed"
    )

    runtime = report.setdefault("runtime_adapter", {})
    backend = model.get("backend") or runtime.get("backend") or "newton"
    runtime["package_versions"] = _package_versions()
    runtime["loader_provenance"] = _loader_provenance(str(backend), str(model.get("format", "")))
    return report


def _validate_entry(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    semantic_maps_dir: Path,
    low_discrepancy_count: int,
    reproduction_command: str,
) -> dict:
    if resolved.status == RobotValidationStatus.LICENSE_BLOCKED.value:
        return _base_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            status=RobotValidationStatus.LICENSE_BLOCKED.value,
            failures=[resolved.reason],
            reproduction_command=reproduction_command,
        )
    if not resolved.available:
        return _base_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            status=RobotValidationStatus.SOURCE_UNAVAILABLE.value,
            failures=[],
            warnings=[resolved.reason],
            reproduction_command=reproduction_command,
        )
    if entry.expected_capability == "negative_control":
        return _validate_negative_control(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            reproduction_command=reproduction_command,
        )
    semantic_map_path_for_entry = _resolve_verified_semantic_map_path(entry, manifest_path=manifest_path)
    if semantic_map_path_for_entry is None:
        return _missing_verified_semantic_map_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            reproduction_command=reproduction_command,
        )
    try:
        semantic_map, semantic_map_resolution = _semantic_map_for_entry(
            entry,
            resolved,
            manifest_path=manifest_path,
            verified_semantic_map_path=semantic_map_path_for_entry,
        )
        semantic_map_path = _write_semantic_map_artifact(
            semantic_maps_dir,
            entry.id,
            semantic_map,
            semantic_map_resolution=semantic_map_resolution,
        )
        profile = compile_kinematic_profile_v3(
            resolved.path,
            semantic_map,
            model_id=entry.id,
            model_format=entry.model_format,
            backend="newton",
            low_discrepancy_count=low_discrepancy_count,
            reproduction_command=reproduction_command,
        )
    except Exception as exc:
        return _model_load_failed_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            reproduction_command=reproduction_command,
            exc=exc,
        )
    report = augment_validation_report_metadata(
        profile.to_json(),
        semantic_map_path=semantic_map_path,
        manifest_path=manifest_path,
    )
    _record_profile_gate_failures(report)
    report["status"] = _profile_status(report)
    report["status_reason"] = _profile_status_reason(report)
    report["manifest_entry"] = entry.to_json()
    report["semantic_map_resolution"] = semantic_map_resolution
    _assert_allowed_status(report["status"])
    return report


def _validate_negative_control(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    reproduction_command: str,
) -> dict:
    try:
        adapter = NewtonRuntimeModelAdapter(resolved.path, model_format=entry.model_format)
        runtime = {
            "backend": "newton",
            "nq": adapter.nq,
            "nv": adapter.nv,
            "body_count": len(adapter.body_names),
            "package_versions": _package_versions(),
            "loader_provenance": _loader_provenance("newton", entry.model_format),
        }
        adapter.close()
    except Exception as exc:
        return _model_load_failed_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            reproduction_command=reproduction_command,
            exc=exc,
        )
    report = _base_report(
        entry,
        resolved,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        status=RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value,
        failures=[],
        warnings=["negative control loaded; humanoid profile compilation intentionally not generated"],
        reproduction_command=reproduction_command,
    )
    report["runtime_adapter"] = runtime
    report["morphology_classification"] = {
        "expected_capability": entry.expected_capability,
        "robot_class": entry.robot_class,
        "humanoid_profile_generated": False,
    }
    return report


def _semantic_map_for_entry(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    *,
    manifest_path: Path,
    verified_semantic_map_path: Path | None = None,
) -> tuple[dict, dict]:
    explicit = verified_semantic_map_path or _resolve_verified_semantic_map_path(entry, manifest_path=manifest_path)
    if explicit is None:
        raise FileNotFoundError(f"missing verified semantic map for {entry.id}")
    return load_semantic_map(explicit), {
        "status": "available",
        "source": "verified_semantic_map",
        "path": display_path(explicit),
    }


def _resolve_verified_semantic_map_path(entry: RobotZooEntry, *, manifest_path: Path) -> Path | None:
    candidates: list[Path] = []
    if entry.semantic_map_path:
        configured = Path(entry.semantic_map_path)
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.append(manifest_path.parent / configured)
            candidates.append(Path.cwd() / configured)
    candidates.append(Path("assets/robot_zoo/semantic_maps") / f"{entry.id}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _missing_verified_semantic_map_report(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    reproduction_command: str,
) -> dict:
    resolution = {
        "status": "missing",
        "source": "verified_semantic_map",
        "path": None,
        "required": True,
        "reason": (
            "positive humanoid validation requires a verified semantic map; "
            "body-name inference is diagnostic only and cannot satisfy Step 2 gates"
        ),
    }
    failure = f"missing verified semantic map for {entry.id}"
    report = _base_report(
        entry,
        resolved,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        status=RobotValidationStatus.SEMANTIC_FAILED.value,
        failures=[failure],
        warnings=[],
        reproduction_command=reproduction_command,
    )
    report["semantic_map_resolution"] = resolution
    report["semantic_map_artifact"] = _unavailable("verified semantic map was not available")
    report.setdefault("failure_taxonomy", {}).setdefault("semantic", {})["verified_semantic_map"] = {
        "status": "failed",
        "classification": RobotValidationStatus.SEMANTIC_FAILED.value,
        "kind": "missing_verified_semantic_map",
        "message": failure,
        "next_action": "add a verified semantic map with topology/site evidence or classify the model as a structured partial/unsupported case",
    }
    return report


def _profile_status(report: dict) -> str:
    failures = report.get("failures", [])
    capability = report.get("capability_status")
    if any("missing required semantics" in failure for failure in failures):
        return RobotValidationStatus.SEMANTIC_FAILED.value
    if _numerical_stability_gate_failures(report):
        return RobotValidationStatus.ALGORITHM_FAILED.value
    if failures:
        return RobotValidationStatus.ALGORITHM_FAILED.value
    if capability == "partial_humanoid":
        return RobotValidationStatus.PARTIAL_PASSED.value
    return RobotValidationStatus.PASSED.value


def _profile_status_reason(report: dict) -> str:
    status = report["status"]
    if status == RobotValidationStatus.PASSED.value:
        return "profile completed without recorded failures"
    if status == RobotValidationStatus.PARTIAL_PASSED.value:
        return "available lower-body/torso semantics compiled with structured partial-humanoid downgrade"
    if status == RobotValidationStatus.SEMANTIC_FAILED.value:
        return "required humanoid semantics were missing or incomplete"
    numerical_failures = _numerical_stability_gate_failures(report)
    if numerical_failures:
        tasks = ", ".join(failure["task"] for failure in numerical_failures)
        return f"numerical stability gate failed for task(s): {tasks}"
    failures = report.get("failures", [])
    if failures:
        first = str(failures[0])
        if len(failures) == 1:
            return first
        return f"{first}; {len(failures) - 1} additional algorithm failure(s)"
    return "compiler recorded algorithm failures"


def _record_profile_gate_failures(report: dict) -> None:
    numerical_failures = _numerical_stability_gate_failures(report)
    if not numerical_failures:
        return

    taxonomy = report.setdefault("failure_taxonomy", {})
    algorithm = taxonomy.setdefault("algorithm", {})
    algorithm["numerical_stability"] = {
        "status": "failed",
        "classification": RobotValidationStatus.ALGORITHM_FAILED.value,
        "tasks": numerical_failures,
    }

    failures = report.setdefault("failures", [])
    existing = set(str(failure) for failure in failures)
    for failure in numerical_failures:
        task = failure["task"]
        message = (
            "numerical stability gate failed: "
            f"{task} has numerical_stability_gate_passed=false"
        )
        if message not in existing:
            failures.append(message)
            existing.add(message)


def _numerical_stability_gate_failures(report: dict) -> list[dict]:
    failures = []
    rank_stability = report.get("rank_stability", {})
    if not isinstance(rank_stability, dict):
        return failures
    severe_classes = {"nonfinite", "unstable_nonsmooth", "engine_fd_mismatch"}
    for task, payload in sorted(rank_stability.items()):
        if not isinstance(payload, dict):
            continue
        gate_failed = payload.get("numerical_stability_gate_passed") is False
        severe_paths = []
        for path, value in _walk_key_values(payload, "class"):
            if value in severe_classes:
                severe_paths.append(f"rank_stability.{task}{path}")
        if not gate_failed and not severe_paths:
            continue
        failures.append(
            {
                "task": str(task),
                "gate": "numerical_stability",
                "status": "failed",
                "false_gate_paths": [
                    f"rank_stability.{task}{path}"
                    for path, value in _walk_key_values(payload, "numerical_stability_gate_passed")
                    if value is False
                ],
                "severe_classification_paths": severe_paths,
                "epsilon_unstable_columns": payload.get("epsilon_unstable_columns", []),
                "stable_sample_fraction": payload.get("stable_sample_fraction"),
                "task_block": payload.get("task_block"),
            }
        )
    return failures


def _epsilon_stability_gate_failures(report: dict) -> list[dict]:
    failures = []
    rank_stability = report.get("rank_stability", {})
    if not isinstance(rank_stability, dict):
        return failures
    for task, payload in sorted(rank_stability.items()):
        if not isinstance(payload, dict):
            continue
        false_paths = [
            f"rank_stability.{task}{path}"
            for path, value in _walk_epsilon_gate_values(payload)
            if value is False
        ]
        if not false_paths:
            continue
        failures.append(
            {
                "task": str(task),
                "gate": "epsilon_stability",
                "status": "failed",
                "false_gate_paths": false_paths,
                "false_gate_count": len(false_paths),
                "epsilon_unstable_columns": payload.get("epsilon_unstable_columns", []),
                "epsilon_unstable_fraction": payload.get("epsilon_unstable_fraction"),
                "epsilon_unstable_sample_fraction": payload.get("epsilon_unstable_sample_fraction"),
            }
        )
    return failures


def _walk_key_values(value: object, key_name: str, *, path: str = "") -> list[tuple[str, object]]:
    hits: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else f".{key}"
            if key == key_name:
                hits.append((child_path, child))
            hits.extend(_walk_key_values(child, key_name, path=child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(_walk_key_values(child, key_name, path=f"{path}[{idx}]"))
    return hits


def _walk_epsilon_gate_values(value: object, *, path: str = "") -> list[tuple[str, object]]:
    hits: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else f".{key}"
            if key == "epsilon_stability_gate_passed":
                hits.append((child_path, child))
            else:
                hits.extend(_walk_epsilon_gate_values(child, path=child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(_walk_epsilon_gate_values(child, path=f"{path}[{idx}]"))
    return hits


def _base_report(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    status: str,
    reproduction_command: str,
    failures: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "schema_version": 4,
        "status": status,
        "status_reason": _status_reason(status, resolved),
        "model": {
            "id": entry.id,
            "format": entry.model_format,
            "backend": "newton",
            "path": display_path(resolved.path),
            "manifest": {
                "status": "available",
                "manifest_path": display_path(manifest_path),
                "manifest_sha256": manifest_sha256,
                "manifest_model_id": entry.id,
                "entry": entry.to_json(),
            },
            "source_resolution": resolved.to_json(manifest_path=manifest_path, manifest_sha256=manifest_sha256),
            "local_file_sha256": sha256_file(resolved.path) if resolved.path and resolved.path.exists() else _unavailable("model path is unavailable"),
            "semantic_map_artifact": _unavailable("semantic map was not generated"),
        },
        "runtime_adapter": {
            "backend": "newton",
            "package_versions": _package_versions(),
            "loader_provenance": _loader_provenance("newton", entry.model_format),
        },
        "manifest_entry": entry.to_json(),
        "failures": failures or [],
        "warnings": warnings or [],
        "reproduction_command": reproduction_command,
    }


def _model_load_failed_report(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    reproduction_command: str,
    exc: BaseException,
) -> dict:
    diagnostic = model_load_failure_diagnostic(exc, reproduction_command=reproduction_command)
    report = _base_report(
        entry,
        resolved,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        status=RobotValidationStatus.MODEL_LOAD_FAILED.value,
        failures=[f"{type(exc).__name__}: {diagnostic['message']}"],
        reproduction_command=reproduction_command,
    )
    report.setdefault("failure_taxonomy", {}).setdefault("model_load", {})[diagnostic["kind"]] = diagnostic
    return report


def _status_reason(status: str, resolved: ResolvedRobotSource) -> str:
    if status == RobotValidationStatus.SOURCE_UNAVAILABLE.value:
        return resolved.reason
    if status == RobotValidationStatus.MODEL_LOAD_FAILED.value:
        return "source resolved but runtime model loading or compilation failed"
    if status == RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value:
        return "negative control source loaded and was not promoted to a humanoid profile"
    if status == RobotValidationStatus.LICENSE_BLOCKED.value:
        return resolved.reason
    return status


def _summary_entry(report: dict) -> dict:
    entry = report["manifest_entry"]
    return {
        "status": report["status"],
        "status_reason": report.get("status_reason", ""),
        "failures": report.get("failures", []),
        "warnings": report.get("warnings", []),
        "robot_class": entry.get("robot_class"),
        "expected_capability": entry.get("expected_capability"),
        "required": entry.get("required"),
        "redistribution": entry.get("redistribution"),
    }


def _write_semantic_map_artifact(
    semantic_maps_dir: Path,
    model_id: str,
    semantic_map: dict,
    *,
    semantic_map_resolution: dict,
) -> Path:
    path = semantic_maps_dir / f"{model_id}.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "model_id": model_id,
            "source": semantic_map_resolution["source"],
            "resolution": semantic_map_resolution,
            "semantics": semantic_map,
        },
    )
    return path


def _cross_format_report(reports: dict[str, dict], per_robot: Path, *, validation_checks: dict | None = None) -> dict:
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for model_id in reports:
        if model_id.endswith("_urdf"):
            groups[model_id[: -len("_urdf")]]["urdf"] = model_id
        elif model_id.endswith("_mjcf"):
            groups[model_id[: -len("_mjcf")]]["mjcf"] = model_id
    pairs = {}
    for family, formats in sorted(groups.items()):
        if {"urdf", "mjcf"} <= set(formats):
            urdf_id = formats["urdf"]
            mjcf_id = formats["mjcf"]
            pairs[family] = {
                "urdf": urdf_id,
                "mjcf": mjcf_id,
                "status": "not_run",
                "reason": "cross-format semantic equivalence scaffolding only; strict comparison is not counted as pass",
                "inputs": {
                    "urdf_status": reports[urdf_id]["status"],
                    "mjcf_status": reports[mjcf_id]["status"],
                    "urdf_report": display_path(per_robot / f"{urdf_id}.json"),
                    "mjcf_report": display_path(per_robot / f"{mjcf_id}.json"),
                },
            }
    return {
        "schema_version": 2,
        "gates": {
            "same_source_strict": _same_source_strict_gate(validation_checks or {}),
            "variant_compatibility": _variant_compatibility_gate(pairs, per_robot=per_robot),
        },
        "pairs": pairs,
    }


def _same_source_strict_gate(validation_checks: dict) -> dict:
    check_name = "g1_mjcf_urdf_equivalence"
    check = validation_checks.get(check_name, {})
    if not isinstance(check, dict) or not check:
        return {
            "status": "not_run",
            "reason": "same-source validation check was not materialized",
            "validation_check": check_name,
        }

    check_status = str(check.get("status", "unknown"))
    gate_status = _cross_format_status_from_same_source_check(check)
    gate = {
        "status": gate_status,
        "reason": _same_source_gate_reason(check, gate_status),
        "validation_check": check_name,
        "validation_check_status": check_status,
    }
    for key in (
        "mode",
        "source",
        "generated",
        "source_sha256",
        "generated_sha256",
        "strict_equivalent",
        "gate_a_status",
        "gate_a_evidence_complete",
        "gate_a_required_sections",
        "evidence_statuses",
        "evidence_incomplete_reasons",
        "differences",
    ):
        if key in check:
            gate[key] = check[key]
    return gate


def _cross_format_status_from_same_source_check(check: dict) -> str:
    status = check.get("status")
    if status == "passed":
        return "passed"
    if status == "incomplete":
        return "incomplete"
    if status in {"failed", "algorithm_failed"}:
        return "failed"
    if status in {"source_unavailable", "blocked"}:
        return "blocked"
    return "not_run"


def _same_source_gate_reason(check: dict, gate_status: str) -> str:
    if gate_status == "passed":
        return "same-source Gate A evidence is complete and passed"
    if gate_status == "incomplete":
        return "same-source conversion was compared, but Gate A semantic/projection evidence is incomplete"
    if gate_status == "failed":
        return "same-source strict comparison failed"
    if gate_status == "blocked":
        return str(check.get("reason") or "same-source strict comparison could not run")
    return "same-source strict comparison has no validation evidence"


def _variant_compatibility_gate(pairs: dict[str, dict], *, per_robot: Path) -> dict:
    pair_statuses = {}
    eligible_count = 0
    passed_count = 0
    for family, pair in sorted(pairs.items()):
        urdf_status = pair["inputs"]["urdf_status"]
        mjcf_status = pair["inputs"]["mjcf_status"]
        if urdf_status not in TERMINAL_PASS_STATUSES or mjcf_status not in TERMINAL_PASS_STATUSES:
            pair_statuses[family] = {
                "status": "not_eligible",
                "urdf_status": urdf_status,
                "mjcf_status": mjcf_status,
                "reason": "variant compatibility only runs when both variants independently pass profile gates",
            }
            continue
        eligible_count += 1
        result = _variant_pair_compatibility(pair, per_robot=per_robot)
        if result["status"] == "passed":
            passed_count += 1
        pair_statuses[family] = result

    if eligible_count == 0:
        status = "blocked"
        reason = "no independently passing URDF/MJCF variant pair is available for compatibility evidence"
    elif passed_count == eligible_count:
        status = "passed"
        reason = "all independently passing URDF/MJCF variant pairs have compatibility evidence"
    else:
        status = "failed"
        reason = "one or more independently passing URDF/MJCF variant pairs failed compatibility evidence"
    return {
        "status": status,
        "reason": reason,
        "pair_count": len(pairs),
        "eligible_pair_count": eligible_count,
        "passed_pair_count": passed_count,
        "pair_statuses": pair_statuses,
    }


def _variant_pair_compatibility(pair: dict, *, per_robot: Path) -> dict:
    urdf_report = _load_pair_report(per_robot, pair["urdf"])
    mjcf_report = _load_pair_report(per_robot, pair["mjcf"])
    evidence = {
        "semantic_sites": _variant_semantic_site_evidence(urdf_report, mjcf_report),
        "common_chains": _variant_chain_evidence(urdf_report, mjcf_report),
        "rank_summary": _variant_rank_evidence(urdf_report, mjcf_report),
        "dof_difference": _variant_dof_evidence(urdf_report, mjcf_report),
        "canonical_projection": _variant_projection_evidence(urdf_report, mjcf_report),
    }
    failures = {
        name: payload.get("failures", [])
        for name, payload in evidence.items()
        if payload.get("status") != "passed"
    }
    return {
        "status": "passed" if not failures else "failed",
        "urdf_status": pair["inputs"]["urdf_status"],
        "mjcf_status": pair["inputs"]["mjcf_status"],
        "urdf": pair["urdf"],
        "mjcf": pair["mjcf"],
        "comparison_mode": "variant_compatibility",
        "strict_equivalence": False,
        "evidence": evidence,
        "failures": failures,
    }


def _load_pair_report(per_robot: Path, model_id: str) -> dict:
    path = per_robot / f"{model_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _variant_semantic_site_evidence(urdf_report: dict, mjcf_report: dict) -> dict:
    required = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
    urdf_sites = urdf_report.get("semantic_sites", {}) if isinstance(urdf_report, dict) else {}
    mjcf_sites = mjcf_report.get("semantic_sites", {}) if isinstance(mjcf_report, dict) else {}
    failures = []
    per_semantic = {}
    for semantic in required:
        left = urdf_sites.get(semantic, {})
        right = mjcf_sites.get(semantic, {})
        left_source = str(left.get("source", ""))
        right_source = str(right.get("source", ""))
        passed = bool(left) and bool(right) and left_source.startswith("verified") and right_source.startswith("verified")
        if semantic != "Hips":
            passed = passed and left.get("body_name") == right.get("body_name")
        if not passed:
            failures.append(f"semantic_site_mismatch:{semantic}")
        per_semantic[semantic] = {
            "passed": passed,
            "urdf_body": left.get("body_name"),
            "mjcf_body": right.get("body_name"),
            "urdf_source": left_source,
            "mjcf_source": right_source,
        }
    return {"status": "passed" if not failures else "failed", "per_semantic": per_semantic, "failures": failures}


def _variant_chain_evidence(urdf_report: dict, mjcf_report: dict) -> dict:
    urdf_chains = urdf_report.get("chains", {}) if isinstance(urdf_report, dict) else {}
    mjcf_chains = mjcf_report.get("chains", {}) if isinstance(mjcf_report, dict) else {}
    common = sorted(set(urdf_chains) & set(mjcf_chains))
    failures = []
    per_task = {}
    for task in common:
        left = urdf_chains[task]
        right = mjcf_chains[task]
        task_failures = []
        for key in ("reference", "target", "coordinate_labels", "joint_types"):
            if left.get(key) != right.get(key):
                task_failures.append(key)
        if task_failures:
            failures.extend(f"chain_mismatch:{task}:{key}" for key in task_failures)
        per_task[task] = {
            "passed": not task_failures,
            "urdf_active_velocity_coordinates": left.get("active_velocity_coordinates"),
            "mjcf_active_velocity_coordinates": right.get("active_velocity_coordinates"),
            "coordinate_labels": left.get("coordinate_labels"),
            "failures": task_failures,
        }
    missing = sorted(set(urdf_chains) ^ set(mjcf_chains))
    failures.extend(f"chain_missing:{task}" for task in missing)
    return {"status": "passed" if common and not failures else "failed", "per_task": per_task, "failures": failures}


def _variant_rank_evidence(urdf_report: dict, mjcf_report: dict) -> dict:
    urdf_rank = urdf_report.get("rank_stability", {}) if isinstance(urdf_report, dict) else {}
    mjcf_rank = mjcf_report.get("rank_stability", {}) if isinstance(mjcf_report, dict) else {}
    common = sorted(set(urdf_rank) & set(mjcf_rank))
    failures = []
    per_task = {}
    for task in common:
        left = urdf_rank[task]
        right = mjcf_rank[task]
        keys = (
            ("nominal_rank_rotation", "regular_rank_rotation")
            if task == "torso"
            else ("nominal_rank_translation", "regular_rank_translation")
        )
        task_failures = [key for key in keys if left.get(key) != right.get(key)]
        if not left.get("epsilon_stability_gate_passed") or not right.get("epsilon_stability_gate_passed"):
            task_failures.append("epsilon_stability_gate_passed")
        failures.extend(f"rank_mismatch:{task}:{key}" for key in task_failures)
        per_task[task] = {
            "passed": not task_failures,
            "urdf": {key: left.get(key) for key in keys},
            "mjcf": {key: right.get(key) for key in keys},
            "rank_contract": "torso_rotation" if task == "torso" else "endpoint_translation",
            "failures": task_failures,
        }
    failures.extend(f"rank_missing:{task}" for task in sorted(set(urdf_rank) ^ set(mjcf_rank)))
    return {"status": "passed" if common and not failures else "failed", "per_task": per_task, "failures": failures}


def _variant_dof_evidence(urdf_report: dict, mjcf_report: dict) -> dict:
    urdf_runtime = urdf_report.get("runtime_adapter", {}) if isinstance(urdf_report, dict) else {}
    mjcf_runtime = mjcf_report.get("runtime_adapter", {}) if isinstance(mjcf_report, dict) else {}
    return {
        "status": "passed" if urdf_runtime and mjcf_runtime else "failed",
        "urdf_nq": urdf_runtime.get("nq"),
        "urdf_nv": urdf_runtime.get("nv"),
        "mjcf_nq": mjcf_runtime.get("nq"),
        "mjcf_nv": mjcf_runtime.get("nv"),
        "reason": "variant DoF differences are recorded and allowed only with matching semantic chain labels",
        "failures": [] if urdf_runtime and mjcf_runtime else ["runtime_dimensions_missing"],
    }


def _variant_projection_evidence(urdf_report: dict, mjcf_report: dict) -> dict:
    required_tasks = ("torso", "left_hand", "right_hand", "left_foot", "right_foot")
    failures = []
    per_side = {}
    for side, report in (("urdf", urdf_report), ("mjcf", mjcf_report)):
        projection = report.get("canonical_projection_reports", {}) if isinstance(report, dict) else {}
        motions = projection.get("motions", {}) if isinstance(projection, dict) else {}
        motion_order = projection.get("motion_order", []) if isinstance(projection, dict) else []
        side_failures = []
        if len(motion_order) < 15:
            side_failures.append("canonical_motion_coverage_insufficient")
        if projection.get("failures"):
            side_failures.append("canonical_projection_failures_present")
        task_coverage = set()
        for motion in motions.values():
            tasks = motion.get("tasks", {}) if isinstance(motion, dict) else {}
            task_coverage.update(tasks)
        missing_tasks = [task for task in required_tasks if task not in task_coverage]
        if missing_tasks:
            side_failures.append("canonical_projection_task_coverage_insufficient")
        failures.extend(f"{side}:{failure}" for failure in side_failures)
        per_side[side] = {
            "passed": not side_failures,
            "motion_count": len(motion_order),
            "task_coverage": sorted(task_coverage),
            "missing_tasks": missing_tasks,
            "failures": side_failures,
        }
    return {"status": "passed" if not failures else "failed", "per_side": per_side, "failures": failures}


def _run_validation_checks(
    output_dir: Path,
    *,
    manifest_path: Path,
    full_reports: dict[str, dict],
    resolved_sources: dict[str, ResolvedRobotSource],
) -> dict:
    """Call validation checks while preserving old one-arg test monkeypatches."""

    kwargs = {
        "manifest_path": manifest_path,
        "full_reports": full_reports,
        "resolved_sources": resolved_sources,
    }
    try:
        signature = inspect.signature(_validation_checks)
    except (TypeError, ValueError):
        return _validation_checks(output_dir)
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    accepted = {
        key: value
        for key, value in kwargs.items()
        if has_var_kwargs or key in signature.parameters
    }
    return _validation_checks(output_dir, **accepted)


def _validation_checks(
    output_dir: Path,
    *,
    manifest_path: Path = ROBOT_ZOO_MANIFEST_PATH,
    full_reports: dict[str, dict] | None = None,
    resolved_sources: dict[str, ResolvedRobotSource] | None = None,
) -> dict:
    return {
        "g1_mjcf_urdf_equivalence": _g1_same_source_strict_check(
            output_dir,
            manifest_path=manifest_path,
            full_reports=full_reports,
            resolved_sources=resolved_sources,
        ),
    }


def _g1_same_source_strict_check(
    output_dir: Path,
    *,
    manifest_path: Path = ROBOT_ZOO_MANIFEST_PATH,
    full_reports: dict[str, dict] | None = None,
    resolved_sources: dict[str, ResolvedRobotSource] | None = None,
) -> dict:
    source = _find_g1_same_source_urdf_compat(manifest_path=manifest_path, resolved_sources=resolved_sources)
    if source is None:
        return {
            "status": "source_unavailable",
            "gate_a_status": "blocked",
            "gate_a_evidence_complete": False,
            "reason": "manifest-backed fixed G1 same-source URDF was not resolved",
            "differences": {"source": "unavailable"},
            "evidence_incomplete_reasons": {"source": "manifest-backed fixed G1 same-source URDF was not resolved"},
        }
    semantic_map, semantic_map_resolution = _g1_same_source_semantic_map(manifest_path=manifest_path, source=source)
    generated = output_dir / "cross_format" / "unitree_g1_same_source_canonical.xml"
    try:
        conversion = convert_urdf_to_canonical_mjcf(source, generated)
        projection_resolution = _g1_same_source_projection_reports(
            source,
            generated,
            semantic_map=semantic_map,
            full_reports=full_reports,
        )
        left = MuJoCoRuntimeModelAdapter(source, model_format="urdf")
        right = MuJoCoRuntimeModelAdapter(generated, model_format="xml")
        try:
            equivalence = _compare_runtime_models_for_gate_a(
                left,
                right,
                semantic_map=semantic_map,
                canonical_projection_reports=projection_resolution["reports"],
            )
        finally:
            left.close()
            right.close()
    except Exception as exc:
        _sanitize_xml_artifact(generated)
        return {
            "status": "algorithm_failed",
            "gate_a_status": "blocked",
            "gate_a_evidence_complete": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "differences": {"exception": type(exc).__name__},
            "evidence_incomplete_reasons": {"runtime_comparison": f"{type(exc).__name__}: {exc}"},
        }
    _sanitize_xml_artifact(generated)
    evidence_sections = _same_source_evidence_sections(equivalence)
    return {
        "status": _same_source_validation_status(equivalence),
        "mode": "same_source_urdf_to_canonical_mjcf",
        "comparison_schema_version": equivalence.get("schema_version"),
        "source": display_path(source),
        "generated": display_path(generated),
        "source_sha256": conversion["source_sha256"],
        "generated_sha256": sha256_file(generated),
        "generated_runtime_sha256": conversion["output_sha256"],
        "source_resolution": _g1_same_source_resolution_payload(source, manifest_path=manifest_path),
        "semantic_map_resolution": semantic_map_resolution,
        "projection_report_resolution": projection_resolution["resolution"],
        "generated_artifact_sanitization": {
            "status": "applied",
            "path_placeholders": [
                "${ROBOT_ZOO_CACHE}",
                "${ROBOT_DESCRIPTIONS_CACHE}",
                "${NEWTON_CACHE}",
                "${LOCAL_SOURCE_PATH}",
            ],
        },
        "strict_equivalent": equivalence["strict_equivalent"],
        "gate_a_status": equivalence["gate_a_status"],
        "gate_a_evidence_complete": equivalence["gate_a_evidence_complete"],
        "gate_a_required_sections": equivalence["gate_a_required_sections"],
        "evidence_statuses": _same_source_evidence_statuses(evidence_sections),
        "evidence_incomplete_reasons": _same_source_evidence_incomplete_reasons(evidence_sections),
        "evidence": evidence_sections,
        "differences": {failure: True for failure in equivalence["failures"]},
        "coordinate_comparison": equivalence.get("coordinate_comparison", {}),
        "runtime_dimensions": {
            "source_nq": equivalence["left_signature"]["nq"],
            "source_nv": equivalence["left_signature"]["nv"],
            "generated_nq": equivalence["right_signature"]["nq"],
            "generated_nv": equivalence["right_signature"]["nv"],
        },
        "tolerances": equivalence["tolerances"],
    }


def _same_source_validation_status(equivalence: dict) -> str:
    gate_a_status = equivalence.get("gate_a_status")
    if gate_a_status == "complete_passed":
        return "passed"
    if gate_a_status == "incomplete":
        return "incomplete"
    return "failed"


def _same_source_evidence_sections(equivalence: dict) -> dict:
    return {section: equivalence.get(section, {}) for section in equivalence.get("gate_a_required_sections", [])}


def _same_source_evidence_statuses(evidence_sections: dict[str, dict]) -> dict:
    return {section: payload.get("status", "missing") for section, payload in evidence_sections.items()}


def _same_source_evidence_incomplete_reasons(evidence_sections: dict[str, dict]) -> dict:
    reasons = {}
    for section, payload in evidence_sections.items():
        if payload.get("status") == "passed":
            continue
        reasons[section] = payload.get("reason") or payload.get("failures") or payload.get("status", "missing")
    return reasons


def _find_g1_same_source_urdf(
    *,
    manifest_path: Path = ROBOT_ZOO_MANIFEST_PATH,
    resolved_sources: dict[str, ResolvedRobotSource] | None = None,
) -> Path | None:
    resolved = _resolve_g1_same_source_urdf(manifest_path=manifest_path, resolved_sources=resolved_sources)
    return resolved.path if resolved and resolved.available else None


def _find_g1_same_source_urdf_compat(
    *,
    manifest_path: Path,
    resolved_sources: dict[str, ResolvedRobotSource] | None,
) -> Path | None:
    """Call the manifest-backed resolver while tolerating old test monkeypatches."""

    try:
        signature = inspect.signature(_find_g1_same_source_urdf)
    except (TypeError, ValueError):
        return _find_g1_same_source_urdf()
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    kwargs = {}
    if has_var_kwargs or "manifest_path" in signature.parameters:
        kwargs["manifest_path"] = manifest_path
    if has_var_kwargs or "resolved_sources" in signature.parameters:
        kwargs["resolved_sources"] = resolved_sources
    return _find_g1_same_source_urdf(**kwargs)


def _resolve_g1_same_source_urdf(
    *,
    manifest_path: Path = ROBOT_ZOO_MANIFEST_PATH,
    resolved_sources: dict[str, ResolvedRobotSource] | None = None,
) -> ResolvedRobotSource | None:
    manifest = load_robot_zoo_manifest(manifest_path)
    entry = manifest.model_by_id.get("unitree_g1_urdf")
    if entry is None:
        return None
    if resolved_sources and "unitree_g1_urdf" in resolved_sources:
        return resolved_sources["unitree_g1_urdf"]
    return resolve_robot_source(entry, allow_fetch=False)


def _g1_same_source_resolution_payload(source: Path, *, manifest_path: Path) -> dict:
    resolved = _resolve_g1_same_source_urdf(manifest_path=manifest_path)
    if resolved is None:
        return {
            "status": "available",
            "resolver": "test_override",
            "path": display_path(source),
            "local_file_sha256": sha256_file(source),
            "manifest_model_id": None,
        }
    manifest = load_robot_zoo_manifest(manifest_path)
    payload = resolved.to_json(manifest_path=manifest.path, manifest_sha256=manifest.sha256)
    payload["manifest_model_id"] = "unitree_g1_urdf"
    return payload


def _g1_same_source_semantic_map(*, manifest_path: Path, source: Path) -> tuple[dict[str, str | dict] | None, dict]:
    try:
        manifest = load_robot_zoo_manifest(manifest_path)
        entry = manifest.model_by_id.get("unitree_g1_urdf")
        if entry is None:
            return None, {
                "status": "missing",
                "source": "verified_semantic_map",
                "reason": "manifest entry unitree_g1_urdf is missing",
            }
        semantic_path = _resolve_verified_semantic_map_path(entry, manifest_path=manifest.path)
        if semantic_path is None:
            return None, {
                "status": "missing",
                "source": "verified_semantic_map",
                "reason": "manifest-backed verified semantic map was not found",
            }
        payload = json.loads(semantic_path.read_text())
        expected_sha = payload.get("source_model", {}).get("local_file_sha256")
        actual_sha = sha256_file(source)
        if expected_sha and expected_sha != actual_sha:
            return None, {
                "status": "failed",
                "source": "verified_semantic_map",
                "path": display_path(semantic_path),
                "manifest_model_id": "unitree_g1_urdf",
                "reason": "verified semantic map source hash does not match manifest-resolved G1 URDF",
                "expected_source_sha256": expected_sha,
                "actual_source_sha256": actual_sha,
            }
        return load_semantic_map(semantic_path), {
            "status": "available",
            "source": "verified_semantic_map",
            "path": display_path(semantic_path),
            "manifest_model_id": "unitree_g1_urdf",
        }
    except Exception as exc:
        return None, {
            "status": "failed",
            "source": "verified_semantic_map",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _g1_same_source_projection_reports(
    source: Path,
    generated: Path,
    *,
    semantic_map: dict[str, str | dict] | None,
    full_reports: dict[str, dict] | None,
) -> dict:
    source_report = _canonical_projection_report_for_model("unitree_g1_urdf", full_reports)
    generated_report = None
    source_error = None
    generated_error = None
    if semantic_map is not None:
        try:
            source_report = _compile_same_source_canonical_projection(
                source,
                semantic_map=semantic_map,
                model_format="urdf",
                model_id="unitree_g1_same_source_source_urdf",
            )
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
        try:
            generated_report = _compile_same_source_canonical_projection(
                generated,
                semantic_map=semantic_map,
                model_format="mjcf",
                model_id="unitree_g1_same_source_canonical_mjcf",
            )
        except Exception as exc:
            generated_error = f"{type(exc).__name__}: {exc}"
    reasons: dict[str, str] = {}
    if source_error:
        reasons["source"] = source_error
    elif source_report is None:
        reasons["source"] = "unitree_g1_urdf canonical_projection_reports were not available from manifest validation"
    if semantic_map is None:
        reasons["semantic_map"] = "verified semantic map was not available for canonical projection"
    if generated_error:
        reasons["generated"] = generated_error
    elif source_report is not None and generated_report is None:
        reasons["generated"] = "generated canonical MJCF projection report was not materialized"
    reports = (source_report, generated_report) if source_report is not None and generated_report is not None else None
    return {
        "reports": reports,
        "resolution": {
            "status": "available" if reports is not None else "incomplete",
            "source_report": "available" if source_report is not None else "missing",
            "generated_report": "available" if generated_report is not None else "missing",
            "generated_model": display_path(generated),
            "source_model": display_path(source),
            "reasons": reasons,
        },
    }


def _canonical_projection_report_for_model(model_id: str, full_reports: dict[str, dict] | None) -> dict | None:
    if not full_reports:
        return None
    report = full_reports.get(model_id, {})
    if not isinstance(report, dict):
        return None
    projection = report.get("canonical_projection_reports")
    return projection if isinstance(projection, dict) and projection else None


def _compile_same_source_canonical_projection(
    model_path: Path,
    *,
    semantic_map: dict[str, str | dict],
    model_format: str,
    model_id: str,
) -> dict:
    profile = compile_kinematic_profile_v3(
        model_path,
        semantic_map,
        model_id=model_id,
        model_format=model_format,
        backend="mujoco",
        low_discrepancy_count=1,
        reproduction_command=(
            "python -m soma_retargeter.tools.compile_kinematic_profile_v3 "
            f"--robot-id {model_id} --backend mujoco"
        ),
    )
    return profile.to_json().get("canonical_projection_reports", {})


def _compare_runtime_models_for_gate_a(
    left,
    right,
    *,
    semantic_map: dict[str, str | dict] | None,
    canonical_projection_reports: tuple[dict, dict] | None,
) -> dict:
    kwargs = {}
    try:
        signature = inspect.signature(compare_runtime_models)
    except (TypeError, ValueError):
        signature = None
    has_var_kwargs = False
    if signature is not None:
        has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if semantic_map is not None and (signature is None or has_var_kwargs or "semantic_map" in signature.parameters):
        kwargs["semantic_map"] = semantic_map
    if (
        canonical_projection_reports is not None
        and (signature is None or has_var_kwargs or "canonical_projection_reports" in signature.parameters)
    ):
        kwargs["canonical_projection_reports"] = canonical_projection_reports
    for key, value in {
        "position_atol": 1e-6,
        "rotation_atol": 2e-6,
        "projection_atol": 1e-7,
    }.items():
        if signature is None or has_var_kwargs or key in signature.parameters:
            kwargs[key] = value
    return compare_runtime_models(left, right, **kwargs)


def _deterministic_rerun_report(
    entries: tuple[RobotZooEntry, ...],
    reports: dict[str, dict],
    resolved_sources: dict[str, ResolvedRobotSource],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    low_discrepancy_count: int,
    deterministic_rerun: bool,
) -> dict:
    models: dict[str, dict] = {}
    if not deterministic_rerun:
        not_run_models = {
            model_id: {
                "status": "not_run",
                "input_status": report["status"],
                "compared": False,
                "reason": "second independent run not requested",
            }
            for model_id, report in sorted(reports.items())
        }
        return {
            "schema_version": 2,
            "status": "not_run",
            "reason": (
                "pass --deterministic-rerun to execute independent two-run comparisons; "
                "single-run reports are not counted as deterministic pass"
            ),
            "comparison_fields": _deterministic_comparison_fields(),
            "totals": _deterministic_totals(not_run_models),
            "models": not_run_models,
        }

    with TemporaryDirectory() as tmp:
        rerun_semantic_maps = Path(tmp) / "semantic_maps"
        for entry in entries:
            first = reports[entry.id]
            input_status = first["status"]
            resolved = resolved_sources[entry.id]
            if input_status == RobotValidationStatus.SOURCE_UNAVAILABLE.value:
                models[entry.id] = {
                    "status": "source_unavailable",
                    "input_status": input_status,
                    "compared": False,
                    "reason": "source unavailable; not counted as deterministic pass",
                }
                continue
            if input_status == RobotValidationStatus.LICENSE_BLOCKED.value:
                models[entry.id] = {
                    "status": "license_blocked",
                    "input_status": input_status,
                    "compared": False,
                    "reason": "license blocked; no model source may be rerun",
                }
                continue
            if input_status == RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value:
                models[entry.id] = _negative_control_determinism_entry(
                    entry,
                    resolved,
                    first,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                    semantic_maps_dir=rerun_semantic_maps,
                    low_discrepancy_count=low_discrepancy_count,
                )
                continue
            if input_status not in {RobotValidationStatus.PASSED.value, RobotValidationStatus.PARTIAL_PASSED.value}:
                models[entry.id] = {
                    "status": "skipped_non_pass_status",
                    "input_status": input_status,
                    "compared": False,
                    "reason": "only terminal profile pass statuses are counted in deterministic pass",
                }
                continue
            rerun = _validate_entry(
                entry,
                resolved,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                semantic_maps_dir=rerun_semantic_maps,
                low_discrepancy_count=low_discrepancy_count,
                reproduction_command=first.get("reproduction_command", ""),
            )
            models[entry.id] = _compare_deterministic_profile_reports(first, rerun)

    totals = _deterministic_totals(models)
    if totals["compared_count"] == 0:
        status = "no_comparable_models"
        reason = "no terminal profile pass reports were available for two-run comparison"
    elif totals["mismatch_count"] == 0 and totals["rerun_failed_count"] == 0:
        status = "passed"
        reason = "all compared terminal profile reports matched; unavailable sources were not counted as pass"
    else:
        status = "failed"
        reason = "one or more deterministic rerun comparisons mismatched or failed to rerun"
    return {
        "schema_version": 2,
        "status": status,
        "reason": reason,
        "comparison_fields": _deterministic_comparison_fields(),
        "tolerances": {"float_abs_tol": 1e-9, "float_rel_tol": 1e-9},
        "totals": totals,
        "models": dict(sorted(models.items())),
    }


def _negative_control_determinism_entry(
    entry: RobotZooEntry,
    resolved: ResolvedRobotSource,
    first: dict,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    semantic_maps_dir: Path,
    low_discrepancy_count: int,
) -> dict:
    rerun = _validate_entry(
        entry,
        resolved,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        semantic_maps_dir=semantic_maps_dir,
        low_discrepancy_count=low_discrepancy_count,
        reproduction_command=first.get("reproduction_command", ""),
    )
    matched = first.get("status") == rerun.get("status") and _negative_runtime_summary(first) == _negative_runtime_summary(rerun)
    return {
        "status": "matched" if matched else "mismatched",
        "input_status": first.get("status"),
        "rerun_status": rerun.get("status"),
        "compared": True,
        "profile_comparison": False,
        "reason": "negative controls have no humanoid profile hash; compared load status and runtime dimensions only",
        "comparisons": {
            "status": {
                "matched": first.get("status") == rerun.get("status"),
                "first": first.get("status"),
                "second": rerun.get("status"),
                "mismatch_paths": [] if first.get("status") == rerun.get("status") else ["status"],
            },
            "runtime_summary": _comparison_result(
                _negative_runtime_summary(first),
                _negative_runtime_summary(rerun),
            ),
        },
    }


def _compare_deterministic_profile_reports(first: dict, rerun: dict) -> dict:
    first_payload = _deterministic_profile_payload(first)
    rerun_payload = _deterministic_profile_payload(rerun)
    comparisons = {
        field: _comparison_result(first_payload[field], rerun_payload[field])
        for field in _deterministic_comparison_fields()
    }
    mismatches = {
        field: comparison["mismatch_paths"]
        for field, comparison in comparisons.items()
        if not comparison["matched"]
    }
    matched = not mismatches
    status = "matched" if matched else "mismatched"
    return {
        "status": status,
        "input_status": first.get("status"),
        "rerun_status": rerun.get("status"),
        "compared": True,
        "profile_comparison": True,
        "first_deterministic_hash": first.get("deterministic_hash"),
        "second_deterministic_hash": rerun.get("deterministic_hash"),
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def _deterministic_comparison_fields() -> list[str]:
    return [
        "status",
        "deterministic_hash",
        "rank_summary",
        "canonical_projection_residuals",
        "semantic_site_evidence",
    ]


def _deterministic_profile_payload(report: dict) -> dict:
    return {
        "status": report.get("status"),
        "deterministic_hash": report.get("deterministic_hash"),
        "rank_summary": _rank_summary(report),
        "canonical_projection_residuals": _canonical_projection_residuals(report),
        "semantic_site_evidence": _semantic_site_evidence(report),
    }


def _rank_summary(report: dict) -> dict:
    keys = (
        "regular_rank_translation",
        "nominal_rank_translation",
        "regular_rank_rotation",
        "nominal_rank_rotation",
        "singularity_fraction_translation",
        "singularity_fraction_rotation",
        "epsilon_unstable_fraction",
        "epsilon_unstable_sample_fraction",
        "epsilon_stability_gate_passed",
        "epsilon_unstable_columns",
        "samples",
        "rank_method",
        "regular_rank_fraction_threshold",
    )
    return {
        task: {key: payload.get(key) for key in keys if key in payload}
        for task, payload in sorted(report.get("rank_stability", {}).items())
    }


def _canonical_projection_residuals(report: dict) -> dict:
    canonical = report.get("canonical_projection_reports", {})
    motions = canonical.get("motions", {})
    return {
        "motion_order": canonical.get("motion_order", []),
        "target_source": canonical.get("target_source"),
        "failures": canonical.get("failures", []),
        "unreachable_demands": canonical.get("unreachable_demands", []),
        "motions": {
            motion_name: {
                task_name: {
                    key: task_payload.get(key)
                    for key in (
                        "status",
                        "converged",
                        "residual",
                        "normalized_residual",
                        "normalization_scale",
                        "iterations",
                        "active_coordinates",
                        "desired_source",
                        "reference",
                        "target",
                    )
                    if key in task_payload
                }
                for task_name, task_payload in sorted(motion_payload.get("tasks", {}).items())
            }
            for motion_name, motion_payload in sorted(motions.items())
        },
    }


def _semantic_site_evidence(report: dict) -> dict:
    return {
        "semantic_map_resolution": {
            key: value
            for key, value in report.get("semantic_map_resolution", {}).items()
            if key in {"status", "source", "path", "warning"}
        },
        "sites": {
            semantic: {
                key: site.get(key)
                for key in (
                    "semantic_name",
                    "body_name",
                    "local_position",
                    "local_rotation_xyzw",
                    "source",
                    "confidence",
                    "reason",
                )
                if key in site
            }
            for semantic, site in sorted(report.get("semantic_sites", {}).items())
        },
    }


def _negative_runtime_summary(report: dict) -> dict:
    runtime = report.get("runtime_adapter", {})
    return {
        "backend": runtime.get("backend"),
        "nq": runtime.get("nq"),
        "nv": runtime.get("nv"),
        "body_count": runtime.get("body_count"),
        "humanoid_profile_generated": report.get("morphology_classification", {}).get("humanoid_profile_generated"),
    }


def _comparison_result(first: object, second: object) -> dict:
    mismatch_paths: list[str] = []
    _compare_values(first, second, path="", mismatch_paths=mismatch_paths)
    return {
        "matched": not mismatch_paths,
        "first": first,
        "second": second,
        "mismatch_paths": mismatch_paths[:50],
    }


def _compare_values(first: object, second: object, *, path: str, mismatch_paths: list[str]) -> None:
    if isinstance(first, bool) or isinstance(second, bool):
        if first != second:
            mismatch_paths.append(path or "$")
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        if not math.isclose(float(first), float(second), rel_tol=1e-9, abs_tol=1e-9):
            mismatch_paths.append(path or "$")
        return
    if isinstance(first, dict) and isinstance(second, dict):
        keys = set(first) | set(second)
        for key in sorted(keys):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in first or key not in second:
                mismatch_paths.append(child_path)
                continue
            _compare_values(first[key], second[key], path=child_path, mismatch_paths=mismatch_paths)
        return
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            mismatch_paths.append(f"{path}.length" if path else "length")
            return
        for idx, (first_item, second_item) in enumerate(zip(first, second)):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            _compare_values(first_item, second_item, path=child_path, mismatch_paths=mismatch_paths)
        return
    if first != second:
        mismatch_paths.append(path or "$")


def _deterministic_totals(models: dict[str, dict]) -> dict:
    compared = [model for model in models.values() if model.get("compared")]
    return {
        "model_count": len(models),
        "compared_count": len(compared),
        "matched_count": sum(1 for model in compared if model.get("status") == "matched"),
        "mismatch_count": sum(1 for model in compared if model.get("status") == "mismatched"),
        "rerun_failed_count": sum(1 for model in compared if model.get("rerun_status") in {
            RobotValidationStatus.MODEL_LOAD_FAILED.value,
            RobotValidationStatus.ALGORITHM_FAILED.value,
            RobotValidationStatus.SEMANTIC_FAILED.value,
        }),
        "source_unavailable_count": sum(
            1 for model in models.values() if model.get("status") == "source_unavailable"
        ),
        "license_blocked_count": sum(1 for model in models.values() if model.get("status") == "license_blocked"),
        "skipped_non_pass_count": sum(
            1 for model in models.values() if model.get("status") == "skipped_non_pass_status"
        ),
    }


def _environment_report(manifest, *, git_snapshot: dict[str, str] | None = None) -> dict:
    git_snapshot = git_snapshot or _git_snapshot()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_snapshot["head"],
        "git_status_short": git_snapshot["status_short"],
        "package_versions": _package_versions(),
        "manifest": {
            "path": display_path(manifest.path),
            "sha256": manifest.sha256,
            "model_count": len(manifest.entries),
        },
        "source_cache_root": "${ROBOT_ZOO_CACHE}",
        "seed": 0,
    }


def _loader_provenance(backend: str, model_format: str) -> dict:
    loader = {
        ("newton", "urdf"): "newton.ModelBuilder.add_urdf",
        ("newton", "xml"): "newton.ModelBuilder.add_mjcf",
        ("newton", "mjcf"): "newton.ModelBuilder.add_mjcf",
        ("mujoco", "urdf"): "mujoco.MjModel.from_xml_path",
        ("mujoco", "xml"): "mujoco.MjModel.from_xml_path",
        ("mujoco", "mjcf"): "mujoco.MjModel.from_xml_path",
    }.get((backend, model_format), f"{backend} loader for {model_format or 'unavailable'}")
    return {
        "loader": loader,
        "backend": backend,
        "model_format": model_format or _unavailable("model format is unavailable"),
        "compiled_model_manifest": _unavailable("runtime backend does not expose a serialized compiled-model manifest"),
    }


def _package_versions() -> dict:
    return {
        "python": platform.python_version(),
        "soma-retargeter": _version_or_unavailable("soma-retargeter"),
        "robot-descriptions": _version_or_unavailable("robot-descriptions"),
        "newton": _version_or_unavailable("newton"),
        "mujoco": _version_or_unavailable("mujoco"),
        "warp": _version_or_unavailable("warp-lang"),
    }


def _version_or_unavailable(package: str) -> str | dict:
    try:
        return importlib.metadata.version(package)
    except Exception as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def _assert_allowed_status(status: str) -> None:
    if status not in allowed_status_values():
        raise ValueError(f"invalid Robot Zoo validation status: {status!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_artifact_payload(payload)
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")


def _sanitize_artifact_payload(value):
    if isinstance(value, dict):
        return {key: _sanitize_artifact_payload(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_artifact_payload(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_artifact_payload(child) for child in value]
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, str):
        return _sanitize_artifact_string(value)
    return value


def _sanitize_artifact_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        path_text = token.rstrip(".,;")
        suffix = token[len(path_text) :]
        return f"{display_path(Path(path_text))}{suffix}"

    return _LOCAL_ABSOLUTE_PATH_RE.sub(replace, value)


def _sanitize_xml_artifact(path: Path) -> None:
    if not path.exists():
        return
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        path.write_text(_sanitize_artifact_string(path.read_text(errors="replace")))
        return
    changed = False
    for element in tree.iter():
        if element.text:
            sanitized = _sanitize_artifact_string(element.text)
            if sanitized != element.text:
                element.text = sanitized
                changed = True
        if element.tail:
            sanitized = _sanitize_artifact_string(element.tail)
            if sanitized != element.tail:
                element.tail = sanitized
                changed = True
        for key, value in list(element.attrib.items()):
            sanitized = _sanitize_artifact_string(value)
            if sanitized != value:
                element.set(key, sanitized)
                changed = True
    if changed:
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode", xml_declaration=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def _external_reproducibility_commands(artifact_root: Path) -> list[str]:
    return _required_reproducibility_artifact_protocol(artifact_root)["commands"]


def _required_reproducibility_artifact_protocol(artifact_root: Path) -> dict:
    root = display_path(artifact_root)
    return {
        "producer": "external_test_protocol",
        "reason": "validation generation does not execute the pytest, JUnit, coverage, or acceptance-audit test protocol",
        "required_files": list(_REQUIRED_REPRODUCIBILITY_ARTIFACTS),
        "commands": [
            "python -m pip install pytest coverage",
            f"python -m coverage run -m pytest tests --junitxml={root}/test_results/junit.xml > {root}/test_results/pytest.txt 2>&1",
            f"python -m coverage json -o {root}/test_results/coverage.json",
            (
                "python scripts/audit_retargeting_v3_step2.py "
                f"--artifact-dir {root} "
                "--source-root . "
                f"--output-json {root}/acceptance_ledger.json "
                f"--junit-xml {root}/test_results/acceptance_audit.junit.xml"
            ),
        ],
    }


def _clear_json_files(path: Path) -> None:
    for stale in path.glob("*.json"):
        stale.unlink()


def _clear_matching_files(path: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for stale in path.glob(pattern):
            stale.unlink()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_snapshot() -> dict[str, str]:
    return {
        "head": _git(["rev-parse", "HEAD"]),
        "status_short": _git(["status", "--short"]),
    }


def isolated_validation_run(*, manifest_path: str | Path, low_discrepancy_count: int = 1) -> dict:
    """Run validation into a temporary directory and return the parsed summary."""

    with TemporaryDirectory() as tmp:
        summary = write_validation_artifacts(
            Path(tmp) / "artifacts",
            manifest_path=manifest_path,
            low_discrepancy_count=low_discrepancy_count,
        )
        return json.loads(json.dumps(summary))
