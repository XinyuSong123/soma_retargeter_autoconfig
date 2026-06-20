"""Manifest-driven Step-2 offline validation artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.metadata
import json
import platform
import subprocess
import sys
import time

from .model_adapter import NewtonRuntimeModelAdapter
from .profile import compile_kinematic_profile_v3
from .robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    RobotValidationStatus,
    RobotZooEntry,
    ResolvedRobotSource,
    allowed_status_values,
    display_path,
    load_robot_zoo_manifest,
    reproduction_compile_command,
    reproduction_validate_command,
    resolve_robot_source,
    sha256_file,
)
from .semantic_sites import default_rpo_semantic_map, infer_semantic_map_from_body_names


DEFAULT_LOW_DISCREPANCY_COUNT = 32
ROBOT_ZOO_MANIFEST_PATH = DEFAULT_ROBOT_ZOO_MANIFEST_PATH

# Compatibility exports for older tests/imports. The manifest remains authoritative.
try:
    _DEFAULT_MANIFEST = load_robot_zoo_manifest(ROBOT_ZOO_MANIFEST_PATH)
    REQUIRED_ARTIFACT_IDS = _DEFAULT_MANIFEST.required_ids()
    MANIFEST_MODEL_ID_BY_REPORT_ID = {entry.id: entry.id for entry in _DEFAULT_MANIFEST.entries}
except Exception:
    REQUIRED_ARTIFACT_IDS = []
    MANIFEST_MODEL_ID_BY_REPORT_ID = {}


def write_validation_artifacts(
    output_dir: str | Path = "artifacts/retargeting_v3_step2",
    *,
    manifest_path: str | Path = ROBOT_ZOO_MANIFEST_PATH,
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

    reports: dict[str, dict] = {}
    commands: list[str] = []
    source_inventory: dict[str, dict] = {}
    command_artifact_root = Path("${RETARGETING_V3_ARTIFACTS}")
    command_manifest_path = Path("${ROBOT_ZOO_MANIFEST}")
    for entry in manifest.entries:
        command = reproduction_compile_command(
            entry.id,
            manifest_path=command_manifest_path,
            output_path=command_artifact_root / "per_robot" / f"{entry.id}.json",
            low_discrepancy_count=low_discrepancy_count,
            backend="newton",
        )
        commands.append(command)
        resolved = resolve_robot_source(entry, allow_fetch=allow_source_fetch)
        source_inventory[entry.id] = resolved.to_json(manifest_path=manifest.path, manifest_sha256=manifest.sha256)
        report = _validate_entry(
            entry,
            resolved,
            manifest_path=manifest.path,
            manifest_sha256=manifest.sha256,
            semantic_maps_dir=semantic_maps_dir,
            low_discrepancy_count=low_discrepancy_count,
            reproduction_command=command,
        )
        _assert_allowed_status(report["status"])
        _write_json(per_robot / f"{entry.id}.json", report)
        reports[entry.id] = _summary_entry(report)
        if report["status"] not in {
            RobotValidationStatus.PASSED.value,
            RobotValidationStatus.PARTIAL_PASSED.value,
            RobotValidationStatus.NEGATIVE_CONTROL_PASSED.value,
            RobotValidationStatus.SOURCE_UNAVAILABLE.value,
        }:
            _write_json(failures_dir / f"{entry.id}.json", report)

    cross_format = _cross_format_report(reports, per_robot)
    deterministic = _deterministic_rerun_report(reports, deterministic_rerun=deterministic_rerun)
    environment = _environment_report(manifest)
    validation_command = reproduction_validate_command(
        manifest_path=command_manifest_path,
        output_dir=command_artifact_root,
        low_discrepancy_count=low_discrepancy_count,
        deterministic_rerun=deterministic_rerun,
    )
    commands.append(validation_command)
    (out / "commands.txt").write_text("\n".join(commands) + "\n")
    _write_json(out / "environment.json", environment)
    _write_json(out / "source_inventory.json", source_inventory)
    _write_json(out / "cross_format.json", cross_format)
    _write_json(out / "deterministic_rerun.json", deterministic)

    status_counts = Counter(item["status"] for item in reports.values())
    class_counts = Counter(item["robot_class"] for item in reports.values())
    capability_counts = Counter(item["expected_capability"] for item in reports.values())
    summary = {
        "schema_version": 4,
        "manifest": {
            "path": display_path(manifest.path),
            "sha256": manifest.sha256,
            "model_count": len(manifest.entries),
            "allowed_statuses": list(allowed_status_values()),
        },
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
        "model_load_failed_count": status_counts[RobotValidationStatus.MODEL_LOAD_FAILED.value],
        "failure_artifacts_count": len(list(failures_dir.glob("*.json"))),
        "cross_format": cross_format,
        "deterministic_rerun": deterministic,
        "notes": [
            "compiled is intentionally not a validation status",
            "source_unavailable is counted separately from algorithm pass/fail",
        ],
    }
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
            "entry": entry.raw,
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
    try:
        semantic_map = _semantic_map_for_entry(entry, resolved)
        semantic_map_path = _write_semantic_map_artifact(semantic_maps_dir, entry.id, semantic_map)
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
        return _base_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            status=RobotValidationStatus.MODEL_LOAD_FAILED.value,
            failures=[f"{type(exc).__name__}: {exc}"],
            reproduction_command=reproduction_command,
        )
    report = augment_validation_report_metadata(
        profile.to_json(),
        semantic_map_path=semantic_map_path,
        manifest_path=manifest_path,
    )
    report["status"] = _profile_status(report)
    report["status_reason"] = _profile_status_reason(report)
    report["manifest_entry"] = entry.raw
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
        return _base_report(
            entry,
            resolved,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            status=RobotValidationStatus.MODEL_LOAD_FAILED.value,
            failures=[f"{type(exc).__name__}: {exc}"],
            reproduction_command=reproduction_command,
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


def _semantic_map_for_entry(entry: RobotZooEntry, resolved: ResolvedRobotSource) -> dict:
    if entry.id == "roboparty_rpo_local":
        return default_rpo_semantic_map()
    adapter = NewtonRuntimeModelAdapter(resolved.path, model_format=entry.model_format)
    try:
        return infer_semantic_map_from_body_names(adapter)
    finally:
        adapter.close()


def _profile_status(report: dict) -> str:
    failures = report.get("failures", [])
    capability = report.get("capability_status")
    if any("missing required semantics" in failure for failure in failures):
        return RobotValidationStatus.SEMANTIC_FAILED.value
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
    return "compiler recorded algorithm failures"


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
                "entry": entry.raw,
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
        "manifest_entry": entry.raw,
        "failures": failures or [],
        "warnings": warnings or [],
        "reproduction_command": reproduction_command,
    }


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


def _write_semantic_map_artifact(semantic_maps_dir: Path, model_id: str, semantic_map: dict) -> Path:
    path = semantic_maps_dir / f"{model_id}.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "model_id": model_id,
            "source": "default_rpo_semantic_map" if model_id == "roboparty_rpo_local" else "inferred_from_newton_body_names",
            "semantics": semantic_map,
        },
    )
    return path


def _cross_format_report(reports: dict[str, dict], per_robot: Path) -> dict:
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
        "schema_version": 1,
        "gates": {
            "same_source_strict": {
                "status": "not_run",
                "reason": "requires Agent A canonical URDF-to-MJCF conversion artifacts",
            },
            "variant_compatibility": {
                "status": "not_run",
                "reason": "requires both variants to pass independent algorithm gates first",
            },
        },
        "pairs": pairs,
    }


def _deterministic_rerun_report(reports: dict[str, dict], *, deterministic_rerun: bool) -> dict:
    return {
        "schema_version": 1,
        "status": "not_run" if not deterministic_rerun else "scaffolded",
        "reason": (
            "pass --deterministic-rerun to reserve a two-run comparison artifact; "
            "the current implementation does not count single-run reports as deterministic pass"
        ),
        "models": {
            model_id: {
                "status": "not_run",
                "input_status": report["status"],
                "reason": "second independent run not executed in this artifact pass",
            }
            for model_id, report in sorted(reports.items())
        },
    }


def _environment_report(manifest) -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _clear_json_files(path: Path) -> None:
    for stale in path.glob("*.json"):
        stale.unlink()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def isolated_validation_run(*, manifest_path: str | Path, low_discrepancy_count: int = 1) -> dict:
    """Run validation into a temporary directory and return the parsed summary."""

    with TemporaryDirectory() as tmp:
        summary = write_validation_artifacts(
            Path(tmp) / "artifacts",
            manifest_path=manifest_path,
            low_discrepancy_count=low_discrepancy_count,
        )
        return json.loads(json.dumps(summary))
