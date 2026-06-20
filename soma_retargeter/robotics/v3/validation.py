"""Artifact generation for Step-2 offline validation."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import time

from .model_adapter import NewtonRuntimeModelAdapter
from .profile import compile_kinematic_profile_v3
from .semantic_sites import default_rpo_semantic_map, infer_semantic_map_from_body_names


DEFAULT_LOW_DISCREPANCY_COUNT = 32
ROBOT_ZOO_MANIFEST_PATH = Path("assets/robot_zoo/robot_zoo_manifest.json")

REQUIRED_ARTIFACT_IDS = [
    "roboparty_rpo",
    "unitree_g1_mjcf",
    "unitree_g1_urdf",
    "unitree_g1_23dof",
    "unitree_h1",
    "robotis_op3",
    "booster_t1",
    "pal_talos",
    "berkeley_humanoid",
]

DESCRIPTION_MODULES = {
    "unitree_g1_mjcf": ("robot_descriptions.g1_mj_description", "MJCF_PATH"),
    "unitree_g1_urdf": ("robot_descriptions.g1_description", "URDF_PATH"),
    "unitree_h1": ("robot_descriptions.h1_mj_description", "MJCF_PATH"),
    "robotis_op3": ("robot_descriptions.op3_mj_description", "MJCF_PATH"),
    "booster_t1": ("robot_descriptions.booster_t1_mj_description", "MJCF_PATH"),
    "pal_talos": ("robot_descriptions.talos_mj_description", "MJCF_PATH"),
    "berkeley_humanoid": ("robot_descriptions.berkeley_humanoid_description", "URDF_PATH"),
}

MANIFEST_MODEL_ID_BY_REPORT_ID = {
    "roboparty_rpo": "roboparty_rpo_local",
    "unitree_g1_mjcf": "unitree_g1_mjcf",
    "unitree_g1_urdf": "unitree_g1_urdf",
    "unitree_h1": "unitree_h1_mjcf",
    "robotis_op3": "robotis_op3_mjcf",
    "booster_t1": "booster_t1_mjcf",
    "pal_talos": "pal_talos_mjcf_direct",
    "berkeley_humanoid": "berkeley_humanoid_urdf",
}

EXTRA_MODEL_FILES = {
    "unitree_g1_23dof": ("robot_descriptions.g1_description", "PACKAGE_PATH", "g1_23dof.xml"),
}


def write_validation_artifacts(
    output_dir: str | Path = "artifacts/retargeting_v3_step2",
    *,
    include_missing_required_reports: bool = True,
    low_discrepancy_count: int = DEFAULT_LOW_DISCREPANCY_COUNT,
) -> dict:
    out = Path(output_dir)
    per_robot = out / "per_robot"
    failures_dir = out / "failures"
    semantic_maps_dir = out / "semantic_maps"
    per_robot.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    semantic_maps_dir.mkdir(parents=True, exist_ok=True)
    _clear_json_files(failures_dir)
    commands = []
    reports = {}
    rpo_path = Path("assets/robots/atom01/mjcf/atom01.xml")
    if rpo_path.exists():
        semantic_map = default_rpo_semantic_map()
        semantic_map_path = _write_semantic_map_artifact(
            semantic_maps_dir,
            "roboparty_rpo",
            semantic_map,
            source="default_rpo_semantic_map",
        )
        cmd = _compile_command(
            model_path=rpo_path,
            model_id="roboparty_rpo",
            output_path=per_robot / "roboparty_rpo.json",
            semantic_map_path=semantic_map_path,
            low_discrepancy_count=low_discrepancy_count,
        )
        commands.append(cmd)
        profile = compile_kinematic_profile_v3(
            rpo_path,
            semantic_map,
            model_id="roboparty_rpo",
            backend="newton",
            low_discrepancy_count=low_discrepancy_count,
            reproduction_command=cmd,
        )
        _write_profile_report(profile, per_robot / "roboparty_rpo.json", semantic_map_path=semantic_map_path)
        reports["roboparty_rpo"] = {"status": "compiled", "failures": profile.failures}
    else:
        reports["roboparty_rpo"] = {"status": "missing", "failures": ["local RPO model not found"]}
    for rid, resolved in _resolve_description_models().items():
        if rid in reports:
            continue
        try:
            adapter = NewtonRuntimeModelAdapter(resolved)
            semantic_map = infer_semantic_map_from_body_names(adapter)
            adapter.close()
            semantic_map_path = _write_semantic_map_artifact(
                semantic_maps_dir,
                rid,
                semantic_map,
                source="inferred_from_newton_body_names",
            )
            cmd = _compile_command(
                model_path=resolved,
                model_id=rid,
                output_path=per_robot / f"{rid}.json",
                semantic_map_path=semantic_map_path,
                low_discrepancy_count=low_discrepancy_count,
            )
            commands.append(cmd)
            profile = compile_kinematic_profile_v3(
                resolved,
                semantic_map,
                model_id=rid,
                backend="newton",
                low_discrepancy_count=low_discrepancy_count,
                reproduction_command=cmd,
            )
            report = _write_profile_report(profile, per_robot / f"{rid}.json", semantic_map_path=semantic_map_path)
            reports[rid] = {"status": "compiled" if not profile.failures else "failed", "failures": profile.failures}
            if profile.failures:
                _write_json(failures_dir / f"{rid}.json", report)
        except Exception as exc:
            cmd = _validate_command(output_dir=out, low_discrepancy_count=low_discrepancy_count)
            report = {
                "schema_version": 3,
                "model": {"id": rid, "path": str(resolved)},
                "failures": [f"model compile failed: {type(exc).__name__}: {exc}"],
                "warnings": [],
                "reproduction_command": cmd,
            }
            report = augment_validation_report_metadata(report, semantic_map_path=None)
            _write_json(per_robot / f"{rid}.json", report)
            _write_json(failures_dir / f"{rid}.json", report)
            reports[rid] = {"status": "failed", "failures": report["failures"]}
    if include_missing_required_reports:
        for rid in REQUIRED_ARTIFACT_IDS:
            if rid in reports:
                continue
            cmd = _validate_command(output_dir=out, low_discrepancy_count=low_discrepancy_count)
            report = {
                "schema_version": 3,
                "model": {"id": rid},
                "failures": [f"required complete model {rid!r} is not present in the local workspace cache"],
                "warnings": ["full Robot Zoo synchronization is out of scope for this Step-2 implementation run"],
                "reproduction_command": cmd,
            }
            report = augment_validation_report_metadata(report, semantic_map_path=None)
            _write_json(per_robot / f"{rid}.json", report)
            _write_json(failures_dir / f"{rid}.json", report)
            reports[rid] = {"status": "missing", "failures": report["failures"]}
    validation_checks = _write_validation_checks(out, per_robot)
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "package_versions": _package_versions(),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    (out / "commands.txt").write_text("\n".join(commands + [_validate_command(output_dir=out, low_discrepancy_count=low_discrepancy_count)]) + "\n")
    summary = {
        "reports": reports,
        "compiled_count": sum(1 for r in reports.values() if r["status"] == "compiled"),
        "missing_count": sum(1 for r in reports.values() if r["status"] == "missing"),
        "failure_artifacts_count": len(list(failures_dir.glob("*.json"))),
        "validation_checks": validation_checks,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def augment_validation_report_metadata(report: dict, *, semantic_map_path: str | Path | None) -> dict:
    """Add reproducibility metadata without changing compiler semantics."""
    model = report.setdefault("model", {})
    model_path = Path(model["path"]) if model.get("path") else None
    model["local_file_sha256"] = _sha256_file(model_path) if model_path else _unavailable("model path is unavailable")
    model["source"] = _model_source_metadata(str(model.get("id", "")), model_path)
    model["semantic_map_artifact"] = str(semantic_map_path) if semantic_map_path else _unavailable("semantic map was not available before model resolution failed")
    model["manifest"] = _robot_zoo_manifest_metadata(str(model.get("id", "")))

    runtime = report.setdefault("runtime_adapter", {})
    backend = model.get("backend") or runtime.get("backend") or "newton"
    runtime["package_versions"] = _package_versions()
    runtime["loader_provenance"] = _loader_provenance(str(backend), str(model.get("format", "")))
    return report


def _resolve_description_models() -> dict[str, Path]:
    resolved = {}
    for rid, (module_name, attr) in DESCRIPTION_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            path = Path(getattr(module, attr))
            if path.exists():
                resolved[rid] = path
        except Exception:
            continue
    for rid, (module_name, root_attr, relative) in EXTRA_MODEL_FILES.items():
        try:
            module = importlib.import_module(module_name)
            path = Path(getattr(module, root_attr)) / relative
            if path.exists():
                resolved[rid] = path
        except Exception:
            continue
    return resolved


def _write_profile_report(profile, output_path: Path, *, semantic_map_path: Path) -> dict:
    report = augment_validation_report_metadata(profile.to_json(), semantic_map_path=semantic_map_path)
    _write_json(output_path, report)
    return report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _clear_json_files(path: Path) -> None:
    for stale in path.glob("*.json"):
        stale.unlink()


def _write_semantic_map_artifact(semantic_maps_dir: Path, model_id: str, semantic_map: dict, *, source: str) -> Path:
    path = semantic_maps_dir / f"{model_id}.json"
    _write_json(path, {"schema_version": 1, "model_id": model_id, "source": source, "semantics": semantic_map})
    return path


def _compile_command(
    *,
    model_path: Path,
    model_id: str,
    output_path: Path,
    semantic_map_path: Path,
    low_discrepancy_count: int,
) -> str:
    return shlex.join(
        [
            "python",
            "-m",
            "soma_retargeter.tools.compile_kinematic_profile_v3",
            "--backend",
            "newton",
            "--model",
            str(model_path),
            "--model-id",
            model_id,
            "--semantic-map",
            str(semantic_map_path),
            "--output",
            str(output_path),
            "--low-discrepancy-count",
            str(low_discrepancy_count),
        ]
    )


def _validate_command(*, output_dir: Path, low_discrepancy_count: int) -> str:
    return shlex.join(
        [
            "python",
            "-m",
            "soma_retargeter.tools.validate_kinematic_profile_v3",
            "--output-dir",
            str(output_dir),
            "--low-discrepancy-count",
            str(low_discrepancy_count),
        ]
    )


def _write_validation_checks(out: Path, per_robot: Path) -> dict:
    checks = {
        "g1_mjcf_urdf_equivalence": _g1_mjcf_urdf_equivalence(per_robot),
        "roboparty_rpo_distal_hand_endpoint": _rpo_distal_hand_endpoint(per_robot),
        "robotis_op3_chest_demand_leakage": _op3_chest_demand_leakage(per_robot),
    }
    _write_json(out / "validation_checks.json", checks)
    return checks


def _read_report(per_robot: Path, report_id: str) -> dict | None:
    path = per_robot / f"{report_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _g1_mjcf_urdf_equivalence(per_robot: Path) -> dict:
    mjcf = _read_report(per_robot, "unitree_g1_mjcf")
    urdf = _read_report(per_robot, "unitree_g1_urdf")
    if not mjcf or not urdf:
        return {
            "status": "unavailable",
            "reason": "one or both G1 reports are unavailable",
            "tolerance": _g1_equivalence_tolerance(),
        }
    differences = {}
    for task in sorted(set(mjcf.get("chains", {})) | set(urdf.get("chains", {}))):
        mjcf_labels = mjcf.get("chains", {}).get(task, {}).get("coordinate_labels", [])
        urdf_labels = urdf.get("chains", {}).get(task, {}).get("coordinate_labels", [])
        if mjcf_labels != urdf_labels:
            differences[f"{task}_coordinate_labels"] = {"mjcf": mjcf_labels, "urdf": urdf_labels}
        for rank_key in ("regular_rank_translation", "regular_rank_rotation"):
            mjcf_rank = mjcf.get("rank_stability", {}).get(task, {}).get(rank_key)
            urdf_rank = urdf.get("rank_stability", {}).get(task, {}).get(rank_key)
            if mjcf_rank != urdf_rank:
                differences[f"{task}_{rank_key}"] = {"mjcf": mjcf_rank, "urdf": urdf_rank}
    return {
        "status": "passed" if not differences else "documented_limitation",
        "tolerance": _g1_equivalence_tolerance(),
        "differences": differences,
        "known_limitation": (
            "The cached G1 URDF and MJCF are not treated as strictly equivalent complete models; "
            "current artifacts document topology/rank differences instead of claiming a pass."
        )
        if differences
        else "",
    }


def _g1_equivalence_tolerance() -> dict:
    return {
        "chain_coordinate_labels": "exact ordered match required for strict topology equivalence",
        "regular_rank_translation": 0,
        "regular_rank_rotation": 0,
        "neutral_semantic_position_m": 1e-3,
    }


def _rpo_distal_hand_endpoint(per_robot: Path) -> dict:
    report = _read_report(per_robot, "roboparty_rpo")
    if not report:
        return {"status": "unavailable", "reason": "roboparty_rpo report is unavailable"}
    checks = {}
    for side, expected_body in (("left", "left_elbow_yaw_link"), ("right", "right_elbow_yaw_link")):
        task = f"{side}_hand"
        semantic = f"{side.capitalize()}Hand"
        site = report.get("semantic_sites", {}).get(semantic, {})
        labels = report.get("chains", {}).get(task, {}).get("coordinate_labels", [])
        checks[task] = {
            "semantic_body": site.get("body_name"),
            "local_position": site.get("local_position"),
            "full_arm_chain_coordinate_count": len(labels),
            "full_arm_chain_includes_shoulder_and_elbow": any("arm_pitch" in label for label in labels)
            and any("elbow_yaw" in label for label in labels),
            "body_endpoint_matches_current_public_model": site.get("body_name") == expected_body,
        }
    passed = all(
        item["body_endpoint_matches_current_public_model"]
        and item["full_arm_chain_coordinate_count"] >= 5
        and item["full_arm_chain_includes_shoulder_and_elbow"]
        for item in checks.values()
    )
    return {
        "status": "passed_with_documented_scope" if passed else "failed",
        "checks": checks,
        "scope_note": (
            "The public RPO model exposes the distal arm endpoint as the elbow-yaw link body with zero local offset; "
            "this check proves full shoulder-to-endpoint chain use, not a separate palm/fingertip mesh anchor."
        ),
    }


def _op3_chest_demand_leakage(per_robot: Path) -> dict:
    report = _read_report(per_robot, "robotis_op3")
    if not report:
        return {"status": "unavailable", "reason": "robotis_op3 report is unavailable"}
    torso_labels = report.get("chains", {}).get("torso", {}).get("coordinate_labels", [])
    projection = report.get("projection_reports", {}).get("torso", {})
    leg_like = [label for label in torso_labels if any(part in label.lower() for part in ("hip", "knee", "ank"))]
    passed = torso_labels == [] and projection.get("status") == "rank_zero" and not leg_like
    return {
        "status": "passed" if passed else "failed",
        "torso_coordinate_labels": torso_labels,
        "leg_coordinate_labels_in_torso_task": leg_like,
        "projection_status": projection.get("status"),
        "scope_note": "Chest demand is represented by a rank-zero torso projection for OP3; no leg coordinates are assigned to the torso task.",
    }


def _sha256_file(path: Path | None) -> str | dict:
    if path is None:
        return _unavailable("path is unavailable")
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        return _unavailable(f"{type(exc).__name__}: {exc}")


def _model_source_metadata(model_id: str, model_path: Path | None) -> dict:
    module_info = _description_module_info(model_id)
    if module_info is None:
        return {
            "type": "local_workspace_file",
            "path": str(model_path) if model_path else None,
            "workspace_git_head": _git(["rev-parse", "HEAD"]) or _unavailable("not in a git checkout"),
            "robot_descriptions": _unavailable("model is not supplied by robot_descriptions"),
        }
    module_name = module_info[0]
    metadata = {
        "type": "robot_descriptions_module",
        "module": module_name,
        "path_attr": module_info[1],
        "robot_descriptions_version": _version_or_unavailable("robot-descriptions"),
    }
    try:
        module = importlib.import_module(module_name)
        metadata["module_file"] = getattr(module, "__file__", None)
        repo = Path(getattr(module, "REPOSITORY_PATH", ""))
        package = Path(getattr(module, "PACKAGE_PATH", "")) if hasattr(module, "PACKAGE_PATH") else None
        metadata["package_path"] = str(package) if package else _unavailable("PACKAGE_PATH is not exposed by module")
        metadata["repository_path"] = str(repo) if str(repo) else _unavailable("REPOSITORY_PATH is not exposed by module")
        metadata["repository_remote"] = _git(["-C", str(repo), "config", "--get", "remote.origin.url"]) if repo.exists() else _unavailable("repository path is unavailable")
        metadata["repository_head"] = _git(["-C", str(repo), "rev-parse", "HEAD"]) if repo.exists() else _unavailable("repository path is unavailable")
        metadata["repository_ref"] = _git(["-C", str(repo), "symbolic-ref", "--short", "HEAD"]) if repo.exists() else _unavailable("repository path is unavailable")
        if not metadata["repository_ref"]:
            metadata["repository_ref"] = _unavailable("repository is detached or ref is unavailable")
    except Exception as exc:
        metadata["module_resolution"] = _unavailable(f"{type(exc).__name__}: {exc}")
    return metadata


def _description_module_info(model_id: str) -> tuple[str, str] | None:
    if model_id in DESCRIPTION_MODULES:
        return DESCRIPTION_MODULES[model_id]
    if model_id in EXTRA_MODEL_FILES:
        module_name, root_attr, _relative = EXTRA_MODEL_FILES[model_id]
        return module_name, root_attr
    return None


def _robot_zoo_manifest_metadata(report_id: str) -> dict:
    manifest_model_id = MANIFEST_MODEL_ID_BY_REPORT_ID.get(report_id)
    if manifest_model_id is None:
        return _unavailable(f"no Robot Zoo manifest mapping is defined for report id {report_id!r}")
    manifest = _robot_zoo_manifest()
    if manifest is None:
        return _unavailable(f"Robot Zoo manifest is unavailable at {ROBOT_ZOO_MANIFEST_PATH}")
    entry = manifest.get("model_by_id", {}).get(manifest_model_id)
    if entry is None:
        return _unavailable(f"Robot Zoo manifest entry {manifest_model_id!r} is not present")
    return {
        "status": "available",
        "manifest_path": str(ROBOT_ZOO_MANIFEST_PATH),
        "manifest_sha256": manifest["sha256"],
        "schema_version": manifest["payload"].get("schema_version"),
        "catalog_name": manifest["payload"].get("catalog_name"),
        "report_id": report_id,
        "manifest_model_id": manifest_model_id,
        "matched_by": "validation_report_id_mapping",
        "entry": entry,
    }


@lru_cache(maxsize=1)
def _robot_zoo_manifest() -> dict | None:
    try:
        payload = json.loads(ROBOT_ZOO_MANIFEST_PATH.read_text())
    except Exception:
        return None
    return {
        "payload": payload,
        "sha256": _sha256_file(ROBOT_ZOO_MANIFEST_PATH),
        "model_by_id": {entry["id"]: entry for entry in payload.get("models", [])},
    }


def _loader_provenance(backend: str, model_format: str) -> dict:
    loader = {
        ("newton", "urdf"): "newton.ModelBuilder.add_urdf",
        ("newton", "xml"): "newton.ModelBuilder.add_mjcf",
        ("newton", "mjcf"): "newton.ModelBuilder.add_mjcf",
        ("mujoco", "urdf"): "mujoco.MjModel.from_xml_path with adapter URDF mesh absolutization fallback",
        ("mujoco", "xml"): "mujoco.MjModel.from_xml_path with adapter XML compatibility fallback",
        ("mujoco", "mjcf"): "mujoco.MjModel.from_xml_path with adapter XML compatibility fallback",
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


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""
