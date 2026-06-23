"""Runtime target-geometry audit for Step 2 differential failures."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
from typing import Iterable

import numpy as np

from .kinematic_paths import discover_paths
from .model_adapter import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter, RuntimeRobotModelAdapter
from .robot_zoo import (
    DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    DEFAULT_RPO_MODEL_PATH,
    display_path,
    load_robot_zoo_manifest,
    sanitize_reproduction_text,
    resolve_robot_source,
    sha256_file,
)
from .semantic_sites import build_semantic_sites, load_semantic_map
from .source_rest import load_soma_source_rest_frames
from .spatial import matrix_to_quat_xyzw, relative_transform


DEFAULT_AUDIT_MODEL_IDS = (
    "booster_t1_urdf",
    "booster_t1_mjcf",
    "fourier_n1_urdf",
    "fourier_n1_mjcf",
    "talos_urdf",
    "pal_talos_mjcf_direct",
    "atlas_drc_urdf",
    "atlas_v4_urdf",
    "unitree_h1_urdf",
    "unitree_h1_mjcf",
    "unitree_g1_urdf",
    "unitree_g1_mjcf",
    "roboparty_rpo_local",
    "robotis_op3_mjcf",
    "jaxon_urdf",
    "valkyrie_urdf",
    "mujoco_humanoid_mjcf",
)

DIFFERENTIAL_PAIRS = {
    "booster_t1": ("booster_t1_urdf", "booster_t1_mjcf"),
    "fourier_n1": ("fourier_n1_urdf", "fourier_n1_mjcf"),
    "talos": ("talos_urdf", "pal_talos_mjcf_direct"),
    "atlas": ("atlas_drc_urdf", "atlas_v4_urdf"),
    "unitree_h1_control": ("unitree_h1_urdf", "unitree_h1_mjcf"),
    "unitree_g1_control": ("unitree_g1_urdf", "unitree_g1_mjcf"),
}

TASK_TO_SOURCE_EDGE = {
    "torso": ("Hips", "Chest"),
    "left_hand": ("Chest", "LeftHand"),
    "right_hand": ("Chest", "RightHand"),
    "left_foot": ("Hips", "LeftFoot"),
    "right_foot": ("Hips", "RightFoot"),
}

DISTAL_SEMANTICS = (
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
    "LeftToe",
    "RightToe",
    "LeftHeel",
    "RightHeel",
)


def audit_target_geometry_matrix(
    *,
    model_ids: Iterable[str] = DEFAULT_AUDIT_MODEL_IDS,
    backend: str = "newton",
    manifest_path: str | Path = DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    semantic_maps_dir: str | Path = Path("assets/robot_zoo/semantic_maps"),
    fallback_artifact_dir: str | Path = Path("artifacts/retargeting_v3_step2_assets44/per_robot"),
) -> dict:
    """Build a JSON-serializable matrix of target-geometry evidence."""

    requested = tuple(model_ids)
    manifest = load_robot_zoo_manifest(manifest_path)
    semantic_maps_root = Path(semantic_maps_dir)
    artifact_root = Path(fallback_artifact_dir)
    try:
        source_rest, source_rest_provenance = load_soma_source_rest_frames()
    except Exception as exc:
        source_rest = {}
        source_rest_provenance = f"unavailable:{type(exc).__name__}:{exc}"

    rows = [
        _audit_one_model(
            model_id,
            backend=backend,
            manifest=manifest,
            semantic_maps_dir=semantic_maps_root,
            fallback_artifact_dir=artifact_root,
            source_rest=source_rest,
        )
        for model_id in requested
    ]
    differentials = _differentials(rows)
    status_counts = Counter(row["status"] for row in rows)
    counts = {
        "requested": len(requested),
        "runtime_loaded": status_counts.get("runtime_loaded", 0),
        "runtime_unavailable_artifact_fallback": status_counts.get("runtime_unavailable_artifact_fallback", 0),
        "runtime_failed_artifact_fallback": status_counts.get("runtime_failed_artifact_fallback", 0),
        "runtime_failed_no_artifact": status_counts.get("runtime_failed_no_artifact", 0),
        "source_unavailable": sum(1 for row in rows if row["source"]["source_status"] != "available"),
        "fingerprint_matches_runtime": sum(
            1 for row in rows if row["semantic_map"].get("fingerprint_matches_runtime") is True
        ),
        "map_change_recommended": sum(1 for row in rows if row.get("map_change_recommendation") not in {"none", "source_unavailable_no_runtime_correction"}),
        "status_counts": dict(sorted(status_counts.items())),
    }
    return {
        "schema_version": 1,
        "backend": backend,
        "scope": {
            "model_ids": list(requested),
            "differential_pairs": {name: list(pair) for name, pair in DIFFERENTIAL_PAIRS.items()},
            "standalone_controls": [
                "roboparty_rpo_local",
                "robotis_op3_mjcf",
                "jaxon_urdf",
                "valkyrie_urdf",
                "mujoco_humanoid_mjcf",
            ],
        },
        "source_rest": {
            "available": bool(source_rest),
            "provenance": source_rest_provenance,
        },
        "counts": counts,
        "rows": rows,
        "differentials": differentials,
    }


def write_target_geometry_matrix(
    output_path: str | Path = Path("artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json"),
    *,
    model_ids: Iterable[str] = DEFAULT_AUDIT_MODEL_IDS,
    backend: str = "newton",
    manifest_path: str | Path = DEFAULT_ROBOT_ZOO_MANIFEST_PATH,
    semantic_maps_dir: str | Path = Path("assets/robot_zoo/semantic_maps"),
    fallback_artifact_dir: str | Path = Path("artifacts/retargeting_v3_step2_assets44/per_robot"),
) -> dict:
    payload = audit_target_geometry_matrix(
        model_ids=model_ids,
        backend=backend,
        manifest_path=manifest_path,
        semantic_maps_dir=semantic_maps_dir,
        fallback_artifact_dir=fallback_artifact_dir,
    )
    payload = _sanitize_artifact_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _sanitize_artifact_payload(value):
    if isinstance(value, dict):
        return {key: _sanitize_artifact_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_artifact_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_reproduction_text(value)
    return value


def _audit_one_model(
    model_id: str,
    *,
    backend: str,
    manifest,
    semantic_maps_dir: Path,
    fallback_artifact_dir: Path,
    source_rest: dict[str, np.ndarray],
) -> dict:
    source = _resolve_audit_source(model_id, manifest)
    semantic_map_path = semantic_maps_dir / f"{model_id}.json"
    semantic_payload = _load_json(semantic_map_path)
    semantic_map = _semantic_map_entries(semantic_payload)
    semantic_summary = _semantic_map_summary(semantic_payload, semantic_map_path)

    if source["source_status"] != "available" or not source["source_path"]:
        return _artifact_fallback_row(
            model_id,
            status="runtime_unavailable_artifact_fallback",
            backend=backend,
            source=source,
            semantic_summary=semantic_summary,
            fallback_artifact_dir=fallback_artifact_dir,
            runtime_error=source.get("reason") or "source unavailable",
        )

    adapter = None
    try:
        adapter = _make_adapter(source["path"], model_format=source["format"], backend=backend)
        sites = build_semantic_sites(
            adapter,
            semantic_map,
            require_distal_site_offsets=True,
        )
        state = adapter.forward_kinematics(adapter.neutral_q())
        paths = discover_paths(adapter, sites)
        site_rows = {
            name: _site_row(adapter, state, site, semantic_map.get(name))
            for name, site in sites.items()
        }
        path_rows = {
            task: _path_row(adapter, state, path, sites, source_rest)
            for task, path in paths.items()
        }
        semantic_summary["runtime_fingerprint"] = adapter.fingerprint
        semantic_summary["fingerprint_matches_runtime"] = semantic_summary.get("model_fingerprint") == adapter.fingerprint
        row = {
            "model_id": model_id,
            "status": "runtime_loaded",
            "backend": backend,
            "source": _source_json(source),
            "semantic_map": semantic_summary,
            "runtime": {
                "nq": adapter.nq,
                "nv": adapter.nv,
                "body_count": len(adapter.body_names),
                "coordinate_count": len(adapter.coordinate_info),
                "loader_provenance": getattr(adapter, "loader_provenance", {}),
            },
            "sites": site_rows,
            "paths": path_rows,
            "symmetry": _symmetry(site_rows, path_rows),
            "classification_flags": _classification_flags(model_id, path_rows),
            "map_change_recommendation": "none",
            "runtime_error": None,
        }
        return row
    except Exception as exc:
        return _artifact_fallback_row(
            model_id,
            status="runtime_failed_artifact_fallback",
            backend=backend,
            source=source,
            semantic_summary=semantic_summary,
            fallback_artifact_dir=fallback_artifact_dir,
            runtime_error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if adapter is not None:
            adapter.close()


def _resolve_audit_source(model_id: str, manifest) -> dict:
    entry = manifest.model_by_id.get(model_id)
    model_format = entry.model_format if entry else ("urdf" if model_id.endswith("_urdf") else "mjcf")
    if model_id == "roboparty_rpo_local":
        path = DEFAULT_RPO_MODEL_PATH
        return _available_source(model_id, model_format, path, resolver="local_workspace")

    snapshot = Path("assets/robot_zoo/snapshots") / model_id / ("model.urdf" if model_format == "urdf" else "model.xml")
    if snapshot.exists():
        return _available_source(model_id, model_format, snapshot, resolver="workspace_snapshot")

    if entry is None:
        return {
            "model_id": model_id,
            "format": model_format,
            "source_status": "source_unavailable",
            "source_path": None,
            "path": None,
            "resolver": "manifest_missing",
            "reason": "model id is not present in the robot zoo manifest",
            "local_file_sha256": None,
        }

    resolved = resolve_robot_source(entry, allow_fetch=False)
    if resolved.available and resolved.path:
        return _available_source(model_id, model_format, resolved.path, resolver=resolved.resolver)
    return {
        "model_id": model_id,
        "format": model_format,
        "source_status": resolved.status,
        "source_path": None,
        "path": None,
        "resolver": resolved.resolver,
        "reason": resolved.reason,
        "local_file_sha256": None,
    }


def _available_source(model_id: str, model_format: str, path: Path, *, resolver: str) -> dict:
    return {
        "model_id": model_id,
        "format": model_format,
        "source_status": "available",
        "source_path": display_path(path),
        "path": path,
        "resolver": resolver,
        "reason": "",
        "local_file_sha256": sha256_file(path),
    }


def _source_json(source: dict) -> dict:
    return {
        "format": source["format"],
        "source_status": source["source_status"],
        "source_path": source["source_path"],
        "resolver": source["resolver"],
        "reason": source.get("reason", ""),
        "local_file_sha256": source.get("local_file_sha256"),
    }


def _make_adapter(path: str | Path, *, model_format: str, backend: str) -> RuntimeRobotModelAdapter:
    if backend == "newton":
        return NewtonRuntimeModelAdapter(path, model_format=model_format)
    if backend == "mujoco":
        return MuJoCoRuntimeModelAdapter(path, model_format=model_format)
    raise ValueError(f"unsupported target geometry audit backend {backend!r}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _semantic_map_entries(payload: dict) -> dict:
    if "semantics" in payload:
        return dict(payload["semantics"])
    if "ik_map" in payload:
        return dict(payload["ik_map"])
    return dict(payload)


def _semantic_map_summary(payload: dict, path: Path) -> dict:
    return {
        "path": display_path(path),
        "exists": path.exists(),
        "model_id": payload.get("model_id"),
        "verification_status": payload.get("verification_status"),
        "model_fingerprint": payload.get("model_fingerprint"),
        "runtime_fingerprint": None,
        "fingerprint_matches_runtime": None,
        "semantic_count": len(_semantic_map_entries(payload)) if payload else 0,
        "auxiliary_semantics": list(payload.get("auxiliary_semantics", [])) if isinstance(payload.get("auxiliary_semantics", []), list) else [],
    }


def _site_row(adapter, state, site, map_entry) -> dict:
    transform = adapter.site_transform(state, site)
    return {
        "semantic_name": site.semantic_name,
        "body": site.body_name,
        "local_position": _array(site.local_position),
        "local_offset_norm": _round(float(np.linalg.norm(site.local_position))),
        "local_rotation_xyzw": _array(site.local_rotation_xyzw),
        "neutral_world_position": _array(transform[:3, 3]),
        "neutral_world_rotation_xyzw": _array(matrix_to_quat_xyzw(transform[:3, :3])),
        "neutral_world_transform": _matrix(transform),
        "source": site.source,
        "confidence": _round(float(site.confidence)),
        "evidence": list(site.evidence),
        "target_frame_convention": _target_frame_convention(map_entry, site),
    }


def _target_frame_convention(map_entry, site) -> str:
    if isinstance(map_entry, dict) and ("site" in map_entry or "model_site" in map_entry):
        return "compiled_model_site_body_local"
    if site.semantic_name in DISTAL_SEMANTICS and float(np.linalg.norm(site.local_position)) > 1e-9:
        return "verified_body_local_offset"
    if float(np.linalg.norm(site.local_position)) > 1e-9:
        return "configured_body_local_offset"
    return "body_origin_frame"


def _path_row(adapter, state, path, sites: dict, source_rest: dict[str, np.ndarray]) -> dict:
    ref_site = sites[path.reference]
    target_site = sites[path.target]
    ref_t = adapter.site_transform(state, ref_site)
    target_t = adapter.site_transform(state, target_site)
    rel_t = relative_transform(ref_t, target_t)
    desired_distance = float(np.linalg.norm(rel_t[:3, 3]))
    chain_length = _chain_length(adapter, state, path.body_path, ref_t[:3, 3], target_t[:3, 3])
    source_distance = _source_distance(path.task, source_rest)
    coordinate_limits = [_coordinate_json(info) for info in adapter.coordinate_limits(path.active_velocity_coordinates)]
    return {
        "task": path.task,
        "reference_semantic": path.reference,
        "target_semantic": path.target,
        "reference_body": path.reference_body,
        "target_body": path.target_body,
        "lca_body": path.lca_body,
        "body_path": path.body_path,
        "reference_branch_bodies": path.reference_branch_bodies,
        "target_branch_bodies": path.target_branch_bodies,
        "active_velocity_coordinates": path.active_velocity_coordinates,
        "active_coordinate_count": len(path.active_velocity_coordinates),
        "coordinate_labels": path.coordinate_labels,
        "joint_types": path.joint_types,
        "limit_source": path.limit_source,
        "coordinate_limits": coordinate_limits,
        "limit_summary": _limit_summary(coordinate_limits),
        "neutral_relative_position": _array(rel_t[:3, 3]),
        "neutral_relative_rotation_xyzw": _array(matrix_to_quat_xyzw(rel_t[:3, :3])),
        "neutral_relative_transform": _matrix(rel_t),
        "desired_distance": _round(desired_distance),
        "chain_length": _round(chain_length),
        "desired_distance_to_chain_length_ratio": _ratio(desired_distance, chain_length),
        "source_distance": _round(source_distance) if source_distance is not None else None,
        "source_to_robot_segment_ratio": _ratio(source_distance, desired_distance) if source_distance is not None else None,
    }


def _coordinate_json(info) -> dict:
    return {
        "index": info.index,
        "label": info.label,
        "joint_name": info.joint_name,
        "joint_type": info.joint_type,
        "qpos_adr": info.qpos_adr,
        "dof_adr": info.dof_adr,
        "limited": info.limited,
        "lower": _finite_or_none(info.lower),
        "upper": _finite_or_none(info.upper),
    }


def _limit_summary(coordinate_limits: list[dict]) -> dict:
    return {
        "coordinate_count": len(coordinate_limits),
        "limited_count": sum(1 for item in coordinate_limits if item["limited"]),
        "unlimited_count": sum(1 for item in coordinate_limits if not item["limited"]),
        "joint_type_counts": dict(sorted(Counter(item["joint_type"] for item in coordinate_limits).items())),
    }


def _chain_length(adapter, state, body_path: list[str], reference_position: np.ndarray, target_position: np.ndarray) -> float:
    if not body_path:
        return float(np.linalg.norm(target_position - reference_position))
    points = [np.asarray(reference_position, dtype=float)]
    for body_name in body_path:
        points.append(adapter.body_transform(state, body_name)[:3, 3])
    points.append(np.asarray(target_position, dtype=float))
    length = 0.0
    for a, b in zip(points, points[1:]):
        length += float(np.linalg.norm(b - a))
    return length


def _source_distance(task: str, source_rest: dict[str, np.ndarray]) -> float | None:
    edge = TASK_TO_SOURCE_EDGE.get(task)
    if not edge:
        return None
    ref_name, target_name = edge
    if ref_name not in source_rest or target_name not in source_rest:
        return None
    return float(np.linalg.norm(source_rest[target_name][:3, 3] - source_rest[ref_name][:3, 3]))


def _symmetry(sites: dict, paths: dict) -> dict:
    def distance(a: str, b: str) -> float | None:
        if a not in sites or b not in sites:
            return None
        pa = np.asarray(sites[a]["neutral_world_position"], dtype=float)
        pb = np.asarray(sites[b]["neutral_world_position"], dtype=float)
        return float(np.linalg.norm(pb - pa))

    left_arm = distance("Chest", "LeftHand")
    right_arm = distance("Chest", "RightHand")
    left_leg = distance("Hips", "LeftFoot")
    right_leg = distance("Hips", "RightFoot")
    return {
        "arm_neutral_distance_left": _round(left_arm) if left_arm is not None else None,
        "arm_neutral_distance_right": _round(right_arm) if right_arm is not None else None,
        "arm_neutral_distance_delta": _round(abs(left_arm - right_arm)) if left_arm is not None and right_arm is not None else None,
        "leg_neutral_distance_left": _round(left_leg) if left_leg is not None else None,
        "leg_neutral_distance_right": _round(right_leg) if right_leg is not None else None,
        "leg_neutral_distance_delta": _round(abs(left_leg - right_leg)) if left_leg is not None and right_leg is not None else None,
        "hand_local_mirror_error": _mirror_error(sites, "LeftHand", "RightHand"),
        "foot_local_mirror_error": _mirror_error(sites, "LeftFoot", "RightFoot"),
        "arm_active_coordinate_delta": _count_delta(paths, "left_hand", "right_hand"),
        "leg_active_coordinate_delta": _count_delta(paths, "left_foot", "right_foot"),
    }


def _mirror_error(sites: dict, left: str, right: str) -> float | None:
    if left not in sites or right not in sites:
        return None
    lpos = np.asarray(sites[left]["local_position"], dtype=float)
    rpos = np.asarray(sites[right]["local_position"], dtype=float)
    mirrored = np.asarray([rpos[0], -rpos[1], rpos[2]], dtype=float)
    return _round(float(np.linalg.norm(lpos - mirrored)))


def _count_delta(paths: dict, left: str, right: str) -> int | None:
    if left not in paths or right not in paths:
        return None
    return int(abs(paths[left]["active_coordinate_count"] - paths[right]["active_coordinate_count"]))


def _classification_flags(model_id: str, paths: dict) -> list[str]:
    flags: list[str] = []
    torso = paths.get("torso")
    if torso:
        if torso["active_coordinate_count"] == 0 and torso["desired_distance"] == 0.0:
            flags.append("fixed_torso_rank_zero_control")
        if torso["active_coordinate_count"] == 1 and torso["joint_types"] == ["revolute"]:
            flags.append("single_revolute_torso_path")
            if model_id == "roboparty_rpo_local":
                flags.append("yaw_only_torso_control")
    if model_id in {"unitree_h1_urdf", "unitree_h1_mjcf", "unitree_g1_urdf", "unitree_g1_mjcf"}:
        flags.append("passing_variant_control")
    if model_id in {"jaxon_urdf", "valkyrie_urdf", "mujoco_humanoid_mjcf"}:
        flags.append("standalone_geometry_probe")
    return flags


def _artifact_fallback_row(
    model_id: str,
    *,
    status: str,
    backend: str,
    source: dict,
    semantic_summary: dict,
    fallback_artifact_dir: Path,
    runtime_error: str,
) -> dict:
    artifact_path = fallback_artifact_dir / f"{model_id}.json"
    artifact = _load_json(artifact_path)
    if not artifact and status == "runtime_failed_artifact_fallback":
        status = "runtime_failed_no_artifact"
    semantic_summary["runtime_fingerprint"] = None
    semantic_summary["fingerprint_matches_runtime"] = None
    sites = _artifact_sites(artifact)
    paths = _artifact_paths(artifact)
    map_recommendation = "source_unavailable_no_runtime_correction"
    if status == "runtime_failed_artifact_fallback":
        map_recommendation = "runtime_failed_no_map_correction"
    return {
        "model_id": model_id,
        "status": status,
        "backend": backend,
        "source": _source_json(source),
        "semantic_map": semantic_summary,
        "runtime": {
            "nq": artifact.get("runtime_adapter", {}).get("nq"),
            "nv": artifact.get("runtime_adapter", {}).get("nv"),
            "body_count": artifact.get("runtime_adapter", {}).get("body_count"),
            "coordinate_count": len(artifact.get("runtime_adapter", {}).get("coordinates", [])),
            "loader_provenance": artifact.get("runtime_adapter", {}).get("loader_provenance", {}),
            "fallback_artifact_path": display_path(artifact_path) if artifact_path.exists() else None,
            "fallback_artifact_status": artifact.get("status"),
        },
        "sites": sites,
        "paths": paths,
        "symmetry": _artifact_symmetry(artifact, sites, paths),
        "classification_flags": _classification_flags(model_id, paths),
        "map_change_recommendation": map_recommendation,
        "runtime_error": runtime_error,
    }


def _artifact_sites(artifact: dict) -> dict:
    out = {}
    for semantic_name, site in artifact.get("semantic_sites", {}).items():
        local_position = site.get("local_position", [0.0, 0.0, 0.0])
        out[semantic_name] = {
            "semantic_name": semantic_name,
            "body": site.get("body_name"),
            "local_position": local_position,
            "local_offset_norm": _round(float(np.linalg.norm(np.asarray(local_position, dtype=float)))),
            "local_rotation_xyzw": site.get("local_rotation_xyzw"),
            "neutral_world_position": None,
            "neutral_world_rotation_xyzw": None,
            "neutral_world_transform": None,
            "source": site.get("source"),
            "confidence": site.get("confidence"),
            "evidence": site.get("evidence", []),
            "target_frame_convention": (
                "verified_body_local_offset"
                if semantic_name in DISTAL_SEMANTICS and float(np.linalg.norm(np.asarray(local_position, dtype=float))) > 1e-9
                else "body_origin_frame"
            ),
        }
    return out


def _artifact_paths(artifact: dict) -> dict:
    out = {}
    for task, chain in artifact.get("chains", {}).items():
        active = chain.get("active_velocity_coordinates", [])
        out[task] = {
            "task": task,
            "reference_semantic": chain.get("reference"),
            "target_semantic": chain.get("target"),
            "reference_body": chain.get("reference_body"),
            "target_body": chain.get("target_body"),
            "lca_body": chain.get("lca_body"),
            "body_path": chain.get("body_path", []),
            "reference_branch_bodies": chain.get("reference_branch_bodies", []),
            "target_branch_bodies": chain.get("target_branch_bodies", []),
            "active_velocity_coordinates": active,
            "active_coordinate_count": len(active),
            "coordinate_labels": chain.get("coordinate_labels", []),
            "joint_types": chain.get("joint_types", []),
            "limit_source": chain.get("limit_source"),
            "coordinate_limits": [],
            "limit_summary": {
                "coordinate_count": len(active),
                "limited_count": None,
                "unlimited_count": None,
                "joint_type_counts": dict(sorted(Counter(chain.get("joint_types", [])).items())),
            },
            "neutral_relative_position": None,
            "neutral_relative_rotation_xyzw": None,
            "neutral_relative_transform": None,
            "desired_distance": None,
            "chain_length": None,
            "desired_distance_to_chain_length_ratio": None,
            "source_distance": None,
            "source_to_robot_segment_ratio": None,
        }
    return out


def _artifact_symmetry(artifact: dict, sites: dict, paths: dict) -> dict:
    symmetry = artifact.get("rest_calibration", {}).get("bilateral_symmetry", {})
    return {
        "arm_neutral_distance_left": None,
        "arm_neutral_distance_right": None,
        "arm_neutral_distance_delta": symmetry.get("arm_length_abs_delta"),
        "leg_neutral_distance_left": None,
        "leg_neutral_distance_right": None,
        "leg_neutral_distance_delta": symmetry.get("leg_length_abs_delta"),
        "hand_local_mirror_error": _mirror_error(sites, "LeftHand", "RightHand") if sites else None,
        "foot_local_mirror_error": _mirror_error(sites, "LeftFoot", "RightFoot") if sites else None,
        "arm_active_coordinate_delta": _count_delta(paths, "left_hand", "right_hand"),
        "leg_active_coordinate_delta": _count_delta(paths, "left_foot", "right_foot"),
    }


def _differentials(rows: list[dict]) -> dict:
    row_by_id = {row["model_id"]: row for row in rows}
    out = {}
    for pair_name, (left_id, right_id) in DIFFERENTIAL_PAIRS.items():
        left = row_by_id.get(left_id)
        right = row_by_id.get(right_id)
        if not left or not right:
            continue
        compared = bool(left.get("sites")) and bool(right.get("sites"))
        out[pair_name] = {
            "left_model_id": left_id,
            "right_model_id": right_id,
            "left_status": left["status"],
            "right_status": right["status"],
            "pair_status": "compared" if compared else "not_comparable",
            "site_differences": _site_differences(left.get("sites", {}), right.get("sites", {})) if compared else {},
            "path_differences": _path_differences(left.get("paths", {}), right.get("paths", {})) if compared else {},
            "fingerprint_match_states": {
                left_id: left["semantic_map"].get("fingerprint_matches_runtime"),
                right_id: right["semantic_map"].get("fingerprint_matches_runtime"),
            },
        }
    return out


def _site_differences(left_sites: dict, right_sites: dict) -> dict:
    out = {}
    for semantic_name in sorted(set(left_sites) & set(right_sites)):
        left = left_sites[semantic_name]
        right = right_sites[semantic_name]
        local_delta = None
        if left.get("local_position") is not None and right.get("local_position") is not None:
            local_delta = _round(float(np.linalg.norm(np.asarray(left["local_position"], dtype=float) - np.asarray(right["local_position"], dtype=float))))
        out[semantic_name] = {
            "body_changed": left.get("body") != right.get("body"),
            "left_body": left.get("body"),
            "right_body": right.get("body"),
            "local_position_delta": local_delta,
            "source_changed": left.get("source") != right.get("source"),
            "target_frame_convention_changed": left.get("target_frame_convention") != right.get("target_frame_convention"),
        }
    return out


def _path_differences(left_paths: dict, right_paths: dict) -> dict:
    out = {}
    for task in sorted(set(left_paths) & set(right_paths)):
        left = left_paths[task]
        right = right_paths[task]
        out[task] = {
            "reference_body_changed": left.get("reference_body") != right.get("reference_body"),
            "target_body_changed": left.get("target_body") != right.get("target_body"),
            "lca_body_changed": left.get("lca_body") != right.get("lca_body"),
            "active_coordinate_count_delta": (
                left.get("active_coordinate_count", 0) - right.get("active_coordinate_count", 0)
            ),
            "joint_types_changed": left.get("joint_types") != right.get("joint_types"),
            "body_path_changed": left.get("body_path") != right.get("body_path"),
            "desired_distance_delta": _nullable_delta(left.get("desired_distance"), right.get("desired_distance")),
            "chain_length_delta": _nullable_delta(left.get("chain_length"), right.get("chain_length")),
            "source_to_robot_segment_ratio_delta": _nullable_delta(
                left.get("source_to_robot_segment_ratio"),
                right.get("source_to_robot_segment_ratio"),
            ),
        }
    return out


def _nullable_delta(left, right):
    if left is None or right is None:
        return None
    return _round(float(left) - float(right))


def _array(value) -> list[float]:
    return [_round(float(v)) for v in np.asarray(value, dtype=float).reshape(-1)]


def _matrix(value) -> list[list[float]]:
    arr = np.asarray(value, dtype=float)
    return [[_round(float(v)) for v in row] for row in arr]


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), 12)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) <= 1e-12:
        return None
    return _round(float(numerator) / float(denominator))


def _finite_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return _round(float(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/retargeting_v3_step2_capability/target_geometry_matrix.json",
        help="path for the JSON matrix artifact",
    )
    parser.add_argument("--backend", default="newton", choices=("newton", "mujoco"))
    parser.add_argument("--model-id", action="append", dest="model_ids", help="model id to audit; may be repeated")
    args = parser.parse_args(argv)
    write_target_geometry_matrix(
        args.output,
        model_ids=tuple(args.model_ids) if args.model_ids else DEFAULT_AUDIT_MODEL_IDS,
        backend=args.backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
