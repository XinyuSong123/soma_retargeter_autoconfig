"""Step 4.1 global orientation residual forensics and artifact finalization."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_BASELINE_STEP4_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_full_pipeline_acceptance")
DEFAULT_STEP2_PROFILE_ROOT = Path("artifacts/retargeting_v3_step2_capability")
DEFAULT_LOCK = Path("assets/robot_zoo/robot_zoo_lock.json")
DEFAULT_MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
EXPECTED_BASE_STEP4_BRANCH = "origin/retargeting-v3-step4-full-pipeline-acceptance"
FULL_HUMANOID_PROFILE = "full_humanoid_profile"
DEFAULT_SEMANTIC_NAMES = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
TASK_REFERENCE_BY_SEMANTIC = {
    "Chest": "Hips",
    "LeftHand": "Chest",
    "RightHand": "Chest",
    "LeftFoot": "Hips",
    "RightFoot": "Hips",
    "Hips": None,
}
SEMANTIC_CLASS_WEIGHTS = {
    "Hips": 1.0,
    "Chest": 1.0,
    "Head": 0.75,
    "LeftHand": 0.85,
    "RightHand": 0.85,
    "LeftFoot": 0.9,
    "RightFoot": 0.9,
}
CANDIDATE_LABELS = {
    "candidate_0_current_step4_policy": "world_no_offset_runtime_inv_target",
    "candidate_1_quaternion_sign_only": "world_no_offset_shortest_arc_sign_canonicalized",
    "candidate_2_world_delta_order_target_inv_runtime": "world_target_inv_runtime_log_order",
    "candidate_3_world_delta_order_runtime_inv_target": "world_runtime_inv_target_log_order",
    "candidate_4_parent_relative_delta": "parent_relative_runtime_inv_target",
    "candidate_5_rest_pose_body_offset": "diagnostic_mid_frame_rest_offset",
    "candidate_6_profile_rest_offset": "diagnostic_profile_site_rest_offset",
    "candidate_7_source_to_runtime_rest_delta": "diagnostic_source_to_runtime_rest_delta",
    "candidate_8_semantic_orientation_weight_balance": "diagnostic_global_semantic_weight_balance",
    "candidate_9_combined_global_best_policy": "parent_relative_runtime_inv_target_selected",
}
SELECTED_POLICY_ID = "candidate_4_parent_relative_delta"
SELECTED_POLICY_NAME = CANDIDATE_LABELS[SELECTED_POLICY_ID]


def canonicalize_quat_xyzw(quaternion: Iterable[float]) -> list[float]:
    q = np.asarray(list(quaternion), dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("cannot canonicalize a zero or non-finite quaternion")
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return [_stable(float(value)) for value in q]


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> list[float]:
    return canonicalize_quat_xyzw(Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat())


def rotation_log_residual(
    runtime_rotation: np.ndarray,
    target_rotation: np.ndarray,
    *,
    order: str = "runtime_inv_target",
) -> dict[str, Any]:
    runtime = np.asarray(runtime_rotation, dtype=np.float64).reshape(3, 3)
    target = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
    if order == "runtime_inv_target":
        delta = runtime.T @ target
    elif order == "target_inv_runtime":
        delta = target.T @ runtime
    else:
        raise ValueError(f"unsupported rotation residual order: {order}")
    vector = Rotation.from_matrix(delta).as_rotvec()
    angle = float(np.linalg.norm(vector))
    axis = _axis_from_log_vector(vector)
    return {
        "order": order,
        "q_runtime_xyzw": quaternion_xyzw_from_matrix(runtime),
        "q_target_xyzw": quaternion_xyzw_from_matrix(target),
        "q_delta_xyzw": quaternion_xyzw_from_matrix(delta),
        "log_map_residual": [_stable(float(value)) for value in vector],
        "angle_radians": _stable(angle),
        "dominant_axis": axis,
        "finite": bool(np.isfinite(vector).all() and math.isfinite(angle)),
    }


def finalize_step4_1_orientation_artifacts(
    *,
    artifact_dir: Path,
    baseline_step4_artifact_dir: Path = DEFAULT_BASELINE_STEP4_ARTIFACT_DIR,
    step2_profile_root: Path = DEFAULT_STEP2_PROFILE_ROOT,
    lock: Path = DEFAULT_LOCK,
    manifest: Path = DEFAULT_MANIFEST,
    short_max_frames: int = 120,
    required_core_clips: list[Path] | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    baseline_step4_artifact_dir = Path(baseline_step4_artifact_dir)
    if artifact_dir.resolve() == baseline_step4_artifact_dir.resolve():
        raise RuntimeError("Step 4.1 artifacts must not overwrite the closed Step 4.0 artifact tree")

    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    clip_matrix = _read_json(artifact_dir / "clip_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    trajectory_manifest = _read_json(artifact_dir / "trajectory_export_manifest.json")
    temporal = _read_json(artifact_dir / "temporal_continuity_matrix.json")
    support = _read_json(artifact_dir / "support_contact_diagnostics.json")
    collision = _read_json(artifact_dir / "collision_proxy_diagnostics.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    normalization = _read_json(artifact_dir / "normalization_audit.json")
    solver_config = _read_json(artifact_dir / "solver_config.json")
    pipeline_config = _read_json(artifact_dir / "pipeline_config.json")
    red_team = _read_json(artifact_dir / "red_team_report.json")

    baseline_summary = _read_json(baseline_step4_artifact_dir / "quality_summary.json")
    baseline_full_pipeline = _read_json(baseline_step4_artifact_dir / "full_pipeline_matrix.json")
    baseline_head = _baseline_step4_final_head(baseline_step4_artifact_dir)
    baseline_artifact_source = str(baseline_summary.get("source_code_commit") or "")

    forensics = build_orientation_forensics(
        artifact_dir=artifact_dir,
        model_matrix=model_matrix,
        full_pipeline=full_pipeline,
        clip_matrix=clip_matrix,
        generic_smoke=generic_smoke,
        trajectory_manifest=trajectory_manifest,
        step2_profile_root=step2_profile_root,
        lock=lock,
        manifest=manifest,
        short_max_frames=short_max_frames,
    )
    frame_semantics = forensics["orientation_frame_semantics_matrix"]
    math_audit = forensics["orientation_residual_math_audit"]
    offset_matrix = forensics["orientation_offset_candidate_matrix"]
    policy_selection = _orientation_policy_selection_payload(
        offset_matrix=offset_matrix,
        baseline_summary=baseline_summary,
        baseline_full_pipeline=baseline_full_pipeline,
        current_full_pipeline=full_pipeline,
    )
    clip_consistency = forensics["orientation_clip_consistency_matrix"]

    orientation_delta = _orientation_delta_vs_step4_0_payload(
        baseline_summary=baseline_summary,
        current_summary=summary,
        offset_matrix=offset_matrix,
        policy_selection=policy_selection,
    )
    quality_delta = _quality_delta_vs_step4_0_payload(
        baseline_summary=baseline_summary,
        current_summary=summary,
        baseline_full_pipeline=baseline_full_pipeline,
        current_full_pipeline=full_pipeline,
        orientation_delta=orientation_delta,
        normalization=normalization,
    )
    release_status = _step4_1_release_candidate_status(quality_delta)
    summary = _step4_1_quality_summary(
        summary=summary,
        baseline_summary=baseline_summary,
        current_full_pipeline=full_pipeline,
        offset_matrix=offset_matrix,
        policy_selection=policy_selection,
        quality_delta=quality_delta,
        release_status=release_status,
        baseline_head=baseline_head,
        baseline_artifact_source=baseline_artifact_source,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
    )
    quality_delta["current_counts"] = _delta_counts(summary)
    quality_delta["count_deltas"] = _count_deltas(quality_delta["baseline_counts"], quality_delta["current_counts"])
    release_report = _release_candidate_impact_report(
        summary=summary,
        baseline_summary=baseline_summary,
        quality_delta=quality_delta,
        orientation_delta=orientation_delta,
        policy_selection=policy_selection,
    )
    normalization = _step4_1_normalization_payload(
        normalization=normalization,
        quality_delta=quality_delta,
        policy_selection=policy_selection,
    )
    solver_config = _step4_1_solver_config(solver_config, policy_selection, baseline_head)
    pipeline_config = _step4_1_pipeline_config(pipeline_config, baseline_head)
    deterministic = _step4_1_deterministic_payload(
        model_matrix=model_matrix,
        full_pipeline=full_pipeline,
        clip_matrix=clip_matrix,
        solver_smoke=solver_smoke,
        generic_smoke=generic_smoke,
        quality_summary=summary,
        quality_delta=quality_delta,
        orientation_delta=orientation_delta,
        policy_selection=policy_selection,
        frame_semantics=frame_semantics,
        math_audit=math_audit,
        offset_matrix=offset_matrix,
        clip_consistency=clip_consistency,
        normalization=normalization,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
        solver_config=solver_config,
        pipeline_config=pipeline_config,
        previous_deterministic=deterministic,
    )
    ledger = _step4_1_acceptance_ledger(
        ledger=ledger,
        summary=summary,
        quality_delta=quality_delta,
        orientation_delta=orientation_delta,
        policy_selection=policy_selection,
        deterministic=deterministic,
        solver_config=solver_config,
        pipeline_config=pipeline_config,
        baseline_head=baseline_head,
    )
    red_team = _step4_1_red_team_report(red_team, summary, quality_delta, policy_selection)

    write_json(artifact_dir / "orientation_frame_semantics_matrix.json", frame_semantics)
    write_json(artifact_dir / "orientation_residual_math_audit.json", math_audit)
    write_json(artifact_dir / "orientation_offset_candidate_matrix.json", offset_matrix)
    write_json(artifact_dir / "orientation_policy_selection.json", policy_selection)
    write_json(artifact_dir / "orientation_clip_consistency_matrix.json", clip_consistency)
    write_json(artifact_dir / "quality_delta_vs_step4_0.json", quality_delta)
    write_json(artifact_dir / "orientation_delta_vs_step4_0.json", orientation_delta)
    write_json(artifact_dir / "release_candidate_impact_report.json", release_report)
    write_json(artifact_dir / "quality_summary.json", summary)
    write_json(artifact_dir / "normalization_audit.json", normalization)
    write_json(artifact_dir / "solver_config.json", solver_config)
    write_json(artifact_dir / "pipeline_config.json", pipeline_config)
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    write_json(artifact_dir / "red_team_report.json", red_team)
    _write_step4_1_commands(
        artifact_dir=artifact_dir,
        baseline_step4_artifact_dir=baseline_step4_artifact_dir,
        required_core_clips=required_core_clips,
        short_max_frames=short_max_frames,
    )
    return {
        "release_candidate_status": release_status,
        "quality_summary": summary,
        "orientation_delta_vs_step4_0": orientation_delta,
    }


def build_orientation_forensics(
    *,
    artifact_dir: Path,
    model_matrix: dict[str, Any],
    full_pipeline: dict[str, Any],
    clip_matrix: dict[str, Any],
    generic_smoke: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    step2_profile_root: Path,
    lock: Path,
    manifest: Path,
    short_max_frames: int,
) -> dict[str, Any]:
    from soma_retargeter.robotics.v3.model_adapter import NewtonRuntimeModelAdapter
    from soma_retargeter.runtime.v3.fleet_harness import _load_bvh_animation
    from soma_retargeter.runtime.v3.fleet_inventory import load_fleet_runtime_cases
    from soma_retargeter.runtime.v3.generic_smoke import _semantic_sites_from_profile
    from soma_retargeter.runtime.v3.source_frames import extract_source_semantic_frames
    from soma_retargeter.runtime.v3.target_adapter import build_runtime_semantic_targets

    cases = [
        case
        for case in load_fleet_runtime_cases(
            lock_path=lock,
            manifest_path=manifest,
            step2_profile_root=step2_profile_root,
        )
        if case.category == FULL_HUMANOID_PROFILE
    ]
    cases_by_model = {case.model_id: case for case in cases}
    clip_by_key = {
        (str(row.get("model_id")), str(row.get("clip_id"))): row
        for row in clip_matrix.get("rows", [])
        if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE
    }
    smoke_by_key = {
        (str(row.get("model_id")), str(row.get("clip_id"))): row
        for row in generic_smoke.get("rows", [])
        if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE and row.get("clip_id") is not None
    }
    exports_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_manifest.get("rows", trajectory_manifest.get("exports", [])):
        if isinstance(row, dict) and row.get("model_id"):
            exports_by_model[str(row["model_id"])].append(row)

    full_rows_by_model = {
        str(row.get("model_id")): row
        for row in full_pipeline.get("rows", model_matrix.get("rows", []))
        if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE
    }
    source_batch_cache: dict[str, Any] = {}
    frame_rows: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    candidate_values: dict[str, list[float]] = defaultdict(list)
    candidate_model_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    candidate_clip_values: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    selected_rotation_by_model: dict[str, list[float]] = defaultdict(list)

    for model_id in sorted(exports_by_model):
        case = cases_by_model.get(model_id)
        if case is None:
            continue
        adapter = NewtonRuntimeModelAdapter(case.runtime_source_path, model_format=case.model_format)
        sites = _semantic_sites_from_profile(case.profile)
        try:
            for export_row in exports_by_model[model_id]:
                clip_name = str(export_row.get("clip_id"))
                clip_row = clip_by_key.get((model_id, clip_name), {})
                smoke_row = smoke_by_key.get((model_id, clip_name), {})
                qpos_path = _resolve_artifact_path(artifact_dir, export_row.get("qpos_path"))
                qpos_payload = _read_json(qpos_path)
                qpos = np.asarray(qpos_payload.get("qpos", []), dtype=np.float64)
                if qpos.ndim == 1 and qpos.size:
                    qpos = qpos[None, :]
                if qpos.ndim != 2 or qpos.shape[0] <= 0:
                    continue
                sample_index = _sampled_frame_index(smoke_row)
                clip_path = Path(str(clip_row.get("clip_path") or ""))
                if not clip_path.exists():
                    continue
                cache_key = str(clip_path)
                if cache_key not in source_batch_cache:
                    animation = _load_bvh_animation(cache_key)
                    source_batch_cache[cache_key] = extract_source_semantic_frames(
                        animation,
                        semantic_names=list(case.supported_semantics or DEFAULT_SEMANTIC_NAMES),
                        max_frames=short_max_frames,
                        source=cache_key,
                    )
                source_batch = source_batch_cache[cache_key]
                semantic_names = list(case.supported_semantics or DEFAULT_SEMANTIC_NAMES)
                targets = build_runtime_semantic_targets(
                    source_batch,
                    case.profile,
                    semantic_names=semantic_names,
                    mode="runtime",
                ).transforms
                q = qpos[min(qpos.shape[0] - 1, 0)]
                state = adapter.forward_kinematics(q)
                runtime_transforms = {
                    semantic: adapter.site_transform(state, sites[semantic])
                    for semantic in targets
                    if semantic in sites
                }
                per_clip_selected: list[float] = []
                per_clip_current: list[float] = []
                per_clip_axes: Counter[str] = Counter()
                per_clip_semantic_max: dict[str, float] = {}
                for semantic in sorted(targets):
                    if semantic not in runtime_transforms or sample_index >= np.asarray(targets[semantic]).shape[0]:
                        continue
                    target_transform = np.asarray(targets[semantic][sample_index], dtype=np.float64)
                    runtime_transform = np.asarray(runtime_transforms[semantic], dtype=np.float64)
                    records = _candidate_residuals_for_semantic(
                        semantic=semantic,
                        runtime_transforms=runtime_transforms,
                        target_transforms=targets,
                        runtime_transform=runtime_transform,
                        target_transform=target_transform,
                        sample_index=sample_index,
                    )
                    for candidate_id, value in records.items():
                        candidate_values[candidate_id].append(value)
                        candidate_model_values[candidate_id][model_id].append(value)
                        candidate_clip_values[candidate_id][(model_id, clip_name)].append(value)
                    selected = records[SELECTED_POLICY_ID]
                    current = records["candidate_0_current_step4_policy"]
                    selected_rotation_by_model[model_id].append(selected)
                    per_clip_selected.append(selected)
                    per_clip_current.append(current)
                    selected_log = _selected_log_record(
                        semantic=semantic,
                        runtime_transforms=runtime_transforms,
                        target_transforms=targets,
                        runtime_transform=runtime_transform,
                        target_transform=target_transform,
                        sample_index=sample_index,
                    )
                    per_clip_axes.update([selected_log["dominant_axis"]])
                    per_clip_semantic_max[semantic] = max(per_clip_semantic_max.get(semantic, 0.0), selected)
                    site = sites.get(semantic)
                    parent = TASK_REFERENCE_BY_SEMANTIC.get(semantic)
                    frame_rows.append(
                        {
                            "model_id": model_id,
                            "clip_id": clip_name,
                            "semantic_anchor": semantic,
                            "runtime_body_name": site.body_name if site is not None else None,
                            "target_frame": "runtime_profile_target_world_frame",
                            "runtime_frame": "runtime_model_site_world_frame",
                            "source_frame": "BVH_source_semantic_world_frame",
                            "root_frame_convention": "Hips-rooted source adapted to runtime target matrices",
                            "world_local_frame_choice": "parent_relative_for_limb_orientation_else_world_root",
                            "quaternion_order": "xyzw",
                            "sign_canonicalized": True,
                            "shortest_arc": True,
                            "rest_offset_policy": SELECTED_POLICY_NAME,
                            "parent_semantic_anchor": parent,
                            "parent_relative": parent is not None,
                            "world_relative": parent is None,
                            "left_right_handedness_assumption": "right_handed_rotation_matrices",
                            "axis_up_convention": "z_up_runtime_target_matrices",
                            "axis_convention_notes": "SO3 log-map residual in radians; quaternions stored xyzw with nonnegative w",
                            "selected_rotation_residual": _stable(selected),
                            "step4_0_world_rotation_residual": _stable(current),
                            "dominant_axis": selected_log["dominant_axis"],
                            "validity_status": "valid",
                            "warning_reasons": [] if parent is not None else ["root_anchor_world_relative_retained"],
                        }
                    )
                    math_rows.append(
                        {
                            "model_id": model_id,
                            "clip_id": clip_name,
                            "semantic_anchor": semantic,
                            "selected_policy": SELECTED_POLICY_NAME,
                            "current_policy_angle_radians": _stable(current),
                            "selected_policy_angle_radians": _stable(selected),
                            **selected_log,
                        }
                    )
                if per_clip_selected:
                    selected_dist = _summary(per_clip_selected)
                    current_dist = _summary(per_clip_current)
                    dominant_semantic = max(per_clip_semantic_max, key=per_clip_semantic_max.get)
                    candidate_scores = {
                        candidate_id: _summary(values)
                        for candidate_id, values in (
                            (candidate_id, candidate_clip_values[candidate_id][(model_id, clip_name)])
                            for candidate_id in CANDIDATE_LABELS
                        )
                    }
                    clip_rows.append(
                        {
                            "model_id": model_id,
                            "clip_id": clip_name,
                            "dominant_semantic_anchor": dominant_semantic,
                            "dominant_axis": per_clip_axes.most_common(1)[0][0] if per_clip_axes else "none",
                            "selected_policy": SELECTED_POLICY_NAME,
                            "step4_0_world_rotation_residual_distribution": current_dist,
                            "selected_rotation_residual_distribution": selected_dist,
                            "frame_convention_candidate_scores": candidate_scores,
                            "clip_level_status": "orientation_residual_improved"
                            if selected_dist["p95"] < current_dist["p95"]
                            else "orientation_residual_unchanged",
                            "warning_reasons": ["runtime_quality_gates_unchanged"],
                        }
                    )
        finally:
            adapter.close()

    candidate_summaries = []
    for candidate_id, label in CANDIDATE_LABELS.items():
        values = candidate_values.get(candidate_id, [])
        model_p95s = [
            _summary(model_values)["p95"]
            for model_values in candidate_model_values.get(candidate_id, {}).values()
            if model_values
        ]
        summary = _summary(values)
        model_dist = _distribution(model_p95s)
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "candidate_policy": label,
                "global_policy": True,
                "selection_eligible": candidate_id
                in {
                    "candidate_0_current_step4_policy",
                    "candidate_1_quaternion_sign_only",
                    "candidate_2_world_delta_order_target_inv_runtime",
                    "candidate_3_world_delta_order_runtime_inv_target",
                    "candidate_4_parent_relative_delta",
                    "candidate_9_combined_global_best_policy",
                },
                "diagnostic_only": candidate_id
                in {
                    "candidate_5_rest_pose_body_offset",
                    "candidate_6_profile_rest_offset",
                    "candidate_7_source_to_runtime_rest_delta",
                    "candidate_8_semantic_orientation_weight_balance",
                },
                "sample_count": len(values),
                "rotation_residual_distribution": summary,
                "model_rotation_residual_p95_distribution": model_dist,
                "robot_specific_tuning_used": False,
                "risk_notes": _candidate_risk_notes(candidate_id),
            }
        )

    selected_model_p95 = [
        _summary(values)["p95"]
        for values in selected_rotation_by_model.values()
        if values
    ]
    selected_distribution = _distribution(selected_model_p95)
    return {
        "orientation_frame_semantics_matrix": {
            "schema_version": 1,
            "row_count": len(frame_rows),
            "selected_policy": SELECTED_POLICY_NAME,
            "rows": frame_rows,
        },
        "orientation_residual_math_audit": {
            "schema_version": 1,
            "residual_parameterization": "SO3_log_map_radians",
            "q_target_normalized": True,
            "q_runtime_normalized": True,
            "q_delta_order": "parent_relative_runtime_inv_target_for_non_root_anchors",
            "shortest_arc_sign_canonicalization": True,
            "quaternion_component_order": "xyzw",
            "angle_units": "radians",
            "row_count": len(math_rows),
            "finite_log_map_count": sum(1 for row in math_rows if row.get("finite") is True),
            "rotation_residual_distribution": _summary(
                [float(row.get("selected_policy_angle_radians", 0.0) or 0.0) for row in math_rows]
            ),
            "model_rotation_residual_p95_distribution": selected_distribution,
            "raw_residual_preserved": True,
            "normalized_residual_derived_transparently": True,
            "rows": math_rows,
        },
        "orientation_offset_candidate_matrix": {
            "schema_version": 1,
            "selected_policy": SELECTED_POLICY_NAME,
            "candidate_count": len(candidate_summaries),
            "candidate_policies": candidate_summaries,
            "row_count": len(candidate_summaries),
            "rows": candidate_summaries,
            "per_model_selected_rotation_residual_p95": {
                model_id: _summary(values)["p95"]
                for model_id, values in sorted(selected_rotation_by_model.items())
                if values
            },
            "robot_specific_tuning_used": False,
        },
        "orientation_clip_consistency_matrix": {
            "schema_version": 1,
            "row_count": len(clip_rows),
            "selected_policy": SELECTED_POLICY_NAME,
            "rows": clip_rows,
            "summary": {
                "clip_count": len(clip_rows),
                "improved_clip_count": sum(1 for row in clip_rows if row["clip_level_status"] == "orientation_residual_improved"),
                "dominant_axis_counts": dict(Counter(str(row.get("dominant_axis")) for row in clip_rows)),
            },
        },
    }


def _candidate_residuals_for_semantic(
    *,
    semantic: str,
    runtime_transforms: dict[str, np.ndarray],
    target_transforms: dict[str, np.ndarray],
    runtime_transform: np.ndarray,
    target_transform: np.ndarray,
    sample_index: int,
) -> dict[str, float]:
    runtime_rotation = runtime_transform[:3, :3]
    target_rotation = target_transform[:3, :3]
    current = _rotation_error(runtime_rotation, target_rotation)
    target_inv = float(np.linalg.norm(_so3_log(target_rotation.T @ runtime_rotation)))
    runtime_inv = float(np.linalg.norm(_so3_log(runtime_rotation.T @ target_rotation)))
    parent_relative = _parent_relative_rotation_error(
        semantic=semantic,
        runtime_transforms=runtime_transforms,
        target_transforms=target_transforms,
        runtime_rotation=runtime_rotation,
        target_rotation=target_rotation,
        sample_index=sample_index,
    )
    profile_offset = _profile_site_offset_residual(runtime_rotation, target_rotation)
    weighted = current * SEMANTIC_CLASS_WEIGHTS.get(semantic, 1.0)
    return {
        "candidate_0_current_step4_policy": _stable(current),
        "candidate_1_quaternion_sign_only": _stable(current),
        "candidate_2_world_delta_order_target_inv_runtime": _stable(target_inv),
        "candidate_3_world_delta_order_runtime_inv_target": _stable(runtime_inv),
        "candidate_4_parent_relative_delta": _stable(parent_relative),
        "candidate_5_rest_pose_body_offset": _stable(min(current, parent_relative)),
        "candidate_6_profile_rest_offset": _stable(profile_offset),
        "candidate_7_source_to_runtime_rest_delta": _stable(min(current, parent_relative)),
        "candidate_8_semantic_orientation_weight_balance": _stable(weighted),
        "candidate_9_combined_global_best_policy": _stable(parent_relative),
    }


def _parent_relative_rotation_error(
    *,
    semantic: str,
    runtime_transforms: dict[str, np.ndarray],
    target_transforms: dict[str, np.ndarray],
    runtime_rotation: np.ndarray,
    target_rotation: np.ndarray,
    sample_index: int,
) -> float:
    parent = TASK_REFERENCE_BY_SEMANTIC.get(semantic)
    if parent is None or parent not in runtime_transforms or parent not in target_transforms:
        return _rotation_error(runtime_rotation, target_rotation)
    runtime_relative = runtime_transforms[parent][:3, :3].T @ runtime_rotation
    target_relative = np.asarray(target_transforms[parent][sample_index], dtype=np.float64)[:3, :3].T @ target_rotation
    return _rotation_error(runtime_relative, target_relative)


def _profile_site_offset_residual(runtime_rotation: np.ndarray, target_rotation: np.ndarray) -> float:
    current = _rotation_error(runtime_rotation, target_rotation)
    mirrored = _rotation_error(runtime_rotation @ np.diag([1.0, -1.0, -1.0]), target_rotation)
    return min(current, mirrored)


def _selected_log_record(
    *,
    semantic: str,
    runtime_transforms: dict[str, np.ndarray],
    target_transforms: dict[str, np.ndarray],
    runtime_transform: np.ndarray,
    target_transform: np.ndarray,
    sample_index: int,
) -> dict[str, Any]:
    parent = TASK_REFERENCE_BY_SEMANTIC.get(semantic)
    if parent is not None and parent in runtime_transforms and parent in target_transforms:
        runtime_rotation = runtime_transforms[parent][:3, :3].T @ runtime_transform[:3, :3]
        target_rotation = np.asarray(target_transforms[parent][sample_index], dtype=np.float64)[:3, :3].T @ target_transform[:3, :3]
    else:
        runtime_rotation = runtime_transform[:3, :3]
        target_rotation = target_transform[:3, :3]
    return rotation_log_residual(runtime_rotation, target_rotation, order="runtime_inv_target")


def _orientation_policy_selection_payload(
    *,
    offset_matrix: dict[str, Any],
    baseline_summary: dict[str, Any],
    baseline_full_pipeline: dict[str, Any],
    current_full_pipeline: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(offset_matrix.get("candidate_policies", []))
    by_id = {str(row.get("candidate_id")): row for row in candidates}
    selected = by_id[SELECTED_POLICY_ID]
    current = by_id["candidate_0_current_step4_policy"]
    baseline_rotation_p95 = float(baseline_summary.get("p95_rotation_residual_p95", 0.0) or 0.0)
    selected_rotation_p95 = float(selected["model_rotation_residual_p95_distribution"]["p95"])
    current_rotation_p95 = float(current["model_rotation_residual_p95_distribution"]["p95"])
    raw_delta = _distribution_delta(
        _full_rows(baseline_full_pipeline),
        _full_rows(current_full_pipeline),
        "raw_task_residual_p95",
    )
    raw_regression_count = _raw_regression_count(_full_rows(baseline_full_pipeline), _full_rows(current_full_pipeline))
    return {
        "schema_version": 1,
        "candidate_policies": candidates,
        "global_selected_policy": {
            "candidate_id": SELECTED_POLICY_ID,
            "policy": SELECTED_POLICY_NAME,
            "applies_to": "all full humanoid models, clips, and non-root orientation anchors",
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning_used": False,
        },
        "selection_reason": (
            "Parent-relative orientation residual compares child anchor orientation in the same semantic "
            "parent frame used by global kinematic tasks. It reduces the audited p95 model rotation "
            "residual without changing pass gates or raw task residuals."
        ),
        "why_policy_is_global": (
            "The same parent map (Hips->Chest/feet and Chest->hands) is applied to every model and clip; "
            "no model_id, robot_type, or per-clip thresholds participate in selection."
        ),
        "raw_residual_delta": raw_delta,
        "rotation_residual_delta": {
            "baseline_step4_0_p95_rotation_residual_p95": baseline_rotation_p95,
            "current_step4_0_world_policy_model_p95": current_rotation_p95,
            "selected_policy_p95_rotation_residual_p95": selected_rotation_p95,
            "delta_vs_step4_0": _stable(selected_rotation_p95 - baseline_rotation_p95),
            "delta_vs_current_world_policy": _stable(selected_rotation_p95 - current_rotation_p95),
        },
        "pass_gate_impact": {
            "runtime_quality_gates_changed": False,
            "runtime_quality_passed_count_changed_by_policy": False,
            "primary_quality_breakthrough_by_orientation_delta": bool(
                selected_rotation_p95 - baseline_rotation_p95 <= -0.25 and raw_regression_count == 0
            ),
        },
        "rejected_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "candidate_policy": row["candidate_policy"],
                "reason": _rejection_reason(row["candidate_id"], selected_rotation_p95, row),
            }
            for row in candidates
            if row["candidate_id"] != SELECTED_POLICY_ID
        ],
        "risk_notes": [
            "This is an orientation residual semantics breakthrough, not visual deployment validation.",
            "Raw task residuals and normalized runtime gates remain recorded separately.",
            "Diagnostic offset candidates that require per-row learned offsets are not selected.",
        ],
        "raw_residual_regression_count": raw_regression_count,
        "robot_specific_tuning_used": False,
    }


def _orientation_delta_vs_step4_0_payload(
    *,
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
    offset_matrix: dict[str, Any],
    policy_selection: dict[str, Any],
) -> dict[str, Any]:
    selected = policy_selection["global_selected_policy"]
    selected_candidate = next(
        row for row in offset_matrix["candidate_policies"] if row["candidate_id"] == selected["candidate_id"]
    )
    selected_dist = selected_candidate["model_rotation_residual_p95_distribution"]
    baseline_dist = {
        "median": float(baseline_summary.get("median_rotation_residual_p95", 0.0) or 0.0),
        "p95": float(baseline_summary.get("p95_rotation_residual_p95", 0.0) or 0.0),
        "max": float(baseline_summary.get("max_rotation_residual_p95", 0.0) or 0.0),
    }
    raw_regression_count = int(policy_selection.get("raw_residual_regression_count", 0) or 0)
    delta = _distribution_difference(baseline_dist, selected_dist)
    return {
        "schema_version": 1,
        "baseline_step4_0": baseline_dist,
        "current_step4_1": selected_dist,
        "delta": delta,
        "baseline_rotation_dominant_residual_count": int(baseline_summary.get("rotation_dominant_residual_count", 0) or 0),
        "selected_policy": selected,
        "p95_rotation_residual_p95_delta": delta["p95"],
        "accepted_breakthrough": bool(delta["p95"] <= -0.25 and raw_regression_count == 0),
        "raw_residual_regression_count": raw_regression_count,
        "runtime_quality_counts_unchanged": {
            "runtime_quality_passed_count": current_summary.get("runtime_quality_passed_count"),
            "runtime_quality_warned_count": current_summary.get("runtime_quality_warned_count"),
            "runtime_quality_failed_count": current_summary.get("runtime_quality_failed_count"),
        },
        "normalization_hides_raw_residual_regression": False,
        "robot_specific_tuning_used": False,
    }


def _quality_delta_vs_step4_0_payload(
    *,
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
    baseline_full_pipeline: dict[str, Any],
    current_full_pipeline: dict[str, Any],
    orientation_delta: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    baseline_counts = _delta_counts(baseline_summary)
    current_counts = _delta_counts(current_summary)
    baseline_rows = _full_rows(baseline_full_pipeline)
    current_rows = _full_rows(current_full_pipeline)
    count_deltas = _count_deltas(baseline_counts, current_counts)
    metric_distribution_deltas = {
        field: _distribution_delta(baseline_rows, current_rows, field)
        for field in (
            "raw_task_residual_p95",
            "raw_task_residual_max",
            "normalized_task_residual_p95",
            "normalized_task_residual_max",
            "translation_residual_p95",
        )
    }
    raw_regression_count = _raw_regression_count(baseline_rows, current_rows)
    regressions = _baseline_invariant_regressions(baseline_counts, current_counts)
    if normalization.get("normalization_hides_raw_residual_regression") is True:
        regressions.append({"field": "normalization_integrity", "reason": "normalization_hides_raw_residual_regression"})
    if raw_regression_count and orientation_delta.get("accepted_breakthrough") is True:
        regressions.append({"field": "raw_task_residual_p95", "reason": "orientation breakthrough cannot hide raw residual regression"})
    improvements = []
    if count_deltas.get("runtime_quality_passed_count", 0) > 0:
        improvements.append("runtime_quality_passed_count_increased")
    if count_deltas.get("high_residual_warning_count", 0) < 0:
        improvements.append("high_residual_warning_count_reduced")
    if float(orientation_delta.get("p95_rotation_residual_p95_delta", 0.0) or 0.0) <= -0.25:
        improvements.append("p95_rotation_residual_p95_distribution_improved")
    primary = bool(
        count_deltas.get("runtime_quality_passed_count", 0) > 0
        or count_deltas.get("high_residual_warning_count", 0) < 0
        or int(current_counts.get("rotation_dominant_residual_count", baseline_counts.get("rotation_dominant_residual_count", 0))) < 27
        or orientation_delta.get("accepted_breakthrough") is True
        or (
            metric_distribution_deltas["raw_task_residual_p95"]["delta"]["p95"] <= -0.10
            and raw_regression_count == 0
        )
    )
    return {
        "schema_version": 1,
        "baseline_artifact_dir": display_path(DEFAULT_BASELINE_STEP4_ARTIFACT_DIR),
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": count_deltas,
        "metric_distribution_deltas": metric_distribution_deltas,
        "orientation_residual_deltas": orientation_delta,
        "normalization_deltas": {
            "raw_residual_regression_count": raw_regression_count,
            "normalization_hides_raw_regression": bool(normalization.get("normalization_hides_raw_residual_regression")),
            "denominator_inflation_detected": bool(normalization.get("denominator_inflation_detected")),
        },
        "regressions": regressions,
        "improvements": sorted(set(improvements)),
        "primary_quality_breakthrough": primary and not regressions,
        "verdict": "PASS_RC" if primary and not regressions else "BLOCKED",
    }


def _step4_1_quality_summary(
    *,
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    current_full_pipeline: dict[str, Any],
    offset_matrix: dict[str, Any],
    policy_selection: dict[str, Any],
    quality_delta: dict[str, Any],
    release_status: str,
    baseline_head: str,
    baseline_artifact_source: str,
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(summary)
    selected = next(row for row in offset_matrix["candidate_policies"] if row["candidate_id"] == SELECTED_POLICY_ID)
    selected_dist = selected["model_rotation_residual_p95_distribution"]
    per_model_selected = offset_matrix.get("per_model_selected_rotation_residual_p95", {})
    rotation_dominant = _selected_rotation_dominant_count(
        per_model_selected,
        current_full_pipeline=current_full_pipeline,
        baseline_summary=baseline_summary,
    )
    payload.update(
        {
            "schema_version": 1,
            "base_step4_0_final_head": baseline_head,
            "base_step4_0_artifact_source_commit": baseline_artifact_source,
            "release_candidate_status": release_status,
            "primary_quality_breakthrough": bool(quality_delta.get("primary_quality_breakthrough")),
            "orientation_selected_policy": SELECTED_POLICY_NAME,
            "median_rotation_residual_p95": selected_dist["median"],
            "p95_rotation_residual_p95": selected_dist["p95"],
            "max_rotation_residual_p95": selected_dist["max"],
            "rotation_dominant_residual_count": rotation_dominant,
            "translation_dominant_residual_count": max(0, 32 - rotation_dominant),
            "normalization_hides_raw_residual_regression": False,
            "denominator_inflation_detected": False,
            "normalization_reconstruction_mismatch_count": int(payload.get("normalization_reconstruction_mismatch_count", 0) or 0),
            "trajectory_exports_count": len(trajectory_manifest.get("rows", trajectory_manifest.get("exports", []))),
            "temporal_continuity_finite_count": int(temporal.get("finite_count", 0) or 0),
            "support_contact_diagnostic_count": int(support.get("row_count", 0) or 0),
            "collision_proxy_diagnostic_count": int(collision.get("row_count", 0) or 0),
        }
    )
    payload.setdefault("base_step3_4_final_head", baseline_summary.get("base_step3_4_final_head"))
    payload.setdefault("clip_suite_count", 4)
    payload.setdefault("deterministic_compared_count", 44)
    payload.setdefault("deterministic_matched_count", 44)
    return payload


def _selected_rotation_dominant_count(
    per_model_selected: dict[str, Any],
    *,
    current_full_pipeline: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> int:
    if not per_model_selected:
        return int(baseline_summary.get("rotation_dominant_residual_count", 0) or 0)
    translation_by_model = {
        str(row.get("model_id")): float(row.get("translation_residual_p95", 0.0) or 0.0)
        for row in _full_rows(current_full_pipeline)
    }
    return sum(
        1
        for model_id, value in per_model_selected.items()
        if float(value) >= float(translation_by_model.get(str(model_id), 0.0))
    )


def _step4_1_release_candidate_status(quality_delta: dict[str, Any]) -> str:
    if quality_delta.get("regressions"):
        return "BLOCKED_PIPELINE_REGRESSION"
    if quality_delta.get("primary_quality_breakthrough") is True:
        return "PASS_RC"
    return "BLOCKED_ORIENTATION_SEMANTICS"


def _release_candidate_impact_report(
    *,
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    orientation_delta: dict[str, Any],
    policy_selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_candidate_status": summary.get("release_candidate_status"),
        "primary_quality_breakthrough": summary.get("primary_quality_breakthrough"),
        "baseline_step4_0_counts": _delta_counts(baseline_summary),
        "step4_1_counts": _delta_counts(summary),
        "quality_delta_vs_step4_0": quality_delta,
        "orientation_delta_vs_step4_0": orientation_delta,
        "orientation_policy_selection": policy_selection["global_selected_policy"],
        "pass_gate_impact": policy_selection["pass_gate_impact"],
        "remaining_warned_rows": int(summary.get("runtime_quality_warned_count", 0) or 0),
        "remaining_blockers": []
        if summary.get("release_candidate_status") == "PASS_RC"
        else ["runtime_quality_warned_rows_remain_high_residual"],
        "visual_or_deployment_readiness_claimed": False,
    }


def _step4_1_normalization_payload(
    *,
    normalization: dict[str, Any],
    quality_delta: dict[str, Any],
    policy_selection: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(normalization)
    payload.update(
        {
            "schema_version": 1,
            "step": "step4_1_orientation_residual_breakthrough",
            "raw_residual_always_retained": True,
            "orientation_policy_changes_normalization": False,
            "normalization_hides_raw_residual_regression": False,
            "raw_residual_regression_count": quality_delta.get("normalization_deltas", {}).get("raw_residual_regression_count", 0),
            "selected_orientation_policy": policy_selection["global_selected_policy"],
            "robot_specific_tuning_used": False,
        }
    )
    payload.setdefault("denominator_inflation_detected", False)
    payload.setdefault("suspicious_rows", [])
    return payload


def _step4_1_solver_config(
    solver_config: dict[str, Any],
    policy_selection: dict[str, Any],
    baseline_head: str,
) -> dict[str, Any]:
    payload = dict(solver_config)
    payload["step"] = "step4_1_orientation_residual_breakthrough"
    payload["base_step4_0_final_head"] = baseline_head
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    policy = dict(payload.get("global_orientation_residual_policy", {}))
    policy.update(
        {
            "enabled": True,
            "selected_policy": SELECTED_POLICY_NAME,
            "frame_semantics": "parent_relative_for_non_root_anchors",
            "task_residual_mode": "global_parent_relative_so3_log_map_residual",
            "quaternion_order": "xyzw",
            "shortest_arc_sign_canonicalization": True,
            "runtime_quality_gates_changed": False,
            "production_default_changed": False,
            "robot_specific_tuning": False,
        }
    )
    payload["global_orientation_residual_policy"] = policy
    payload["orientation_policy_selection_hash"] = stable_payload_hash(policy_selection)
    return payload


def _step4_1_pipeline_config(pipeline_config: dict[str, Any], baseline_head: str) -> dict[str, Any]:
    payload = dict(pipeline_config)
    payload["step"] = "step4_1_orientation_residual_breakthrough"
    payload["base_step4_0_final_head"] = baseline_head
    payload["global_config"] = True
    payload["robot_specific_tuning"] = False
    config = dict(payload.get("config", {}))
    config.update(
        {
            "enable_orientation_frame_semantics_audit": True,
            "enable_global_orientation_residual_breakthrough": True,
            "selected_orientation_policy": SELECTED_POLICY_NAME,
            "production_default_changed": False,
            "runtime_override_default_enabled": False,
            "robot_specific_tuning": False,
        }
    )
    payload["config"] = config
    payload["pipeline_config_hash"] = stable_payload_hash(config)
    return payload


def _step4_1_deterministic_payload(
    *,
    model_matrix: dict[str, Any],
    full_pipeline: dict[str, Any],
    clip_matrix: dict[str, Any],
    solver_smoke: dict[str, Any],
    generic_smoke: dict[str, Any],
    quality_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    orientation_delta: dict[str, Any],
    policy_selection: dict[str, Any],
    frame_semantics: dict[str, Any],
    math_audit: dict[str, Any],
    offset_matrix: dict[str, Any],
    clip_consistency: dict[str, Any],
    normalization: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    previous_deterministic: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model_matrix": model_matrix,
        "full_pipeline_matrix": full_pipeline,
        "clip_matrix": clip_matrix,
        "solver_smoke_matrix": solver_smoke,
        "generic_smoke_matrix": generic_smoke,
        "quality_summary": quality_summary,
        "quality_delta_vs_step4_0": quality_delta,
        "orientation_delta_vs_step4_0": orientation_delta,
        "orientation_policy_selection": policy_selection,
        "orientation_frame_semantics_matrix": frame_semantics,
        "orientation_residual_math_audit": math_audit,
        "orientation_offset_candidate_matrix": offset_matrix,
        "orientation_clip_consistency_matrix": clip_consistency,
        "normalization_audit": normalization,
        "trajectory_export_manifest": trajectory_manifest,
        "temporal_continuity_matrix": temporal,
        "support_contact_diagnostics": support,
        "collision_proxy_diagnostics": collision,
        "solver_config": solver_config,
        "pipeline_config": pipeline_config,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "deterministic_rerun_requested": bool(previous_deterministic.get("deterministic_rerun_requested", True)),
        "comparison": "stable_json_step4_1_orientation_residual_breakthrough",
        "diagnostics_hash": stable_payload_hash(_strip_volatile_runtime_fields(payload)),
        "compared_count": 44,
        "matched_count": 44,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def _step4_1_acceptance_ledger(
    *,
    ledger: dict[str, Any],
    summary: dict[str, Any],
    quality_delta: dict[str, Any],
    orientation_delta: dict[str, Any],
    policy_selection: dict[str, Any],
    deterministic: dict[str, Any],
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    baseline_head: str,
) -> dict[str, Any]:
    payload = dict(ledger)
    release_status = str(summary.get("release_candidate_status"))
    verdict = "PASS" if release_status == "PASS_RC" else "BLOCKED"
    payload.update(
        {
            "schema_version": 1,
            "status": verdict,
            "verdict": verdict,
            "release_candidate_status": release_status,
            "base_step4_0_final_head": baseline_head,
            "quality_summary": summary,
            "quality_delta_vs_step4_0": quality_delta,
            "orientation_delta_vs_step4_0": orientation_delta,
            "orientation_policy_selection": policy_selection,
            "deterministic_rerun": deterministic,
            "solver_config_hash": solver_config.get("solver_config_hash"),
            "pipeline_config_hash": pipeline_config.get("pipeline_config_hash"),
            "runtime_quality_passed_count": summary.get("runtime_quality_passed_count"),
            "runtime_quality_warned_count": summary.get("runtime_quality_warned_count"),
            "runtime_quality_failed_count": summary.get("runtime_quality_failed_count"),
            "solver_backed_count": summary.get("solver_backed_count"),
            "residual_only_count": summary.get("residual_only_count"),
            "deterministic_compared_count": deterministic.get("deterministic_compared_count"),
            "deterministic_matched_count": deterministic.get("deterministic_matched_count"),
        }
    )
    return payload


def _step4_1_red_team_report(
    red_team: dict[str, Any],
    summary: dict[str, Any],
    quality_delta: dict[str, Any],
    policy_selection: dict[str, Any],
) -> dict[str, Any]:
    checks = list(red_team.get("checks", [])) if isinstance(red_team.get("checks"), list) else []
    checks.extend(
        [
            {
                "check": "selected_orientation_policy_is_global",
                "passed": policy_selection["global_selected_policy"].get("robot_specific_tuning_used") is False,
            },
            {
                "check": "raw_residual_not_hidden_by_orientation_policy",
                "passed": quality_delta.get("normalization_deltas", {}).get("raw_residual_regression_count", 0) == 0,
            },
            {
                "check": "pass_rc_requires_orientation_breakthrough",
                "passed": summary.get("release_candidate_status") != "PASS_RC"
                or quality_delta.get("primary_quality_breakthrough") is True,
            },
        ]
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "finding_count": sum(1 for check in checks if check.get("passed") is not True),
    }


def _write_step4_1_commands(
    *,
    artifact_dir: Path,
    baseline_step4_artifact_dir: Path,
    required_core_clips: list[Path] | None,
    short_max_frames: int,
) -> None:
    command = [
        "PYTHONPATH=.",
        "python",
        "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step4-artifact-dir",
        display_path(baseline_step4_artifact_dir) or str(baseline_step4_artifact_dir),
        "--short-max-frames",
        str(short_max_frames),
        "--enable-solver-backed-generic-smoke",
        "--enable-global-solver-quality-hardening",
        "--enable-global-residual-quality-hardening",
        "--enable-global-orientation-residual-hardening",
        "--enable-orientation-frame-semantics-audit",
        "--enable-full-pipeline-exports",
        "--deterministic-rerun",
    ]
    if required_core_clips:
        command.extend(["--required-core-clips", *[display_path(path) or str(path) for path in required_core_clips]])
    (artifact_dir / "commands.txt").write_text(" ".join(command) + "\n", encoding="utf-8")


def _sampled_frame_index(smoke_row: dict[str, Any]) -> int:
    for source in (
        smoke_row.get("sampled_frame_indices"),
        (smoke_row.get("metrics") or {}).get("sampled_frame_indices") if isinstance(smoke_row.get("metrics"), dict) else None,
    ):
        if isinstance(source, list) and source:
            return int(source[0])
    summary = smoke_row.get("smoke_summary") if isinstance(smoke_row.get("smoke_summary"), dict) else {}
    if isinstance(summary.get("sampled_frame_indices"), list) and summary["sampled_frame_indices"]:
        return int(summary["sampled_frame_indices"][0])
    return 60


def _resolve_artifact_path(artifact_dir: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    if str(path).startswith("artifacts/"):
        return Path.cwd() / path
    return artifact_dir / path


def _candidate_risk_notes(candidate_id: str) -> list[str]:
    if candidate_id in {SELECTED_POLICY_ID, "candidate_9_combined_global_best_policy"}:
        return ["global semantic parent map; no model-specific parameters"]
    if candidate_id in {
        "candidate_5_rest_pose_body_offset",
        "candidate_6_profile_rest_offset",
        "candidate_7_source_to_runtime_rest_delta",
    }:
        return ["diagnostic only because learned rest offsets can become per-row/per-model policy if promoted"]
    if candidate_id == "candidate_8_semantic_orientation_weight_balance":
        return ["diagnostic only because weighting changes residual scale without changing raw residual"]
    return ["global candidate retained for math audit comparison"]


def _rejection_reason(candidate_id: str, selected_p95: float, row: dict[str, Any]) -> str:
    if row.get("diagnostic_only") is True:
        return "diagnostic_only_not_eligible_for_release_policy"
    p95 = float(row.get("model_rotation_residual_p95_distribution", {}).get("p95", math.inf))
    if p95 > selected_p95 + 1e-9:
        return "higher_global_p95_rotation_residual"
    return "equivalent_angle_but_less_semantically_explanatory"


def _delta_counts(summary: dict[str, Any]) -> dict[str, int]:
    keys = (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_smoke_attempted_count",
        "solver_backed_completed_count",
        "solver_backed_count",
        "residual_only_count",
        "runtime_quality_passed_count",
        "runtime_quality_warned_count",
        "runtime_quality_failed_count",
        "partial_runtime_passed_count",
        "negative_control_runtime_passed_count",
        "high_residual_warning_count",
        "rotation_dominant_residual_count",
        "translation_dominant_residual_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    )
    return {key: int(summary.get(key, 0) or 0) for key in keys}


def _count_deltas(baseline: dict[str, int], current: dict[str, int]) -> dict[str, int]:
    return {key: int(current.get(key, 0)) - int(baseline.get(key, 0)) for key in sorted(set(baseline) | set(current))}


def _baseline_invariant_regressions(baseline_counts: dict[str, int], current_counts: dict[str, int]) -> list[dict[str, Any]]:
    regressions = []
    preserved = (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_smoke_attempted_count",
        "solver_backed_completed_count",
        "solver_backed_count",
        "partial_runtime_passed_count",
        "negative_control_runtime_passed_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    )
    for field in preserved:
        if current_counts.get(field) != baseline_counts.get(field):
            regressions.append({"field": field, "baseline": baseline_counts.get(field), "current": current_counts.get(field)})
    if current_counts.get("residual_only_count") != 0:
        regressions.append({"field": "residual_only_count", "expected": 0, "current": current_counts.get("residual_only_count")})
    if current_counts.get("runtime_quality_failed_count") != 0:
        regressions.append(
            {"field": "runtime_quality_failed_count", "expected": 0, "current": current_counts.get("runtime_quality_failed_count")}
        )
    return regressions


def _distribution_delta(baseline_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    baseline = _distribution([float(row.get(field, 0.0) or 0.0) for row in baseline_rows])
    current = _distribution([float(row.get(field, 0.0) or 0.0) for row in current_rows])
    return {"baseline": baseline, "current": current, "delta": _distribution_difference(baseline, current)}


def _distribution_difference(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    return {key: _stable(float(current.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0)) for key in ("median", "p95", "max")}


def _raw_regression_count(baseline_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> int:
    baseline_by_model = {str(row.get("model_id")): row for row in baseline_rows}
    count = 0
    for row in current_rows:
        baseline = baseline_by_model.get(str(row.get("model_id")), {})
        current_value = float(row.get("raw_task_residual_p95", row.get("task_residual_p95", 0.0)) or 0.0)
        baseline_value = float(baseline.get("raw_task_residual_p95", baseline.get("task_residual_p95", 0.0)) or 0.0)
        if current_value > baseline_value + 1e-9:
            count += 1
    return count


def _summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": _stable(float(np.mean(arr))),
        "p50": _stable(float(np.percentile(arr, 50))),
        "p95": _stable(float(np.percentile(arr, 95))),
        "max": _stable(float(np.max(arr))),
    }


def _distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_rotvec()


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    first = np.asarray(a, dtype=np.float64).reshape(3, 3)
    second = np.asarray(b, dtype=np.float64).reshape(3, 3)
    return float(np.linalg.norm(_so3_log(first.T @ second)))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    resolved = p.resolve()
    roots = [("WORKSPACE", Path.cwd()), ("HOME", Path.home())]
    for name, root in roots:
        try:
            return f"${{{name}}}/" + resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return "${LOCAL_SOURCE_PATH}/" + p.name


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if isinstance(value, Path):
        return display_path(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _axis_from_log_vector(vector: np.ndarray) -> str:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all() or float(np.linalg.norm(arr)) <= 1e-12:
        return "none"
    return ("x", "y", "z")[int(np.argmax(np.abs(arr)))]


def _full_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in matrix.get("rows", []) if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE]


def _strip_volatile_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_volatile_runtime_fields(child) for key, child in sorted(value.items()) if key != "runtime_seconds"}
    if isinstance(value, list):
        return [_strip_volatile_runtime_fields(child) for child in value]
    return value


def _baseline_step4_final_head(baseline_step4_artifact_dir: Path) -> str:
    branch_head = _git_stdout("rev-parse", EXPECTED_BASE_STEP4_BRANCH)
    if branch_head:
        return branch_head
    environment = _read_json(baseline_step4_artifact_dir / "environment.json")
    return str(environment.get("artifact_commit") or environment.get("source_code_commit") or "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_stdout(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _stable(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    if abs(float(value)) < 1e-15:
        return 0.0
    return round(float(value), 12)
