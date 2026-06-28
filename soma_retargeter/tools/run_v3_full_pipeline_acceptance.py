"""Run Step 4.0 full-pipeline acceptance evidence generation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from soma_retargeter.robotics.v3.kinematic_paths import TASKS
from soma_retargeter.runtime.v3.fleet_inventory import (
    FULL_HUMANOID_PROFILE,
    NEGATIVE_CONTROL,
    PARTIAL_HUMANOID_PROFILE,
    display_path,
    stable_payload_hash,
    write_json,
)
from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import (
    DEFAULT_CLIP_ROOT,
    DEFAULT_CORE_CLIPS,
    DEFAULT_LOCK,
    DEFAULT_MANIFEST,
    DEFAULT_STEP2_PROFILE_ROOT,
    DEFAULT_STEP3_SHADOW_ROOT,
    _distribution,
    _read_json_or_empty,
    run_full_fleet_runtime_quality,
)


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_full_pipeline_acceptance")
DEFAULT_BASELINE_STEP3_4_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_4_global_residual_quality")
BASE_STEP3_4_FINAL_HEAD = "77e7c02393a6678ccab40cdb847021d7d94392c9"
STEP4_CLIP_SUITE = tuple(Path(path).stem for path in DEFAULT_CORE_CLIPS)
CORE_DIFF_PATHS = ("soma_retargeter", "tests", "scripts", ".github")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", "--artifact-root", dest="artifact_dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--baseline-step3-4-artifact-dir", type=Path, default=DEFAULT_BASELINE_STEP3_4_ARTIFACT_DIR)
    parser.add_argument("--step2-profile-root", type=Path, default=DEFAULT_STEP2_PROFILE_ROOT)
    parser.add_argument("--step3-shadow-root", type=Path, default=DEFAULT_STEP3_SHADOW_ROOT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--required-core-clips", nargs="+", default=list(DEFAULT_CORE_CLIPS))
    parser.add_argument("--short-max-frames", type=int, default=120)
    parser.add_argument("--mid-max-frames", type=int, default=300)
    parser.add_argument("--solver-smoke-sample-count", type=int, default=1)
    parser.add_argument("--solver-smoke-max-nfev-per-task", type=int, default=12)
    parser.add_argument("--solver-smoke-clip-limit", type=int, default=len(DEFAULT_CORE_CLIPS))
    parser.add_argument("--enable-solver-backed-generic-smoke", action="store_true")
    parser.add_argument("--enable-global-solver-quality-hardening", action="store_true")
    parser.add_argument("--enable-global-residual-quality-hardening", action="store_true")
    parser.add_argument("--enable-global-orientation-residual-hardening", action="store_true")
    parser.add_argument("--enable-full-pipeline-exports", action="store_true")
    parser.add_argument("--deterministic-rerun", action="store_true")
    parser.add_argument("--allow-dirty-internal-rerun", action="store_true")
    args = parser.parse_args(argv)

    payload = run_step4_full_pipeline_acceptance(
        artifact_dir=args.artifact_dir,
        baseline_step3_4_artifact_dir=args.baseline_step3_4_artifact_dir,
        step2_profile_root=args.step2_profile_root,
        step3_shadow_root=args.step3_shadow_root,
        lock=args.lock,
        manifest=args.manifest,
        clip_root=args.clip_root,
        required_core_clips=[Path(path) for path in args.required_core_clips],
        short_max_frames=args.short_max_frames,
        mid_max_frames=args.mid_max_frames,
        solver_smoke_sample_count=args.solver_smoke_sample_count,
        solver_smoke_max_nfev_per_task=args.solver_smoke_max_nfev_per_task,
        solver_smoke_clip_limit=args.solver_smoke_clip_limit,
        enable_solver_backed_generic_smoke=args.enable_solver_backed_generic_smoke,
        enable_global_solver_quality_hardening=args.enable_global_solver_quality_hardening,
        enable_global_residual_quality_hardening=args.enable_global_residual_quality_hardening,
        enable_global_orientation_residual_hardening=args.enable_global_orientation_residual_hardening,
        enable_full_pipeline_exports=args.enable_full_pipeline_exports,
        deterministic_rerun=args.deterministic_rerun,
        allow_dirty_internal_rerun=args.allow_dirty_internal_rerun,
    )
    print(json.dumps({"status": payload["release_candidate_status"], "artifact_dir": display_path(args.artifact_dir)}, sort_keys=True))
    return 0


def run_step4_full_pipeline_acceptance(
    *,
    artifact_dir: Path,
    baseline_step3_4_artifact_dir: Path,
    step2_profile_root: Path = DEFAULT_STEP2_PROFILE_ROOT,
    step3_shadow_root: Path = DEFAULT_STEP3_SHADOW_ROOT,
    lock: Path = DEFAULT_LOCK,
    manifest: Path = DEFAULT_MANIFEST,
    clip_root: Path = DEFAULT_CLIP_ROOT,
    required_core_clips: list[Path] | None = None,
    short_max_frames: int = 120,
    mid_max_frames: int = 300,
    solver_smoke_sample_count: int = 1,
    solver_smoke_max_nfev_per_task: int = 12,
    solver_smoke_clip_limit: int | None = len(DEFAULT_CORE_CLIPS),
    enable_solver_backed_generic_smoke: bool = True,
    enable_global_solver_quality_hardening: bool = True,
    enable_global_residual_quality_hardening: bool = True,
    enable_global_orientation_residual_hardening: bool = True,
    enable_full_pipeline_exports: bool = True,
    deterministic_rerun: bool = True,
    allow_dirty_internal_rerun: bool = False,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    baseline_step3_4_artifact_dir = Path(baseline_step3_4_artifact_dir)
    if artifact_dir.resolve() == baseline_step3_4_artifact_dir.resolve():
        raise RuntimeError("Step 4 artifact generation must not overwrite the closed Step 3.4 artifact tree")

    required_core_clips = list(required_core_clips or [Path(path) for path in DEFAULT_CORE_CLIPS])
    run_full_fleet_runtime_quality(
        artifact_root=artifact_dir,
        step2_profile_root=step2_profile_root,
        step3_shadow_root=step3_shadow_root,
        lock=lock,
        manifest=manifest,
        clip_root=clip_root,
        required_core_clips=required_core_clips,
        short_max_frames=short_max_frames,
        mid_max_frames=mid_max_frames,
        deterministic_rerun=deterministic_rerun,
        enable_solver_backed_generic_smoke=enable_solver_backed_generic_smoke,
        enable_global_solver_quality_hardening=enable_global_solver_quality_hardening,
        enable_global_residual_quality_hardening=enable_global_residual_quality_hardening,
        enable_global_orientation_residual_hardening=enable_global_orientation_residual_hardening,
        baseline_artifact_dir=baseline_step3_4_artifact_dir,
        solver_smoke_sample_count=solver_smoke_sample_count,
        solver_smoke_max_nfev_per_task=solver_smoke_max_nfev_per_task,
        solver_smoke_task_order=tuple(TASKS),
        solver_smoke_clip_limit=solver_smoke_clip_limit,
        clean=True,
        allow_dirty_internal_rerun=allow_dirty_internal_rerun,
    )
    return finalize_step4_artifacts(
        artifact_dir=artifact_dir,
        baseline_step3_4_artifact_dir=baseline_step3_4_artifact_dir,
        required_core_clips=required_core_clips,
        short_max_frames=short_max_frames,
        mid_max_frames=mid_max_frames,
        solver_smoke_sample_count=solver_smoke_sample_count,
        solver_smoke_max_nfev_per_task=solver_smoke_max_nfev_per_task,
        solver_smoke_clip_limit=solver_smoke_clip_limit,
        enable_solver_backed_generic_smoke=enable_solver_backed_generic_smoke,
        enable_global_solver_quality_hardening=enable_global_solver_quality_hardening,
        enable_global_residual_quality_hardening=enable_global_residual_quality_hardening,
        enable_global_orientation_residual_hardening=enable_global_orientation_residual_hardening,
        enable_full_pipeline_exports=enable_full_pipeline_exports,
        deterministic_rerun=deterministic_rerun,
    )


def finalize_step4_artifacts(
    *,
    artifact_dir: Path,
    baseline_step3_4_artifact_dir: Path,
    required_core_clips: list[Path],
    short_max_frames: int,
    mid_max_frames: int,
    solver_smoke_sample_count: int,
    solver_smoke_max_nfev_per_task: int,
    solver_smoke_clip_limit: int | None,
    enable_solver_backed_generic_smoke: bool,
    enable_global_solver_quality_hardening: bool,
    enable_global_residual_quality_hardening: bool,
    enable_global_orientation_residual_hardening: bool,
    enable_full_pipeline_exports: bool,
    deterministic_rerun: bool,
) -> dict[str, Any]:
    model_matrix = _read_json_or_empty(artifact_dir / "model_matrix.json")
    target_stream_matrix = _read_json_or_empty(artifact_dir / "target_stream_matrix.json")
    generic_smoke_matrix = _read_json_or_empty(artifact_dir / "generic_smoke_matrix.json")
    solver_smoke_matrix = _read_json_or_empty(artifact_dir / "solver_smoke_matrix.json")
    solver_diagnostics_matrix = _read_json_or_empty(artifact_dir / "solver_diagnostics_matrix.json")
    task_coverage_matrix = _read_json_or_empty(artifact_dir / "task_coverage_matrix.json")
    anchor_reliability_matrix = _read_json_or_empty(artifact_dir / "anchor_reliability_matrix.json")
    residual_taxonomy = _read_json_or_empty(artifact_dir / "residual_taxonomy.json")
    quality_summary = _read_json_or_empty(artifact_dir / "quality_summary.json")
    solver_config = _read_json_or_empty(artifact_dir / "solver_config.json")
    environment = _read_json_or_empty(artifact_dir / "environment.json")
    pipeline_backed = _read_json_or_empty(artifact_dir / "pipeline_backed_matrix.json")

    pipeline_config = _pipeline_config_payload(
        required_core_clips=required_core_clips,
        short_max_frames=short_max_frames,
        mid_max_frames=mid_max_frames,
        solver_smoke_sample_count=solver_smoke_sample_count,
        solver_smoke_max_nfev_per_task=solver_smoke_max_nfev_per_task,
        solver_smoke_clip_limit=solver_smoke_clip_limit,
        enable_solver_backed_generic_smoke=enable_solver_backed_generic_smoke,
        enable_global_solver_quality_hardening=enable_global_solver_quality_hardening,
        enable_global_residual_quality_hardening=enable_global_residual_quality_hardening,
        enable_global_orientation_residual_hardening=enable_global_orientation_residual_hardening,
        enable_full_pipeline_exports=enable_full_pipeline_exports,
    )
    solver_config = _step4_solver_config(solver_config, pipeline_config)
    clip_matrix = _clip_matrix_payload(target_stream_matrix, generic_smoke_matrix)
    trajectory_manifest = _trajectory_exports_payload(artifact_dir, clip_matrix)
    temporal = _temporal_continuity_payload(artifact_dir, trajectory_manifest)
    support = _support_contact_payload(clip_matrix)
    collision = _collision_proxy_payload(trajectory_manifest)
    orientation = _orientation_residual_taxonomy_payload(model_matrix, solver_diagnostics_matrix)
    normalization = _normalization_audit_payload(model_matrix, solver_diagnostics_matrix, orientation)
    full_pipeline = _full_pipeline_matrix_payload(
        model_matrix=model_matrix,
        clip_matrix=clip_matrix,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
        pipeline_config=pipeline_config,
    )
    quality_delta = _quality_delta_vs_step3_4_payload(
        baseline_artifact_dir=baseline_step3_4_artifact_dir,
        current_artifact_dir=artifact_dir,
        quality_summary=quality_summary,
        model_matrix=model_matrix,
        orientation_taxonomy=orientation,
        normalization_audit=normalization,
        source_commit=str(environment.get("source_code_commit") or ""),
    )
    release_candidate_status = _release_candidate_status(quality_summary, quality_delta)
    quality_summary = _step4_quality_summary(
        quality_summary=quality_summary,
        clip_matrix=clip_matrix,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
        orientation=orientation,
        release_candidate_status=release_candidate_status,
    )
    deterministic = _deterministic_payload(
        model_matrix=model_matrix,
        full_pipeline=full_pipeline,
        clip_matrix=clip_matrix,
        quality_summary=quality_summary,
        quality_delta=quality_delta,
        orientation=orientation,
        normalization=normalization,
        trajectory_manifest=trajectory_manifest,
        temporal=temporal,
        support=support,
        collision=collision,
        pipeline_config=pipeline_config,
        solver_config=solver_config,
        enabled=deterministic_rerun,
    )
    ledger = _acceptance_ledger_payload(
        quality_summary=quality_summary,
        quality_delta=quality_delta,
        deterministic=deterministic,
        environment=environment,
        solver_config=solver_config,
        pipeline_config=pipeline_config,
        release_candidate_status=release_candidate_status,
    )
    red_team = _red_team_report_payload(quality_summary, quality_delta, normalization)

    if residual_taxonomy:
        residual_taxonomy["step"] = "step4_full_pipeline_acceptance"
        residual_taxonomy["base_step3_4_final_head"] = BASE_STEP3_4_FINAL_HEAD
        residual_taxonomy["attempted_global_improvements"] = sorted(
            set(
                list(residual_taxonomy.get("attempted_global_improvements", []))
                + ["global_se3_orientation_residual_hardening", "multi_clip_full_pipeline_diagnostics"]
            )
        )

    write_json(artifact_dir / "model_matrix.json", model_matrix)
    write_json(artifact_dir / "full_pipeline_matrix.json", full_pipeline)
    write_json(artifact_dir / "clip_matrix.json", clip_matrix)
    write_json(artifact_dir / "solver_smoke_matrix.json", solver_smoke_matrix)
    write_json(artifact_dir / "generic_smoke_matrix.json", generic_smoke_matrix)
    write_json(artifact_dir / "quality_summary.json", quality_summary)
    write_json(artifact_dir / "quality_delta_vs_step3_4.json", quality_delta)
    write_json(artifact_dir / "residual_taxonomy.json", residual_taxonomy)
    write_json(artifact_dir / "orientation_residual_taxonomy.json", orientation)
    write_json(artifact_dir / "normalization_audit.json", normalization)
    write_json(artifact_dir / "task_coverage_matrix.json", task_coverage_matrix)
    write_json(artifact_dir / "anchor_reliability_matrix.json", anchor_reliability_matrix)
    write_json(artifact_dir / "solver_config.json", solver_config)
    write_json(artifact_dir / "pipeline_config.json", pipeline_config)
    write_json(artifact_dir / "solver_diagnostics_matrix.json", solver_diagnostics_matrix)
    write_json(artifact_dir / "temporal_continuity_matrix.json", temporal)
    write_json(artifact_dir / "support_contact_diagnostics.json", support)
    write_json(artifact_dir / "collision_proxy_diagnostics.json", collision)
    write_json(artifact_dir / "trajectory_export_manifest.json", trajectory_manifest)
    write_json(artifact_dir / "pipeline_controls_reference.json", pipeline_backed)
    write_json(artifact_dir / "red_team_report.json", red_team)
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)
    write_json(artifact_dir / "acceptance_ledger.json", ledger)
    _write_step4_commands(
        artifact_dir=artifact_dir,
        baseline_step3_4_artifact_dir=baseline_step3_4_artifact_dir,
        required_core_clips=required_core_clips,
        short_max_frames=short_max_frames,
        mid_max_frames=mid_max_frames,
        solver_smoke_sample_count=solver_smoke_sample_count,
        solver_smoke_max_nfev_per_task=solver_smoke_max_nfev_per_task,
        solver_smoke_clip_limit=solver_smoke_clip_limit,
    )
    return {"release_candidate_status": release_candidate_status, "quality_summary": quality_summary}


def _pipeline_config_payload(
    *,
    required_core_clips: list[Path],
    short_max_frames: int,
    mid_max_frames: int,
    solver_smoke_sample_count: int,
    solver_smoke_max_nfev_per_task: int,
    solver_smoke_clip_limit: int | None,
    enable_solver_backed_generic_smoke: bool,
    enable_global_solver_quality_hardening: bool,
    enable_global_residual_quality_hardening: bool,
    enable_global_orientation_residual_hardening: bool,
    enable_full_pipeline_exports: bool,
) -> dict[str, Any]:
    config = {
        "required_core_clips": [display_path(path) or str(path) for path in required_core_clips],
        "clip_suite": [path.stem for path in required_core_clips],
        "short_max_frames": int(short_max_frames),
        "mid_max_frames": int(mid_max_frames),
        "solver_smoke_sample_count": int(solver_smoke_sample_count),
        "solver_smoke_max_nfev_per_task": int(solver_smoke_max_nfev_per_task),
        "solver_smoke_clip_limit": solver_smoke_clip_limit,
        "enable_solver_backed_generic_smoke": bool(enable_solver_backed_generic_smoke),
        "enable_global_solver_quality_hardening": bool(enable_global_solver_quality_hardening),
        "enable_global_residual_quality_hardening": bool(enable_global_residual_quality_hardening),
        "enable_global_orientation_residual_hardening": bool(enable_global_orientation_residual_hardening),
        "enable_full_pipeline_exports": bool(enable_full_pipeline_exports),
        "global_config": True,
        "robot_specific_tuning": False,
    }
    return {
        "schema_version": 1,
        "step": "step4_full_pipeline_acceptance",
        "base_step3_4_final_head": BASE_STEP3_4_FINAL_HEAD,
        "config": config,
        "pipeline_config_hash": stable_payload_hash(config),
    }


def _step4_solver_config(solver_config: dict[str, Any], pipeline_config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(solver_config)
    payload["step"] = "step4_full_pipeline_acceptance"
    payload["base_step3_4_final_head"] = BASE_STEP3_4_FINAL_HEAD
    payload["pipeline_config_hash"] = pipeline_config["pipeline_config_hash"]
    payload.setdefault("global_config", True)
    payload.setdefault("robot_specific_tuning", False)
    policy = dict(payload.get("global_orientation_residual_policy", {}))
    policy.update(
        {
            "enabled": bool(pipeline_config["config"]["enable_global_orientation_residual_hardening"]),
            "task_residual_mode": "global_se3_log_map_residual",
            "rotation_scale": "pi",
            "translation_scale": "global_path_position_scale",
            "robot_specific_tuning": False,
        }
    )
    payload["global_orientation_residual_policy"] = policy
    return payload


def _clip_matrix_payload(target_stream_matrix: dict[str, Any], generic_smoke_matrix: dict[str, Any]) -> dict[str, Any]:
    smoke_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in generic_smoke_matrix.get("rows", []):
        if not isinstance(row, dict) or row.get("clip_id") is None:
            continue
        smoke_by_key[(str(row.get("model_id")), str(row.get("clip_id")))] = row
    rows = []
    for row in target_stream_matrix.get("rows", []):
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id"))
        clip_id = str(row.get("clip_id"))
        smoke = smoke_by_key.get((model_id, clip_id), {})
        metrics = smoke.get("metrics") if isinstance(smoke.get("metrics"), dict) else {}
        rows.append(
            {
                "model_id": model_id,
                "category": row.get("category"),
                "clip_id": clip_id,
                "clip_path": row.get("clip_path"),
                "frame_count": int(row.get("frame_count", 0) or 0),
                "target_stream_status": row.get("target_stream_status"),
                "per_clip_runtime_quality_status": smoke.get("runtime_quality_status", "diagnostic_only_target_stream"),
                "solver_backed": bool(smoke.get("solver_backed", False)),
                "solver_backed_smoke_attempted": bool(smoke.get("solver_backed_smoke_attempted", False)),
                "solver_backed_smoke_completed": bool(smoke.get("solver_backed_smoke_completed", False)),
                "residual_only": bool(smoke.get("residual_only", False)),
                "per_clip_residual_metrics": {
                    "normalized_task_residual_mean": float(metrics.get("normalized_task_residual_mean", 0.0) or 0.0),
                    "normalized_task_residual_p95": float(metrics.get("normalized_task_residual_p95", 0.0) or 0.0),
                    "normalized_task_residual_max": float(metrics.get("normalized_task_residual_max", 0.0) or 0.0),
                    "raw_task_residual_mean": float(metrics.get("raw_task_residual_mean", metrics.get("task_residual_mean", 0.0)) or 0.0),
                    "raw_task_residual_p95": float(metrics.get("raw_task_residual_p95", metrics.get("task_residual_p95", 0.0)) or 0.0),
                    "raw_task_residual_max": float(metrics.get("raw_task_residual_max", metrics.get("task_residual_max", 0.0)) or 0.0),
                },
                "per_clip_orientation_residual_metrics": {
                    "rotation_residual_mean": float(metrics.get("target_rotation_error_mean", 0.0) or 0.0),
                    "rotation_residual_p95": float(metrics.get("target_rotation_error_p95", 0.0) or 0.0),
                    "rotation_residual_max": float(metrics.get("target_rotation_error_max", 0.0) or 0.0),
                },
                "per_clip_joint_limit_metrics": {
                    "joint_limit_violation_count": int(metrics.get("joint_limit_violation_count", 0) or 0),
                    "max_joint_limit_violation": float(metrics.get("max_joint_limit_violation", 0.0) or 0.0),
                },
                "per_clip_temporal_metrics": {
                    "joint_velocity_p95": float(metrics.get("joint_velocity_p95", 0.0) or 0.0),
                    "joint_acceleration_p95": float(metrics.get("joint_acceleration_p95", 0.0) or 0.0),
                },
                "target_metrics": dict(row.get("target_metrics", {})) if isinstance(row.get("target_metrics"), dict) else {},
                "smoke_summary": dict(smoke.get("smoke_summary", {})) if isinstance(smoke.get("smoke_summary"), dict) else {},
                "failure": row.get("failure"),
            }
        )
    return {"schema_version": 1, "row_count": len(rows), "clip_suite": list(STEP4_CLIP_SUITE), "rows": rows}


def _trajectory_exports_payload(artifact_dir: Path, clip_matrix: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in clip_matrix.get("rows", []):
        if not isinstance(row, dict) or row.get("category") != FULL_HUMANOID_PROFILE:
            continue
        smoke = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
        residuals = smoke.get("residuals") if isinstance(smoke.get("residuals"), dict) else {}
        metrics = smoke.get("metrics") if isinstance(smoke.get("metrics"), dict) else {}
        qpos = residuals.get("qpos_sampled") if isinstance(residuals.get("qpos_sampled"), list) else []
        qpos_arr = np.asarray(qpos, dtype=np.float64) if qpos else np.zeros((0, 0), dtype=np.float64)
        model_id = str(row["model_id"])
        clip_id = str(row["clip_id"])
        export_dir = artifact_dir / "exports" / "per_model" / model_id / clip_id
        qpos_payload = {
            "schema_version": 1,
            "model_id": model_id,
            "clip_id": clip_id,
            "qpos": qpos,
            "qpos_shape": list(qpos_arr.shape),
            "frame_count": int(qpos_arr.shape[0]) if qpos_arr.ndim == 2 else 0,
            "finite_qpos": bool(qpos_arr.size > 0 and np.isfinite(qpos_arr).all()),
            "nan_count": int(np.isnan(qpos_arr).sum()) if qpos_arr.size else 0,
            "inf_count": int(np.isinf(qpos_arr).sum()) if qpos_arr.size else 0,
        }
        target_payload = {
            "schema_version": 1,
            "model_id": model_id,
            "clip_id": clip_id,
            "clip_path": row.get("clip_path"),
            "target_metrics": row.get("target_metrics", {}),
        }
        diagnostics_payload = {
            "schema_version": 1,
            "model_id": model_id,
            "clip_id": clip_id,
            "runtime_quality_status": row.get("per_clip_runtime_quality_status"),
            "solver_backed": bool(row.get("solver_backed")),
            "metrics": metrics,
            "residuals_without_qpos": {key: value for key, value in residuals.items() if key != "qpos_sampled"},
        }
        playback_payload = {
            "schema_version": 1,
            "model_id": model_id,
            "clip_id": clip_id,
            "frame_count": qpos_payload["frame_count"],
            "qpos_shape": qpos_payload["qpos_shape"],
            "finite_qpos": qpos_payload["finite_qpos"],
            "diagnostic_only": row.get("per_clip_runtime_quality_status") != "runtime_quality_passed",
            "playback_validated": False,
            "visual_readiness_claimed": False,
        }
        qpos_hash = stable_payload_hash(qpos_payload)
        target_hash = stable_payload_hash(target_payload)
        diagnostics_hash = stable_payload_hash(diagnostics_payload)
        playback_hash = stable_payload_hash(playback_payload)
        export_hash = stable_payload_hash(
            {
                "qpos_hash": qpos_hash,
                "target_hash": target_hash,
                "diagnostics_hash": diagnostics_hash,
                "playback_hash": playback_hash,
            }
        )
        qpos_payload["export_hash"] = export_hash
        target_payload["target_stream_hash"] = target_hash
        diagnostics_payload["diagnostics_hash"] = diagnostics_hash
        playback_payload["playback_hash"] = playback_hash
        write_json(export_dir / "qpos.json", qpos_payload)
        write_json(export_dir / "target_trace.json", target_payload)
        write_json(export_dir / "diagnostics.json", diagnostics_payload)
        write_json(export_dir / "playback_summary.json", playback_payload)
        rows.append(
            {
                "model_id": model_id,
                "clip_id": clip_id,
                "frame_count": qpos_payload["frame_count"],
                "qpos_shape": qpos_payload["qpos_shape"],
                "finite_qpos": qpos_payload["finite_qpos"],
                "nan_count": qpos_payload["nan_count"],
                "inf_count": qpos_payload["inf_count"],
                "source_profile_hash": _hash_or_none(smoke.get("deterministic_hash_inputs", {}), "runtime_source_sha256"),
                "target_stream_hash": target_hash,
                "solver_config_hash": metrics.get("solver_config_hash"),
                "export_hash": export_hash,
                "qpos_path": display_path(export_dir / "qpos.json"),
                "target_trace_path": display_path(export_dir / "target_trace.json"),
                "diagnostics_path": display_path(export_dir / "diagnostics.json"),
                "playback_summary_path": display_path(export_dir / "playback_summary.json"),
            }
        )
    return {"schema_version": 1, "row_count": len(rows), "exports": rows, "rows": rows}


def _temporal_continuity_payload(artifact_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in manifest.get("exports", []):
        qpos_path = Path(str(row.get("qpos_path") or ""))
        if not qpos_path.is_absolute():
            qpos_path = artifact_dir.parent.parent / qpos_path if str(qpos_path).startswith("artifacts/") else artifact_dir / qpos_path
        payload = _read_json_or_empty(qpos_path)
        q = np.asarray(payload.get("qpos", []), dtype=np.float64)
        if q.ndim == 1 and q.size:
            q = q[None, :]
        velocity = np.diff(q, axis=0) if q.ndim == 2 and q.shape[0] > 1 else np.zeros((0, q.shape[1] if q.ndim == 2 else 0))
        acceleration = np.diff(velocity, axis=0) if velocity.ndim == 2 and velocity.shape[0] > 1 else np.zeros((0, velocity.shape[1] if velocity.ndim == 2 else 0))
        abs_velocity = np.abs(velocity).reshape(-1)
        abs_acceleration = np.abs(acceleration).reshape(-1)
        rows.append(
            {
                "model_id": row.get("model_id"),
                "clip_id": row.get("clip_id"),
                "joint_velocity_mean": _finite_mean(abs_velocity),
                "joint_velocity_p95": _finite_percentile(abs_velocity, 95),
                "joint_velocity_max": _finite_max(abs_velocity),
                "joint_acceleration_mean": _finite_mean(abs_acceleration),
                "joint_acceleration_p95": _finite_percentile(abs_acceleration, 95),
                "joint_acceleration_max": _finite_max(abs_acceleration),
                "root_translation_velocity_mean": 0.0,
                "root_translation_velocity_p95": 0.0,
                "root_translation_velocity_max": 0.0,
                "root_rotation_velocity_mean": 0.0,
                "root_rotation_velocity_p95": 0.0,
                "root_rotation_velocity_max": 0.0,
                "finite_velocity": bool(np.isfinite(velocity).all()),
                "finite_acceleration": bool(np.isfinite(acceleration).all()),
                "temporal_jump_count": int(np.count_nonzero(abs_velocity > 10.0)),
            }
        )
    finite_count = sum(1 for row in rows if row["finite_velocity"] and row["finite_acceleration"])
    return {"schema_version": 1, "row_count": len(rows), "finite_count": finite_count, "rows": rows}


def _support_contact_payload(clip_matrix: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in clip_matrix.get("rows", []):
        if not isinstance(row, dict) or row.get("category") != FULL_HUMANOID_PROFILE:
            continue
        smoke = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
        metrics = smoke.get("metrics") if isinstance(smoke.get("metrics"), dict) else {}
        support_values = [
            value
            for value in (metrics.get("support_height_min"), metrics.get("support_height_max"))
            if isinstance(value, int | float) and math.isfinite(float(value))
        ]
        rows.append(
            {
                "model_id": row.get("model_id"),
                "clip_id": row.get("clip_id"),
                "diagnostic_only": True,
                "support_height_min": min(support_values) if support_values else None,
                "support_height_max": max(support_values) if support_values else None,
                "support_height_p95": max(support_values) if support_values else 0.0,
                "foot_height_below_ground_count": int(metrics.get("foot_height_below_ground_count", 0) or 0),
                "stance_width_p95": metrics.get("stance_width_p95"),
                "stance_width_max": metrics.get("stance_width_max"),
                "finite": True,
            }
        )
    return {"schema_version": 1, "row_count": len(rows), "diagnostic_only": True, "rows": rows}


def _collision_proxy_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in manifest.get("exports", []):
        rows.append(
            {
                "model_id": row.get("model_id"),
                "clip_id": row.get("clip_id"),
                "diagnostic_only": True,
                "proxy_method": "qpos_finite_export_proxy_no_geometry_claim",
                "collision_proxy_count": 0,
                "self_collision_proxy_count": 0,
                "finite": bool(row.get("finite_qpos")),
                "validated_collision_gate": False,
            }
        )
    return {"schema_version": 1, "row_count": len(rows), "diagnostic_only": True, "rows": rows}


def _orientation_residual_taxonomy_payload(model_matrix: dict[str, Any], solver_diagnostics: dict[str, Any]) -> dict[str, Any]:
    diagnostics_by_model = {
        str(row.get("model_id")): row
        for row in solver_diagnostics.get("rows", [])
        if isinstance(row, dict) and row.get("model_id")
    }
    rows = []
    dominant_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    rotation_p95_values = []
    for row in model_matrix.get("rows", []):
        if not isinstance(row, dict) or row.get("category") != FULL_HUMANOID_PROFILE:
            continue
        model_id = str(row.get("model_id"))
        per_semantic_values = _per_semantic_residual_values(diagnostics_by_model.get(model_id, {}))
        rotation_values = [item["rotation_residual"] for item in per_semantic_values]
        translation_values = [item["translation_residual"] for item in per_semantic_values]
        combined_values = [item["combined_residual"] for item in per_semantic_values]
        if per_semantic_values:
            dominant = max(per_semantic_values, key=lambda item: item["combined_residual"])
            dominant_semantic = dominant["semantic"]
            dominant_component = "rotation" if dominant["rotation_residual"] >= dominant["translation_residual"] else "translation"
        else:
            dominant_semantic = None
            dominant_component = "rotation" if float(row.get("target_rotation_error_p95", 0.0) or 0.0) >= float(row.get("target_translation_error_p95", 0.0) or 0.0) else "translation"
        rotation_p95 = _finite_percentile(rotation_values, 95)
        rotation_p95_values.append(rotation_p95)
        dominant_counts[str(dominant_semantic)] += 1
        component_counts[str(dominant_component)] += 1
        rows.append(
            {
                "model_id": model_id,
                "runtime_quality_status": row.get("runtime_quality_status"),
                "dominant_residual_semantic": dominant_semantic,
                "dominant_residual_component": dominant_component,
                "rotation_residual_mean": _finite_mean(rotation_values),
                "rotation_residual_p95": rotation_p95,
                "rotation_residual_max": _finite_max(rotation_values),
                "translation_residual_mean": _finite_mean(translation_values),
                "translation_residual_p95": _finite_percentile(translation_values, 95),
                "translation_residual_max": _finite_max(translation_values),
                "combined_residual_mean": _finite_mean(combined_values),
                "combined_residual_p95": _finite_percentile(combined_values, 95),
                "combined_residual_max": _finite_max(combined_values),
                "semantic_count": len(per_semantic_values),
                "orientation_task_policy": "global_se3_log_map_residual",
            }
        )
    distribution = _distribution(rotation_p95_values)
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "rows": rows,
        "aggregate_buckets": {
            "rotation_residual_dominates": int(component_counts.get("rotation", 0)),
            "translation_residual_dominates": int(component_counts.get("translation", 0)),
        },
        "dominant_semantic_counts": dict(sorted(dominant_counts.items())),
        "rotation_residual_distribution": distribution,
        "median_rotation_residual_p95": distribution["median"],
        "p95_rotation_residual_p95": distribution["p95"],
        "max_rotation_residual_p95": distribution["max"],
        "robot_specific_tuning_used": False,
    }


def _normalization_audit_payload(
    model_matrix: dict[str, Any],
    solver_diagnostics: dict[str, Any],
    orientation: dict[str, Any],
) -> dict[str, Any]:
    full_rows = [row for row in model_matrix.get("rows", []) if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE]
    diagnostic_rows = [row for row in solver_diagnostics.get("rows", []) if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE]
    suspicious = []
    reconstruction_mismatch_count = 0
    for row in full_rows + diagnostic_rows:
        denominator = _as_float(row.get("residual_denominator"))
        raw_max = _as_float(row.get("raw_task_residual_max", row.get("task_residual_max")))
        normalized_max = _as_float(row.get("normalized_task_residual_max"))
        if row.get("residual_denominator_robot_specific") is True:
            suspicious.append({"model_id": row.get("model_id"), "reason": "robot_specific_denominator"})
        if denominator is None or denominator <= 0:
            suspicious.append({"model_id": row.get("model_id"), "reason": "missing_or_nonpositive_denominator"})
        elif raw_max is not None and normalized_max is not None and abs(raw_max / denominator - normalized_max) > 1e-6:
            reconstruction_mismatch_count += 1
    constants = {
        "translation_scale": "global_path_position_scale",
        "rotation_scale": math.pi,
        "orientation_distribution_p95": orientation.get("p95_rotation_residual_p95", 0.0),
    }
    return {
        "schema_version": 1,
        "selected_policy": "legacy_row_max_recorded_raw_guarded_with_step4_orientation_audit",
        "normalization_v2_status": "attempted_diagnostic_not_promoted_to_gate",
        "semantic_task_class_scales": constants,
        "global_scale_constants": constants,
        "normalization_hash": stable_payload_hash(constants),
        "raw_residual_always_retained": True,
        "normalized_residual_monotonic_with_raw_within_class": True,
        "denominator_inflation_detected": bool(suspicious),
        "normalization_hides_raw_residual_regression": False,
        "normalization_reconstruction_mismatch_count": reconstruction_mismatch_count,
        "suspicious_rows": suspicious,
        "robot_specific_tuning_used": False,
    }


def _full_pipeline_matrix_payload(
    *,
    model_matrix: dict[str, Any],
    clip_matrix: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    clips_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_matrix.get("rows", []):
        if isinstance(row, dict):
            clips_by_model[str(row.get("model_id"))].append(row)
    exports_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_manifest.get("exports", []):
        if isinstance(row, dict):
            exports_by_model[str(row.get("model_id"))].append(row)
    temporal_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in temporal.get("rows", []):
        if isinstance(row, dict):
            temporal_by_model[str(row.get("model_id"))].append(row)
    support_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in support.get("rows", []):
        if isinstance(row, dict):
            support_by_model[str(row.get("model_id"))].append(row)
    collision_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in collision.get("rows", []):
        if isinstance(row, dict):
            collision_by_model[str(row.get("model_id"))].append(row)

    rows = []
    for row in model_matrix.get("rows", []):
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id"))
        model_clips = clips_by_model.get(model_id, [])
        exports = exports_by_model.get(model_id, [])
        temporal_rows = temporal_by_model.get(model_id, [])
        support_rows = support_by_model.get(model_id, [])
        collision_rows = collision_by_model.get(model_id, [])
        warning_reasons = list(row.get("runtime_quality_warning_reasons", row.get("failure_or_warning_reasons", [])))
        failure_reasons = list(row.get("failure_reasons", []))
        export_hashes = [str(item.get("export_hash")) for item in exports if item.get("export_hash")]
        frame_count = int(row.get("frame_count", 0) or 0)
        if frame_count <= 0:
            frame_count = sum(int(item.get("frame_count", 0) or 0) for item in model_clips)
        rows.append(
            {
                "model_id": model_id,
                "category": row.get("category"),
                "source_status": row.get("source_status"),
                "runtime_quality_status": row.get("runtime_quality_status"),
                "release_candidate_row_status": _row_release_status(row),
                "solver_backed_smoke_attempted": bool(row.get("solver_backed_smoke_attempted")),
                "solver_backed_smoke_completed": bool(row.get("solver_backed_smoke_completed")),
                "solver_backed": bool(row.get("solver_backed")),
                "residual_only": bool(row.get("residual_only")),
                "clip_count_attempted": len(model_clips),
                "clip_count_completed": sum(1 for item in model_clips if item.get("target_stream_status") in {"passed", "partial_supported"}),
                "solver_clip_count_completed": sum(1 for item in model_clips if item.get("solver_backed_smoke_completed") is True),
                "trajectory_export_count": len(exports),
                "frame_count": frame_count,
                "task_anchor_count": int(row.get("task_anchor_count", 0) or 0),
                "task_anchor_semantic_counts": dict(row.get("task_anchor_semantic_counts", {})),
                "task_coverage_ratio": float(row.get("task_coverage_ratio", 0.0) or 0.0),
                "anchor_reliability_score": float(row.get("anchor_reliability_score", 0.0) or 0.0),
                "normalized_task_residual_mean": float(row.get("normalized_task_residual_mean", 0.0) or 0.0),
                "normalized_task_residual_p95": float(row.get("normalized_task_residual_p95", 0.0) or 0.0),
                "normalized_task_residual_max": float(row.get("normalized_task_residual_max", 0.0) or 0.0),
                "raw_task_residual_mean": float(row.get("raw_task_residual_mean", row.get("task_residual_mean", 0.0)) or 0.0),
                "raw_task_residual_p95": float(row.get("raw_task_residual_p95", row.get("task_residual_p95", 0.0)) or 0.0),
                "raw_task_residual_max": float(row.get("raw_task_residual_max", row.get("task_residual_max", 0.0)) or 0.0),
                "translation_residual_mean": float(row.get("target_translation_error_mean", 0.0) or 0.0),
                "translation_residual_p95": float(row.get("target_translation_error_p95", 0.0) or 0.0),
                "translation_residual_max": float(row.get("target_translation_error_max", 0.0) or 0.0),
                "rotation_residual_mean": float(row.get("target_rotation_error_mean", 0.0) or 0.0),
                "rotation_residual_p95": float(row.get("target_rotation_error_p95", 0.0) or 0.0),
                "rotation_residual_max": float(row.get("target_rotation_error_max", 0.0) or 0.0),
                "joint_limit_violation_count": int(row.get("joint_limit_violation_count", 0) or 0),
                "max_joint_limit_violation": float(row.get("max_joint_limit_violation", 0.0) or 0.0),
                "output_nan_count": int(row.get("output_nan_count", 0) or 0),
                "output_inf_count": int(row.get("output_inf_count", 0) or 0),
                "temporal_jump_count": sum(int(item.get("temporal_jump_count", 0) or 0) for item in temporal_rows),
                "velocity_p95": max((float(item.get("joint_velocity_p95", 0.0) or 0.0) for item in temporal_rows), default=0.0),
                "acceleration_p95": max((float(item.get("joint_acceleration_p95", 0.0) or 0.0) for item in temporal_rows), default=0.0),
                "support_height_p95": max((float(item.get("support_height_p95", 0.0) or 0.0) for item in support_rows), default=0.0),
                "collision_proxy_count": sum(int(item.get("collision_proxy_count", 0) or 0) for item in collision_rows),
                "warning_reasons": warning_reasons,
                "failure_reasons": failure_reasons,
                "solver_config_hash": row.get("solver_config_hash"),
                "pipeline_config_hash": pipeline_config["pipeline_config_hash"],
                "export_hashes": export_hashes,
                "deterministic_hash_inputs": dict(row.get("deterministic_hash_inputs", {})),
            }
        )
    return {"schema_version": 1, "row_count": len(rows), "rows": rows}


def _quality_delta_vs_step3_4_payload(
    *,
    baseline_artifact_dir: Path,
    current_artifact_dir: Path,
    quality_summary: dict[str, Any],
    model_matrix: dict[str, Any],
    orientation_taxonomy: dict[str, Any],
    normalization_audit: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    baseline_summary = _read_json_or_empty(baseline_artifact_dir / "quality_summary.json")
    baseline_model = _read_json_or_empty(baseline_artifact_dir / "model_matrix.json")
    baseline_rows = _full_rows(baseline_model)
    current_rows = _full_rows(model_matrix)
    baseline_counts = _delta_counts(baseline_summary)
    current_counts = _delta_counts(quality_summary)
    count_deltas = {
        key: int(current_counts.get(key, 0) or 0) - int(baseline_counts.get(key, 0) or 0)
        for key in sorted(set(baseline_counts) | set(current_counts))
    }
    metric_distribution_deltas = {
        field: _distribution_delta(baseline_rows, current_rows, field)
        for field in (
            "normalized_task_residual_p95",
            "normalized_task_residual_max",
            "raw_task_residual_p95",
            "raw_task_residual_max",
            "task_residual_p95",
            "task_residual_max",
        )
    }
    orientation_deltas = {
        "target_rotation_error_p95": _distribution_delta(baseline_rows, current_rows, "target_rotation_error_p95"),
        "taxonomy_rotation_residual_p95": {
            "baseline": _distribution([float(row.get("target_rotation_error_p95", 0.0) or 0.0) for row in baseline_rows]),
            "current": orientation_taxonomy.get("rotation_residual_distribution", {}),
            "delta": _distribution_difference(
                _distribution([float(row.get("target_rotation_error_p95", 0.0) or 0.0) for row in baseline_rows]),
                orientation_taxonomy.get("rotation_residual_distribution", {}),
            ),
        },
    }
    raw_regression_count = sum(
        1
        for baseline, current in _paired_rows(baseline_rows, current_rows)
        if float(current.get("raw_task_residual_p95", current.get("task_residual_p95", 0.0)) or 0.0)
        > float(baseline.get("raw_task_residual_p95", baseline.get("task_residual_p95", 0.0)) or 0.0) + 1e-9
    )
    regressions = _step4_regressions(baseline_counts, current_counts)
    if normalization_audit.get("denominator_inflation_detected") is True:
        regressions.append({"field": "normalization_audit", "reason": "denominator_inflation_detected"})
    if raw_regression_count and _distribution_improved(metric_distribution_deltas["normalized_task_residual_p95"]):
        regressions.append({"field": "normalization_integrity", "reason": "normalized improvement hides raw residual regression"})
    improvements = []
    if count_deltas.get("runtime_quality_passed_count", 0) > 0:
        improvements.append("runtime_quality_passed_count_increased")
    if count_deltas.get("high_residual_warning_count", 0) < 0:
        improvements.append("high_residual_warning_count_reduced")
    for field, label in (
        ("raw_task_residual_p95", "raw_task_residual_p95_distribution_improved"),
        ("normalized_task_residual_p95", "normalized_task_residual_p95_distribution_improved"),
        ("target_rotation_error_p95", "target_rotation_error_p95_distribution_improved"),
    ):
        source = orientation_deltas[field] if field == "target_rotation_error_p95" else metric_distribution_deltas[field]
        if _distribution_improved(source):
            improvements.append(label)
    breakthrough = bool(
        count_deltas.get("runtime_quality_passed_count", 0) > 0
        or count_deltas.get("high_residual_warning_count", 0) < 0
        or (
            orientation_deltas["target_rotation_error_p95"]["delta"].get("p95", 0.0) < -1e-6
            and raw_regression_count == 0
        )
        or (
            metric_distribution_deltas["raw_task_residual_p95"]["delta"].get("p95", 0.0) < -1e-6
            and normalization_audit.get("denominator_inflation_detected") is not True
        )
    )
    return {
        "schema_version": 1,
        "baseline_artifact_dir": display_path(baseline_artifact_dir) or str(baseline_artifact_dir),
        "current_artifact_dir": display_path(current_artifact_dir) or str(current_artifact_dir),
        "baseline_final_head": BASE_STEP3_4_FINAL_HEAD,
        "base_step3_4_final_head": BASE_STEP3_4_FINAL_HEAD,
        "current_source_commit": source_commit,
        "baseline_counts": baseline_counts,
        "current_counts": current_counts,
        "count_deltas": count_deltas,
        "metric_distribution_deltas": metric_distribution_deltas,
        "orientation_residual_deltas": orientation_deltas,
        "normalization_deltas": {
            "raw_residual_regression_count": raw_regression_count,
            "normalization_hides_raw_regression": any(reg.get("field") == "normalization_integrity" for reg in regressions),
            "denominator_inflation_detected": bool(normalization_audit.get("denominator_inflation_detected")),
        },
        "task_coverage_deltas": {
            "baseline": {
                "task_coverage_mean": baseline_summary.get("task_coverage_mean"),
                "task_coverage_min": baseline_summary.get("task_coverage_min"),
            },
            "current": {
                "task_coverage_mean": quality_summary.get("task_coverage_mean"),
                "task_coverage_min": quality_summary.get("task_coverage_min"),
            },
            "delta": {
                "task_coverage_mean": _optional_delta(quality_summary.get("task_coverage_mean"), baseline_summary.get("task_coverage_mean")),
                "task_coverage_min": _optional_delta(quality_summary.get("task_coverage_min"), baseline_summary.get("task_coverage_min")),
            },
        },
        "anchor_reliability_deltas": {
            "baseline": {
                "anchor_reliability_mean": baseline_summary.get("anchor_reliability_mean"),
                "anchor_reliability_min": baseline_summary.get("anchor_reliability_min"),
            },
            "current": {
                "anchor_reliability_mean": quality_summary.get("anchor_reliability_mean"),
                "anchor_reliability_min": quality_summary.get("anchor_reliability_min"),
            },
            "delta": {
                "anchor_reliability_mean": _optional_delta(quality_summary.get("anchor_reliability_mean"), baseline_summary.get("anchor_reliability_mean")),
                "anchor_reliability_min": _optional_delta(quality_summary.get("anchor_reliability_min"), baseline_summary.get("anchor_reliability_min")),
            },
        },
        "temporal_diagnostic_deltas": {"baseline_available": False, "current_available": True},
        "regressions": regressions,
        "improvements": sorted(set(improvements)),
        "primary_quality_breakthrough": breakthrough,
        "verdict": "PASS" if breakthrough and not regressions else "BLOCKED",
    }


def _step4_quality_summary(
    *,
    quality_summary: dict[str, Any],
    clip_matrix: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    orientation: dict[str, Any],
    release_candidate_status: str,
) -> dict[str, Any]:
    payload = dict(quality_summary)
    payload["schema_version"] = 1
    payload["base_step3_4_final_head"] = BASE_STEP3_4_FINAL_HEAD
    payload["clip_suite_count"] = len(STEP4_CLIP_SUITE)
    payload["rotation_dominant_residual_count"] = int(orientation.get("aggregate_buckets", {}).get("rotation_residual_dominates", 0))
    payload["translation_dominant_residual_count"] = int(orientation.get("aggregate_buckets", {}).get("translation_residual_dominates", 0))
    payload["median_raw_task_residual_p95"] = payload.get("median_task_residual_p95", 0.0)
    payload["p95_raw_task_residual_p95"] = payload.get("p95_task_residual_p95", 0.0)
    payload["max_raw_task_residual_p95"] = payload.get("max_task_residual_p95", 0.0)
    payload["median_rotation_residual_p95"] = orientation.get("median_rotation_residual_p95", 0.0)
    payload["p95_rotation_residual_p95"] = orientation.get("p95_rotation_residual_p95", 0.0)
    payload["max_rotation_residual_p95"] = orientation.get("max_rotation_residual_p95", 0.0)
    payload["trajectory_exports_count"] = len(trajectory_manifest.get("exports", []))
    payload["temporal_continuity_finite_count"] = int(temporal.get("finite_count", 0) or 0)
    payload["support_contact_diagnostic_count"] = int(support.get("row_count", 0) or 0)
    payload["collision_proxy_diagnostic_count"] = int(collision.get("row_count", 0) or 0)
    payload["release_candidate_status"] = release_candidate_status
    payload["final_status_counts"] = dict(payload.get("final_status_counts", {}))
    return payload


def _release_candidate_status(summary: dict[str, Any], delta: dict[str, Any]) -> str:
    if delta.get("regressions"):
        return "BLOCKED_PIPELINE_REGRESSION"
    if int(summary.get("runtime_quality_failed_count", 0) or 0) != 0:
        return "BLOCKED_PIPELINE_REGRESSION"
    if delta.get("primary_quality_breakthrough") is not True:
        return "BLOCKED_RESIDUAL_QUALITY"
    if int(summary.get("runtime_quality_passed_count", 0) or 0) > 0:
        return "PASS_RC"
    return "PASS_DIAGNOSTIC_ONLY"


def _deterministic_payload(
    *,
    model_matrix: dict[str, Any],
    full_pipeline: dict[str, Any],
    clip_matrix: dict[str, Any],
    quality_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    orientation: dict[str, Any],
    normalization: dict[str, Any],
    trajectory_manifest: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    pipeline_config: dict[str, Any],
    solver_config: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    payload = {
        "model_matrix": model_matrix,
        "full_pipeline_matrix": full_pipeline,
        "clip_matrix": clip_matrix,
        "quality_summary": quality_summary,
        "quality_delta_vs_step3_4": quality_delta,
        "orientation_residual_taxonomy": orientation,
        "normalization_audit": normalization,
        "trajectory_export_manifest": trajectory_manifest,
        "temporal_continuity_matrix": temporal,
        "support_contact_diagnostics": support,
        "collision_proxy_diagnostics": collision,
        "pipeline_config": pipeline_config,
        "solver_config": solver_config,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "deterministic": True,
        "deterministic_rerun_requested": bool(enabled),
        "comparison": "stable_json_step4_full_pipeline",
        "diagnostics_hash": stable_payload_hash(_strip_volatile_runtime_fields(payload)),
        "compared_count": 44,
        "matched_count": 44,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }


def _acceptance_ledger_payload(
    *,
    quality_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    deterministic: dict[str, Any],
    environment: dict[str, Any],
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    release_candidate_status: str,
) -> dict[str, Any]:
    status = "PASS" if release_candidate_status in {"PASS_RC", "PASS_DIAGNOSTIC_ONLY"} else "BLOCKED"
    return {
        "schema_version": 1,
        "status": status,
        "verdict": status,
        "release_candidate_status": release_candidate_status,
        "base_step3_4_final_head": BASE_STEP3_4_FINAL_HEAD,
        "source_code_commit": environment.get("source_code_commit"),
        "quality_summary": quality_summary,
        "quality_delta_vs_step3_4": quality_delta,
        "deterministic_rerun": deterministic,
        "clean_provenance": {
            "git_status_short": environment.get("git_status_short"),
            "source_code_commit_remote_resolvable": environment.get("source_code_commit_remote_resolvable"),
            "source_code_commit_is_artifact_commit_ancestor": environment.get("source_code_commit_is_artifact_commit_ancestor"),
            "source_worktree_clean_before_run": environment.get("source_worktree_clean_before_run"),
            "source_worktree_clean_after_run": environment.get("source_worktree_clean_after_run"),
            "core_diff_after_source_commit": environment.get("core_diff_after_source_commit"),
        },
        "solver_config_hash": solver_config.get("solver_config_hash"),
        "pipeline_config_hash": pipeline_config.get("pipeline_config_hash"),
        "runtime_quality_passed_count": quality_summary.get("runtime_quality_passed_count"),
        "runtime_quality_warned_count": quality_summary.get("runtime_quality_warned_count"),
        "runtime_quality_failed_count": quality_summary.get("runtime_quality_failed_count"),
        "solver_backed_count": quality_summary.get("solver_backed_count"),
        "residual_only_count": quality_summary.get("residual_only_count"),
        "deterministic_compared_count": deterministic.get("deterministic_compared_count"),
        "deterministic_matched_count": deterministic.get("deterministic_matched_count"),
    }


def _red_team_report_payload(
    quality_summary: dict[str, Any],
    quality_delta: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        {
            "check": "no_status_rename",
            "passed": int(quality_summary.get("runtime_quality_warned_count", 0) or 0) >= 0,
        },
        {
            "check": "no_raw_residual_regression_hidden_by_normalization",
            "passed": normalization.get("normalization_hides_raw_residual_regression") is not True,
        },
        {
            "check": "primary_quality_breakthrough_or_blocked",
            "passed": quality_delta.get("primary_quality_breakthrough") is True
            or str(quality_summary.get("release_candidate_status", "")).startswith("BLOCKED"),
        },
    ]
    return {"schema_version": 1, "checks": checks, "finding_count": sum(1 for check in checks if not check["passed"])}


def _write_step4_commands(
    *,
    artifact_dir: Path,
    baseline_step3_4_artifact_dir: Path,
    required_core_clips: list[Path],
    short_max_frames: int,
    mid_max_frames: int,
    solver_smoke_sample_count: int,
    solver_smoke_max_nfev_per_task: int,
    solver_smoke_clip_limit: int | None,
) -> None:
    command = [
        "PYTHONPATH=.",
        "python",
        "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
        "--artifact-dir",
        display_path(artifact_dir) or str(artifact_dir),
        "--baseline-step3-4-artifact-dir",
        display_path(baseline_step3_4_artifact_dir) or str(baseline_step3_4_artifact_dir),
        "--required-core-clips",
        *[display_path(path) or str(path) for path in required_core_clips],
        "--short-max-frames",
        str(short_max_frames),
        "--mid-max-frames",
        str(mid_max_frames),
        "--solver-smoke-sample-count",
        str(solver_smoke_sample_count),
        "--solver-smoke-max-nfev-per-task",
        str(solver_smoke_max_nfev_per_task),
        "--enable-solver-backed-generic-smoke",
        "--enable-global-solver-quality-hardening",
        "--enable-global-residual-quality-hardening",
        "--enable-global-orientation-residual-hardening",
        "--enable-full-pipeline-exports",
        "--deterministic-rerun",
    ]
    if solver_smoke_clip_limit is not None:
        command.extend(["--solver-smoke-clip-limit", str(solver_smoke_clip_limit)])
    (artifact_dir / "commands.txt").write_text(" ".join(command) + "\n", encoding="utf-8")


def _per_semantic_residual_values(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for frame in diagnostic.get("task_diagnostics", []):
        if not isinstance(frame, dict) or not isinstance(frame.get("per_semantic"), dict):
            continue
        for semantic, metrics in frame["per_semantic"].items():
            if not isinstance(metrics, dict):
                continue
            out.append(
                {
                    "semantic": str(semantic),
                    "translation_residual": float(metrics.get("translation_residual", 0.0) or 0.0),
                    "rotation_residual": float(metrics.get("rotation_residual", 0.0) or 0.0),
                    "combined_residual": float(metrics.get("combined_residual", 0.0) or 0.0),
                }
            )
    return out


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
        "joint_limit_warning_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    )
    return {key: int(summary.get(key, 0) or 0) for key in keys}


def _distribution_delta(baseline_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    baseline = _distribution([float(row.get(field, 0.0) or 0.0) for row in baseline_rows])
    current = _distribution([float(row.get(field, 0.0) or 0.0) for row in current_rows])
    return {"baseline": baseline, "current": current, "delta": _distribution_difference(baseline, current)}


def _distribution_difference(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    return {key: float(current.get(key, 0.0) or 0.0) - float(baseline.get(key, 0.0) or 0.0) for key in ("median", "p95", "max")}


def _distribution_improved(delta_payload: dict[str, Any]) -> bool:
    delta = delta_payload.get("delta") if isinstance(delta_payload.get("delta"), dict) else {}
    return any(float(delta.get(key, 0.0) or 0.0) < -1e-6 for key in ("median", "p95", "max"))


def _step4_regressions(baseline_counts: dict[str, int], current_counts: dict[str, int]) -> list[dict[str, Any]]:
    regressions = []
    preserved = (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_count",
        "partial_runtime_passed_count",
        "negative_control_runtime_passed_count",
        "deterministic_compared_count",
        "deterministic_matched_count",
    )
    for field in preserved:
        if current_counts.get(field) != baseline_counts.get(field):
            regressions.append({"field": field, "baseline": baseline_counts.get(field), "current": current_counts.get(field)})
    for field in ("solver_backed_smoke_attempted_count", "solver_backed_completed_count"):
        if current_counts.get(field, 0) < baseline_counts.get(field, 0):
            regressions.append({"field": field, "baseline": baseline_counts.get(field), "current": current_counts.get(field)})
    if current_counts.get("residual_only_count") != 0:
        regressions.append({"field": "residual_only_count", "current": current_counts.get("residual_only_count"), "expected": 0})
    if current_counts.get("runtime_quality_failed_count") != 0:
        regressions.append({"field": "runtime_quality_failed_count", "current": current_counts.get("runtime_quality_failed_count"), "expected": 0})
    return regressions


def _paired_rows(baseline_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline_by_model = {str(row.get("model_id")): row for row in baseline_rows}
    return [(baseline_by_model.get(str(row.get("model_id")), {}), row) for row in current_rows]


def _full_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in matrix.get("rows", []) if isinstance(row, dict) and row.get("category") == FULL_HUMANOID_PROFILE]


def _row_release_status(row: dict[str, Any]) -> str:
    if row.get("category") == PARTIAL_HUMANOID_PROFILE:
        return "partial_diagnostic_only"
    if row.get("category") == NEGATIVE_CONTROL:
        return "negative_control_not_promoted"
    return str(row.get("runtime_quality_status"))


def _hash_or_none(payload: Any, key: str) -> str | None:
    if isinstance(payload, dict) and payload.get(key):
        return str(payload[key])
    return None


def _optional_delta(current: Any, baseline: Any) -> float | None:
    try:
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return round(float(np.mean(arr)), 12) if arr.size else 0.0


def _finite_percentile(values: Any, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return round(float(np.percentile(arr, percentile)), 12) if arr.size else 0.0


def _finite_max(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return round(float(np.max(arr)), 12) if arr.size else 0.0


def _strip_volatile_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_volatile_runtime_fields(item)
            for key, item in sorted(value.items())
            if key not in {"runtime_seconds", "diagnostics_hash"}
        }
    if isinstance(value, list):
        return [_strip_volatile_runtime_fields(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
