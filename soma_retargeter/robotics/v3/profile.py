"""KinematicProfileV3 compiler orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from .canonical_projection import project_canonical_motion_sequence, project_temporal_motion_sequences
from .engine_jacobian import engine_relative_jacobian
from .kinematic_paths import TASKS, KinematicPath, discover_paths
from .model_adapter import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter, SemanticSite
from .numerical_jacobian import engine_translation_jacobian_crosscheck, numerical_relative_jacobian
from .reachability import ReachabilityReport, analyze_reachability
from .rest_frames import RestCalibration, calibrate_rest_frames, neutral_exactness_passed
from .semantic_sites import DISTAL_SEMANTICS, build_semantic_sites, missing_required_semantics
from .source_rest import load_soma_source_rest_frames
from .target_builder import canonical_motion_targets, validate_canonical_targets


@dataclass(frozen=True)
class KinematicProfileV3:
    schema_version: int
    model: dict
    runtime_adapter: dict
    semantic_sites: dict[str, SemanticSite]
    chains: dict[str, KinematicPath]
    neutral_jacobians: dict[str, dict]
    rank_stability: dict[str, ReachabilityReport]
    rest_calibration: RestCalibration
    canonical_targets: dict
    canonical_target_validation: dict
    canonical_projection_reports: dict
    temporal_projection_reports: dict
    projection_reports: dict
    failures: list[str]
    warnings: list[str]
    capability_status: str
    timing: dict[str, float]
    reproduction_command: str

    def to_json(self) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "model": self.model,
            "runtime_adapter": self.runtime_adapter,
            "semantic_sites": {k: v.to_json() for k, v in self.semantic_sites.items()},
            "chains": {k: v.to_json() for k, v in self.chains.items()},
            "neutral_jacobians": self.neutral_jacobians,
            "rank_stability": {k: v.to_json() for k, v in self.rank_stability.items()},
            "rest_calibration": self.rest_calibration.to_json(),
            "canonical_targets": self.canonical_targets,
            "canonical_target_validation": self.canonical_target_validation,
            "canonical_projection_reports": self.canonical_projection_reports,
            "temporal_projection_reports": self.temporal_projection_reports,
            "projection_reports": self.projection_reports,
            "failures": self.failures,
            "warnings": self.warnings,
            "capability_status": self.capability_status,
            "timing": self.timing,
            "reproduction_command": self.reproduction_command,
        }
        stable = dict(payload)
        stable["timing"] = {}
        stable["deterministic_hash"] = ""
        payload["deterministic_hash"] = hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


def compile_kinematic_profile_v3(
    model_path: str | Path,
    semantic_map: dict[str, str | dict],
    *,
    model_id: str | None = None,
    model_format: str | None = None,
    backend: str = "mujoco",
    low_discrepancy_count: int = 32,
    reproduction_command: str = "",
    require_distal_site_offsets: bool | None = None,
) -> KinematicProfileV3:
    start = time.perf_counter()
    failures: list[str] = []
    warnings: list[str] = []
    adapter = _make_adapter(model_path, model_format=model_format, backend=backend)
    try:
        if require_distal_site_offsets is None:
            require_distal_site_offsets = _requires_verified_distal_offsets(semantic_map)
        sites = build_semantic_sites(
            adapter,
            semantic_map,
            require_distal_site_offsets=require_distal_site_offsets,
        )
        missing = missing_required_semantics(sites)
        capability_status = _capability_status(sites, missing)
        if missing and capability_status == "partial_humanoid":
            warnings.append(f"partial humanoid downgrade; unavailable semantics: {', '.join(missing)}")
        elif missing:
            failures.append(f"missing required semantics: {', '.join(missing)}")
        paths = discover_paths(adapter, sites)
        q0 = adapter.neutral_q()
        neutral_jacobians = {}
        reachability = {}
        for task, path in paths.items():
            ref = sites[path.reference]
            target = sites[path.target]
            jac = numerical_relative_jacobian(adapter, q0, ref, target, path.active_velocity_coordinates)
            jac_json = jac.to_json()
            try:
                engine_jac = engine_relative_jacobian(adapter, q0, ref, target, path.active_velocity_coordinates)
                jac_json["engine_relative_jacobian"] = engine_jac.to_json()
                jac_json["primary_jacobian_source"] = "engine_relative_jacobian"
            except Exception as exc:
                jac_json["engine_relative_jacobian"] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                jac_json["primary_jacobian_source"] = "finite_difference_unavailable_engine"
            jac_json["engine_translation_crosscheck"] = engine_translation_jacobian_crosscheck(
                adapter, q0, ref, target, path.active_velocity_coordinates, jac.translation
            )
            neutral_jacobians[task] = jac_json
            reachability[task] = analyze_reachability(
                adapter,
                ref,
                target,
                path.active_velocity_coordinates,
                low_discrepancy_count=low_discrepancy_count,
                task_name=task,
            )
        try:
            source_rest_transforms, source_provenance = load_soma_source_rest_frames()
        except Exception as exc:
            source_rest_transforms = None
            source_provenance = "robot_neutral_proxy_no_external_source_rest_supplied"
            warnings.append(f"source rest load failed; using robot-neutral proxy: {type(exc).__name__}: {exc}")
        calibration = calibrate_rest_frames(
            adapter,
            sites,
            source_rest_transforms=source_rest_transforms,
            source_provenance=source_provenance,
        )
        if not neutral_exactness_passed(calibration):
            failures.append("neutral exactness gate failed")
        canonical_objects = canonical_motion_targets(calibration)
        canonical_validation = validate_canonical_targets(calibration, canonical_objects)
        if canonical_validation["failures"]:
            failures.extend(f"canonical target validation failed: {failure}" for failure in canonical_validation["failures"])
        canonical_projection = project_canonical_motion_sequence(
            adapter,
            sites,
            paths,
            canonical_objects,
            neutral_q=q0,
        )
        if canonical_projection.failures:
            failures.extend(f"canonical projection failed: {failure}" for failure in canonical_projection.failures)
        canonical_projection_json = canonical_projection.to_json()
        temporal_projection_json = project_temporal_motion_sequences(
            adapter,
            sites,
            paths,
            canonical_objects,
            neutral_q=q0,
        )
        quality_failures = _projection_quality_failures(canonical_projection_json)
        failures.extend(quality_failures)
        projection_reports = _motion_projection_reports(canonical_projection_json)
        canonical = {k: v.to_json() for k, v in canonical_objects.items()}
        model_payload = {
            "id": model_id or Path(model_path).stem,
            "path": str(model_path),
            "format": adapter.model_format,
            "backend": backend,
            "fingerprint": adapter.fingerprint,
        }
        runtime_payload = {
            "backend": backend,
            "nq": adapter.nq,
            "nv": adapter.nv,
            "body_count": len(adapter.body_names),
            "coordinates": [c.to_json() for c in adapter.coordinate_info],
        }
    finally:
        adapter.close()
    elapsed = time.perf_counter() - start
    return KinematicProfileV3(
        schema_version=3,
        model=model_payload,
        runtime_adapter=runtime_payload,
        semantic_sites=sites,
        chains=paths,
        neutral_jacobians=neutral_jacobians,
        rank_stability=reachability,
        rest_calibration=calibration,
        canonical_targets=canonical,
        canonical_target_validation=canonical_validation,
        canonical_projection_reports=canonical_projection_json,
        temporal_projection_reports=temporal_projection_json,
        projection_reports=projection_reports,
        failures=failures,
        warnings=warnings,
        capability_status=capability_status,
        timing={"compile_seconds": elapsed},
        reproduction_command=reproduction_command,
    )


def write_profile(profile: KinematicProfileV3, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_json(), indent=2, sort_keys=True) + "\n")


def _make_adapter(model_path: str | Path, *, model_format: str | None, backend: str):
    if backend == "newton":
        return NewtonRuntimeModelAdapter(model_path, model_format=model_format)
    if backend == "mujoco":
        return MuJoCoRuntimeModelAdapter(model_path, model_format=model_format)
    raise ValueError(f"unknown v3 kinematic backend {backend!r}")


def _capability_status(sites: dict[str, SemanticSite], missing: list[str]) -> str:
    if not missing:
        return "full_humanoid_ready"
    lower_body_ready = {"Hips", "Chest", "LeftFoot", "RightFoot"} <= set(sites)
    only_upper_missing = set(missing) <= {"LeftHand", "RightHand"}
    if lower_body_ready and only_upper_missing:
        return "partial_humanoid"
    return "semantic_incomplete"


def _requires_verified_distal_offsets(semantic_map: dict[str, str | dict]) -> bool:
    for semantic_name, entry in semantic_map.items():
        if semantic_name not in DISTAL_SEMANTICS or not isinstance(entry, dict):
            continue
        source = str(entry.get("source", ""))
        if source.startswith("verified"):
            return True
    return False


def _motion_projection_reports(canonical_projection: dict) -> dict:
    reports: dict[str, dict] = {}
    for motion_name, motion in canonical_projection.get("motions", {}).items():
        task_reports = {}
        for task_name, task_report in motion.get("tasks", {}).items():
            task_payload = dict(task_report)
            task_payload["motion"] = motion_name
            task_payload["task"] = task_name
            task_payload["reference_semantic"] = task_payload.pop("reference", None)
            task_payload["target_semantic"] = task_payload.pop("target", None)
            task_payload["canonical_desired_source"] = task_payload.get("desired_source")
            task_payload["desired_source"] = "canonical_semantic_target_relative_transform"
            task_payload["neutral_as_desired"] = False
            task_reports[task_name] = task_payload
        reports[motion_name] = task_reports
    return reports


def _projection_quality_failures(canonical_projection: dict) -> list[str]:
    failures: list[str] = []
    motions = canonical_projection.get("motions", {})
    if not isinstance(motions, dict):
        return failures
    for motion_name, motion in sorted(motions.items()):
        tasks = motion.get("tasks", {}) if isinstance(motion, dict) else {}
        if not isinstance(tasks, dict):
            continue
        for task_name, payload in sorted(tasks.items()):
            if not isinstance(payload, dict):
                continue
            if payload.get("status") in {"rank_zero", "unreachable/rank_zero"}:
                continue
            if motion_name == "extreme_but_valid_joint_limit_stress":
                continue
            normalized = payload.get("normalized_residual")
            residual = payload.get("residual")
            threshold = _projection_quality_threshold(str(task_name), str(motion_name))
            if normalized is None or not np.isfinite(float(normalized)):
                failures.append(f"projection residual gate failed: {motion_name}.{task_name} normalized_residual nonfinite")
                continue
            if float(normalized) > threshold:
                failures.append(
                    "projection residual gate failed: "
                    f"{motion_name}.{task_name} normalized_residual={float(normalized):.6g} threshold={threshold:.6g} residual={float(residual or 0.0):.6g}"
                )
    return failures


def _projection_quality_threshold(task_name: str, motion_name: str) -> float:
    if motion_name == "neutral":
        return 1e-3
    if "foot" in task_name:
        return 0.06
    if "hand" in task_name:
        return 0.12
    if "torso" in task_name:
        return 0.08
    return 0.05
