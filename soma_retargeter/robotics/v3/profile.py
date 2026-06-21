"""KinematicProfileV3 compiler orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

from .canonical_projection import canonical_projection_json, project_canonical_targets
from .kinematic_paths import TASKS, KinematicPath, discover_paths
from .model_adapter import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter, SemanticSite
from .numerical_jacobian import engine_translation_jacobian_crosscheck, numerical_relative_jacobian
from .reachability import ReachabilityReport, analyze_reachability
from .rest_frames import RestCalibration, calibrate_rest_frames, neutral_exactness_passed
from .semantic_sites import build_semantic_sites, missing_required_semantics
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
) -> KinematicProfileV3:
    start = time.perf_counter()
    failures: list[str] = []
    warnings: list[str] = []
    adapter = _make_adapter(model_path, model_format=model_format, backend=backend)
    try:
        sites = build_semantic_sites(adapter, semantic_map)
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
        canonical_projections = project_canonical_targets(adapter, sites, paths, canonical_objects)
        projection_reports = canonical_projection_json(canonical_projections)
        failures.extend(_projection_failures(paths, projection_reports))
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


def _projection_failures(paths: dict[str, KinematicPath], projection_reports: dict) -> list[str]:
    failures: list[str] = []
    if "neutral" not in projection_reports:
        failures.append("canonical projection coverage missing neutral motion")
    expected_motions = {"torso_pitch", "torso_roll", "torso_yaw", "arms_forward", "overhead_reach", "squat", "single_step_target"}
    missing_motions = sorted(expected_motions - set(projection_reports))
    if missing_motions:
        failures.append(f"canonical projection coverage missing motions: {', '.join(missing_motions)}")
    for motion, task_reports in projection_reports.items():
        missing_tasks = sorted(set(paths) - set(task_reports))
        if missing_tasks:
            failures.append(f"{motion}: canonical projection missing tasks: {', '.join(missing_tasks)}")
        for task, report in task_reports.items():
            if report.get("desired_source") != "canonical_semantic_target_relative_transform":
                failures.append(f"{motion}/{task}: projection desired_source is not canonical semantic target")
            if report.get("neutral_as_desired") is not False:
                failures.append(f"{motion}/{task}: neutral_as_desired must be false")
            if not report.get("active_coordinates") and "rank_zero" not in str(report.get("status", "")):
                failures.append(f"{motion}/{task}: rank-zero projection lacks rank_zero status")
    return failures
