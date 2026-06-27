"""Generic runtime FK smoke for Step 3.1 full-fleet evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from soma_retargeter.robotics.v3.engine_jacobian import engine_relative_jacobian
from soma_retargeter.robotics.v3.kinematic_paths import TASKS, KinematicPath, discover_paths
from soma_retargeter.robotics.v3.model_adapter import NewtonRuntimeModelAdapter, SemanticSite
from soma_retargeter.robotics.v3.numerical_jacobian import numerical_relative_jacobian
from soma_retargeter.robotics.v3.projection_solver import so3_log_residual_jacobian
from soma_retargeter.robotics.v3.spatial import relative_transform, rotation_error, so3_log

from .fleet_inventory import FleetRuntimeCase, stable_payload_hash
from .quality_metrics import contact_diagnostics, smoke_output_metrics
from .runtime_quality_gates import (
    BLOCKED_SOURCE_OR_PROFILE,
    NEGATIVE_CONTROL_RUNTIME_PASSED,
    PARTIAL_RUNTIME_PASSED,
    RESIDUAL_ONLY_SOLVER_TYPE,
    RUNTIME_QUALITY_FAILED,
    classify_runtime_quality,
)


SOLVER_BACKED_GENERIC_SOLVER_TYPE = "generic_chain_projection_least_squares_smoke"


@dataclass(frozen=True)
class SolverBackedSmokeConfig:
    sample_count: int = 1
    max_nfev_per_task: int = 12
    xtol: float = 1e-8
    ftol: float = 1e-8
    gtol: float = 1e-8
    task_order: tuple[str, ...] = ("torso",)

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "max_nfev_per_task": int(self.max_nfev_per_task),
            "xtol": float(self.xtol),
            "ftol": float(self.ftol),
            "gtol": float(self.gtol),
            "task_order": list(self.task_order),
            "global_config": True,
        }


@dataclass(frozen=True)
class GenericSmokeResult:
    status: str
    mode: str
    metrics: dict[str, Any]
    residuals: dict[str, Any]
    error: str | None = None
    solver_type: str | None = None
    solver_backed: bool = False
    residual_only: bool = False
    quality_pass_allowed: bool = False
    quality_classification: str | None = None
    classification_reason: str | None = None
    quality_gate_results: dict[str, Any] | None = None
    failure_or_warning_reasons: list[str] | None = None
    solver_backed_smoke_attempted: bool = False
    solver_backed_smoke_completed: bool = False
    solver_failure_reason: str | None = None
    sampled_frame_indices: list[int] | None = None
    deterministic_hash_inputs: dict[str, Any] | None = None
    task_diagnostics: list[dict[str, Any]] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "solver_type": self.solver_type or self.residuals.get("solver"),
            "solver_backed": self.solver_backed,
            "residual_only": self.residual_only,
            "quality_pass_allowed": self.quality_pass_allowed,
            "quality_classification": self.quality_classification or self.status,
            "classification_reason": self.classification_reason,
            "quality_gate_results": self.quality_gate_results or {},
            "failure_or_warning_reasons": list(self.failure_or_warning_reasons or []),
            "solver_backed_smoke_attempted": bool(self.solver_backed_smoke_attempted),
            "solver_backed_smoke_completed": bool(self.solver_backed_smoke_completed),
            "solver_failure_reason": self.solver_failure_reason,
            "sampled_frame_indices": list(self.sampled_frame_indices or []),
            "deterministic_hash_inputs": dict(self.deterministic_hash_inputs or {}),
            "task_diagnostics": list(self.task_diagnostics or []),
            "metrics": self.metrics,
            "residuals": self.residuals,
            "error": self.error,
        }


def run_full_humanoid_fk_smoke(
    case: FleetRuntimeCase,
    target_transforms: Mapping[str, np.ndarray],
    *,
    adapter: NewtonRuntimeModelAdapter | None = None,
    profile: dict[str, Any] | None = None,
) -> GenericSmokeResult:
    """Evaluate target stream against runtime FK at neutral pose.

    This is intentionally a generic runtime smoke, not production retargeting:
    it loads the runtime model, performs FK with runtime truth, and quantifies
    residuals against the V3 target stream without per-robot tuning.
    """

    owns_adapter = adapter is None
    adapter = adapter or NewtonRuntimeModelAdapter(case.runtime_source_path, model_format=case.model_format)
    profile = profile or case.profile
    started = time.perf_counter()
    try:
        q0 = adapter.neutral_q()
        state = adapter.forward_kinematics(q0)
        sites = _semantic_sites_from_profile(profile)
        frame_count = _frame_count(target_transforms)
        residual_values: list[float] = []
        per_semantic: dict[str, dict[str, Any]] = {}
        for semantic, target_stack in sorted(target_transforms.items()):
            if semantic not in sites:
                continue
            runtime_transform = adapter.site_transform(state, sites[semantic])
            stack = np.asarray(target_stack, dtype=np.float64)
            translation = np.linalg.norm(stack[:, :3, 3] - runtime_transform[:3, 3], axis=1)
            rotation = np.asarray(
                [rotation_error(runtime_transform[:3, :3], stack[index, :3, :3]) for index in range(stack.shape[0])],
                dtype=np.float64,
            )
            combined = translation + rotation
            residual_values.extend(float(v) for v in combined)
            per_semantic[semantic] = {
                "translation_residual_mean": float(np.mean(translation)) if translation.size else 0.0,
                "translation_residual_p95": float(np.percentile(translation, 95)) if translation.size else 0.0,
                "translation_residual_max": float(np.max(translation)) if translation.size else 0.0,
                "rotation_residual_mean": float(np.mean(rotation)) if rotation.size else 0.0,
                "rotation_residual_p95": float(np.percentile(rotation, 95)) if rotation.size else 0.0,
                "rotation_residual_max": float(np.max(rotation)) if rotation.size else 0.0,
            }
        q_sequence = np.repeat(q0[None, :], frame_count, axis=0)
        metrics = smoke_output_metrics(
            q_sequence=q_sequence,
            coordinate_info=adapter.coordinate_info,
            residuals=np.asarray(residual_values, dtype=np.float64),
            runtime_seconds=time.perf_counter() - started,
            solver_iterations=np.zeros(frame_count),
        )
        metrics.update(contact_diagnostics(target_transforms))
        metrics["target_se3_orthogonality_error_max"] = _target_se3_orthogonality_error_max(target_transforms)
        classification = classify_runtime_quality(
            metrics,
            solver_type=RESIDUAL_ONLY_SOLVER_TYPE,
            solver_backed=False,
        )
        metrics.update(_classification_metrics(classification))
        return GenericSmokeResult(
            status=str(classification["classification"]),
            mode="generic_fk_residual_smoke",
            metrics=metrics,
            residuals={
                "solver": RESIDUAL_ONLY_SOLVER_TYPE,
                "solver_type": RESIDUAL_ONLY_SOLVER_TYPE,
                "per_semantic": per_semantic,
                "residual_sample_count": len(residual_values),
            },
            solver_type=str(classification["solver_type"]),
            solver_backed=bool(classification["solver_backed"]),
            residual_only=bool(classification["residual_only"]),
            quality_pass_allowed=bool(classification["quality_pass_allowed"]),
            quality_classification=str(classification["quality_classification"]),
            classification_reason=str(classification["classification_reason"]),
            quality_gate_results=dict(classification["quality_gate_results"]),
            failure_or_warning_reasons=list(classification["failure_or_warning_reasons"]),
        )
    except Exception as exc:
        metrics = {
            "output_frame_count": 0,
            "joint_coord_count": int(getattr(adapter, "nq", 0)),
            "nan_count": 0,
            "inf_count": 0,
            "output_finite": False,
            "joint_limit_violation_count": 0,
            "max_joint_limit_violation": 0.0,
            "normalized_task_residual_mean": 0.0,
            "normalized_task_residual_p95": 0.0,
            "normalized_task_residual_max": 0.0,
            "solver_success_fraction": 0.0,
            "runtime_seconds": 0.0,
        }
        classification = classify_runtime_quality(
            metrics,
            solver_type=RESIDUAL_ONLY_SOLVER_TYPE,
            solver_backed=False,
        )
        metrics.update(_classification_metrics(classification))
        metrics["classification_reason"] = f"runtime_model_fk_residual_evaluation_failed: {type(exc).__name__}: {exc}"
        return GenericSmokeResult(
            status=RUNTIME_QUALITY_FAILED,
            mode="generic_fk_residual_smoke",
            metrics=metrics,
            residuals={"solver": RESIDUAL_ONLY_SOLVER_TYPE, "solver_type": RESIDUAL_ONLY_SOLVER_TYPE, "per_semantic": {}},
            error=f"{type(exc).__name__}: {exc}",
            solver_type=str(classification["solver_type"]),
            solver_backed=bool(classification["solver_backed"]),
            residual_only=bool(classification["residual_only"]),
            quality_pass_allowed=bool(classification["quality_pass_allowed"]),
            quality_classification=RUNTIME_QUALITY_FAILED,
            classification_reason=str(metrics["classification_reason"]),
            quality_gate_results=dict(classification["quality_gate_results"]),
            failure_or_warning_reasons=list(classification["failure_or_warning_reasons"])
            or ["runtime_model_fk_residual_evaluation_failed"],
        )
    finally:
        if owns_adapter:
            adapter.close()


def run_full_humanoid_solver_backed_smoke(
    case: FleetRuntimeCase,
    target_transforms: Mapping[str, np.ndarray],
    *,
    adapter: NewtonRuntimeModelAdapter | None = None,
    profile: dict[str, Any] | None = None,
    config: SolverBackedSmokeConfig | None = None,
) -> GenericSmokeResult:
    """Run an isolated globally bounded solver-backed projection smoke."""

    cfg = config or SolverBackedSmokeConfig()
    owns_adapter = adapter is None
    adapter = adapter or NewtonRuntimeModelAdapter(case.runtime_source_path, model_format=case.model_format)
    profile = profile or case.profile
    started = time.perf_counter()
    sampled_frame_indices: list[int] = []
    task_diagnostics: list[dict[str, Any]] = []
    try:
        sites = _semantic_sites_from_profile(profile)
        missing_sites = sorted(set(_target_semantics(target_transforms)) - set(sites))
        if missing_sites:
            return _solver_backed_failed_result(
                adapter=adapter,
                started=started,
                reason="solver_anchor_sites_missing:" + ",".join(missing_sites),
                sampled_frame_indices=[],
                task_diagnostics=[],
                config=cfg,
            )

        paths = discover_paths(adapter, sites)
        missing_tasks = [task for task in cfg.task_order if task in TASKS and task not in paths]
        frame_count = _frame_count(target_transforms)
        sampled_frame_indices = _sampled_frame_indices(frame_count, cfg.sample_count)
        if not sampled_frame_indices:
            return _solver_backed_failed_result(
                adapter=adapter,
                started=started,
                reason="target_stream_empty",
                sampled_frame_indices=[],
                task_diagnostics=[],
                config=cfg,
            )

        q0 = adapter.neutral_q()
        q_sequence: list[np.ndarray] = []
        residual_values: list[float] = []
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        solver_iterations: list[float] = []
        solver_successes = 0
        solver_tasks = 0

        for frame_index in sampled_frame_indices:
            q = q0.copy()
            frame_task_reports: list[dict[str, Any]] = []
            for task in cfg.task_order:
                path = paths.get(task)
                if path is None:
                    continue
                if path.reference not in target_transforms or path.target not in target_transforms:
                    continue
                desired_relative = relative_transform(
                    np.asarray(target_transforms[path.reference][frame_index], dtype=np.float64),
                    np.asarray(target_transforms[path.target][frame_index], dtype=np.float64),
                )
                task_result = _solve_projection_task(
                    adapter,
                    q,
                    sites[path.reference],
                    sites[path.target],
                    path,
                    desired_relative,
                    config=cfg,
                )
                solver_tasks += 1
                solver_iterations.append(float(task_result["iterations"]))
                if task_result["success"]:
                    solver_successes += 1
                q = np.asarray(task_result["q"], dtype=np.float64)
                frame_task_reports.append({key: value for key, value in task_result.items() if key != "q"})

            state = adapter.forward_kinematics(q)
            per_semantic: dict[str, dict[str, Any]] = {}
            for semantic, stack_value in sorted(target_transforms.items()):
                if semantic not in sites:
                    continue
                stack = np.asarray(stack_value, dtype=np.float64)
                runtime_transform = adapter.site_transform(state, sites[semantic])
                target_transform = stack[frame_index]
                translation = float(np.linalg.norm(runtime_transform[:3, 3] - target_transform[:3, 3]))
                rotation = rotation_error(runtime_transform[:3, :3], target_transform[:3, :3])
                translation_errors.append(translation)
                rotation_errors.append(rotation)
                residual_values.append(translation + rotation)
                per_semantic[semantic] = {
                    "translation_residual": _stable(translation),
                    "rotation_residual": _stable(rotation),
                    "combined_residual": _stable(translation + rotation),
                }
            q_sequence.append(q)
            task_diagnostics.append(
                {
                    "frame_index": int(frame_index),
                    "tasks": frame_task_reports,
                    "per_semantic": per_semantic,
                }
            )

        if solver_tasks <= 0 or not q_sequence:
            reason = "solver_no_projectable_tasks"
            if missing_tasks:
                reason += ":" + ",".join(missing_tasks)
            return _solver_backed_failed_result(
                adapter=adapter,
                started=started,
                reason=reason,
                sampled_frame_indices=sampled_frame_indices,
                task_diagnostics=task_diagnostics,
                config=cfg,
            )

        q_arr = np.vstack(q_sequence)
        metrics = smoke_output_metrics(
            q_sequence=q_arr,
            coordinate_info=adapter.coordinate_info,
            residuals=np.asarray(residual_values, dtype=np.float64),
            runtime_seconds=time.perf_counter() - started,
            solver_iterations=np.asarray(solver_iterations, dtype=np.float64),
        )
        sampled_targets = {
            name: np.asarray(value, dtype=np.float64)[sampled_frame_indices]
            for name, value in target_transforms.items()
        }
        metrics.update(contact_diagnostics(sampled_targets))
        metrics.update(_summary_fields("target_translation_error", translation_errors))
        metrics.update(_summary_fields("target_rotation_error", rotation_errors))
        metrics["target_se3_orthogonality_error_max"] = _target_se3_orthogonality_error_max(target_transforms)
        metrics["solver_success_fraction"] = _stable(float(solver_successes / solver_tasks))
        metrics["solver_task_count"] = int(solver_tasks)
        metrics["solver_task_success_count"] = int(solver_successes)
        metrics["solver_backed_smoke_attempted"] = True
        metrics["solver_backed_smoke_completed"] = True
        metrics["sampled_frame_count"] = len(sampled_frame_indices)
        metrics["sampled_frame_indices"] = list(sampled_frame_indices)
        classification = classify_runtime_quality(
            metrics,
            solver_type=SOLVER_BACKED_GENERIC_SOLVER_TYPE,
            solver_backed=True,
            residual_only=False,
        )
        warning_reasons = list(classification["failure_or_warning_reasons"])
        if solver_successes < solver_tasks and "solver_convergence_weak" not in warning_reasons:
            warning_reasons.append("solver_convergence_weak")
        if missing_tasks and "solver_task_anchor_missing" not in warning_reasons:
            warning_reasons.append("solver_task_anchor_missing")
        metrics.update(_classification_metrics({**classification, "failure_or_warning_reasons": warning_reasons}))
        deterministic_hash_inputs = _solver_deterministic_hash_inputs(
            case=case,
            sampled_frame_indices=sampled_frame_indices,
            config=cfg,
            task_diagnostics=task_diagnostics,
            metrics={key: value for key, value in metrics.items() if key != "runtime_seconds"},
        )
        return GenericSmokeResult(
            status=str(classification["classification"]),
            mode="solver_backed_generic_chain_projection_smoke",
            metrics=metrics,
            residuals={
                "solver": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
                "solver_type": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
                "sampled_frame_indices": sampled_frame_indices,
                "frame_reports": task_diagnostics,
                "missing_tasks": missing_tasks,
            },
            solver_type=SOLVER_BACKED_GENERIC_SOLVER_TYPE,
            solver_backed=True,
            residual_only=False,
            quality_pass_allowed=bool(classification["quality_pass_allowed"]),
            quality_classification=str(classification["quality_classification"]),
            classification_reason=str(classification["classification_reason"]),
            quality_gate_results=dict(classification["quality_gate_results"]),
            failure_or_warning_reasons=warning_reasons,
            solver_backed_smoke_attempted=True,
            solver_backed_smoke_completed=True,
            sampled_frame_indices=sampled_frame_indices,
            deterministic_hash_inputs=deterministic_hash_inputs,
            task_diagnostics=task_diagnostics,
        )
    except Exception as exc:
        return _solver_backed_failed_result(
            adapter=adapter,
            started=started,
            reason=f"{type(exc).__name__}: {exc}",
            sampled_frame_indices=sampled_frame_indices,
            task_diagnostics=task_diagnostics,
            config=cfg,
        )
    finally:
        if owns_adapter:
            adapter.close()


def run_partial_supported_smoke(
    case: FleetRuntimeCase,
    source_transforms: Mapping[str, np.ndarray],
    *,
    adapter: NewtonRuntimeModelAdapter | None = None,
) -> GenericSmokeResult:
    owns_adapter = adapter is None
    adapter = adapter or NewtonRuntimeModelAdapter(case.runtime_source_path, model_format=case.model_format)
    started = time.perf_counter()
    try:
        frame_count = _frame_count(source_transforms)
        q0 = adapter.neutral_q()
        q_sequence = np.repeat(q0[None, :], frame_count, axis=0)
        metrics = smoke_output_metrics(
            q_sequence=q_sequence,
            coordinate_info=adapter.coordinate_info,
            residuals=np.zeros(frame_count, dtype=np.float64),
            runtime_seconds=time.perf_counter() - started,
        )
        metrics.update(contact_diagnostics(source_transforms))
        return GenericSmokeResult(
            status="passed",
            mode="partial_supported_semantic_fk_smoke",
            metrics=metrics,
            residuals={
                "solver": "partial_supported_semantic_runtime_load_and_fk",
                "supported_semantics": list(case.supported_semantics),
                "missing_required_semantics": list(case.missing_required_semantics),
                "full_override_attempted": False,
            },
            solver_type="partial_supported_semantic_runtime_load_and_fk",
            solver_backed=False,
            residual_only=False,
            quality_pass_allowed=False,
            quality_classification=PARTIAL_RUNTIME_PASSED,
            classification_reason="partial_supported_semantics_smoke_completed",
        )
    except Exception as exc:
        return GenericSmokeResult(
            status="failed",
            mode="partial_supported_semantic_fk_smoke",
            metrics={"output_frame_count": 0, "joint_coord_count": 0, "nan_count": 0, "inf_count": 0, "runtime_seconds": 0.0},
            residuals={"solver": "partial_supported_semantic_runtime_load_and_fk"},
            error=f"{type(exc).__name__}: {exc}",
            solver_type="partial_supported_semantic_runtime_load_and_fk",
            solver_backed=False,
            residual_only=False,
            quality_pass_allowed=False,
            quality_classification=BLOCKED_SOURCE_OR_PROFILE,
            classification_reason="partial_supported_semantic_runtime_load_failed",
            failure_or_warning_reasons=["partial_supported_semantic_runtime_load_failed"],
        )
    finally:
        if owns_adapter:
            adapter.close()


def run_negative_control_rejection_smoke(case: FleetRuntimeCase) -> GenericSmokeResult:
    started = time.perf_counter()
    try:
        adapter = NewtonRuntimeModelAdapter(case.runtime_source_path, model_format=case.model_format)
    except Exception as exc:
        return GenericSmokeResult(
            status="failed",
            mode="negative_control_rejection",
            metrics={"output_frame_count": 0, "joint_coord_count": 0, "nan_count": 0, "inf_count": 0, "runtime_seconds": 0.0},
            residuals={"solver": "negative_control_runtime_load", "humanoid_profile_generated": False},
            error=f"{type(exc).__name__}: {exc}",
            solver_type="negative_control_runtime_load",
            solver_backed=False,
            residual_only=False,
            quality_pass_allowed=False,
            quality_classification=BLOCKED_SOURCE_OR_PROFILE,
            classification_reason="negative_control_runtime_load_failed",
            failure_or_warning_reasons=["negative_control_runtime_load_failed"],
        )
    try:
        q0 = adapter.neutral_q()
        adapter.forward_kinematics(q0)
        metrics = smoke_output_metrics(
            q_sequence=q0,
            coordinate_info=adapter.coordinate_info,
            residuals=np.zeros(1, dtype=np.float64),
            runtime_seconds=time.perf_counter() - started,
        )
        return GenericSmokeResult(
            status="passed",
            mode="negative_control_rejection",
            metrics=metrics,
            residuals={
                "solver": "negative_control_runtime_load_and_reject_humanoid_profile",
                "humanoid_profile_generated": False,
                "target_stream_override_generated": False,
            },
            solver_type="negative_control_runtime_load_and_reject_humanoid_profile",
            solver_backed=False,
            residual_only=False,
            quality_pass_allowed=False,
            quality_classification=NEGATIVE_CONTROL_RUNTIME_PASSED,
            classification_reason="negative_control_loaded_and_rejected_from_humanoid_quality",
        )
    finally:
        adapter.close()


def _solve_projection_task(
    adapter: NewtonRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    path: KinematicPath,
    desired_relative: np.ndarray,
    *,
    config: SolverBackedSmokeConfig,
) -> dict[str, Any]:
    active, x0, lower, upper = _active_values_and_bounds(adapter, q_seed, path.active_velocity_coordinates)
    task_kind = "rotation" if path.task == "torso" else "translation"
    scale = math.pi if task_kind == "rotation" else _position_scale(adapter, q_seed, reference, target, active, desired_relative)

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        q = adapter.set_velocity_coordinates(q_seed, active, values) if active else np.asarray(q_seed, dtype=np.float64)
        current = adapter.relative_transform(adapter.forward_kinematics(q), reference, target)
        jacobian_source = "rank_zero_no_active_coordinates"
        if task_kind == "rotation":
            error = so3_log(current[:3, :3].T @ desired_relative[:3, :3])
            if active:
                jac = _relative_jacobian(adapter, q, reference, target, active)
                jacobian_source = str(jac["source"])
                task_jac = so3_log_residual_jacobian(current[:3, :3], error, jac["rotation"]) / scale
            else:
                task_jac = np.zeros((3, 0), dtype=np.float64)
            return error / scale, task_jac, q, jacobian_source
        residual = current[:3, 3] - desired_relative[:3, 3]
        if active:
            jac = _relative_jacobian(adapter, q, reference, target, active)
            jacobian_source = str(jac["source"])
            task_jac = jac["translation"] / scale
        else:
            task_jac = np.zeros((3, 0), dtype=np.float64)
        return residual / scale, task_jac, q, jacobian_source

    seed_residual, _, seed_q, jacobian_source = evaluate(x0)
    if not active:
        residual_norm = float(np.linalg.norm(seed_residual))
        return {
            "task": path.task,
            "reference": path.reference,
            "target": path.target,
            "task_kind": task_kind,
            "success": bool(residual_norm <= 1e-10),
            "status": "rank_zero" if residual_norm <= 1e-10 else "unreachable_rank_zero",
            "iterations": 0,
            "nfev": 0,
            "active_coordinate_count": 0,
            "residual_norm": _stable(residual_norm),
            "residual_norm_seed": _stable(residual_norm),
            "jacobian_source": jacobian_source,
            "q": seed_q,
        }

    def residual_fun(values: np.ndarray) -> np.ndarray:
        residual, _, _, _ = evaluate(values)
        return residual

    def jacobian_fun(values: np.ndarray) -> np.ndarray:
        _, jacobian, _, _ = evaluate(values)
        return jacobian

    try:
        result = least_squares(
            residual_fun,
            x0,
            jac=jacobian_fun,
            bounds=(lower, upper),
            xtol=config.xtol,
            ftol=config.ftol,
            gtol=config.gtol,
            max_nfev=config.max_nfev_per_task,
        )
        final_residual, final_jacobian, final_q, jacobian_source = evaluate(result.x)
        finite = bool(
            np.all(np.isfinite(result.x))
            and np.all(np.isfinite(final_residual))
            and np.all(np.isfinite(final_jacobian))
            and np.all(np.isfinite(final_q))
        )
        residual_norm = float(np.linalg.norm(final_residual))
        seed_norm = float(np.linalg.norm(seed_residual))
        success = bool(finite and residual_norm <= seed_norm + 1e-9)
        return {
            "task": path.task,
            "reference": path.reference,
            "target": path.target,
            "task_kind": task_kind,
            "success": success,
            "status": "solver_completed" if success else "solver_completed_with_residual_or_limit",
            "iterations": int(result.nfev),
            "nfev": int(result.nfev),
            "active_coordinate_count": len(active),
            "active_velocity_coordinates": list(active),
            "residual_norm": _stable(residual_norm),
            "residual_norm_seed": _stable(seed_norm),
            "cost": _stable(float(result.cost)),
            "optimality": _stable(float(result.optimality)),
            "jacobian_source": jacobian_source,
            "message": str(result.message),
            "q": final_q,
        }
    except Exception as exc:
        seed_norm = float(np.linalg.norm(seed_residual))
        return {
            "task": path.task,
            "reference": path.reference,
            "target": path.target,
            "task_kind": task_kind,
            "success": False,
            "status": "solver_failed_closed",
            "iterations": 0,
            "nfev": 0,
            "active_coordinate_count": len(active),
            "active_velocity_coordinates": list(active),
            "residual_norm": _stable(seed_norm),
            "residual_norm_seed": _stable(seed_norm),
            "jacobian_source": jacobian_source,
            "message": f"{type(exc).__name__}: {exc}",
            "q": seed_q,
        }


def _active_values_and_bounds(
    adapter: NewtonRuntimeModelAdapter,
    q_seed: np.ndarray,
    active_coordinates: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    active: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type in {"revolute", "prismatic"}:
            value = float(q_seed[info.qpos_adr])
        else:
            value = 0.0
        lo = float(info.lower) if np.isfinite(info.lower) else -math.pi
        hi = float(info.upper) if np.isfinite(info.upper) else math.pi
        if hi - lo <= 1e-12:
            continue
        active.append(int(dof))
        values.append(float(np.clip(value, lo, hi)))
        lower.append(lo)
        upper.append(hi)
    return active, np.asarray(values, dtype=np.float64), np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _relative_jacobian(
    adapter: NewtonRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> dict[str, np.ndarray | str]:
    try:
        engine = engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
        return {"translation": engine.translation, "rotation": engine.rotation, "source": engine.source}
    except Exception:
        fd = numerical_relative_jacobian(
            adapter,
            q,
            reference,
            target,
            active_coordinates,
            engine_validation=False,
        )
        return {"translation": fd.translation, "rotation": fd.rotation, "source": "finite_difference_fallback"}


def _position_scale(
    adapter: NewtonRuntimeModelAdapter,
    q_seed: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    desired_relative: np.ndarray,
) -> float:
    state = adapter.forward_kinematics(q_seed)
    try:
        body_path = adapter.body_path(reference.body_name, target.body_name)
    except Exception:
        body_path = []
    points = [adapter.site_transform(state, reference)[:3, 3]]
    for body_name in body_path:
        points.append(adapter.body_transform(state, body_name)[:3, 3])
    points.append(adapter.site_transform(state, target)[:3, 3])
    length = 0.0
    for lhs, rhs in zip(points, points[1:]):
        length += float(np.linalg.norm(np.asarray(rhs) - np.asarray(lhs)))
    direct = float(np.linalg.norm(points[-1] - points[0])) if len(points) >= 2 else 0.0
    desired = float(np.linalg.norm(np.asarray(desired_relative, dtype=np.float64)[:3, 3]))
    prismatic_span = 0.0
    for dof in active_coordinates:
        info = adapter.coordinate(dof)
        if info.joint_type == "prismatic" and np.isfinite(info.lower) and np.isfinite(info.upper):
            prismatic_span += abs(float(info.upper) - float(info.lower))
    return max(length, direct, desired, prismatic_span, 1e-6)


def _solver_backed_failed_result(
    *,
    adapter: NewtonRuntimeModelAdapter,
    started: float,
    reason: str,
    sampled_frame_indices: list[int],
    task_diagnostics: list[dict[str, Any]],
    config: SolverBackedSmokeConfig,
) -> GenericSmokeResult:
    metrics = {
        "output_frame_count": 0,
        "joint_coord_count": int(getattr(adapter, "nq", 0)),
        "nan_count": 0,
        "inf_count": 0,
        "output_finite": False,
        "joint_limit_violation_count": 0,
        "max_joint_limit_violation": 0.0,
        "normalized_task_residual_mean": 0.0,
        "normalized_task_residual_p95": 0.0,
        "normalized_task_residual_max": 0.0,
        "solver_success_fraction": 0.0,
        "runtime_seconds": _stable(time.perf_counter() - started),
        "solver_backed_smoke_attempted": True,
        "solver_backed_smoke_completed": False,
        "sampled_frame_indices": list(sampled_frame_indices),
        "sampled_frame_count": len(sampled_frame_indices),
    }
    classification = classify_runtime_quality(
        metrics,
        solver_type=SOLVER_BACKED_GENERIC_SOLVER_TYPE,
        solver_backed=True,
        residual_only=False,
    )
    reasons = _dedupe([*classification["failure_or_warning_reasons"], "solver_failed_closed"])
    metrics.update(_classification_metrics({**classification, "failure_or_warning_reasons": reasons}))
    metrics["classification_reason"] = f"solver_failed_closed:{reason}"
    deterministic_hash_inputs = {
        "schema_version": 1,
        "solver_type": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
        "solver_config": config.to_json(),
        "sampled_frame_indices": list(sampled_frame_indices),
        "solver_failure_reason": str(reason),
        "stable_hash": stable_payload_hash(
            {
                "solver_type": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
                "solver_config": config.to_json(),
                "sampled_frame_indices": list(sampled_frame_indices),
                "solver_failure_reason": str(reason),
            }
        ),
    }
    return GenericSmokeResult(
        status=RUNTIME_QUALITY_FAILED,
        mode="solver_backed_generic_chain_projection_smoke",
        metrics=metrics,
        residuals={
            "solver": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
            "solver_type": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
            "sampled_frame_indices": sampled_frame_indices,
            "frame_reports": task_diagnostics,
        },
        error=str(reason),
        solver_type=SOLVER_BACKED_GENERIC_SOLVER_TYPE,
        solver_backed=False,
        residual_only=False,
        quality_pass_allowed=False,
        quality_classification=RUNTIME_QUALITY_FAILED,
        classification_reason=str(metrics["classification_reason"]),
        quality_gate_results=dict(classification["quality_gate_results"]),
        failure_or_warning_reasons=reasons,
        solver_backed_smoke_attempted=True,
        solver_backed_smoke_completed=False,
        solver_failure_reason=str(reason),
        sampled_frame_indices=sampled_frame_indices,
        deterministic_hash_inputs=deterministic_hash_inputs,
        task_diagnostics=task_diagnostics,
    )


def _semantic_sites_from_profile(profile: dict[str, Any]) -> dict[str, SemanticSite]:
    sites = {}
    for semantic, payload in profile.get("semantic_sites", {}).items():
        if not isinstance(payload, dict):
            continue
        sites[str(semantic)] = SemanticSite(
            semantic_name=str(semantic),
            body_name=str(payload["body_name"]),
            local_position=np.asarray(payload["local_position"], dtype=np.float64),
            local_rotation_xyzw=np.asarray(payload["local_rotation_xyzw"], dtype=np.float64),
            source=str(payload.get("source", "step2_profile")),
            confidence=float(payload.get("confidence", 1.0)),
            reason=str(payload.get("reason", "step2_profile")),
            evidence=tuple(str(v) for v in payload.get("evidence", [])),
            provenance=dict(payload.get("provenance", {})),
        )
    return sites


def _frame_count(transforms: Mapping[str, np.ndarray]) -> int:
    for value in transforms.values():
        return int(np.asarray(value).shape[0])
    return 0


def _target_semantics(transforms: Mapping[str, np.ndarray]) -> list[str]:
    return [str(key) for key in transforms]


def _sampled_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    frame_count = int(frame_count)
    sample_count = max(0, int(sample_count))
    if frame_count <= 0 or sample_count <= 0:
        return []
    if sample_count >= frame_count:
        return list(range(frame_count))
    if sample_count == 1:
        return [frame_count // 2]
    values = np.linspace(0, frame_count - 1, sample_count)
    out: list[int] = []
    for value in values:
        index = int(round(float(value)))
        if index not in out:
            out.append(index)
    return out


def _target_se3_orthogonality_error_max(transforms: Mapping[str, np.ndarray]) -> float:
    orthogonality_error_max = 0.0
    for stack_value in transforms.values():
        stack = np.asarray(stack_value, dtype=np.float64)
        if stack.ndim != 3 or stack.shape[1:] != (4, 4) or stack.shape[0] == 0:
            continue
        rotations = stack[:, :3, :3]
        errors = np.linalg.norm(np.matmul(np.swapaxes(rotations, 1, 2), rotations) - np.eye(3), axis=(1, 2))
        orthogonality_error_max = max(orthogonality_error_max, float(np.max(errors)) if errors.size else 0.0)
    if abs(orthogonality_error_max) < 1e-15:
        return 0.0
    return round(orthogonality_error_max, 12)


def _summary_fields(prefix: str, values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": _stable(float(np.mean(arr))),
        f"{prefix}_p95": _stable(float(np.percentile(arr, 95))),
        f"{prefix}_max": _stable(float(np.max(arr))),
    }


def _solver_deterministic_hash_inputs(
    *,
    case: FleetRuntimeCase,
    sampled_frame_indices: list[int],
    config: SolverBackedSmokeConfig,
    task_diagnostics: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "model_id": case.model_id,
        "runtime_source_sha256": case.runtime_source_sha256,
        "solver_type": SOLVER_BACKED_GENERIC_SOLVER_TYPE,
        "solver_config": config.to_json(),
        "sampled_frame_indices": list(sampled_frame_indices),
        "task_diagnostics": task_diagnostics,
        "metrics_without_runtime_seconds": metrics,
    }
    return {**payload, "stable_hash": stable_payload_hash(payload)}


def _stable(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    if abs(float(value)) < 1e-15:
        return 0.0
    return round(float(value), 12)


def _classification_metrics(classification: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "solver_type": str(classification["solver_type"]),
        "solver_backed": bool(classification["solver_backed"]),
        "residual_only": bool(classification["residual_only"]),
        "quality_pass_allowed": bool(classification["quality_pass_allowed"]),
        "quality_classification": str(classification["quality_classification"]),
        "classification_reason": str(classification["classification_reason"]),
        "quality_gate_results": dict(classification["quality_gate_results"]),
        "failure_or_warning_reasons": list(classification["failure_or_warning_reasons"]),
        "runtime_quality_gates": dict(classification["gates"]),
    }


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
