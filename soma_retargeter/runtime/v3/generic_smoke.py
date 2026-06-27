"""Generic runtime FK smoke for Step 3.1 full-fleet evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from soma_retargeter.robotics.v3.model_adapter import NewtonRuntimeModelAdapter, SemanticSite
from soma_retargeter.robotics.v3.spatial import rotation_error

from .fleet_inventory import FleetRuntimeCase
from .quality_metrics import contact_diagnostics, smoke_output_metrics
from .runtime_quality_gates import (
    BLOCKED_SOURCE_OR_PROFILE,
    NEGATIVE_CONTROL_RUNTIME_PASSED,
    PARTIAL_RUNTIME_PASSED,
    RESIDUAL_ONLY_SOLVER_TYPE,
    RUNTIME_QUALITY_FAILED,
    classify_runtime_quality,
)


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
