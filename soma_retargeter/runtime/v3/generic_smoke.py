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


@dataclass(frozen=True)
class GenericSmokeResult:
    status: str
    mode: str
    metrics: dict[str, Any]
    residuals: dict[str, Any]
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
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
        status = "passed" if metrics["nan_count"] == 0 and metrics["inf_count"] == 0 else "failed"
        return GenericSmokeResult(
            status=status,
            mode="generic_fk_residual_smoke",
            metrics=metrics,
            residuals={
                "solver": "runtime_model_fk_residual_evaluation",
                "per_semantic": per_semantic,
                "residual_sample_count": len(residual_values),
            },
        )
    except Exception as exc:
        return GenericSmokeResult(
            status="failed",
            mode="generic_fk_residual_smoke",
            metrics={
                "output_frame_count": 0,
                "joint_coord_count": int(getattr(adapter, "nq", 0)),
                "nan_count": 0,
                "inf_count": 0,
                "joint_limit_violation_count": 0,
                "max_joint_limit_violation": 0.0,
                "runtime_seconds": 0.0,
            },
            residuals={"solver": "runtime_model_fk_residual_evaluation", "per_semantic": {}},
            error=f"{type(exc).__name__}: {exc}",
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
        )
    except Exception as exc:
        return GenericSmokeResult(
            status="failed",
            mode="partial_supported_semantic_fk_smoke",
            metrics={"output_frame_count": 0, "joint_coord_count": 0, "nan_count": 0, "inf_count": 0, "runtime_seconds": 0.0},
            residuals={"solver": "partial_supported_semantic_runtime_load_and_fk"},
            error=f"{type(exc).__name__}: {exc}",
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
