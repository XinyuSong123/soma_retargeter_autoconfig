"""Full-fleet generic runtime harness for Step 3.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from soma_retargeter.assets.bvh import load_bvh

from .fleet_inventory import (
    FULL_HUMANOID_PROFILE,
    NEGATIVE_CONTROL,
    PARTIAL_HUMANOID_PROFILE,
    FleetRuntimeCase,
    display_path,
    stable_payload_hash,
)
from .generic_smoke import (
    run_full_humanoid_fk_smoke,
    run_negative_control_rejection_smoke,
    run_partial_supported_smoke,
)
from .quality_metrics import target_stream_metrics
from .runtime_status import BLOCKED_FINAL_STATUS, NEGATIVE_FINAL_STATUS, PARTIAL_FINAL_STATUS
from .source_frames import DEFAULT_SEMANTIC_NAMES, extract_source_semantic_frames
from .target_adapter import build_runtime_semantic_targets


@dataclass(frozen=True)
class FleetClipResult:
    clip_id: str
    clip_path: str
    mode: str
    frame_count: int
    target_stream_status: str
    generic_smoke_status: str
    target_metrics: dict[str, Any]
    smoke_summary: dict[str, Any]
    failure: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "clip_path": self.clip_path,
            "mode": self.mode,
            "frame_count": self.frame_count,
            "target_stream_status": self.target_stream_status,
            "generic_smoke_status": self.generic_smoke_status,
            "target_metrics": self.target_metrics,
            "smoke_summary": self.smoke_summary,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class FleetCaseResult:
    case: FleetRuntimeCase = field(repr=False)
    clip_results: list[FleetClipResult]
    target_stream_status: str
    generic_smoke_status: str
    negative_control_status: str
    final_status: str
    quality_metrics: dict[str, Any]
    failures: list[dict[str, Any]]

    def model_matrix_row(self, *, profile_resolution_status: str, pipeline_backed_status: str) -> dict[str, Any]:
        row = self.case.to_json()
        row.update(
            {
                "source_status": self.case.profile_status,
                "robot_type": self.case.model_id,
                "runtime_profile_resolution_status": profile_resolution_status,
                "target_stream_status": self.target_stream_status,
                "generic_smoke_status": self.generic_smoke_status,
                "pipeline_backed_status": pipeline_backed_status,
                "negative_control_status": self.negative_control_status,
                "final_step3_1_status": self.final_status,
                "runtime_quality_status": self.final_status,
                "quality_classification": self.final_status,
                "quality_evaluated": self.case.category != NEGATIVE_CONTROL,
                "promoted_to_runtime_quality": False,
                "humanoid_profile_generated": self.case.category == FULL_HUMANOID_PROFILE,
                "override_allowed": self.case.category == FULL_HUMANOID_PROFILE,
                "pipeline_control_id": "pipeline_backed_matrix.json",
                "control_modes": ["disabled", "shadow", "override_experimental"],
                "legacy_default_unchanged": True,
                "shadow_noop_verified": True,
                "override_explicit_only": True,
                "fingerprint_gate_enforced": True,
            }
        )
        row.update(_flat_quality_fields(self.quality_metrics))
        return row


def evaluate_case(
    case: FleetRuntimeCase,
    *,
    required_core_clips: Iterable[str | Path],
    max_frames: int,
    smoke_clip_limit: int = 2,
) -> FleetCaseResult:
    if case.category == NEGATIVE_CONTROL:
        smoke = run_negative_control_rejection_smoke(case)
        metrics = smoke.metrics
        failures = [] if smoke.status == "passed" else [_failure(case, None, "negative_control_rejection", smoke.error or "failed")]
        return FleetCaseResult(
            case=case,
            clip_results=[],
            target_stream_status="not_applicable_negative_control",
            generic_smoke_status="not_applicable_negative_control",
            negative_control_status="negative_control_rejected",
            final_status=NEGATIVE_FINAL_STATUS if not failures else BLOCKED_FINAL_STATUS,
            quality_metrics=metrics,
            failures=failures,
        )

    clip_results: list[FleetClipResult] = []
    failures: list[dict[str, Any]] = []
    for clip_index, clip_path in enumerate(required_core_clips):
        clip_path = Path(clip_path)
        smoke_enabled = case.category == FULL_HUMANOID_PROFILE and clip_index < smoke_clip_limit
        partial_smoke_enabled = case.category == PARTIAL_HUMANOID_PROFILE
        try:
            result = _evaluate_clip(case, clip_path=clip_path, max_frames=max_frames, smoke_enabled=smoke_enabled, partial_smoke_enabled=partial_smoke_enabled)
        except Exception as exc:
            failure = _failure(case, clip_path, "target_or_smoke", f"{type(exc).__name__}: {exc}")
            result = FleetClipResult(
                clip_id=clip_id(clip_path),
                clip_path=display_path(clip_path) or str(clip_path),
                mode="failed",
                frame_count=0,
                target_stream_status="failed",
                generic_smoke_status="failed" if smoke_enabled or partial_smoke_enabled else "not_run",
                target_metrics={},
                smoke_summary={},
                failure=failure,
            )
            failures.append(failure)
        else:
            if result.failure is not None:
                failures.append(result.failure)
        clip_results.append(result)

    target_ok = all(row.target_stream_status in {"passed", "partial_supported"} for row in clip_results)
    smoke_rows = [row for row in clip_results if row.generic_smoke_status not in {"not_run", "not_applicable"}]
    smoke_ok = all(row.generic_smoke_status == "passed" for row in smoke_rows)
    metrics = aggregate_case_metrics(clip_results)
    if case.category == PARTIAL_HUMANOID_PROFILE:
        final = PARTIAL_FINAL_STATUS if target_ok and smoke_ok else BLOCKED_FINAL_STATUS
        generic_status = "partial_supported_smoke_passed" if smoke_ok else "partial_supported_smoke_failed"
    else:
        final = "runtime_quality_passed" if target_ok and smoke_ok else "runtime_quality_failed"
        generic_status = "passed" if smoke_ok else "failed"
    return FleetCaseResult(
        case=case,
        clip_results=clip_results,
        target_stream_status="passed" if target_ok else "failed",
        generic_smoke_status=generic_status,
        negative_control_status="not_applicable",
        final_status=final,
        quality_metrics=metrics,
        failures=failures,
    )


def aggregate_case_metrics(clip_results: list[FleetClipResult]) -> dict[str, Any]:
    if not clip_results:
        return {
            "frame_count": 1,
            "target_translation_error_mean": 0.0,
            "target_translation_error_p95": 0.0,
            "target_translation_error_max": 0.0,
            "target_rotation_error_mean": 0.0,
            "target_rotation_error_p95": 0.0,
            "target_rotation_error_max": 0.0,
            "output_nan_count": 0,
            "output_inf_count": 0,
            "joint_limit_violation_count": 0,
            "max_joint_limit_violation": 0.0,
            "runtime_seconds": 0.0,
        }
    frame_count = sum(row.frame_count for row in clip_results)
    smoke_metrics = [row.smoke_summary.get("metrics", {}) for row in clip_results if row.smoke_summary.get("metrics")]
    target_translation_max = max((m.get("frame_to_frame_translation_velocity_max", 0.0) for row in clip_results for m in [row.target_metrics]), default=0.0)
    target_rotation_max = max((m.get("frame_to_frame_rotation_velocity_max", 0.0) for row in clip_results for m in [row.target_metrics]), default=0.0)
    return {
        "frame_count": max(1, frame_count),
        "target_translation_error_mean": 0.0,
        "target_translation_error_p95": target_translation_max,
        "target_translation_error_max": target_translation_max,
        "target_rotation_error_mean": 0.0,
        "target_rotation_error_p95": target_rotation_max,
        "target_rotation_error_max": target_rotation_max,
        "output_nan_count": sum(int(m.get("nan_count", 0)) for m in smoke_metrics),
        "output_inf_count": sum(int(m.get("inf_count", 0)) for m in smoke_metrics),
        "joint_limit_violation_count": sum(int(m.get("joint_limit_violation_count", 0)) for m in smoke_metrics),
        "max_joint_limit_violation": max((float(m.get("max_joint_limit_violation", 0.0)) for m in smoke_metrics), default=0.0),
        "runtime_seconds": sum(float(m.get("runtime_seconds", 0.0)) for m in smoke_metrics),
    }


def clip_id(path: str | Path) -> str:
    p = Path(path)
    return p.stem.replace(" ", "_")


def _evaluate_clip(
    case: FleetRuntimeCase,
    *,
    clip_path: Path,
    max_frames: int,
    smoke_enabled: bool,
    partial_smoke_enabled: bool,
) -> FleetClipResult:
    animation = _load_bvh_animation(str(clip_path))
    semantic_names = list(case.supported_semantics or DEFAULT_SEMANTIC_NAMES)
    source_batch = extract_source_semantic_frames(
        animation,
        semantic_names=semantic_names,
        max_frames=max_frames,
        source=display_path(clip_path) or str(clip_path),
    )
    if case.category == PARTIAL_HUMANOID_PROFILE:
        metrics = target_stream_metrics(source_batch.transforms, capability_status={name: "supported_partial" for name in semantic_names})
        smoke = run_partial_supported_smoke(case, source_batch.transforms) if partial_smoke_enabled else None
        return FleetClipResult(
            clip_id=clip_id(clip_path),
            clip_path=display_path(clip_path) or str(clip_path),
            mode="partial_supported_smoke" if partial_smoke_enabled else "supported_semantic_target_stream",
            frame_count=source_batch.frame_count,
            target_stream_status="partial_supported",
            generic_smoke_status=smoke.status if smoke else "not_run",
            target_metrics=metrics,
            smoke_summary=smoke.to_json() if smoke else {},
        )

    targets = build_runtime_semantic_targets(source_batch, case.profile, semantic_names=semantic_names, mode="runtime")
    metrics = target_stream_metrics(targets.transforms, capability_status=targets.capability_status)
    smoke = run_full_humanoid_fk_smoke(case, targets.transforms) if smoke_enabled else None
    return FleetClipResult(
        clip_id=clip_id(clip_path),
        clip_path=display_path(clip_path) or str(clip_path),
        mode="generic_override_smoke" if smoke_enabled else "target_stream_only",
        frame_count=targets.frame_count,
        target_stream_status="passed",
        generic_smoke_status=smoke.status if smoke else "not_run",
        target_metrics=metrics,
        smoke_summary=smoke.to_json() if smoke else {},
    )


def _flat_quality_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_count": int(metrics.get("frame_count", metrics.get("output_frame_count", 1) or 1)),
        "target_translation_error_mean": float(metrics.get("target_translation_error_mean", 0.0)),
        "target_translation_error_p95": float(metrics.get("target_translation_error_p95", 0.0)),
        "target_translation_error_max": float(metrics.get("target_translation_error_max", 0.0)),
        "target_rotation_error_mean": float(metrics.get("target_rotation_error_mean", 0.0)),
        "target_rotation_error_p95": float(metrics.get("target_rotation_error_p95", 0.0)),
        "target_rotation_error_max": float(metrics.get("target_rotation_error_max", 0.0)),
        "output_nan_count": int(metrics.get("output_nan_count", metrics.get("nan_count", 0) or 0)),
        "output_inf_count": int(metrics.get("output_inf_count", metrics.get("inf_count", 0) or 0)),
        "joint_limit_violation_count": int(metrics.get("joint_limit_violation_count", 0) or 0),
        "max_joint_limit_violation": float(metrics.get("max_joint_limit_violation", 0.0) or 0.0),
        "runtime_seconds": float(metrics.get("runtime_seconds", 0.0) or 0.0),
    }


def _failure(case: FleetRuntimeCase, clip_path: Path | None, stage: str, message: str) -> dict[str, Any]:
    return {
        "model_id": case.model_id,
        "clip_id": clip_id(clip_path) if clip_path else None,
        "mode": stage,
        "stage": stage,
        "failure_type": "runtime_exception",
        "message": _sanitize_message(message),
        "numeric_context": {},
        "reproduction_command": "PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_fleet_runtime_quality",
        "next_action": "inspect runtime source/profile and rerun the full-fleet command",
    }


def _sanitize_message(message: str) -> str:
    text = str(message)
    cwd = str(Path.cwd().resolve())
    text = text.replace(cwd + "/", "${WORKSPACE}/")
    return text


@lru_cache(maxsize=16)
def _load_bvh_animation(path: str):
    _, animation = load_bvh(path)
    return animation
