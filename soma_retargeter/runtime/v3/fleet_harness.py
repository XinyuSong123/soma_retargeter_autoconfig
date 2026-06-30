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
from .quality_metrics import target_stream_metrics
from .runtime_quality_gates import RUNTIME_QUALITY_FAILED, combine_full_humanoid_classifications
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
        profile_blocked = _profile_resolution_blocks_quality(profile_resolution_status)
        final_status = BLOCKED_FINAL_STATUS if profile_blocked else self.final_status
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
                "final_step3_1_status": final_status,
                "runtime_quality_status": final_status,
                "runtime_quality_classification": final_status,
                "quality_classification": final_status,
                "quality_evaluated": self.case.category != NEGATIVE_CONTROL and not profile_blocked,
                "promoted_to_runtime_quality": False,
                "humanoid_profile_generated": profile_resolution_status == "runtime_local_profile_generated",
                "override_allowed": self.case.category == FULL_HUMANOID_PROFILE and not profile_blocked,
                "smoke_type": self.quality_metrics.get("smoke_type", "not_applicable"),
                "solver_type": self.quality_metrics.get("solver_type", "not_applicable"),
                "solver_backed": bool(self.quality_metrics.get("solver_backed", False)),
                "residual_only": bool(self.quality_metrics.get("residual_only", False)),
                "solver_backed_smoke_attempted": bool(self.quality_metrics.get("solver_backed_smoke_attempted", False)),
                "solver_backed_smoke_completed": bool(self.quality_metrics.get("solver_backed_smoke_completed", False)),
                "solver_mode": self.quality_metrics.get("solver_type", self.quality_metrics.get("smoke_type", "not_applicable")),
                "solver_failure_reason": self.quality_metrics.get("solver_failure_reason"),
                "sampled_frame_indices": list(self.quality_metrics.get("sampled_frame_indices", [])),
                "deterministic_hash_inputs": dict(self.quality_metrics.get("deterministic_hash_inputs", {})),
                "quality_pass_allowed": bool(self.quality_metrics.get("quality_pass_allowed", False)),
                "quality_gate_results": dict(self.quality_metrics.get("quality_gate_results", {})),
                "failure_or_warning_reasons": list(self.quality_metrics.get("failure_or_warning_reasons", [])),
                "pipeline_control_id": "pipeline_backed_matrix.json",
                "control_modes": ["disabled", "shadow", "override_experimental"],
                "legacy_default_unchanged": True,
                "shadow_noop_verified": True,
                "override_explicit_only": True,
                "fingerprint_gate_enforced": True,
                "profile_resolution_blocked": profile_blocked,
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
    enable_solver_backed_generic_smoke: bool = False,
    solver_smoke_config: Any | None = None,
) -> FleetCaseResult:
    if case.category == NEGATIVE_CONTROL:
        from .generic_smoke import run_negative_control_rejection_smoke

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
            result = _evaluate_clip(
                case,
                clip_path=clip_path,
                max_frames=max_frames,
                smoke_enabled=smoke_enabled,
                partial_smoke_enabled=partial_smoke_enabled,
                enable_solver_backed_generic_smoke=enable_solver_backed_generic_smoke,
                solver_smoke_config=solver_smoke_config,
            )
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
    metrics = aggregate_case_metrics(clip_results)
    if case.category == PARTIAL_HUMANOID_PROFILE:
        smoke_ok = all(row.generic_smoke_status == "passed" for row in smoke_rows)
        final = PARTIAL_FINAL_STATUS if target_ok and smoke_ok else BLOCKED_FINAL_STATUS
        generic_status = "partial_supported_smoke_passed" if smoke_ok else "partial_supported_smoke_failed"
    else:
        classifications = [
            str(row.smoke_summary.get("quality_classification") or row.generic_smoke_status)
            for row in smoke_rows
            if row.smoke_summary
        ]
        final = combine_full_humanoid_classifications(classifications) if target_ok else RUNTIME_QUALITY_FAILED
        generic_status = final
        metrics["runtime_quality_classification"] = final
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
            "normalized_task_residual_mean": 0.0,
            "normalized_task_residual_p50": 0.0,
            "normalized_task_residual_p95": 0.0,
            "normalized_task_residual_max": 0.0,
            "task_residual_mean": 0.0,
            "task_residual_p50": 0.0,
            "task_residual_p95": 0.0,
            "task_residual_max": 0.0,
            "raw_task_residual_mean": 0.0,
            "raw_task_residual_p50": 0.0,
            "raw_task_residual_p95": 0.0,
            "raw_task_residual_max": 0.0,
            "solver_success_fraction": 0.0,
            "solver_type": "not_applicable",
            "smoke_type": "not_applicable",
            "solver_backed": False,
            "residual_only": False,
            "solver_backed_smoke_attempted": False,
            "solver_backed_smoke_completed": False,
            "solver_failure_reason": None,
            "sampled_frame_indices": [],
            "deterministic_hash_inputs": {},
            "quality_pass_allowed": False,
            "quality_gate_results": {},
            "failure_or_warning_reasons": [],
            "task_anchor_count": 0,
            "task_anchor_semantic_counts": {},
            "task_coverage_ratio": 0.0,
            "successful_task_coverage_ratio": 0.0,
            "anchor_reliability_score": 0.0,
            "anchor_rejection_reasons": [],
            "residual_normalization_version": "",
            "residual_normalization_formula": "",
            "residual_denominator": 0.0,
            "residual_denominator_source": "",
            "residual_denominator_scope": "",
            "residual_denominator_units": "",
            "residual_denominator_robot_specific": False,
            "residual_denominator_from_current_row_max": False,
            "runtime_seconds": 0.0,
        }
    frame_count = sum(row.frame_count for row in clip_results)
    smoke_metrics = [row.smoke_summary.get("metrics", {}) for row in clip_results if row.smoke_summary.get("metrics")]
    smoke_summaries = [row.smoke_summary for row in clip_results if row.smoke_summary]
    target_translation_max = max((m.get("frame_to_frame_translation_velocity_max", 0.0) for row in clip_results for m in [row.target_metrics]), default=0.0)
    target_rotation_max = max((m.get("frame_to_frame_rotation_velocity_max", 0.0) for row in clip_results for m in [row.target_metrics]), default=0.0)
    reasons: list[str] = []
    gate_results: dict[str, Any] = {}
    for summary in smoke_summaries:
        reasons.extend(str(v) for v in summary.get("failure_or_warning_reasons", []))
        for key, value in summary.get("quality_gate_results", {}).items():
            gate_results[key] = bool(gate_results.get(key, True) and value)
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
        "normalized_task_residual_mean": max((float(m.get("normalized_task_residual_mean", 0.0)) for m in smoke_metrics), default=0.0),
        "normalized_task_residual_p50": max((float(m.get("normalized_task_residual_p50", 0.0)) for m in smoke_metrics), default=0.0),
        "normalized_task_residual_p95": max((float(m.get("normalized_task_residual_p95", 0.0)) for m in smoke_metrics), default=0.0),
        "normalized_task_residual_max": max((float(m.get("normalized_task_residual_max", 0.0)) for m in smoke_metrics), default=0.0),
        "task_residual_mean": max((float(m.get("task_residual_mean", 0.0)) for m in smoke_metrics), default=0.0),
        "task_residual_p50": max((float(m.get("task_residual_p50", 0.0)) for m in smoke_metrics), default=0.0),
        "task_residual_p95": max((float(m.get("task_residual_p95", 0.0)) for m in smoke_metrics), default=0.0),
        "task_residual_max": max((float(m.get("task_residual_max", 0.0)) for m in smoke_metrics), default=0.0),
        "raw_task_residual_mean": max((float(m.get("raw_task_residual_mean", m.get("task_residual_mean", 0.0))) for m in smoke_metrics), default=0.0),
        "raw_task_residual_p50": max((float(m.get("raw_task_residual_p50", m.get("task_residual_p50", 0.0))) for m in smoke_metrics), default=0.0),
        "raw_task_residual_p95": max((float(m.get("raw_task_residual_p95", m.get("task_residual_p95", 0.0))) for m in smoke_metrics), default=0.0),
        "raw_task_residual_max": max((float(m.get("raw_task_residual_max", m.get("task_residual_max", 0.0))) for m in smoke_metrics), default=0.0),
        "solver_success_fraction": min((float(m.get("solver_success_fraction", 1.0)) for m in smoke_metrics), default=1.0),
        "solver_type": _aggregate_solver_type(smoke_summaries),
        "smoke_type": _aggregate_smoke_type(smoke_summaries),
        "solver_backed": any(bool(summary.get("solver_backed", False)) for summary in smoke_summaries),
        "residual_only": any(bool(summary.get("residual_only", False)) for summary in smoke_summaries),
        "solver_backed_smoke_attempted": any(bool(summary.get("solver_backed_smoke_attempted", False)) for summary in smoke_summaries),
        "solver_backed_smoke_completed": any(bool(summary.get("solver_backed_smoke_completed", False)) for summary in smoke_summaries),
        "solver_failure_reason": _aggregate_solver_failure_reason(smoke_summaries),
        "sampled_frame_indices": _aggregate_sampled_frame_indices(smoke_summaries),
        "deterministic_hash_inputs": _aggregate_deterministic_hash_inputs(smoke_summaries),
        "quality_pass_allowed": all(bool(summary.get("quality_pass_allowed", False)) for summary in smoke_summaries) if smoke_summaries else False,
        "quality_gate_results": gate_results,
        "failure_or_warning_reasons": _dedupe(reasons),
        "task_anchor_count": max((int(m.get("task_anchor_count", 0) or 0) for m in smoke_metrics), default=0),
        "task_anchor_semantic_counts": _aggregate_semantic_counts(smoke_metrics, "task_anchor_semantic_counts"),
        "task_coverage_ratio": min((float(m.get("task_coverage_ratio", 0.0) or 0.0) for m in smoke_metrics), default=0.0),
        "successful_task_coverage_ratio": min((float(m.get("successful_task_coverage_ratio", 0.0) or 0.0) for m in smoke_metrics), default=0.0),
        "anchor_reliability_score": min((float(m.get("anchor_reliability_score", 0.0) or 0.0) for m in smoke_metrics), default=0.0),
        "anchor_rejection_reasons": _dedupe(
            [str(reason) for m in smoke_metrics for reason in m.get("anchor_rejection_reasons", [])]
        ),
        "residual_normalization_version": _first_metric_text(smoke_metrics, "residual_normalization_version"),
        "residual_normalization_formula": _first_metric_text(smoke_metrics, "residual_normalization_formula"),
        "residual_denominator": max((float(m.get("residual_denominator", 0.0) or 0.0) for m in smoke_metrics), default=0.0),
        "residual_denominator_source": _first_metric_text(smoke_metrics, "residual_denominator_source"),
        "residual_denominator_scope": _first_metric_text(smoke_metrics, "residual_denominator_scope"),
        "residual_denominator_units": _first_metric_text(smoke_metrics, "residual_denominator_units"),
        "residual_denominator_robot_specific": any(bool(m.get("residual_denominator_robot_specific", False)) for m in smoke_metrics),
        "residual_denominator_from_current_row_max": any(bool(m.get("residual_denominator_from_current_row_max", False)) for m in smoke_metrics),
        "runtime_seconds": sum(float(m.get("runtime_seconds", 0.0)) for m in smoke_metrics),
    }


def _profile_resolution_blocks_quality(profile_resolution_status: str) -> bool:
    return profile_resolution_status in {
        "runtime_local_profile_failed",
        "runtime_model_load_failed",
        "source_or_cache_unavailable",
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
    enable_solver_backed_generic_smoke: bool = False,
    solver_smoke_config: Any | None = None,
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
        from .generic_smoke import run_partial_supported_smoke

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
    from .generic_smoke import run_full_humanoid_fk_smoke, run_full_humanoid_solver_backed_smoke

    if smoke_enabled and enable_solver_backed_generic_smoke:
        smoke = run_full_humanoid_solver_backed_smoke(case, targets.transforms, config=solver_smoke_config)
    else:
        smoke = run_full_humanoid_fk_smoke(case, targets.transforms) if smoke_enabled else None
    return FleetClipResult(
        clip_id=clip_id(clip_path),
        clip_path=display_path(clip_path) or str(clip_path),
        mode="solver_backed_generic_smoke" if smoke_enabled and enable_solver_backed_generic_smoke else ("generic_override_smoke" if smoke_enabled else "target_stream_only"),
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
        "normalized_task_residual_mean": float(metrics.get("normalized_task_residual_mean", 0.0) or 0.0),
        "normalized_task_residual_p50": float(metrics.get("normalized_task_residual_p50", 0.0) or 0.0),
        "normalized_task_residual_p95": float(metrics.get("normalized_task_residual_p95", 0.0) or 0.0),
        "normalized_task_residual_max": float(metrics.get("normalized_task_residual_max", 0.0) or 0.0),
        "task_residual_mean": float(metrics.get("task_residual_mean", metrics.get("raw_task_residual_mean", 0.0)) or 0.0),
        "task_residual_p50": float(metrics.get("task_residual_p50", metrics.get("raw_task_residual_p50", 0.0)) or 0.0),
        "task_residual_p95": float(metrics.get("task_residual_p95", metrics.get("raw_task_residual_p95", 0.0)) or 0.0),
        "task_residual_max": float(metrics.get("task_residual_max", metrics.get("raw_task_residual_max", 0.0)) or 0.0),
        "raw_task_residual_mean": float(metrics.get("raw_task_residual_mean", metrics.get("task_residual_mean", 0.0)) or 0.0),
        "raw_task_residual_p50": float(metrics.get("raw_task_residual_p50", metrics.get("task_residual_p50", 0.0)) or 0.0),
        "raw_task_residual_p95": float(metrics.get("raw_task_residual_p95", metrics.get("task_residual_p95", 0.0)) or 0.0),
        "raw_task_residual_max": float(metrics.get("raw_task_residual_max", metrics.get("task_residual_max", 0.0)) or 0.0),
        "orientation_integrated_residual_mean": float(metrics.get("orientation_integrated_residual_mean", metrics.get("task_residual_mean", 0.0)) or 0.0),
        "orientation_integrated_residual_p95": float(metrics.get("orientation_integrated_residual_p95", metrics.get("task_residual_p95", 0.0)) or 0.0),
        "orientation_integrated_residual_max": float(metrics.get("orientation_integrated_residual_max", metrics.get("task_residual_max", 0.0)) or 0.0),
        "legacy_world_task_residual_mean": float(metrics.get("legacy_world_task_residual_mean", metrics.get("task_residual_mean", 0.0)) or 0.0),
        "legacy_world_task_residual_p95": float(metrics.get("legacy_world_task_residual_p95", metrics.get("task_residual_p95", 0.0)) or 0.0),
        "legacy_world_task_residual_max": float(metrics.get("legacy_world_task_residual_max", metrics.get("task_residual_max", 0.0)) or 0.0),
        "legacy_world_rotation_residual_mean": float(metrics.get("legacy_world_rotation_residual_mean", metrics.get("target_rotation_error_mean", 0.0)) or 0.0),
        "legacy_world_rotation_residual_p95": float(metrics.get("legacy_world_rotation_residual_p95", metrics.get("target_rotation_error_p95", 0.0)) or 0.0),
        "legacy_world_rotation_residual_max": float(metrics.get("legacy_world_rotation_residual_max", metrics.get("target_rotation_error_max", 0.0)) or 0.0),
        "active_runtime_scoring_orientation_policy": str(metrics.get("active_runtime_scoring_orientation_policy", "")),
        "diagnostic_orientation_policy": str(metrics.get("diagnostic_orientation_policy", "")),
        "production_default_orientation_policy": str(metrics.get("production_default_orientation_policy", "")),
        "orientation_policy_active_for_scoring": bool(metrics.get("orientation_policy_active_for_scoring", False)),
        "orientation_policy_production_default_changed": bool(metrics.get("orientation_policy_production_default_changed", False)),
        "runtime_override_default_enabled": bool(metrics.get("runtime_override_default_enabled", False)),
        "solver_success_fraction": float(metrics.get("solver_success_fraction", 0.0) or 0.0),
        "task_anchor_count": int(metrics.get("task_anchor_count", 0) or 0),
        "task_anchor_semantic_counts": dict(metrics.get("task_anchor_semantic_counts", {})),
        "task_coverage_ratio": float(metrics.get("task_coverage_ratio", 0.0) or 0.0),
        "successful_task_coverage_ratio": float(metrics.get("successful_task_coverage_ratio", 0.0) or 0.0),
        "anchor_reliability_score": float(metrics.get("anchor_reliability_score", 0.0) or 0.0),
        "anchor_rejection_reasons": list(metrics.get("anchor_rejection_reasons", [])),
        "residual_normalization_version": str(metrics.get("residual_normalization_version", "")),
        "residual_normalization_formula": str(metrics.get("residual_normalization_formula", "")),
        "residual_denominator": float(metrics.get("residual_denominator", 0.0) or 0.0),
        "residual_denominator_source": str(metrics.get("residual_denominator_source", "")),
        "residual_denominator_scope": str(metrics.get("residual_denominator_scope", "")),
        "residual_denominator_units": str(metrics.get("residual_denominator_units", "")),
        "residual_denominator_robot_specific": bool(metrics.get("residual_denominator_robot_specific", False)),
        "residual_denominator_from_current_row_max": bool(metrics.get("residual_denominator_from_current_row_max", False)),
        "runtime_seconds": float(metrics.get("runtime_seconds", 0.0) or 0.0),
    }


def _aggregate_semantic_counts(metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in metrics:
        value = row.get(key)
        if not isinstance(value, dict):
            continue
        for semantic, count in value.items():
            counts[str(semantic)] = max(counts.get(str(semantic), 0), int(count or 0))
    return dict(sorted(counts.items()))


def _first_metric_text(metrics: list[dict[str, Any]], key: str) -> str:
    for row in metrics:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _aggregate_solver_type(smoke_summaries: list[dict[str, Any]]) -> str:
    values = [str(summary.get("solver_type") or summary.get("residuals", {}).get("solver") or "") for summary in smoke_summaries]
    values = [value for value in values if value]
    if not values:
        return "not_applicable"
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "+".join(unique)


def _aggregate_smoke_type(smoke_summaries: list[dict[str, Any]]) -> str:
    values = [str(summary.get("mode") or "") for summary in smoke_summaries]
    values = [value for value in values if value]
    if not values:
        return "not_applicable"
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else "+".join(unique)


def _aggregate_solver_failure_reason(smoke_summaries: list[dict[str, Any]]) -> str | None:
    values = [str(summary.get("solver_failure_reason") or "") for summary in smoke_summaries]
    values = [value for value in values if value]
    if not values:
        return None
    return ";".join(_dedupe(values))


def _aggregate_sampled_frame_indices(smoke_summaries: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for summary in smoke_summaries:
        for value in summary.get("sampled_frame_indices", []):
            index = int(value)
            if index not in out:
                out.append(index)
    return out


def _aggregate_deterministic_hash_inputs(smoke_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        summary.get("deterministic_hash_inputs")
        for summary in smoke_summaries
        if isinstance(summary.get("deterministic_hash_inputs"), dict) and summary.get("deterministic_hash_inputs")
    ]
    if not rows:
        return {}
    return {"smoke_rows": rows}


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


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
