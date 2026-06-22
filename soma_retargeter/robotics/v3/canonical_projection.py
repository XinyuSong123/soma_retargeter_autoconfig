"""Canonical-motion chain projection using real robot-space semantic targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain_projection import project_endpoint_position, project_torso_orientation
from .kinematic_paths import TASKS, KinematicPath
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import relative_transform
from .target_builder import SemanticTargets

TEMPORAL_MOTION_SEQUENCES: dict[str, list[str]] = {
    "arms_overhead_return": ["neutral", "arms_forward", "overhead_reach", "neutral"],
    "squat_stand": ["neutral", "squat", "neutral"],
    "step_support": ["neutral", "single_step", "neutral"],
}


@dataclass(frozen=True)
class CanonicalProjectionReport:
    motion_order: list[str]
    motions: dict[str, dict]
    warnings: list[str]
    failures: list[str]
    unreachable_demands: list[str]
    target_source: str = "canonical_semantic_targets"

    def to_json(self) -> dict:
        return {
            "motion_order": self.motion_order,
            "motions": self.motions,
            "warnings": self.warnings,
            "failures": self.failures,
            "unreachable_demands": self.unreachable_demands,
            "target_source": self.target_source,
        }


def project_canonical_motion_sequence(
    adapter: MuJoCoRuntimeModelAdapter,
    sites: dict[str, SemanticSite],
    paths: dict[str, KinematicPath],
    canonical_targets: dict[str, SemanticTargets],
    *,
    neutral_q: np.ndarray | None = None,
    motion_order: list[str] | None = None,
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 0.0,
    use_continuity_prior: bool = False,
) -> CanonicalProjectionReport:
    """Project torso, hand and foot chain targets for each canonical motion.

    The desired values come from ``SemanticTargets.transforms`` in robot world
    coordinates. They are converted to each task's semantic reference frame
    before calling the bounded chain-only projection routines.
    """

    q0 = adapter.neutral_q() if neutral_q is None else np.asarray(neutral_q, dtype=float).copy()
    order = motion_order or _default_motion_order(canonical_targets)
    previous_q_by_task: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    failures: list[str] = []
    unreachable_demands: list[str] = []
    motions: dict[str, dict] = {}

    for motion_name in order:
        targets = canonical_targets.get(motion_name)
        if targets is None:
            warnings.append(f"{motion_name}: missing canonical targets")
            continue
        motion_report = {
            "mode": targets.mode,
            "tasks": {},
            "skipped": {},
            "target_source": "SemanticTargets.transforms",
        }
        for task_name in _ordered_tasks(paths):
            path = paths[task_name]
            if path.reference not in sites or path.target not in sites:
                motion_report["skipped"][task_name] = "missing semantic site"
                continue
            if path.reference not in targets.transforms or path.target not in targets.transforms:
                motion_report["skipped"][task_name] = "missing canonical target transform"
                continue
            desired_relative = relative_transform(targets.transforms[path.reference], targets.transforms[path.target])
            previous_q = previous_q_by_task.get(task_name) if use_continuity_prior else None
            q_seed = previous_q if previous_q is not None else q0
            reference = sites[path.reference]
            target = sites[path.target]
            if task_name == "torso":
                result = project_torso_orientation(
                    adapter,
                    q_seed,
                    reference,
                    target,
                    path.active_velocity_coordinates,
                    desired_relative[:3, :3],
                    neutral_q=q0,
                    previous_q=previous_q,
                    neutral_prior_weight=neutral_prior_weight,
                    continuity_prior_weight=continuity_prior_weight if use_continuity_prior else 0.0,
                )
            else:
                result = project_endpoint_position(
                    adapter,
                    q_seed,
                    reference,
                    target,
                    path.active_velocity_coordinates,
                    desired_relative[:3, 3],
                    neutral_q=q0,
                    previous_q=previous_q,
                    neutral_prior_weight=neutral_prior_weight,
                    continuity_prior_weight=continuity_prior_weight if use_continuity_prior else 0.0,
                )
            if use_continuity_prior:
                previous_q_by_task[task_name] = result.chain_q
            result_json = result.to_json()
            result_json["reference"] = path.reference
            result_json["target"] = path.target
            result_json["desired_source"] = "canonical_targets.transforms"
            motion_report["tasks"][task_name] = result_json
            if result.status == "unreachable/rank_zero":
                unreachable_demands.append(f"{motion_name}:{task_name}: rank-zero chain has nonzero demand")
        motions[motion_name] = motion_report

    return CanonicalProjectionReport(
        motion_order=order,
        motions=motions,
        warnings=warnings,
        failures=failures,
        unreachable_demands=unreachable_demands,
    )


def project_temporal_motion_sequences(
    adapter: MuJoCoRuntimeModelAdapter,
    sites: dict[str, SemanticSite],
    paths: dict[str, KinematicPath],
    canonical_targets: dict[str, SemanticTargets],
    *,
    neutral_q: np.ndarray | None = None,
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-3,
) -> dict[str, dict]:
    """Project named temporal benchmarks with continuity isolated from capability motions."""

    reports: dict[str, dict] = {}
    for sequence_name, order in TEMPORAL_MOTION_SEQUENCES.items():
        available_order = [name for name in order if name in canonical_targets]
        report = project_canonical_motion_sequence(
            adapter,
            sites,
            paths,
            canonical_targets,
            neutral_q=neutral_q,
            motion_order=available_order,
            neutral_prior_weight=neutral_prior_weight,
            continuity_prior_weight=continuity_prior_weight,
            use_continuity_prior=True,
        )
        payload = report.to_json()
        payload["benchmark_type"] = "temporal_sequence"
        payload["continuity_prior_enabled"] = True
        reports[sequence_name] = payload
    return reports


def _default_motion_order(canonical_targets: dict[str, SemanticTargets]) -> list[str]:
    names = list(canonical_targets)
    if "neutral" in canonical_targets:
        return ["neutral", *[name for name in names if name != "neutral"]]
    return names


def _ordered_tasks(paths: dict[str, KinematicPath]) -> list[str]:
    ordered = [task for task in TASKS if task in paths]
    ordered.extend(sorted(task for task in paths if task not in TASKS))
    return ordered
