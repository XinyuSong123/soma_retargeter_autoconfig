"""Canonical-motion chain projection scaffolding.

This module consumes already-built canonical semantic targets and projects each
task through the robot chain. Desired targets come from the canonical motion's
semantic transforms, not from neutral FK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .chain_projection import ProjectionResult, project_endpoint_position, project_torso_orientation
from .kinematic_paths import KinematicPath
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import relative_transform


@dataclass(frozen=True)
class CanonicalTaskProjection:
    motion: str
    task: str
    reference_semantic: str
    target_semantic: str
    desired_source: str
    neutral_as_desired: bool
    result: ProjectionResult

    def to_json(self) -> dict:
        payload = self.result.to_json()
        payload.update(
            {
                "motion": self.motion,
                "task": self.task,
                "reference_semantic": self.reference_semantic,
                "target_semantic": self.target_semantic,
                "desired_source": self.desired_source,
                "neutral_as_desired": self.neutral_as_desired,
            }
        )
        return payload


def project_canonical_targets(
    adapter: MuJoCoRuntimeModelAdapter,
    sites: Mapping[str, SemanticSite],
    paths: Mapping[str, KinematicPath],
    canonical_targets: Mapping[str, object],
    *,
    use_continuity_prior: bool = True,
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-8,
) -> dict[str, dict[str, CanonicalTaskProjection]]:
    """Project all available chain tasks for each canonical motion.

    The returned structure is motion -> task -> projection. When continuity is
    enabled, each task's previous projected q seeds the same task in the next
    canonical motion in deterministic iteration order.
    """

    previous_by_task: dict[str, np.ndarray] = {}
    output: dict[str, dict[str, CanonicalTaskProjection]] = {}
    for motion, semantic_targets in canonical_targets.items():
        transforms = _target_transforms(semantic_targets)
        output[motion] = project_canonical_motion(
            adapter,
            sites,
            paths,
            transforms,
            motion=motion,
            previous_q_by_task=previous_by_task if use_continuity_prior else None,
            neutral_prior_weight=neutral_prior_weight,
            continuity_prior_weight=continuity_prior_weight,
        )
        if use_continuity_prior:
            for task, projection in output[motion].items():
                previous_by_task[task] = projection.result.chain_q
    return output


def project_canonical_motion(
    adapter: MuJoCoRuntimeModelAdapter,
    sites: Mapping[str, SemanticSite],
    paths: Mapping[str, KinematicPath],
    target_transforms: Mapping[str, np.ndarray],
    *,
    motion: str,
    previous_q_by_task: Mapping[str, np.ndarray] | None = None,
    neutral_prior_weight: float = 1e-8,
    continuity_prior_weight: float = 1e-8,
) -> dict[str, CanonicalTaskProjection]:
    q0 = adapter.neutral_q()
    previous_q_by_task = previous_q_by_task or {}
    projections: dict[str, CanonicalTaskProjection] = {}
    for task, path in paths.items():
        if path.reference not in target_transforms or path.target not in target_transforms:
            continue
        ref_site = sites[path.reference]
        target_site = sites[path.target]
        desired_relative = relative_transform(target_transforms[path.reference], target_transforms[path.target])
        q_seed = previous_q_by_task.get(task, q0)
        common_kwargs = {
            "neutral_q": q0,
            "previous_q": previous_q_by_task.get(task),
            "neutral_prior_weight": neutral_prior_weight,
            "continuity_prior_weight": continuity_prior_weight,
        }
        if task == "torso":
            result = project_torso_orientation(
                adapter,
                q_seed,
                ref_site,
                target_site,
                path.active_velocity_coordinates,
                desired_relative[:3, :3],
                **common_kwargs,
            )
        else:
            result = project_endpoint_position(
                adapter,
                q_seed,
                ref_site,
                target_site,
                path.active_velocity_coordinates,
                desired_relative[:3, 3],
                **common_kwargs,
            )
        projections[task] = CanonicalTaskProjection(
            motion=motion,
            task=task,
            reference_semantic=path.reference,
            target_semantic=path.target,
            desired_source="canonical_semantic_target_relative_transform",
            neutral_as_desired=False,
            result=result,
        )
    return projections


def canonical_projection_json(projections: Mapping[str, Mapping[str, CanonicalTaskProjection]]) -> dict:
    return {motion: {task: result.to_json() for task, result in tasks.items()} for motion, tasks in projections.items()}


def _target_transforms(semantic_targets: object) -> dict[str, np.ndarray]:
    if hasattr(semantic_targets, "transforms"):
        raw = getattr(semantic_targets, "transforms")
    elif isinstance(semantic_targets, Mapping) and "transforms" in semantic_targets:
        raw = semantic_targets["transforms"]
    else:
        raise TypeError("canonical target must expose a transforms mapping")
    return {name: np.asarray(transform, dtype=float) for name, transform in raw.items()}
