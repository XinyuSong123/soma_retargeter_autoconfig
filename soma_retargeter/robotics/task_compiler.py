# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from soma_retargeter.robotics.morphology import MorphologyAnalysis
from soma_retargeter.robotics.retarget_profile import (
    COMPILER_VERSION,
    QUATERNION_ORDER,
    SCHEMA_VERSION,
    CompiledRetargetProfile,
    KinematicChainProfile,
    SemanticSite,
    TaskSpec,
    stable_hash_payload,
)


_END_EFFECTOR_SEMANTICS = {"LeftHand", "RightHand", "LeftFoot", "RightFoot", "Hips"}
_MIDDLE_LIMB_SEMANTICS = {"LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftLeg", "RightLeg", "LeftShin", "RightShin"}


def _semantic_confidence(semantic: str, body: str, morphology: MorphologyAnalysis) -> float:
    if body in morphology.body_names:
        return 1.0
    body_tokens = body.lower().replace("-", "_")
    semantic_tokens = semantic.lower().replace("left", "l").replace("right", "r")
    return 0.55 if any(token and token in body_tokens for token in semantic_tokens.split("_")) else 0.25


def _make_site(semantic: str, body: str, confidence: float) -> SemanticSite:
    orientation_supported = semantic not in {"LeftHand", "RightHand"}
    return SemanticSite(
        semantic_name=semantic,
        body_name=body,
        local_position=np.zeros(3, dtype=float),
        local_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        source="explicit" if confidence >= 1.0 else "explicit_unverified",
        confidence=confidence,
        orientation_supported=orientation_supported,
    )


def _task_for_site(site: SemanticSite) -> TaskSpec:
    if site.semantic_name == "Chest":
        return TaskSpec(
            name="torso_projected_relative_rotation",
            task_type="projected_relative_rotation",
            source_semantic=site.semantic_name,
            target_site=site.semantic_name,
            reference_site="Hips",
            priority=2,
            position_mask_or_basis=None,
            rotation_mask_or_basis=[[0.0], [0.0], [1.0]],
            normalized_weight=100.0,
            characteristic_length=1.0,
            robust_loss="huber",
            enabled=site.confidence >= 0.7,
            reason="reachable yaw-only default until Jacobian rank is available",
        )
    if site.semantic_name in _MIDDLE_LIMB_SEMANTICS:
        return TaskSpec(
            name=f"{site.semantic_name}_direction",
            task_type="direction",
            source_semantic=site.semantic_name,
            target_site=site.semantic_name,
            reference_site=None,
            priority=3,
            position_mask_or_basis=None,
            rotation_mask_or_basis=None,
            normalized_weight=10.0,
            characteristic_length=1.0,
            robust_loss="huber",
            enabled=site.confidence >= 0.7,
            reason="middle semantic uses direction rather than absolute link position",
        )
    return TaskSpec(
        name=f"{site.semantic_name}_position",
        task_type="normalized_position",
        source_semantic=site.semantic_name,
        target_site=site.semantic_name,
        reference_site=None,
        priority=1 if site.semantic_name in {"Hips", "LeftFoot", "RightFoot"} else 2,
        position_mask_or_basis=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        rotation_mask_or_basis=None,
        normalized_weight=1000.0 if site.semantic_name in {"Hips", "LeftFoot", "RightFoot"} else 100.0,
        characteristic_length=1.0,
        robust_loss="none" if site.semantic_name in {"LeftFoot", "RightFoot"} else "huber",
        enabled=site.confidence >= 0.7 and site.semantic_name in _END_EFFECTOR_SEMANTICS,
        reason="end-effector position task" if site.semantic_name in _END_EFFECTOR_SEMANTICS else "non-endpoint absolute position disabled",
    )


def compile_retarget_profile(
    *,
    robot_name: str,
    raw_config: dict[str, Any],
    morphology: MorphologyAnalysis,
    source_config_path: str | Path | None = None,
) -> CompiledRetargetProfile:
    warnings = list(morphology.warnings)
    ik_map = raw_config.get("ik_map", {})
    if not isinstance(ik_map, dict):
        ik_map = {}
        warnings.append({"code": "invalid_ik_map", "message": "ik_map must be a mapping"})

    semantic_sites: dict[str, SemanticSite] = {}
    for semantic, body in sorted(ik_map.items()):
        confidence = _semantic_confidence(str(semantic), str(body), morphology)
        if confidence < 0.7:
            warnings.append({"code": "low_semantic_confidence", "semantic": str(semantic), "body": str(body), "confidence": confidence})
        semantic_sites[str(semantic)] = _make_site(str(semantic), str(body), confidence)

    tasks = [_task_for_site(site) for _, site in sorted(semantic_sites.items())]

    chains: dict[str, KinematicChainProfile] = {}
    for semantic in sorted(semantic_sites):
        chains[semantic] = KinematicChainProfile(
            name=semantic,
            root_body=semantic_sites.get("Hips", semantic_sites[semantic]).body_name,
            tip_site=semantic,
            joint_names=[],
            segment_lengths=[],
            total_length=1.0,
            translational_basis=np.eye(3, dtype=float),
            rotational_basis=np.eye(3, dtype=float) if semantic_sites[semantic].orientation_supported else np.zeros((3, 0), dtype=float),
            translational_rank=3,
            rotational_rank=3 if semantic_sites[semantic].orientation_supported else 0,
            singular_values_translation=[],
            singular_values_rotation=[],
            confidence=semantic_sites[semantic].confidence,
        )

    confidences = [site.confidence for site in semantic_sites.values()]
    profile_confidence = float(min(confidences)) if confidences else 0.0
    source_hash = stable_hash_payload(raw_config if source_config_path is None else {"path": str(source_config_path), "config": raw_config})

    return CompiledRetargetProfile(
        schema_version=SCHEMA_VERSION,
        compiler_version=COMPILER_VERSION,
        quaternion_order=QUATERNION_ORDER,
        robot_fingerprint=morphology.robot_fingerprint,
        source_skeleton_fingerprint="soma-v1",
        morphology_summary={"robot_name": robot_name, **morphology.summary()},
        semantic_sites=semantic_sites,
        chains=chains,
        rest_frame_alignment={},
        segment_ratios={},
        tasks=tasks,
        contact=raw_config.get("contact_aware_foot_ik", {}),
        solver={"priority_weight_bands": {"0": 10000.0, "1": 1000.0, "2": 100.0, "3": 10.0, "4": 1.0}},
        warnings=warnings,
        confidence=profile_confidence,
        source_config_hash=source_hash,
    )
