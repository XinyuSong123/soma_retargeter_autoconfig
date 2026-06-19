# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from soma_retargeter.robotics.morphology import MorphologyAnalysis
from soma_retargeter.robotics.reachability import orthonormal_basis_from_jacobian
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
_SEMANTIC_PARENTS = {
    "Chest": "Hips",
    "LeftArm": "Chest",
    "RightArm": "Chest",
    "LeftForeArm": "LeftArm",
    "RightForeArm": "RightArm",
    "LeftHand": "LeftForeArm",
    "RightHand": "RightForeArm",
    "LeftLeg": "Hips",
    "RightLeg": "Hips",
    "LeftShin": "LeftLeg",
    "RightShin": "RightLeg",
    "LeftFoot": "LeftShin",
    "RightFoot": "RightShin",
}


def _semantic_confidence(semantic: str, body: str, morphology: MorphologyAnalysis) -> float:
    if body in morphology.body_names:
        return 1.0
    body_tokens = body.lower().replace("-", "_")
    semantic_tokens = semantic.lower().replace("left", "l").replace("right", "r")
    return 0.55 if any(token and token in body_tokens for token in semantic_tokens.split("_")) else 0.25


def _hand_orientation_supported(semantic: str, body_path: list[str], joints: list[str]) -> bool:
    if semantic not in {"LeftHand", "RightHand"}:
        return True
    searchable = " ".join([*body_path, *joints]).lower()
    return "wrist" in searchable


def _make_site(semantic: str, body: str, confidence: float, orientation_supported: bool) -> SemanticSite:
    return SemanticSite(
        semantic_name=semantic,
        body_name=body,
        local_position=np.zeros(3, dtype=float),
        local_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        source="explicit" if confidence >= 1.0 else "explicit_unverified",
        confidence=confidence,
        orientation_supported=orientation_supported,
    )


def _task_for_site(site: SemanticSite, chain: KinematicChainProfile | None) -> TaskSpec:
    if site.semantic_name == "Chest":
        basis = chain.rotational_basis.tolist() if chain is not None and chain.rotational_rank > 0 else None
        return TaskSpec(
            name="torso_projected_relative_rotation",
            task_type="projected_relative_rotation",
            source_semantic=site.semantic_name,
            target_site=site.semantic_name,
            reference_site="Hips",
            priority=2,
            position_mask_or_basis=None,
            rotation_mask_or_basis=basis,
            normalized_weight=100.0,
            characteristic_length=chain.total_length if chain is not None else 1.0,
            robust_loss="huber",
            enabled=site.confidence >= 0.7 and chain is not None and chain.rotational_rank > 0,
            reason="projected to reachable torso rotational basis" if chain is not None and chain.rotational_rank > 0 else "disabled: no reachable torso rotation basis",
        )
    if site.semantic_name in _MIDDLE_LIMB_SEMANTICS:
        reference_site = _SEMANTIC_PARENTS.get(site.semantic_name)
        return TaskSpec(
            name=f"{site.semantic_name}_direction",
            task_type="direction",
            source_semantic=site.semantic_name,
            target_site=site.semantic_name,
            reference_site=reference_site,
            priority=3,
            position_mask_or_basis=None,
            rotation_mask_or_basis=None,
            normalized_weight=10.0,
            characteristic_length=chain.total_length if chain is not None else 1.0,
            robust_loss="huber",
            enabled=site.confidence >= 0.7 and reference_site is not None,
            reason="middle semantic uses parent-to-child direction rather than absolute link position",
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
        characteristic_length=chain.total_length if chain is not None else 1.0,
        robust_loss="none" if site.semantic_name in {"LeftFoot", "RightFoot"} else "huber",
        enabled=site.confidence >= 0.7 and site.semantic_name in _END_EFFECTOR_SEMANTICS,
        reason="end-effector position task" if site.semantic_name in _END_EFFECTOR_SEMANTICS else "non-endpoint absolute position disabled",
    )


def _semantic_root_body(semantic: str, semantic_sites: dict[str, SemanticSite]) -> str:
    parent_semantic = _SEMANTIC_PARENTS.get(semantic)
    if parent_semantic is not None and parent_semantic in semantic_sites:
        return semantic_sites[parent_semantic].body_name
    if "Hips" in semantic_sites and semantic != "Hips":
        return semantic_sites["Hips"].body_name
    return semantic_sites[semantic].body_name


def _chain_profile_for_site(
    semantic: str,
    site: SemanticSite,
    semantic_sites: dict[str, SemanticSite],
    morphology: MorphologyAnalysis,
) -> tuple[KinematicChainProfile, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    root_body = _semantic_root_body(semantic, semantic_sites)
    body_path = morphology.body_path(root_body, site.body_name)
    if not body_path:
        warnings.append({"code": "missing_body_path", "semantic": semantic, "root_body": root_body, "tip_body": site.body_name})
        body_path = [site.body_name] if site.body_name in morphology.bodies else []

    joints = morphology.joints_on_path(body_path)
    joint_names = [j.joint_name for j in joints]

    segment_lengths: list[float] = []
    for parent, child in zip(body_path, body_path[1:]):
        length = float(np.linalg.norm(morphology.bodies[child].world_position - morphology.bodies[parent].world_position))
        segment_lengths.append(length)
    total_length = float(sum(segment_lengths)) if segment_lengths else 0.0

    if joints:
        rotational_j = np.stack([j.axis_world_rest for j in joints], axis=1)
        rotational_basis, singular_rot, rotational_rank = orthonormal_basis_from_jacobian(rotational_j)
        tip_pos = morphology.bodies[site.body_name].world_position if site.body_name in morphology.bodies else np.zeros(3)
        translational_cols = []
        for joint in joints:
            joint_pos = morphology.bodies[joint.body_name].world_position
            translational_cols.append(np.cross(joint.axis_world_rest, tip_pos - joint_pos))
        translational_j = np.stack(translational_cols, axis=1)
        translational_basis, singular_trans, translational_rank = orthonormal_basis_from_jacobian(translational_j)
    else:
        rotational_basis = np.zeros((3, 0), dtype=float)
        translational_basis = np.zeros((3, 0), dtype=float)
        singular_rot = []
        singular_trans = []
        rotational_rank = 0
        translational_rank = 0

    if not site.orientation_supported:
        rotational_basis = np.zeros((3, 0), dtype=float)
        singular_rot = []
        rotational_rank = 0

    return (
        KinematicChainProfile(
            name=semantic,
            root_body=root_body,
            tip_site=semantic,
            joint_names=joint_names,
            segment_lengths=segment_lengths,
            total_length=max(total_length, 1e-6),
            translational_basis=translational_basis,
            rotational_basis=rotational_basis,
            translational_rank=translational_rank,
            rotational_rank=rotational_rank,
            singular_values_translation=singular_trans,
            singular_values_rotation=singular_rot,
            confidence=site.confidence,
        ),
        warnings,
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

    pending_sites: dict[str, tuple[str, float]] = {}
    for semantic, body in sorted(ik_map.items()):
        confidence = _semantic_confidence(str(semantic), str(body), morphology)
        if confidence < 0.7:
            warnings.append({"code": "low_semantic_confidence", "semantic": str(semantic), "body": str(body), "confidence": confidence})
        pending_sites[str(semantic)] = (str(body), confidence)

    semantic_sites: dict[str, SemanticSite] = {}
    for semantic, (body, confidence) in pending_sites.items():
        root_body = pending_sites.get(_SEMANTIC_PARENTS.get(semantic, ""), pending_sites.get("Hips", (body, confidence)))[0]
        body_path = morphology.body_path(root_body, body)
        path_joints = [j.joint_name for j in morphology.joints_on_path(body_path)]
        semantic_sites[semantic] = _make_site(
            semantic,
            body,
            confidence,
            _hand_orientation_supported(semantic, body_path, path_joints),
        )

    chains: dict[str, KinematicChainProfile] = {}
    for semantic, site in sorted(semantic_sites.items()):
        chain, chain_warnings = _chain_profile_for_site(semantic, site, semantic_sites, morphology)
        chains[semantic] = chain
        warnings.extend(chain_warnings)

    tasks = [_task_for_site(site, chains.get(site.semantic_name)) for _, site in sorted(semantic_sites.items())]

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
