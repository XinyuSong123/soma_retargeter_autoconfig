# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
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
_SEMANTIC_CHILDREN = {parent: child for child, parent in _SEMANTIC_PARENTS.items()}
_COLLISION_PROXY_SEMANTICS = {
    "Chest",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftLeg",
    "RightLeg",
    "LeftShin",
    "RightShin",
    "LeftHand",
    "RightHand",
}
_COLLISION_PAIR_SEMANTICS = (
    ("Chest", "LeftArm"),
    ("Chest", "RightArm"),
    ("Chest", "LeftForeArm"),
    ("Chest", "RightForeArm"),
    ("Chest", "LeftHand"),
    ("Chest", "RightHand"),
    ("LeftArm", "RightArm"),
    ("LeftForeArm", "RightForeArm"),
    ("LeftHand", "RightHand"),
    ("LeftLeg", "RightLeg"),
    ("LeftShin", "RightShin"),
    ("LeftHand", "LeftLeg"),
    ("LeftHand", "LeftShin"),
    ("LeftHand", "RightLeg"),
    ("LeftHand", "RightShin"),
    ("RightHand", "LeftLeg"),
    ("RightHand", "LeftShin"),
    ("RightHand", "RightLeg"),
    ("RightHand", "RightShin"),
)
_DISTAL_SITE_SEMANTICS = {"LeftHand", "RightHand", "LeftFoot", "RightFoot"}
_DISTAL_SITE_TOKENS = {
    "LeftHand": ("left_hand", "lefthand", "hand", "palm"),
    "RightHand": ("right_hand", "righthand", "hand", "palm"),
    "LeftFoot": ("left_foot", "leftfoot", "foot", "sole", "toe", "heel"),
    "RightFoot": ("right_foot", "rightfoot", "foot", "sole", "toe", "heel"),
}
_SEMANTIC_INFERENCE_SPECS = {
    "Hips": {
        "tokens": ("pelvis", "hips", "hip", "base_link", "base", "root"),
        "reject": ("left", "right", "shoulder", "arm", "elbow", "wrist", "hand", "knee", "ankle", "foot"),
    },
    "Chest": {
        "tokens": ("torso", "chest", "trunk", "spine"),
        "reject": ("left", "right", "arm", "elbow", "wrist", "hand", "leg", "knee", "ankle", "foot"),
    },
    "LeftArm": {"side": "left", "tokens": ("shoulder", "upper_arm", "arm_roll", "arm"), "reject": ("forearm", "elbow", "wrist", "hand")},
    "RightArm": {"side": "right", "tokens": ("shoulder", "upper_arm", "arm_roll", "arm"), "reject": ("forearm", "elbow", "wrist", "hand")},
    "LeftForeArm": {"side": "left", "tokens": ("forearm", "lower_arm", "elbow"), "reject": ("wrist", "hand")},
    "RightForeArm": {"side": "right", "tokens": ("forearm", "lower_arm", "elbow"), "reject": ("wrist", "hand")},
    "LeftHand": {"side": "left", "tokens": ("hand", "wrist", "palm"), "reject": ()},
    "RightHand": {"side": "right", "tokens": ("hand", "wrist", "palm"), "reject": ()},
    "LeftLeg": {"side": "left", "tokens": ("hip_roll", "thigh", "upper_leg", "leg"), "reject": ("knee", "shin", "ankle", "foot")},
    "RightLeg": {"side": "right", "tokens": ("hip_roll", "thigh", "upper_leg", "leg"), "reject": ("knee", "shin", "ankle", "foot")},
    "LeftShin": {"side": "left", "tokens": ("shin", "knee", "lower_leg"), "reject": ("ankle", "foot")},
    "RightShin": {"side": "right", "tokens": ("shin", "knee", "lower_leg"), "reject": ("ankle", "foot")},
    "LeftFoot": {"side": "left", "tokens": ("foot", "ankle_roll", "ankle", "toe", "sole"), "reject": ()},
    "RightFoot": {"side": "right", "tokens": ("foot", "ankle_roll", "ankle", "toe", "sole"), "reject": ()},
}


def _normalize_body_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _side_matches(body_name: str, side: str | None) -> bool:
    if side is None:
        return True
    normalized = _normalize_body_token(body_name)
    compact = normalized.replace("_", "")
    if side == "left":
        return "left" in normalized or normalized.startswith("l_") or compact.startswith("l")
    if side == "right":
        return "right" in normalized or normalized.startswith("r_") or compact.startswith("r")
    return True


def _semantic_body_match_score(semantic: str, body_name: str) -> int:
    spec = _SEMANTIC_INFERENCE_SPECS.get(semantic)
    if spec is None:
        return 0
    normalized = _normalize_body_token(body_name)
    compact = normalized.replace("_", "")
    if not _side_matches(normalized, spec.get("side")):
        return 0
    for reject in spec.get("reject", ()):
        reject_norm = _normalize_body_token(reject)
        if reject_norm in normalized or reject_norm.replace("_", "") in compact:
            return 0

    score = 0
    for token in spec.get("tokens", ()):
        token_norm = _normalize_body_token(token)
        token_compact = token_norm.replace("_", "")
        if normalized == token_norm or compact == token_compact:
            score += 4
        elif token_norm in normalized or token_compact in compact:
            score += 2
    return score


def _infer_semantic_body(semantic: str, morphology: MorphologyAnalysis) -> tuple[str, float] | None:
    best: tuple[int, str] | None = None
    for body_name in morphology.body_names:
        score = _semantic_body_match_score(semantic, body_name)
        if score <= 0:
            continue
        if best is None or score > best[0] or (score == best[0] and body_name < best[1]):
            best = (score, body_name)
    if best is None:
        return None
    confidence = min(0.9, 0.7 + 0.05 * best[0])
    return best[1], confidence


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


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    q_vec = np.array([x, y, z], dtype=float)
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (w * uv + uuv)


def _quat_inverse_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    inverse = np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)
    return _quat_rotate_wxyz(inverse, vec)


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _normalize_site_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _explicit_site_for_semantic(semantic: str, body: str, morphology: MorphologyAnalysis):
    if semantic not in _DISTAL_SITE_SEMANTICS:
        return None
    sites = morphology.sites_by_body.get(body, [])
    if not sites:
        return None
    semantic_token = _normalize_site_token(semantic)
    preferred_tokens = (semantic_token, *_DISTAL_SITE_TOKENS.get(semantic, ()))
    for site in sites:
        site_name = _normalize_site_token(site.site_name)
        compact_name = site_name.replace("_", "")
        if any(token in site_name or token.replace("_", "") in compact_name for token in preferred_tokens):
            return site
    if len(sites) == 1:
        return sites[0]
    return None


def _child_anchor_for_semantic(semantic: str, body: str, morphology: MorphologyAnalysis) -> tuple[np.ndarray, str | None]:
    if semantic not in _DISTAL_SITE_SEMANTICS:
        return np.zeros(3, dtype=float), None
    children = [
        child
        for child in morphology.bodies.values()
        if child.parent_body_name == body and float(np.linalg.norm(child.local_position)) > 1.0e-8
    ]
    if not children:
        return np.zeros(3, dtype=float), None

    preferred_tokens = _DISTAL_SITE_TOKENS.get(semantic, ())
    named_children = []
    for child in children:
        child_name = _normalize_site_token(child.body_name)
        compact_name = child_name.replace("_", "")
        if any(token in child_name or token.replace("_", "") in compact_name for token in preferred_tokens):
            named_children.append(child)
    candidates = named_children or (children if len(children) == 1 else [])
    if not candidates:
        return np.zeros(3, dtype=float), None

    body_info = morphology.bodies.get(body)
    if body_info is None:
        return np.zeros(3, dtype=float), None
    local_direction = np.zeros(3, dtype=float)
    if body_info.parent_body_name in morphology.bodies:
        parent_info = morphology.bodies[body_info.parent_body_name]
        world_direction = body_info.world_position - parent_info.world_position
        local_direction = _quat_inverse_rotate_wxyz(body_info.world_rotation_wxyz, world_direction)
    if float(np.linalg.norm(local_direction)) <= 1.0e-8:
        local_direction = candidates[0].local_position
    local_direction = local_direction / max(float(np.linalg.norm(local_direction)), 1.0e-12)

    best_child = max(candidates, key=lambda child: float(np.dot(child.local_position, local_direction)))
    projection = float(np.dot(best_child.local_position, local_direction))
    if projection <= 1.0e-8:
        return np.zeros(3, dtype=float), None
    return best_child.local_position, "inferred_child"


def _geom_extent_along_local_direction(geom_type: str, size: np.ndarray, local_direction: np.ndarray) -> float:
    if len(size) == 0:
        return 0.0
    direction = local_direction / max(float(np.linalg.norm(local_direction)), 1.0e-12)
    if geom_type == "sphere":
        return float(size[0])
    if geom_type in {"capsule", "cylinder"}:
        radius = float(size[0])
        half_length = float(size[1]) if len(size) > 1 else 0.0
        axial = abs(float(direction[2])) * half_length
        radial = float(np.linalg.norm(direction[:2])) * radius
        return axial + radial
    if geom_type in {"box", "mesh"}:
        return float(np.dot(np.abs(direction[:3]), size[:3]))
    if geom_type == "ellipsoid":
        return float(np.dot(np.abs(direction[:3]), size[:3]))
    return 0.0


def _distal_site_offset_from_geom_bounds(
    semantic: str,
    body: str,
    root_body: str,
    morphology: MorphologyAnalysis,
) -> tuple[np.ndarray, str | None]:
    if semantic not in _DISTAL_SITE_SEMANTICS:
        return np.zeros(3, dtype=float), None
    body_info = morphology.bodies.get(body)
    root_info = morphology.bodies.get(root_body)
    if body_info is None:
        return np.zeros(3, dtype=float), None

    world_direction = body_info.world_position - root_info.world_position if root_info is not None else np.zeros(3, dtype=float)
    if float(np.linalg.norm(world_direction)) <= 1.0e-8 and body_info.parent_body_name in morphology.bodies:
        parent_info = morphology.bodies[body_info.parent_body_name]
        world_direction = body_info.world_position - parent_info.world_position
    if float(np.linalg.norm(world_direction)) <= 1.0e-8:
        return np.zeros(3, dtype=float), None

    local_direction = _quat_inverse_rotate_wxyz(body_info.world_rotation_wxyz, world_direction)
    local_direction = local_direction / max(float(np.linalg.norm(local_direction)), 1.0e-12)

    best_offset: np.ndarray | None = None
    best_projection = -float("inf")
    for geom in morphology.geoms_by_body.get(body, []):
        extent = _geom_extent_along_local_direction(geom.geom_type, geom.size, local_direction)
        if extent <= 0.0:
            continue
        candidate = geom.local_position + local_direction * extent
        projection = float(np.dot(candidate, local_direction))
        if projection > best_projection:
            best_projection = projection
            best_offset = candidate

    if best_offset is None:
        return np.zeros(3, dtype=float), None
    return best_offset, "geom_bounds"


def _make_site(
    semantic: str,
    body: str,
    confidence: float,
    orientation_supported: bool,
    morphology: MorphologyAnalysis,
    root_body: str,
) -> SemanticSite:
    explicit_site = _explicit_site_for_semantic(semantic, body, morphology)
    if explicit_site is not None:
        local_position = explicit_site.local_position
        local_rotation_xyzw = _quat_wxyz_to_xyzw(explicit_site.local_rotation_wxyz)
        source = "explicit_site"
    else:
        local_position, source_override = _child_anchor_for_semantic(semantic, body, morphology)
        if source_override is None:
            local_position, source_override = _distal_site_offset_from_geom_bounds(semantic, body, root_body, morphology)
        local_rotation_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        source = source_override or ("explicit" if confidence >= 1.0 else "explicit_unverified")
    return SemanticSite(
        semantic_name=semantic,
        body_name=body,
        local_position=local_position,
        local_rotation_xyzw=local_rotation_xyzw,
        source=source,
        confidence=confidence,
        orientation_supported=orientation_supported,
    )


def _task_for_site(site: SemanticSite, chain: KinematicChainProfile | None) -> TaskSpec:
    if site.semantic_name == "Chest":
        basis = chain.rotational_basis.tolist() if chain is not None and chain.rotational_rank > 0 else None
        weight = 100.0
        if chain is not None and chain.rotational_rank == 1:
            weight = 1.0
        elif chain is not None and chain.rotational_rank > 1:
            weight = 50.0
        return TaskSpec(
            name="torso_projected_relative_rotation",
            task_type="projected_relative_rotation",
            source_semantic=site.semantic_name,
            target_site=site.semantic_name,
            reference_site="Hips",
            priority=2,
            position_mask_or_basis=None,
            rotation_mask_or_basis=basis,
            normalized_weight=weight,
            characteristic_length=chain.total_length if chain is not None else 1.0,
            robust_loss="huber",
            enabled=site.confidence >= 0.7 and chain is not None and chain.rotational_rank > 0,
            reason=(
                "projected to reachable torso rotational basis; softened for multi-axis torso"
                if chain is not None and chain.rotational_rank > 1
                else "projected to reachable torso rotational basis; weak style target for single-axis torso"
                if chain is not None and chain.rotational_rank == 1
                else "projected to reachable torso rotational basis"
                if chain is not None and chain.rotational_rank > 0
                else "disabled: no reachable torso rotation basis"
            ),
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
    position_weight = 100.0
    if site.semantic_name == "Hips":
        position_weight = 1000.0
    elif site.semantic_name in {"LeftFoot", "RightFoot"}:
        position_weight = 1000.0
    return TaskSpec(
        name=f"{site.semantic_name}_position",
        task_type="normalized_position",
        source_semantic=site.semantic_name,
        target_site=site.semantic_name,
        reference_site=None,
        priority=1 if site.semantic_name in {"Hips", "LeftFoot", "RightFoot"} else 2,
        position_mask_or_basis=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        rotation_mask_or_basis=None,
        normalized_weight=position_weight,
        characteristic_length=chain.total_length if chain is not None else 1.0,
        robust_loss="none" if site.semantic_name in {"LeftFoot", "RightFoot"} else "huber",
        enabled=site.confidence >= 0.7 and site.semantic_name in _END_EFFECTOR_SEMANTICS,
        reason="end-effector position task" if site.semantic_name in _END_EFFECTOR_SEMANTICS else "non-endpoint absolute position disabled",
    )


def _schedule_task_weights_for_morphology(
    tasks: list[TaskSpec],
    chains: dict[str, KinematicChainProfile],
) -> list[TaskSpec]:
    chest_chain = chains.get("Chest")
    scheduled: list[TaskSpec] = []
    for task in tasks:
        if task.task_type == "normalized_position" and task.target_site in {"LeftHand", "RightHand"}:
            hand_chain = chains.get(task.target_site)
            if hand_chain is not None and hand_chain.translational_rank <= 1 and hand_chain.rotational_rank == 0:
                scheduled.append(
                    dataclasses.replace(
                        task,
                        normalized_weight=50.0,
                        reason=f"{task.reason}; softened for low-rank no-wrist hand",
                    )
                )
                continue
        scheduled.append(task)
    tasks = scheduled
    if chest_chain is not None and chest_chain.rotational_rank <= 1:
        scheduled: list[TaskSpec] = []
        for task in tasks:
            if task.task_type == "normalized_position" and task.target_site in {"LeftFoot", "RightFoot"}:
                scheduled.append(
                    dataclasses.replace(
                        task,
                        normalized_weight=500.0,
                        reason=f"{task.reason}; softened for single-axis torso profile",
                    )
                )
            else:
                scheduled.append(task)
        return scheduled
    if chest_chain is None:
        return tasks
    scheduled: list[TaskSpec] = []
    for task in tasks:
        if task.task_type == "normalized_position" and task.target_site in {"LeftFoot", "RightFoot"}:
            scheduled.append(dataclasses.replace(task, normalized_weight=1500.0))
        else:
            scheduled.append(task)
    return scheduled


def _pole_task_for_site(
    site: SemanticSite,
    semantic_sites: dict[str, SemanticSite],
    chain: KinematicChainProfile | None,
) -> TaskSpec | None:
    if site.semantic_name not in _MIDDLE_LIMB_SEMANTICS:
        return None
    reference_site = _SEMANTIC_PARENTS.get(site.semantic_name)
    target_site = _SEMANTIC_CHILDREN.get(site.semantic_name)
    if reference_site is None or target_site is None:
        return None
    enabled = (
        site.confidence >= 0.7
        and reference_site in semantic_sites
        and target_site in semantic_sites
        and semantic_sites[reference_site].confidence >= 0.7
        and semantic_sites[target_site].confidence >= 0.7
    )
    return TaskSpec(
        name=f"{site.semantic_name}_pole_vector",
        task_type="pole_vector",
        source_semantic=site.semantic_name,
        target_site=target_site,
        reference_site=reference_site,
        priority=3,
        position_mask_or_basis=None,
        rotation_mask_or_basis=None,
        normalized_weight=10.0,
        characteristic_length=chain.total_length if chain is not None else 1.0,
        robust_loss="huber",
        enabled=enabled,
        reason="bend plane normal from parent-middle-child semantics" if enabled else "disabled: pole-vector triplet incomplete or low confidence",
    )


def _semantic_root_body(semantic: str, semantic_sites: dict[str, SemanticSite]) -> str:
    parent_semantic = _SEMANTIC_PARENTS.get(semantic)
    if parent_semantic is not None and parent_semantic in semantic_sites:
        return semantic_sites[parent_semantic].body_name
    if "Hips" in semantic_sites and semantic != "Hips":
        return semantic_sites["Hips"].body_name
    return semantic_sites[semantic].body_name


def _site_world_position(site: SemanticSite, morphology: MorphologyAnalysis) -> np.ndarray | None:
    body_info = morphology.bodies.get(site.body_name)
    if body_info is None:
        return None
    return body_info.world_position + _quat_rotate_wxyz(body_info.world_rotation_wxyz, site.local_position)


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
    if site.body_name in morphology.bodies:
        site_offset_length = float(np.linalg.norm(site.local_position))
        if site_offset_length > 1.0e-8:
            segment_lengths.append(site_offset_length)
    total_length = float(sum(segment_lengths)) if segment_lengths else 0.0

    if joints:
        rotational_j = np.stack([j.axis_world_rest for j in joints], axis=1)
        rotational_basis, singular_rot, rotational_rank = orthonormal_basis_from_jacobian(rotational_j)
        tip_pos = _site_world_position(site, morphology)
        if tip_pos is None:
            tip_pos = np.zeros(3)
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


def _bodies_are_adjacent(lhs_body: str, rhs_body: str, morphology: MorphologyAnalysis) -> bool:
    lhs = morphology.bodies.get(lhs_body)
    rhs = morphology.bodies.get(rhs_body)
    if lhs is None or rhs is None:
        return False
    return lhs.parent_body_name == rhs_body or rhs.parent_body_name == lhs_body


def _collision_proxy_for_site(
    semantic: str,
    site: SemanticSite,
    chain: KinematicChainProfile | None,
    morphology: MorphologyAnalysis,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    geoms = morphology.geoms_by_body.get(site.body_name, [])
    if geoms:
        radius = max(float(geom.bounding_radius) for geom in geoms)
        weighted_center = np.zeros(3, dtype=float)
        radius_sum = 0.0
        for geom in geoms:
            weighted_center += geom.local_position * float(geom.bounding_radius)
            radius_sum += float(geom.bounding_radius)
        local_center = weighted_center / max(radius_sum, 1.0e-8)
        return (
            {
                "semantic": semantic,
                "body": site.body_name,
                "shape": "sphere",
                "local_center": local_center.tolist(),
                "radius": max(radius, 1.0e-4),
                "source": "geom_bounds",
                "geom_count": len(geoms),
            },
            None,
        )

    if chain is not None and chain.total_length > 1.0e-5:
        return (
            {
                "semantic": semantic,
                "body": site.body_name,
                "shape": "sphere",
                "local_center": [0.0, 0.0, 0.0],
                "radius": max(float(chain.total_length) * 0.15, 1.0e-4),
                "source": "chain_length_fallback",
                "geom_count": 0,
            },
            {"code": "collision_proxy_from_chain_length", "semantic": semantic, "body": site.body_name},
        )

    return None, {"code": "collision_proxy_unavailable", "semantic": semantic, "body": site.body_name}


def _compile_collision_config(
    semantic_sites: dict[str, SemanticSite],
    chains: dict[str, KinematicChainProfile],
    morphology: MorphologyAnalysis,
    raw_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    raw_collision = raw_config.get("self_collision", raw_config.get("collision", {}))
    if isinstance(raw_collision, dict) and raw_collision.get("enabled") is False:
        return {"enabled": False, "reason": "disabled_by_config", "proxies": [], "pairs": [], "margin": 0.03}, []

    margin = 0.03
    if isinstance(raw_collision, dict):
        try:
            margin = float(raw_collision.get("margin", margin))
        except (TypeError, ValueError):
            warnings.append({"code": "invalid_collision_margin", "value": raw_collision.get("margin")})
            margin = 0.03

    proxies: list[dict[str, Any]] = []
    proxy_by_semantic: dict[str, dict[str, Any]] = {}
    for semantic in sorted(_COLLISION_PROXY_SEMANTICS):
        site = semantic_sites.get(semantic)
        if site is None:
            continue
        proxy, warning = _collision_proxy_for_site(semantic, site, chains.get(semantic), morphology)
        if warning is not None:
            warnings.append(warning)
        if proxy is not None:
            proxies.append(proxy)
            proxy_by_semantic[semantic] = proxy

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for lhs_semantic, rhs_semantic in _COLLISION_PAIR_SEMANTICS:
        lhs = proxy_by_semantic.get(lhs_semantic)
        rhs = proxy_by_semantic.get(rhs_semantic)
        if lhs is None or rhs is None:
            continue
        if _bodies_are_adjacent(lhs["body"], rhs["body"], morphology):
            continue
        key = tuple(sorted((lhs_semantic, rhs_semantic)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "a": lhs_semantic,
                "b": rhs_semantic,
                "body_a": lhs["body"],
                "body_b": rhs["body"],
                "margin": margin,
                "priority": 0,
                "barrier": "smooth_soft_sphere",
            }
        )

    enabled = bool(proxies and pairs)
    if not enabled:
        warnings.append({
            "code": "collision_proxy_disabled",
            "reason": "no usable non-adjacent proxy pairs",
            "proxy_count": len(proxies),
            "pair_count": len(pairs),
        })

    return (
        {
            "enabled": enabled,
            "margin": margin,
            "proxies": proxies,
            "pairs": pairs,
            "source": "geom_bounds_or_chain_length",
            "runtime_barrier": "sphere_pair_optional",
        },
        warnings,
    )


def _compile_root_ground_metadata(
    semantic_sites: dict[str, SemanticSite],
    chains: dict[str, KinematicChainProfile],
    morphology: MorphologyAnalysis,
    raw_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    source_leg_length = 0.9
    raw_root_motion = raw_config.get("root_motion", {})
    if isinstance(raw_root_motion, dict):
        try:
            source_leg_length = float(raw_root_motion.get("source_leg_length_m", source_leg_length))
        except (TypeError, ValueError):
            warnings.append({"code": "invalid_source_leg_length", "value": raw_root_motion.get("source_leg_length_m")})
            source_leg_length = 0.9
    if source_leg_length <= 1.0e-6:
        warnings.append({"code": "invalid_source_leg_length", "value": source_leg_length})
        source_leg_length = 0.9

    hips_site = semantic_sites.get("Hips")
    foot_sites = [semantic_sites[name] for name in ("LeftFoot", "RightFoot") if name in semantic_sites]
    robot_leg_length = 0.0
    nominal_pelvis_height = None
    ground_height = 0.0
    ground_height_source = "default_world_z0"
    confidence = 0.0

    hips_pos = _site_world_position(hips_site, morphology) if hips_site is not None else None
    if hips_site is not None and hips_pos is not None and foot_sites:
        foot_positions = [pos for site in foot_sites if (pos := _site_world_position(site, morphology)) is not None]
        if foot_positions:
            foot_center = np.mean(np.stack(foot_positions, axis=0), axis=0)
            robot_leg_length = float(np.linalg.norm(hips_pos - foot_center))
            min_foot_z = float(min(pos[2] for pos in foot_positions))
            nominal_pelvis_height = float(hips_pos[2] - min_foot_z)
            ground_height = min_foot_z
            ground_height_source = "semantic_foot_rest_min_z"
            confidence = min([hips_site.confidence, *[site.confidence for site in foot_sites]])

    if robot_leg_length <= 1.0e-6:
        candidate_lengths = [
            float(chains[name].total_length)
            for name in ("LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot")
            if name in chains and chains[name].total_length > 1.0e-6
        ]
        if candidate_lengths:
            robot_leg_length = max(candidate_lengths)
            ground_height_source = "default_world_z0"
            confidence = min(confidence or 1.0, 0.5)
            warnings.append({"code": "root_leg_length_from_chain_fallback", "robot_leg_length_m": robot_leg_length})
        else:
            robot_leg_length = source_leg_length
            confidence = 0.0
            warnings.append({"code": "root_leg_length_unavailable", "fallback_m": robot_leg_length})

    raw_ground_barrier = raw_config.get("ground_barrier", {})
    if isinstance(raw_ground_barrier, dict) and "ground_height" in raw_ground_barrier:
        try:
            ground_height = float(raw_ground_barrier["ground_height"])
            ground_height_source = "explicit_ground_barrier"
        except (TypeError, ValueError):
            warnings.append({"code": "invalid_ground_height", "value": raw_ground_barrier.get("ground_height")})

    horizontal_scale = float(robot_leg_length / source_leg_length)
    segment_ratios = {
        "root_horizontal": horizontal_scale,
        "leg_length": horizontal_scale,
    }
    root_motion = {
        "source": "semantic_hips_feet_rest_pose" if confidence > 0.0 else "fallback",
        "horizontal_scale": horizontal_scale,
        "robot_leg_length_m": float(robot_leg_length),
        "source_leg_length_m": float(source_leg_length),
        "robot_nominal_pelvis_height_m": nominal_pelvis_height,
        "vertical_height_source": "robot_nominal_pelvis_height+stance_foot+ground+crouch_ratio",
        "ground_height_m": float(ground_height),
        "ground_height_source": ground_height_source,
        "confidence": float(confidence),
    }
    return {"root_motion": root_motion}, segment_ratios, warnings


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
    for semantic in sorted(_SEMANTIC_INFERENCE_SPECS):
        if semantic in pending_sites:
            continue
        inferred = _infer_semantic_body(semantic, morphology)
        if inferred is None:
            continue
        body, confidence = inferred
        pending_sites[semantic] = (body, confidence)
        warnings.append({
            "code": "inferred_semantic_mapping",
            "semantic": semantic,
            "body": body,
            "confidence": confidence,
        })

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
            morphology,
            root_body,
        )

    chains: dict[str, KinematicChainProfile] = {}
    for semantic, site in sorted(semantic_sites.items()):
        chain, chain_warnings = _chain_profile_for_site(semantic, site, semantic_sites, morphology)
        chains[semantic] = chain
        warnings.extend(chain_warnings)

    tasks = [_task_for_site(site, chains.get(site.semantic_name)) for _, site in sorted(semantic_sites.items())]
    for _, site in sorted(semantic_sites.items()):
        pole_task = _pole_task_for_site(site, semantic_sites, chains.get(site.semantic_name))
        if pole_task is not None:
            tasks.append(pole_task)
    tasks = _schedule_task_weights_for_morphology(tasks, chains)
    collision, collision_warnings = _compile_collision_config(semantic_sites, chains, morphology, raw_config)
    warnings.extend(collision_warnings)
    rest_frame_alignment, segment_ratios, root_warnings = _compile_root_ground_metadata(
        semantic_sites,
        chains,
        morphology,
        raw_config,
    )
    warnings.extend(root_warnings)

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
        rest_frame_alignment=rest_frame_alignment,
        segment_ratios=segment_ratios,
        tasks=tasks,
        contact=raw_config.get("contact_aware_foot_ik", {}),
        collision=collision,
        solver={"priority_weight_bands": {"0": 10000.0, "1": 1000.0, "2": 100.0, "3": 10.0, "4": 1.0}},
        warnings=warnings,
        confidence=profile_confidence,
        source_config_hash=source_hash,
    )
