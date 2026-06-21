"""Offset-free semantic target construction for canonical source motions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rest_frames import EDGES, RestCalibration
from .spatial import invert_transform, normalize, rotation_error, so3_exp


@dataclass(frozen=True)
class SemanticTargets:
    transforms: dict[str, np.ndarray]
    segment_length_errors: dict[str, float]
    mode: str = "source_rest_calibrated_reference"
    provenance: dict[str, str] | None = None

    def to_json(self) -> dict:
        return {
            "transforms": {k: v.tolist() for k, v in self.transforms.items()},
            "segment_length_errors": self.segment_length_errors,
            "mode": self.mode,
            "provenance": self.provenance or {},
        }


def build_neutral_targets(calibration: RestCalibration) -> SemanticTargets:
    return build_targets_from_source_semantic_frames(
        calibration,
        calibration.source_rest_semantic_frames,
        mode="neutral",
    )


def canonical_motion_targets(calibration: RestCalibration) -> dict[str, SemanticTargets]:
    base = {k: v.copy() for k, v in calibration.source_rest_semantic_frames.items()}
    source_motions = {
        "neutral": base,
        "root_translation": _translated(base, np.array([0.1, 0.0, 0.0])),
        "global_root_yaw": _rotated_about(base, "Hips", so3_exp(np.array([0.0, 0.0, 0.35]))),
        "torso_pitch": _rotate_child(base, "Hips", "Chest", so3_exp(np.array([0.25, 0.0, 0.0]))),
        "torso_roll": _rotate_child(base, "Hips", "Chest", so3_exp(np.array([0.0, 0.25, 0.0]))),
        "torso_yaw": _rotate_child(base, "Hips", "Chest", so3_exp(np.array([0.0, 0.0, 0.25]))),
        "mixed_torso_rotation": _rotate_child(base, "Hips", "Chest", so3_exp(np.array([0.15, -0.1, 0.2]))),
        "arms_forward": _move_arms(base, np.array([0.18, 0.0, 0.08])),
        "elbow_bend": _move_arms(base, np.array([0.08, 0.0, -0.05])),
        "overhead_reach": _move_arms(base, np.array([0.05, 0.0, 0.22])),
        "squat": _squat(base),
        "single_step_target": _single_step(base),
    }
    return {
        name: build_targets_from_source_semantic_frames(calibration, source_pose, mode=name)
        for name, source_pose in source_motions.items()
    }


def build_targets_from_source_semantic_frames(
    calibration: RestCalibration,
    source_pose: dict[str, np.ndarray],
    *,
    mode: str,
) -> SemanticTargets:
    """Reconstruct robot-space semantic targets from calibrated source semantic frames."""
    root = "Hips" if "Hips" in calibration.robot_neutral_site_transforms else next(iter(calibration.robot_neutral_site_transforms))
    neutral_source = calibration.source_rest_semantic_frames

    transforms: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {
        "builder": "source_rest_semantic_frames+edge_alignment_rotations+robot_segment_lengths",
        "source_provenance": calibration.source_provenance,
    }
    if calibration.source_provenance == "robot_neutral_proxy_no_external_source_rest_supplied":
        provenance["limitation"] = "source rest unavailable; source semantics are robot-neutral proxy frames"

    transforms[root] = _root_target_transform(calibration, source_pose, root)
    for edge_name, (parent, child) in EDGES.items():
        if parent not in calibration.robot_neutral_site_transforms or child not in calibration.robot_neutral_site_transforms:
            continue
        if parent not in transforms:
            transforms[parent] = calibration.robot_neutral_site_transforms[parent].copy()
        transforms[child] = _edge_target_transform(calibration, source_pose, edge_name, parent, child, transforms[parent])

    for name, neutral_t in calibration.robot_neutral_site_transforms.items():
        if name not in transforms:
            transforms[name] = neutral_t.copy()
            provenance[f"{name}:fallback"] = "not_on_calibrated_semantic_edge"
    return SemanticTargets(
        transforms=transforms,
        segment_length_errors=_segment_length_errors(calibration, transforms),
        mode=mode,
        provenance=provenance,
    )


def validate_canonical_targets(
    calibration: RestCalibration,
    targets: dict[str, SemanticTargets],
) -> dict:
    neutral = targets.get("neutral")
    if neutral is None:
        return {"failures": ["missing neutral canonical target"]}
    neutral_errors = _neutral_reconstruction_errors(calibration, neutral.transforms)
    calibration_errors = _calibration_error_check(calibration, neutral.transforms)
    per_motion = {}
    failures = []
    for name, target in targets.items():
        max_len = max((abs(v) for v in target.segment_length_errors.values()), default=0.0)
        edge_delta = _max_parent_frame_edge_delta(neutral.transforms, target.transforms)
        per_motion[name] = {
            "max_segment_length_error": float(max_len),
            "max_parent_frame_edge_delta_from_neutral": float(edge_delta),
            "mode": target.mode,
        }
        if max_len > 1e-9:
            failures.append(f"{name}: segment length error {max_len:g}")
    if neutral_errors["max_position_error"] > 1e-9 or neutral_errors["max_orientation_error"] > 1e-9:
        failures.append("neutral target does not reconstruct robot neutral")
    root_translation_equivariant = per_motion.get("root_translation", {}).get("max_parent_frame_edge_delta_from_neutral", np.inf) < 1e-9
    global_yaw_equivariant = per_motion.get("global_root_yaw", {}).get("max_parent_frame_edge_delta_from_neutral", np.inf) < 1e-9
    if not root_translation_equivariant:
        failures.append("root_translation changes local semantic edge articulation")
    if not global_yaw_equivariant:
        failures.append("global_root_yaw changes local semantic edge articulation")
    return {
        "neutral_reconstruction": neutral_errors,
        "neutral_reconstruction_reference": "robot_neutral_site_transforms",
        "calibration_error_check": calibration_errors,
        "target_builder_reference": "source_rest_semantic_frames+edge_alignment_rotations+robot_segment_lengths",
        "source_provenance": calibration.source_provenance,
        "per_motion": per_motion,
        "root_translation_equivariant": bool(root_translation_equivariant),
        "global_root_yaw_equivariant": bool(global_yaw_equivariant),
        "failures": failures,
    }


def _targets(calibration: RestCalibration, transforms: dict[str, np.ndarray], mode: str) -> SemanticTargets:
    return SemanticTargets(transforms=transforms, segment_length_errors=_segment_length_errors(calibration, transforms), mode=mode)


def _edge_target_transform(
    calibration: RestCalibration,
    source_pose: dict[str, np.ndarray],
    edge_name: str,
    parent: str,
    child: str,
    parent_target: np.ndarray,
) -> np.ndarray:
    robot_parent_neutral = calibration.robot_neutral_site_transforms[parent]
    robot_child_neutral = calibration.robot_neutral_site_transforms[child]
    neutral_rel_robot = invert_transform(robot_parent_neutral) @ robot_child_neutral
    out = parent_target @ neutral_rel_robot

    length = calibration.segment_lengths.get(edge_name)
    source_parent = source_pose.get(parent)
    source_child = source_pose.get(child)
    neutral_source_parent = calibration.source_rest_semantic_frames.get(parent)
    neutral_source_child = calibration.source_rest_semantic_frames.get(child)
    alignment = calibration.edge_alignment_rotations.get(edge_name)
    if (
        length is None
        or length <= 1e-9
        or source_parent is None
        or source_child is None
        or neutral_source_parent is None
        or neutral_source_child is None
        or alignment is None
    ):
        return parent_target @ neutral_rel_robot

    source_rel = invert_transform(source_parent) @ source_child
    neutral_source_rel = invert_transform(neutral_source_parent) @ neutral_source_child
    source_edge = source_rel[:3, 3]
    if np.linalg.norm(source_edge) <= 1e-9:
        return _apply_global_delta(robot_child_neutral, global_delta)

    mapped_direction_parent = alignment @ normalize(source_edge)
    out[:3, 3] = parent_target[:3, 3] + parent_target[:3, :3] @ mapped_direction_parent * float(length)
    source_rotation_delta = neutral_source_rel[:3, :3].T @ source_rel[:3, :3]
    mapped_rotation_delta = alignment @ source_rotation_delta @ alignment.T
    out[:3, :3] = parent_target[:3, :3] @ neutral_rel_robot[:3, :3] @ mapped_rotation_delta
    return out


def _root_target_transform(
    calibration: RestCalibration,
    source_pose: dict[str, np.ndarray],
    root: str,
) -> np.ndarray:
    robot_root_neutral = calibration.robot_neutral_site_transforms[root]
    source_root = source_pose.get(root)
    neutral_source_root = calibration.source_rest_semantic_frames.get(root)
    if source_root is None or neutral_source_root is None:
        return robot_root_neutral.copy()

    root_alignment = calibration.edge_alignment_rotations.get("torso", np.eye(3))
    source_local_delta = neutral_source_root[:3, :3].T @ (source_root[:3, 3] - neutral_source_root[:3, 3])
    mapped_delta = root_alignment @ source_local_delta
    root_offset = np.array(
        [
            mapped_delta[0] * calibration.root_horizontal_scale,
            mapped_delta[1] * calibration.root_horizontal_scale,
            _source_support_height_delta(calibration, source_pose) * calibration.vertical_root_scale,
        ],
        dtype=float,
    )
    source_rotation_delta = neutral_source_root[:3, :3].T @ source_root[:3, :3]
    mapped_rotation_delta = root_alignment @ source_rotation_delta @ root_alignment.T
    out = robot_root_neutral.copy()
    out[:3, 3] = robot_root_neutral[:3, 3] + robot_root_neutral[:3, :3] @ root_offset
    out[:3, :3] = robot_root_neutral[:3, :3] @ mapped_rotation_delta
    return out.copy()


def _source_support_height_delta(calibration: RestCalibration, source_pose: dict[str, np.ndarray]) -> float:
    if "Hips" not in source_pose:
        return 0.0
    hips = source_pose["Hips"]
    heights = []
    for foot in ("LeftFoot", "RightFoot"):
        if foot not in source_pose:
            continue
        rel = hips[:3, :3].T @ (source_pose[foot][:3, 3] - hips[:3, 3])
        heights.append(float(rel[2]))
    if not heights:
        return 0.0
    return float(-min(heights) - calibration.source_support_height)


def _translated(base: dict[str, np.ndarray], delta: np.ndarray, names: list[str] | None = None) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in base.items()}
    selected = names or list(out)
    for name in selected:
        if name in out:
            out[name][:3, 3] += delta
    return out


def _rotated_about(base: dict[str, np.ndarray], pivot_name: str, rotation: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in base.items()}
    if pivot_name not in base:
        return out
    pivot = base[pivot_name][:3, 3]
    for name, t in out.items():
        t[:3, 3] = pivot + rotation @ (t[:3, 3] - pivot)
        t[:3, :3] = rotation @ t[:3, :3]
    return out


def _rotate_child(base: dict[str, np.ndarray], parent: str, child: str, rotation_parent: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in base.items()}
    if parent not in out or child not in out:
        return out
    parent_t = out[parent]
    child_t = out[child]
    old_child_position = child_t[:3, 3].copy()
    rel = child_t[:3, 3] - parent_t[:3, 3]
    world_r = parent_t[:3, :3] @ rotation_parent @ parent_t[:3, :3].T
    child_t[:3, 3] = parent_t[:3, 3] + world_r @ rel
    child_t[:3, :3] = world_r @ child_t[:3, :3]
    if child == "Chest":
        delta = child_t[:3, 3] - old_child_position
        for descendant in ("LeftHand", "RightHand"):
            if descendant in out:
                out[descendant][:3, 3] = child_t[:3, 3] + world_r @ (out[descendant][:3, 3] - old_child_position)
                out[descendant][:3, :3] = world_r @ out[descendant][:3, :3]
    return out


def _move_arms(base: dict[str, np.ndarray], delta_chest: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in base.items()}
    if "Chest" not in out:
        return out
    chest_r = out["Chest"][:3, :3]
    for hand in ("LeftHand", "RightHand"):
        if hand not in out:
            continue
        rel = out[hand][:3, 3] - out["Chest"][:3, 3]
        length = np.linalg.norm(rel)
        if length < 1e-9:
            continue
        desired = rel + chest_r @ delta_chest
        out[hand][:3, 3] = out["Chest"][:3, 3] + desired / np.linalg.norm(desired) * length
    return out


def _single_step(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in base.items()}
    _redirect_endpoint(out, "Hips", "LeftFoot", np.array([0.12, 0.0, 0.03]))
    _redirect_endpoint(out, "Hips", "RightFoot", np.array([-0.04, 0.0, 0.0]))
    return out


def _squat(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = _translated(base, np.array([0.0, 0.0, -0.12]), names=["Hips", "Chest", "LeftHand", "RightHand"])
    for foot in ("LeftFoot", "RightFoot"):
        if "Hips" not in out or foot not in out or "Hips" not in base or foot not in base:
            continue
        original = base[foot][:3, 3] - base["Hips"][:3, 3]
        length = np.linalg.norm(original)
        direction = original + np.array([0.05 if foot == "LeftFoot" else -0.05, 0.0, 0.12])
        if length > 1e-9 and np.linalg.norm(direction) > 1e-9:
            out[foot][:3, 3] = out["Hips"][:3, 3] + direction / np.linalg.norm(direction) * length
    return out


def _redirect_endpoint(out: dict[str, np.ndarray], parent: str, child: str, delta_parent: np.ndarray) -> None:
    if parent not in out or child not in out:
        return
    rel = out[child][:3, 3] - out[parent][:3, 3]
    length = np.linalg.norm(rel)
    if length < 1e-9:
        return
    desired = rel + out[parent][:3, :3] @ delta_parent
    out[child][:3, 3] = out[parent][:3, 3] + desired / np.linalg.norm(desired) * length


def _segment_length_errors(calibration: RestCalibration, transforms: dict[str, np.ndarray]) -> dict[str, float]:
    errors = {}
    for edge_name, (parent, child) in EDGES.items():
        if edge_name not in calibration.segment_lengths or parent not in transforms or child not in transforms:
            continue
        current = float(np.linalg.norm(transforms[child][:3, 3] - transforms[parent][:3, 3]))
        errors[edge_name] = current - calibration.segment_lengths[edge_name]
    return errors


def _neutral_reconstruction_errors(calibration: RestCalibration, transforms: dict[str, np.ndarray]) -> dict[str, float]:
    pos_errors = []
    rot_errors = []
    for name, neutral_t in calibration.robot_neutral_site_transforms.items():
        if name not in transforms:
            continue
        pos_errors.append(float(np.linalg.norm(transforms[name][:3, 3] - neutral_t[:3, 3])))
        rot_errors.append(rotation_error(transforms[name][:3, :3], neutral_t[:3, :3]))
    return {
        "max_position_error": _zero_small(max(pos_errors, default=0.0)),
        "max_orientation_error": _zero_small(max(rot_errors, default=0.0)),
    }


def _calibration_error_check(calibration: RestCalibration, neutral_transforms: dict[str, np.ndarray]) -> dict[str, float | bool]:
    reconstructed = _neutral_reconstruction_errors(calibration, neutral_transforms)
    position_consistent = abs(reconstructed["max_position_error"] - calibration.max_position_error) < 1e-12
    orientation_consistent = abs(reconstructed["max_orientation_error"] - calibration.max_orientation_error) < 1e-12
    return {
        "recorded_max_position_error": float(calibration.max_position_error),
        "recorded_max_orientation_error": float(calibration.max_orientation_error),
        "reconstructed_max_position_error": reconstructed["max_position_error"],
        "reconstructed_max_orientation_error": reconstructed["max_orientation_error"],
        "matches_recorded_calibration_errors": bool(position_consistent and orientation_consistent),
    }


def _max_parent_frame_edge_delta(neutral: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> float:
    deltas = []
    for parent, child in EDGES.values():
        if parent not in neutral or child not in neutral or parent not in target or child not in target:
            continue
        neutral_rel = invert_transform(neutral[parent]) @ neutral[child]
        target_rel = invert_transform(target[parent]) @ target[child]
        deltas.append(float(np.linalg.norm(neutral_rel[:3, 3] - target_rel[:3, 3])))
    return _zero_small(max(deltas, default=0.0))


def _zero_small(value: float, eps: float = 1e-15) -> float:
    value = float(value)
    if abs(value) < eps:
        return 0.0
    return value
