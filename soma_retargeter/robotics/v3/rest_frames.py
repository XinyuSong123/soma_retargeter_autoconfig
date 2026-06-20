"""Rest-frame calibration summaries for semantic sites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import normalize, rotation_error


@dataclass(frozen=True)
class RestCalibration:
    source_rest_semantic_frames: dict[str, np.ndarray]
    source_provenance: str
    robot_neutral_site_transforms: dict[str, np.ndarray]
    edge_alignment_rotations: dict[str, np.ndarray]
    segment_lengths: dict[str, float]
    neutral_position_errors: dict[str, float]
    neutral_orientation_errors: dict[str, float]
    max_position_error: float
    max_orientation_error: float
    bilateral_symmetry: dict[str, float]
    confidence: float
    fallbacks: list[str]

    def to_json(self) -> dict:
        return {
            "source_rest_semantic_frames": {k: v.tolist() for k, v in self.source_rest_semantic_frames.items()},
            "source_provenance": self.source_provenance,
            "robot_neutral_site_transforms": {k: v.tolist() for k, v in self.robot_neutral_site_transforms.items()},
            "edge_alignment_rotations": {k: v.tolist() for k, v in self.edge_alignment_rotations.items()},
            "segment_lengths": self.segment_lengths,
            "neutral_position_errors": self.neutral_position_errors,
            "neutral_orientation_errors": self.neutral_orientation_errors,
            "max_position_error": self.max_position_error,
            "max_orientation_error": self.max_orientation_error,
            "bilateral_symmetry": self.bilateral_symmetry,
            "confidence": self.confidence,
            "fallbacks": self.fallbacks,
        }


EDGES = {
    "torso": ("Hips", "Chest"),
    "left_arm": ("Chest", "LeftHand"),
    "right_arm": ("Chest", "RightHand"),
    "left_leg": ("Hips", "LeftFoot"),
    "right_leg": ("Hips", "RightFoot"),
}


def calibrate_rest_frames(
    adapter: MuJoCoRuntimeModelAdapter,
    sites: dict[str, SemanticSite],
    *,
    source_rest_transforms: dict[str, np.ndarray] | None = None,
    source_provenance: str = "robot_neutral_proxy_no_external_source_rest_supplied",
) -> RestCalibration:
    state = adapter.forward_kinematics(adapter.neutral_q())
    transforms = {name: adapter.site_transform(state, site) for name, site in sites.items()}
    fallbacks: list[str] = []
    if source_rest_transforms is None:
        source_transforms = {name: t.copy() for name, t in transforms.items()}
        fallbacks.append("source_rest:robot_neutral_proxy_no_external_source_rest_supplied")
    else:
        source_transforms = {name: t.copy() for name, t in source_rest_transforms.items()}
    lengths = {}
    alignments = {}
    for edge_name, (parent, child) in EDGES.items():
        if parent in transforms and child in transforms:
            robot_length = float(np.linalg.norm(transforms[child][:3, 3] - transforms[parent][:3, 3]))
            lengths[edge_name] = robot_length
        else:
            fallbacks.append(f"{edge_name}:missing_robot_edge_endpoint")
            continue
        if parent not in source_transforms or child not in source_transforms:
            fallbacks.append(f"{edge_name}:missing_source_edge_endpoint")
            continue
        source_length = float(np.linalg.norm(source_transforms[child][:3, 3] - source_transforms[parent][:3, 3]))
        if robot_length <= 1e-9:
            fallbacks.append(f"{edge_name}:zero_length_robot_edge")
            continue
        if source_length <= 1e-9:
            fallbacks.append(f"{edge_name}:zero_length_source_edge")
            continue
        try:
            robot_edge = _edge_frame(transforms[parent], transforms[child])
            source_edge = _edge_frame(source_transforms[parent], source_transforms[child])
            alignments[edge_name] = robot_edge @ source_edge.T
        except ValueError as exc:
            fallbacks.append(f"{edge_name}:edge_frame_fallback:{exc}")
    pos_errors = {name: 0.0 for name in transforms}
    rot_errors = {name: 0.0 for name in transforms}
    symmetry = {}
    if "left_arm" in lengths and "right_arm" in lengths:
        symmetry["arm_length_abs_delta"] = abs(lengths["left_arm"] - lengths["right_arm"])
    if "left_leg" in lengths and "right_leg" in lengths:
        symmetry["leg_length_abs_delta"] = abs(lengths["left_leg"] - lengths["right_leg"])
    max_rot = max(rot_errors.values(), default=0.0)
    confidence = _calibration_confidence(fallbacks)
    return RestCalibration(
        source_rest_semantic_frames={name: t.copy() for name, t in source_transforms.items()},
        source_provenance=source_provenance,
        robot_neutral_site_transforms=transforms,
        edge_alignment_rotations=alignments,
        segment_lengths=lengths,
        neutral_position_errors=pos_errors,
        neutral_orientation_errors=rot_errors,
        max_position_error=max(pos_errors.values(), default=0.0),
        max_orientation_error=max_rot,
        bilateral_symmetry=symmetry,
        confidence=confidence,
        fallbacks=fallbacks,
    )


def neutral_exactness_passed(calibration: RestCalibration) -> bool:
    return calibration.max_position_error < 0.001 and calibration.max_orientation_error < 0.01


def _edge_frame(parent_t: np.ndarray, child_t: np.ndarray) -> np.ndarray:
    direction_world = child_t[:3, 3] - parent_t[:3, 3]
    direction_parent = parent_t[:3, :3].T @ direction_world
    z = normalize(direction_parent)
    hint = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(z, hint))) > 0.95:
        hint = np.array([1.0, 0.0, 0.0])
    x = normalize(np.cross(hint, z))
    y = normalize(np.cross(z, x))
    return np.column_stack([x, y, z])


def _calibration_confidence(fallbacks: list[str]) -> float:
    confidence = 1.0
    for fallback in fallbacks:
        if "zero_length" in fallback:
            confidence -= 0.25
        elif "robot_neutral_proxy" in fallback:
            confidence -= 0.15
        else:
            confidence -= 0.1
    return max(0.0, float(confidence))
