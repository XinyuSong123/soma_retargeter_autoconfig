"""Rest-frame calibration summaries for semantic sites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import invert_transform, normalize, rotation_error, wahba_alignment


@dataclass(frozen=True)
class RestCalibration:
    source_rest_semantic_frames: dict[str, np.ndarray]
    source_provenance: str
    robot_neutral_site_transforms: dict[str, np.ndarray]
    edge_alignment_rotations: dict[str, np.ndarray]
    edge_conditioning: dict[str, float]
    edge_frame_sources: dict[str, str]
    segment_lengths: dict[str, float]
    root_horizontal_scale: float
    vertical_root_scale: float
    source_support_height: float
    robot_support_height: float
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
            "edge_conditioning": self.edge_conditioning,
            "edge_frame_sources": self.edge_frame_sources,
            "segment_lengths": self.segment_lengths,
            "root_horizontal_scale": self.root_horizontal_scale,
            "vertical_root_scale": self.vertical_root_scale,
            "source_support_height": self.source_support_height,
            "robot_support_height": self.robot_support_height,
            "neutral_position_errors": self.neutral_position_errors,
            "neutral_orientation_errors": self.neutral_orientation_errors,
            "recomputed_neutral_errors": {
                "source": "independent_source_rest_reconstruction_vs_runtime_neutral_fk",
                "position_errors": self.neutral_position_errors,
                "orientation_errors": self.neutral_orientation_errors,
                "max_position_error": self.max_position_error,
                "max_orientation_error": self.max_orientation_error,
            },
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
    conditioning = {}
    frame_sources = {}
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
            alignment, cond, source = _semantic_alignment(
                transforms,
                source_transforms,
                parent,
                child,
                edge_name,
            )
            alignments[edge_name] = alignment
            conditioning[edge_name] = cond
            frame_sources[edge_name] = source
            if "fallback" in source:
                fallbacks.append(f"{edge_name}:edge_frame_fallback:{source}")
        except ValueError as exc:
            fallbacks.append(f"{edge_name}:edge_frame_fallback:{exc}")
    root_horizontal_scale, vertical_root_scale = _root_scales(transforms, source_transforms, lengths, fallbacks)
    source_support_height = _support_height(source_transforms)
    robot_support_height = _support_height(transforms)
    reconstructed = _reconstruct_neutral_targets(transforms, source_transforms, alignments, lengths)
    pos_errors, rot_errors = _neutral_errors(transforms, reconstructed)
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
        edge_conditioning=conditioning,
        edge_frame_sources=frame_sources,
        segment_lengths=lengths,
        root_horizontal_scale=root_horizontal_scale,
        vertical_root_scale=vertical_root_scale,
        source_support_height=source_support_height,
        robot_support_height=robot_support_height,
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


def _semantic_alignment(
    robot_transforms: dict[str, np.ndarray],
    source_transforms: dict[str, np.ndarray],
    parent: str,
    child: str,
    edge_name: str,
) -> tuple[np.ndarray, float, str]:
    source_vectors, robot_vectors, labels = _semantic_vector_pairs(
        robot_transforms,
        source_transforms,
        parent,
        child,
    )
    if len(source_vectors) >= 2 and _vector_rank(source_vectors) >= 2 and _vector_rank(robot_vectors) >= 2:
        source_frame, robot_frame, label = _conditioned_edge_frames(source_vectors, robot_vectors, labels)
        rotation = robot_frame @ source_frame.T
        source = np.vstack(source_vectors)
        robot = np.vstack(robot_vectors)
        wahba_rotation = wahba_alignment(source, robot)
        residual = float(np.max(np.linalg.norm((wahba_rotation @ source.T).T - robot, axis=1)))
        conditioning = _conditioning(source_vectors)
        return rotation, conditioning, f"conditioned_edge_frame:{label}:wahba_residual={residual:.3g}"

    robot_edge = _edge_frame(robot_transforms[parent], robot_transforms[child])
    source_edge = _edge_frame(source_transforms[parent], source_transforms[child])
    return robot_edge @ source_edge.T, float("inf"), f"fallback_single_edge_frame:{edge_name}"


def _semantic_vector_pairs(
    robot_transforms: dict[str, np.ndarray],
    source_transforms: dict[str, np.ndarray],
    parent: str,
    child: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    source_vectors: list[np.ndarray] = []
    robot_vectors: list[np.ndarray] = []
    labels: list[str] = []

    def add(label: str, source_vec_world: np.ndarray, robot_vec_world: np.ndarray, parent_name: str = parent) -> None:
        if parent_name not in source_transforms or parent_name not in robot_transforms:
            return
        source_vec = source_transforms[parent_name][:3, :3].T @ source_vec_world
        robot_vec = robot_transforms[parent_name][:3, :3].T @ robot_vec_world
        if np.linalg.norm(source_vec) <= 1e-9 or np.linalg.norm(robot_vec) <= 1e-9:
            return
        source_vectors.append(normalize(source_vec))
        robot_vectors.append(normalize(robot_vec))
        labels.append(label)

    add(
        "edge_direction",
        source_transforms[child][:3, 3] - source_transforms[parent][:3, 3],
        robot_transforms[child][:3, 3] - robot_transforms[parent][:3, 3],
    )
    if {"LeftHand", "RightHand"} <= source_transforms.keys() and {"LeftHand", "RightHand"} <= robot_transforms.keys():
        add(
            "bilateral_hands",
            source_transforms["LeftHand"][:3, 3] - source_transforms["RightHand"][:3, 3],
            robot_transforms["LeftHand"][:3, 3] - robot_transforms["RightHand"][:3, 3],
        )
    if {"LeftFoot", "RightFoot"} <= source_transforms.keys() and {"LeftFoot", "RightFoot"} <= robot_transforms.keys():
        add(
            "bilateral_feet",
            source_transforms["LeftFoot"][:3, 3] - source_transforms["RightFoot"][:3, 3],
            robot_transforms["LeftFoot"][:3, 3] - robot_transforms["RightFoot"][:3, 3],
        )
    if {"Hips", "Chest"} <= source_transforms.keys() and {"Hips", "Chest"} <= robot_transforms.keys():
        add(
            "torso_axis",
            source_transforms["Chest"][:3, 3] - source_transforms["Hips"][:3, 3],
            robot_transforms["Chest"][:3, 3] - robot_transforms["Hips"][:3, 3],
        )
    return source_vectors, robot_vectors, labels


def _conditioned_edge_frames(
    source_vectors: list[np.ndarray],
    robot_vectors: list[np.ndarray],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    source_primary = normalize(source_vectors[0])
    robot_primary = normalize(robot_vectors[0])
    best_index = -1
    best_score = 0.0
    for index in range(1, len(source_vectors)):
        source_proj = source_vectors[index] - source_primary * float(np.dot(source_primary, source_vectors[index]))
        robot_proj = robot_vectors[index] - robot_primary * float(np.dot(robot_primary, robot_vectors[index]))
        score = min(float(np.linalg.norm(source_proj)), float(np.linalg.norm(robot_proj)))
        if score > best_score:
            best_index = index
            best_score = score
    if best_index < 0 or best_score <= 1e-8:
        raise ValueError("semantic vectors are rank deficient after primary edge projection")
    source_frame = _frame_from_primary_secondary(source_primary, source_vectors[best_index])
    robot_frame = _frame_from_primary_secondary(robot_primary, robot_vectors[best_index])
    return source_frame, robot_frame, f"edge_direction+{labels[best_index]}"


def _frame_from_primary_secondary(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    z = normalize(primary)
    y_raw = np.asarray(secondary, dtype=float) - z * float(np.dot(z, secondary))
    y = normalize(y_raw)
    x = normalize(np.cross(y, z))
    y = normalize(np.cross(z, x))
    return np.column_stack([x, y, z])


def _vector_rank(vectors: list[np.ndarray]) -> int:
    if not vectors:
        return 0
    singular_values = np.linalg.svd(np.vstack(vectors), compute_uv=False)
    return int(np.sum(singular_values > max(1e-8, singular_values[0] * 1e-6)))


def _conditioning(vectors: list[np.ndarray]) -> float:
    singular_values = np.linalg.svd(np.vstack(vectors), compute_uv=False)
    if len(singular_values) < 2 or singular_values[1] <= 1e-12:
        return float("inf")
    return float(singular_values[0] / singular_values[1])


def _root_scales(
    robot_transforms: dict[str, np.ndarray],
    source_transforms: dict[str, np.ndarray],
    robot_lengths: dict[str, float],
    fallbacks: list[str],
) -> tuple[float, float]:
    robot_legs = [robot_lengths[name] for name in ("left_leg", "right_leg") if name in robot_lengths]
    source_legs = []
    for name in ("left_leg", "right_leg"):
        parent, child = EDGES[name]
        if parent in source_transforms and child in source_transforms:
            source_legs.append(float(np.linalg.norm(source_transforms[child][:3, 3] - source_transforms[parent][:3, 3])))
    if not robot_legs or not source_legs or float(np.mean(source_legs)) <= 1e-9:
        fallbacks.append("root_scale:missing_or_zero_leg_length")
        return 1.0, 1.0
    scale = float(np.mean(robot_legs) / np.mean(source_legs))
    return scale, scale


def _support_height(transforms: dict[str, np.ndarray]) -> float:
    if "Hips" not in transforms:
        return 0.0
    foot_heights = []
    hips = transforms["Hips"]
    for foot in ("LeftFoot", "RightFoot"):
        if foot not in transforms:
            continue
        rel = hips[:3, :3].T @ (transforms[foot][:3, 3] - hips[:3, 3])
        foot_heights.append(float(rel[2]))
    if not foot_heights:
        return 0.0
    return float(-min(foot_heights))


def _reconstruct_neutral_targets(
    robot_transforms: dict[str, np.ndarray],
    source_transforms: dict[str, np.ndarray],
    alignments: dict[str, np.ndarray],
    lengths: dict[str, float],
) -> dict[str, np.ndarray]:
    if not robot_transforms:
        return {}
    root = "Hips" if "Hips" in robot_transforms else next(iter(robot_transforms))
    reconstructed = {root: robot_transforms[root].copy()}
    for edge_name, (parent, child) in EDGES.items():
        if parent not in reconstructed or parent not in robot_transforms or child not in robot_transforms:
            continue
        robot_rel = invert_transform(robot_transforms[parent]) @ robot_transforms[child]
        if (
            parent not in source_transforms
            or child not in source_transforms
            or edge_name not in alignments
            or edge_name not in lengths
            or lengths[edge_name] <= 1e-9
        ):
            reconstructed[child] = reconstructed[parent] @ robot_rel
            continue
        source_rel = invert_transform(source_transforms[parent]) @ source_transforms[child]
        source_direction = source_rel[:3, 3]
        if np.linalg.norm(source_direction) <= 1e-9:
            reconstructed[child] = reconstructed[parent] @ robot_rel
            continue
        child_t = np.eye(4)
        child_t[:3, 3] = (
            reconstructed[parent][:3, 3]
            + reconstructed[parent][:3, :3] @ (alignments[edge_name] @ normalize(source_direction)) * lengths[edge_name]
        )
        child_t[:3, :3] = reconstructed[parent][:3, :3] @ robot_rel[:3, :3]
        reconstructed[child] = child_t
    return reconstructed


def _neutral_errors(
    robot_transforms: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    pos_errors = {}
    rot_errors = {}
    for name, robot_t in robot_transforms.items():
        if name not in reconstructed:
            pos_errors[name] = float("inf")
            rot_errors[name] = float("inf")
            continue
        pos_errors[name] = float(np.linalg.norm(reconstructed[name][:3, 3] - robot_t[:3, 3]))
        rot_errors[name] = rotation_error(reconstructed[name][:3, :3], robot_t[:3, :3])
    return pos_errors, rot_errors


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
