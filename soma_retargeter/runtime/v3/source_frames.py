"""Runtime extraction of SOMA source semantic frames."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import numpy as np

from soma_retargeter.robotics.v3.spatial import quat_xyzw_to_matrix


DEFAULT_SEMANTIC_NAMES = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")

SEMANTIC_JOINT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Hips": ("Hips", "Pelvis", "Hip", "Root"),
    "Chest": ("Chest", "UpperChest", "Spine2", "Spine3", "Torso"),
    "LeftHand": ("LeftHand", "LeftWrist", "LHand", "LWrist"),
    "RightHand": ("RightHand", "RightWrist", "RHand", "RWrist"),
    "LeftFoot": ("LeftFoot", "LeftAnkle", "LFoot", "LAnkle"),
    "RightFoot": ("RightFoot", "RightAnkle", "RFoot", "RAnkle"),
}


@dataclass(frozen=True)
class SourceSemanticFrameBatch:
    semantic_names: list[str]
    joint_names: dict[str, str]
    transforms: dict[str, np.ndarray]
    frame_count: int
    sample_rate: float
    source: str
    joint_evidence: dict[str, dict] = field(default_factory=dict)


def resolve_soma_semantic_joints(
    skeleton,
    semantic_names: Iterable[str] = DEFAULT_SEMANTIC_NAMES,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Resolve runtime skeleton joints for required SOMA semantics.

    Resolution is deterministic: exact candidate names win first, followed by
    normalized candidate names. Missing semantics raise instead of silently
    dropping targets.
    """
    joint_names = list(getattr(skeleton, "joint_names", ()))
    exact = {name: name for name in joint_names}
    normalized = {_normalize_joint_name(name): name for name in joint_names}
    resolved: dict[str, str] = {}
    evidence: dict[str, dict] = {}
    missing: list[str] = []
    for semantic in semantic_names:
        candidates = SEMANTIC_JOINT_CANDIDATES.get(semantic, (semantic,))
        joint_name = None
        match_type = ""
        matched_candidate = None
        for candidate in candidates:
            if candidate in exact:
                joint_name = exact[candidate]
                match_type = "exact" if candidate == semantic else "candidate"
                matched_candidate = candidate
                break
        if joint_name is None:
            for candidate in candidates:
                normalized_candidate = _normalize_joint_name(candidate)
                if normalized_candidate in normalized:
                    joint_name = normalized[normalized_candidate]
                    match_type = "normalized_candidate"
                    matched_candidate = candidate
                    break
        if joint_name is None:
            missing.append(semantic)
            continue
        joint_index = int(skeleton.joint_index(joint_name))
        resolved[semantic] = joint_name
        evidence[semantic] = {
            "semantic_name": semantic,
            "joint_name": joint_name,
            "joint_index": joint_index,
            "match_type": match_type,
            "matched_candidate": matched_candidate,
            "candidates": list(candidates),
        }
    if missing:
        preview = ", ".join(joint_names[:16])
        suffix = "..." if len(joint_names) > 16 else ""
        raise ValueError(
            "missing source semantic joint(s): "
            + ", ".join(missing)
            + f"; available joints: {preview}{suffix}"
        )
    return resolved, evidence


def extract_source_semantic_frames(
    animation_buffer,
    *,
    semantic_names: Iterable[str] = DEFAULT_SEMANTIC_NAMES,
    frame_start: int = 0,
    frame_stop: int | None = None,
    max_frames: int | None = None,
    offset_transform: np.ndarray | None = None,
    source: str = "AnimationBuffer",
) -> SourceSemanticFrameBatch:
    """Extract semantic world transforms from an ``AnimationBuffer``."""
    names = list(semantic_names)
    joint_names, evidence = resolve_soma_semantic_joints(animation_buffer.skeleton, names)
    frame_indices = _frame_indices(animation_buffer.num_frames, frame_start, frame_stop, max_frames)
    offset = np.eye(4, dtype=np.float64) if offset_transform is None else np.asarray(offset_transform, dtype=np.float64)
    validate_se3_transform(offset, context="offset_transform")

    output = {semantic: np.empty((len(frame_indices), 4, 4), dtype=np.float64) for semantic in names}
    for out_index, frame_index in enumerate(frame_indices):
        globals_ = animation_buffer.compute_global_transforms(frame_index)
        for semantic in names:
            joint_index = int(animation_buffer.skeleton.joint_index(joint_names[semantic]))
            transform = offset @ transform7_to_matrix(globals_[joint_index])
            validate_se3_transform(transform, context=f"{semantic}[frame={frame_index}]")
            output[semantic][out_index] = transform
    return SourceSemanticFrameBatch(
        semantic_names=names,
        joint_names=joint_names,
        transforms=output,
        frame_count=len(frame_indices),
        sample_rate=float(animation_buffer.sample_rate),
        source=source,
        joint_evidence=evidence,
    )


def transform7_to_matrix(value: np.ndarray) -> np.ndarray:
    """Convert Warp transform numpy layout ``[x, y, z, qx, qy, qz, qw]`` to 4x4."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 7:
        raise ValueError(f"expected transform7 with 7 scalars, got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite transform7 values")
    quat = arr[3:7]
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("near-zero transform quaternion")
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = arr[:3]
    out[:3, :3] = quat_xyzw_to_matrix(quat / norm)
    return out


def validate_se3_transform(transform: np.ndarray, *, context: str = "transform", atol: float = 1e-6) -> None:
    arr = np.asarray(transform, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{context} must be shaped (4, 4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{context} contains non-finite values")
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError(f"{context} is not homogeneous SE(3): invalid last row")
    rotation = arr[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError(f"{context} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(f"{context} rotation determinant is {determinant:g}, expected 1")


def _frame_indices(
    num_frames: int,
    frame_start: int,
    frame_stop: int | None,
    max_frames: int | None,
) -> list[int]:
    start = int(frame_start)
    stop = num_frames if frame_stop is None else int(frame_stop)
    if start < 0:
        raise ValueError("frame_start must be non-negative")
    if stop < start:
        raise ValueError("frame_stop must be greater than or equal to frame_start")
    if start > num_frames:
        raise ValueError(f"frame_start {start} exceeds buffer frame count {num_frames}")
    stop = min(stop, num_frames)
    indices = list(range(start, stop))
    if max_frames is not None:
        limit = int(max_frames)
        if limit < 0:
            raise ValueError("max_frames must be non-negative")
        indices = indices[:limit]
    return indices


def _normalize_joint_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
