"""Small SO(3) and transform helpers for the offline v3 compiler."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("cannot normalize near-zero vector")
    return np.asarray(v, dtype=float) / n


def transform(position: np.ndarray | None = None, rotation: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4)
    if position is not None:
        out[:3, 3] = np.asarray(position, dtype=float)
    if rotation is not None:
        out[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    return out


def invert_transform(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def relative_transform(reference_world: np.ndarray, target_world: np.ndarray) -> np.ndarray:
    return invert_transform(reference_world) @ target_world


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q[3] < 0:
        q = -q
    return Rotation.from_quat(q).as_matrix()


def matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(r).as_quat()
    if q[3] < 0:
        q = -q
    return q


def so3_log(r: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(r).as_rotvec()


def so3_exp(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(v, dtype=float)).as_matrix()


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log(a.T @ b)))


def wahba_alignment(source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    """Return R that best maps source vectors onto target vectors."""
    source = np.asarray(source_vectors, dtype=float)
    target = np.asarray(target_vectors, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be shaped (N, 3)")
    if source.shape[0] < 2:
        raise ValueError("at least two vectors are required")
    rotation, _ = Rotation.align_vectors(target, source)
    return rotation.as_matrix()


def frame_from_yz(y_hint: np.ndarray, z_hint: np.ndarray) -> tuple[np.ndarray, float, str]:
    y = normalize(y_hint)
    z_raw = np.asarray(z_hint, dtype=float) - y * float(np.dot(y, z_hint))
    if np.linalg.norm(z_raw) < 1e-8:
        return np.eye(3), 0.0, "degenerate_yz"
    z = normalize(z_raw)
    x = normalize(np.cross(y, z))
    y = normalize(np.cross(z, x))
    return np.column_stack([x, y, z]), 1.0, "primary"
