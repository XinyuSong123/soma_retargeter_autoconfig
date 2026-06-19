# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


def orthonormal_basis_from_jacobian(
    jacobian: np.ndarray,
    *,
    abs_threshold: float = 1e-6,
    rel_threshold: float = 1e-3,
) -> tuple[np.ndarray, list[float], int]:
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 3:
        raise ValueError("jacobian must have shape [3, dof]")
    if matrix.shape[1] == 0:
        return np.zeros((3, 0), dtype=float), [], 0

    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(abs_threshold, rel_threshold * sigma_max)
    rank = int(np.count_nonzero(singular_values > threshold))
    return u[:, :rank].copy(), [float(v) for v in singular_values], rank


def projector_from_basis(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != 3:
        raise ValueError("basis must have shape [3, rank]")
    if basis.shape[1] == 0:
        return np.zeros((3, 3), dtype=float)
    return basis @ basis.T


def project_vector(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float)
    if vec.shape != (3,):
        raise ValueError("vector must have shape [3]")
    return projector_from_basis(basis) @ vec


def rotation_vector_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = rotvec / angle
    half = angle * 0.5
    return np.array([axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half), np.cos(half)], dtype=float)


def quat_xyzw_to_rotation_vector(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    if quat.shape != (4,):
        raise ValueError("quat must have shape [4] in xyzw order")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    x, y, z, w = quat / norm
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = float(np.linalg.norm([x, y, z]))
    if sin_half <= 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(sin_half, w)
    return np.array([x, y, z], dtype=float) / sin_half * angle


def project_relative_rotation_quat_xyzw(quat_xyzw: np.ndarray, rotational_basis: np.ndarray) -> np.ndarray:
    reachable_rotvec = project_vector(quat_xyzw_to_rotation_vector(quat_xyzw), rotational_basis)
    return rotation_vector_to_quat_xyzw(reachable_rotvec)
