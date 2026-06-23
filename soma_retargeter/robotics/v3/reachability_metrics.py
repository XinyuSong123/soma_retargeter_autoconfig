"""Rank and subspace metrics used by v3 reachability gates."""

from __future__ import annotations

import numpy as np


def backend_rank_abs_floor(adapter_or_backend: object) -> float:
    """Return the Step 2.1 backend-specific absolute rank floor."""

    backend = (
        str(adapter_or_backend).lower()
        if isinstance(adapter_or_backend, str)
        else adapter_or_backend.__class__.__name__.lower()
    )
    if "newton" in backend:
        return 1e-5
    return 1e-9


def rank_with_uncertainty(
    matrix: np.ndarray,
    *,
    noise_norm: float = 0.0,
    abs_floor: float = 1e-8,
    rel_tol: float = 1e-5,
    k_uncertainty: float = 10.0,
) -> tuple[int, np.ndarray, float]:
    if matrix.size == 0:
        return 0, np.zeros(0), abs_floor
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0, singular_values, abs_floor
    threshold = max(float(abs_floor), float(rel_tol * singular_values[0]), float(k_uncertainty * noise_norm))
    return int(np.sum(singular_values > threshold)), singular_values, threshold


def retained_left_singular_vectors(matrix: np.ndarray, rank: int) -> np.ndarray:
    if matrix.size == 0 or rank <= 0:
        return np.zeros((matrix.shape[0] if matrix.ndim == 2 else 0, 0))
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank]


def projector_distance(u_a: np.ndarray, u_b: np.ndarray) -> float | None:
    if u_a.size == 0 and u_b.size == 0:
        return 0.0
    if u_a.shape[0] != u_b.shape[0]:
        return None
    p_a = u_a @ u_a.T if u_a.size else np.zeros((u_b.shape[0], u_b.shape[0]))
    p_b = u_b @ u_b.T if u_b.size else np.zeros((u_a.shape[0], u_a.shape[0]))
    return float(np.linalg.norm(p_a - p_b, ord=2))


def principal_angles(u_a: np.ndarray, u_b: np.ndarray) -> list[float]:
    if u_a.size == 0 or u_b.size == 0:
        return []
    singular_values = np.linalg.svd(u_a.T @ u_b, compute_uv=False)
    clipped = np.clip(singular_values, -1.0, 1.0)
    return [float(x) for x in np.arccos(clipped)]


def percentile(values: list[float], q: float) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite), q))
