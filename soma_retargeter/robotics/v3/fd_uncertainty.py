"""Finite-difference uncertainty helpers for relative Jacobian validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ColumnClass = Literal[
    "stable_nonzero",
    "numerically_zero",
    "engine_fd_mismatch",
    "unstable_roundoff",
    "unstable_nonsmooth",
    "nonfinite",
]


@dataclass(frozen=True)
class FiniteDifferenceEstimate:
    translation: np.ndarray
    rotation: np.ndarray
    epsilons: np.ndarray
    noise_norm: float
    method: str = "central-difference-richardson-2h-h-h2"

    def to_json(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "epsilons": self.epsilons.tolist(),
            "noise_norm": self.noise_norm,
            "method": self.method,
        }


@dataclass(frozen=True)
class JacobianColumnClassification:
    dof: int
    relevant_block: str
    classification: ColumnClass
    engine_norm: float
    fd_norm: float
    error_estimate: float
    normalized_error: float | None

    def to_json(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class JacobianValidationReport:
    columns: list[JacobianColumnClassification]
    relevant_rank_engine: int
    relevant_rank_fd: int
    rank_agreement: bool
    projector_distance: float | None
    passed: bool

    def to_json(self) -> dict:
        return {
            "columns": [column.to_json() for column in self.columns],
            "relevant_rank_engine": self.relevant_rank_engine,
            "relevant_rank_fd": self.relevant_rank_fd,
            "rank_agreement": self.rank_agreement,
            "projector_distance": self.projector_distance,
            "passed": self.passed,
        }


def backend_machine_epsilon(adapter) -> float:
    backend = adapter.__class__.__name__.lower()
    if "newton" in backend:
        return float(np.finfo(np.float32).eps)
    return float(np.finfo(np.float64).eps)


def coordinate_step(adapter, dof_index: int, *, q: np.ndarray | None = None, chain_length: float | None = None) -> float:
    info = adapter.coordinate(dof_index)
    if np.isfinite(info.lower) and np.isfinite(info.upper):
        span = max(abs(float(info.upper) - float(info.lower)), 1.0)
    else:
        span = 1.0
    current = abs(float(q[info.qpos_adr])) if q is not None and info.joint_type in {"revolute", "prismatic"} else 0.0
    scale = max(span, current, 1.0)
    if info.joint_type == "prismatic" and chain_length is not None:
        scale = max(scale, abs(float(chain_length)))
    base = backend_machine_epsilon(adapter) ** (1.0 / 3.0)
    if info.joint_type == "prismatic":
        return float(np.clip(base * scale, 1e-6, 5e-3))
    return float(np.clip(base * scale, 1e-6, 5e-3))


def classify_column(
    *,
    dof: int,
    relevant_block: str,
    fd_vector: np.ndarray,
    engine_vector: np.ndarray | None,
    error_estimate: float,
    finite: bool,
    zero_tolerance: float,
    stability_tolerance: float,
) -> JacobianColumnClassification:
    fd_norm = float(np.linalg.norm(fd_vector))
    engine_norm = float(np.linalg.norm(engine_vector)) if engine_vector is not None else fd_norm
    reference_norm = max(engine_norm, fd_norm)
    normalized_error = None
    if engine_vector is not None:
        normalized_error = float(np.linalg.norm(np.asarray(engine_vector) - np.asarray(fd_vector)) / max(reference_norm, zero_tolerance))
    if not finite:
        classification: ColumnClass = "nonfinite"
    elif reference_norm <= zero_tolerance and error_estimate <= stability_tolerance:
        classification = "numerically_zero"
    elif normalized_error is not None and normalized_error > 0.02 and reference_norm > zero_tolerance:
        classification = "engine_fd_mismatch"
    elif error_estimate <= stability_tolerance:
        classification = "stable_nonzero"
    elif error_estimate <= 0.25 * max(reference_norm, zero_tolerance):
        classification = "unstable_roundoff"
    else:
        classification = "unstable_nonsmooth"
    return JacobianColumnClassification(
        dof=int(dof),
        relevant_block=relevant_block,
        classification=classification,
        engine_norm=engine_norm,
        fd_norm=fd_norm,
        error_estimate=float(error_estimate),
        normalized_error=normalized_error,
    )
