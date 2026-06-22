"""Central-difference relative Jacobians in velocity coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine_jacobian import engine_relative_jacobian
from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import so3_log


@dataclass(frozen=True)
class RelativeJacobian:
    translation: np.ndarray
    rotation: np.ndarray
    epsilons: np.ndarray
    unstable_columns: list[int]
    active_coordinates: list[int]
    epsilon_discrepancies: list[dict[str, float | int | bool]]
    stability_gate_passed: bool
    column_classifications: list[dict] | None = None
    jacobian_noise_norm: float = 0.0
    validation_method: str = "backend-aware-multiscale-fd"

    def to_json(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "epsilons": self.epsilons.tolist(),
            "unstable_columns": self.unstable_columns,
            "active_coordinates": self.active_coordinates,
            "epsilon_discrepancies": self.epsilon_discrepancies,
            "stability_gate_passed": self.stability_gate_passed,
            "column_classifications": self.column_classifications or [],
            "jacobian_noise_norm": self.jacobian_noise_norm,
            "validation_method": self.validation_method,
        }


def coordinate_epsilon(adapter: MuJoCoRuntimeModelAdapter, dof_index: int) -> float:
    info = adapter.coordinate(dof_index)
    if np.isfinite(info.lower) and np.isfinite(info.upper):
        span = max(abs(info.upper - info.lower), 1.0)
    else:
        span = 1.0
    backend = adapter.__class__.__name__.lower()
    dtype_floor = np.sqrt(np.finfo(np.float32).eps) if "newton" in backend else np.sqrt(np.finfo(np.float64).eps)
    base = max(1e-6, float(dtype_floor))
    if info.joint_type == "prismatic":
        return float(np.clip(base * span, 5e-6, 2e-3))
    return float(np.clip(base * span, 5e-6, 2e-3))


def numerical_relative_jacobian(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    *,
    stability_rtol: float = 5e-2,
    stability_atol: float = 1e-6,
    raise_on_unstable: bool = False,
) -> RelativeJacobian:
    cols_p = []
    cols_r = []
    eps = []
    unstable = []
    discrepancies: list[dict[str, float | int | bool]] = []
    classifications: list[dict] = []
    noise_norms: list[float] = []
    for dof in active_coordinates:
        e = coordinate_epsilon(adapter, dof)
        c_2h = _column(adapter, q, reference, target, dof, 2.0 * e)
        c_h = _column(adapter, q, reference, target, dof, e)
        c_h2 = _column(adapter, q, reference, target, dof, 0.5 * e)
        jp_2h, jr_2h = c_2h
        jp, jr = c_h
        jp_half, jr_half = c_h2
        richardson_p = (4.0 * jp_half - jp) / 3.0
        richardson_r = (4.0 * jr_half - jr) / 3.0
        translation_discrepancy = float(max(np.max(np.abs(jp - jp_half)), np.max(np.abs(jp_2h - jp)))) if jp.size else 0.0
        rotation_discrepancy = float(max(np.max(np.abs(jr - jr_half)), np.max(np.abs(jr_2h - jr)))) if jr.size else 0.0
        translation_norm_discrepancy = float(max(np.linalg.norm(jp - jp_half), np.linalg.norm(jp_2h - jp)))
        rotation_norm_discrepancy = float(max(np.linalg.norm(jr - jr_half), np.linalg.norm(jr_2h - jr)))
        translation_tolerance = float(stability_atol + stability_rtol * max(np.max(np.abs(jp_half)), np.max(np.abs(jp)), np.max(np.abs(jp_2h))))
        rotation_tolerance = float(stability_atol + stability_rtol * max(np.max(np.abs(jr_half)), np.max(np.abs(jr)), np.max(np.abs(jr_2h))))
        finite = all(np.all(np.isfinite(x)) for x in (jp_2h, jr_2h, jp, jr, jp_half, jr_half))
        col_norm = float(max(np.linalg.norm(richardson_p), np.linalg.norm(richardson_r)))
        max_column_norm = float(max(np.linalg.norm(jp_2h), np.linalg.norm(jr_2h), np.linalg.norm(jp), np.linalg.norm(jr), np.linalg.norm(jp_half), np.linalg.norm(jr_half)))
        noise = float(max(translation_norm_discrepancy, rotation_norm_discrepancy))
        zero_tol = max(10.0 * stability_atol, 10.0 * np.finfo(float).eps / max(0.5 * e, 1e-12))
        if not finite:
            classification = "nonfinite"
            stable = False
        elif max_column_norm <= zero_tol:
            classification = "numerically_zero"
            stable = True
            richardson_p = np.zeros(3)
            richardson_r = np.zeros(3)
        elif translation_discrepancy <= translation_tolerance and rotation_discrepancy <= rotation_tolerance:
            classification = "stable_nonzero"
            stable = True
        elif noise <= 0.25 * col_norm:
            classification = "unstable_roundoff"
            stable = True
        else:
            classification = "unstable_nonsmooth"
            stable = False
        discrepancies.append(
            {
                "dof": int(dof),
                "epsilon": float(e),
                "translation_max_abs": translation_discrepancy,
                "rotation_max_abs": rotation_discrepancy,
                "translation_l2": translation_norm_discrepancy,
                "rotation_l2": rotation_norm_discrepancy,
                "translation_tolerance": translation_tolerance,
                "rotation_tolerance": rotation_tolerance,
                "stable": stable,
                "classification": classification,
                "column_norm": col_norm,
                "noise_norm": noise,
            }
        )
        if not stable:
            unstable.append(dof)
        classifications.append(
            {
                "dof": int(dof),
                "class": classification,
                "stable": bool(stable),
                "column_norm": col_norm,
                "noise_norm": noise,
                "epsilons": [float(2.0 * e), float(e), float(0.5 * e)],
            }
        )
        noise_norms.append(noise)
        cols_p.append(richardson_p)
        cols_r.append(richardson_r)
        eps.append(e)
    if cols_p:
        jp_mat = np.column_stack(cols_p)
        jr_mat = np.column_stack(cols_r)
    else:
        jp_mat = np.zeros((3, 0))
        jr_mat = np.zeros((3, 0))
    if not np.all(np.isfinite(jp_mat)) or not np.all(np.isfinite(jr_mat)):
        raise FloatingPointError("non-finite numerical Jacobian")
    if unstable and raise_on_unstable:
        raise FloatingPointError(f"multi-scale stability gate failed for velocity coordinates {unstable}")
    return RelativeJacobian(
        jp_mat,
        jr_mat,
        np.asarray(eps),
        unstable,
        list(active_coordinates),
        discrepancies,
        stability_gate_passed=not unstable,
        column_classifications=classifications,
        jacobian_noise_norm=float(max(noise_norms, default=0.0)),
    )


def _column(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    dof: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros(adapter.nv)
    delta[dof] = epsilon
    q_plus = adapter.integrate(q, delta)
    q_minus = adapter.integrate(q, -delta)
    t_plus = adapter.relative_transform(adapter.forward_kinematics(q_plus), reference, target)
    t_minus = adapter.relative_transform(adapter.forward_kinematics(q_minus), reference, target)
    jp = (t_plus[:3, 3] - t_minus[:3, 3]) / (2.0 * epsilon)
    jr = so3_log(t_plus[:3, :3] @ t_minus[:3, :3].T) / (2.0 * epsilon)
    return jp, jr


def matrix_rank_and_singular_values(mat: np.ndarray, abs_tol: float = 1e-8, rel_tol: float = 1e-5) -> tuple[int, np.ndarray]:
    if mat.size == 0:
        return 0, np.zeros(0)
    sv = np.linalg.svd(mat, compute_uv=False)
    if sv.size == 0:
        return 0, sv
    threshold = max(abs_tol, rel_tol * float(sv[0]))
    return int(np.sum(sv > threshold)), sv


def engine_translation_jacobian_crosscheck(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    finite_difference_translation: np.ndarray,
) -> dict:
    try:
        engine_rel = engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
    except Exception as exc:
        return {"available": False, "note": f"engine relative Jacobian failed: {type(exc).__name__}: {exc}"}
    engine = engine_rel.translation
    diff = engine - finite_difference_translation
    return {
        "available": True,
        "finite": engine_rel.finite,
        "source": engine_rel.source,
        "convention": engine_rel.convention,
        "scalar_dtype": engine_rel.scalar_dtype,
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "frobenius_error": float(np.linalg.norm(diff)),
        "engine_translation": engine.tolist(),
        "note": "engine relative site translation compared against finite-difference uncertainty estimate",
    }
