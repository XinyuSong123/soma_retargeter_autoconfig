"""Central-difference relative Jacobians in velocity coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine_jacobian import engine_relative_jacobian
from .fd_uncertainty import classify_column, coordinate_step
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
    return coordinate_step(adapter, dof_index)


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
    engine_validation: bool = True,
) -> RelativeJacobian:
    cols_p = []
    cols_r = []
    eps = []
    unstable = []
    discrepancies: list[dict[str, float | int | bool]] = []
    classifications: list[dict] = []
    noise_norms: list[float] = []
    chain_length = _characteristic_length(adapter, q, reference, target)
    engine_cols = None
    if engine_validation:
        try:
            engine = engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
            engine_cols = np.vstack([engine.translation, engine.rotation])
        except Exception:
            engine_cols = None
    for dof in active_coordinates:
        e = coordinate_step(adapter, dof, q=q, chain_length=chain_length)
        c_2h = _column(adapter, q, reference, target, dof, 2.0 * e)
        c_h = _column(adapter, q, reference, target, dof, e)
        c_h2 = _column(adapter, q, reference, target, dof, 0.5 * e)
        jp_2h, jr_2h = c_2h
        jp, jr = c_h
        jp_half, jr_half = c_h2
        diff_2h_h = max(float(np.linalg.norm(jp - jp_2h)), float(np.linalg.norm(jr - jr_2h)))
        diff_h_h2 = max(float(np.linalg.norm(jp_half - jp)), float(np.linalg.norm(jr_half - jr)))
        if diff_h_h2 <= diff_2h_h:
            richardson_p = (4.0 * jp_half - jp) / 3.0
            richardson_r = (4.0 * jr_half - jr) / 3.0
            error_estimate = diff_h_h2 / 3.0
            selected_plateau = "h/h2"
        else:
            richardson_p = (4.0 * jp - jp_2h) / 3.0
            richardson_r = (4.0 * jr - jr_2h) / 3.0
            error_estimate = diff_2h_h / 3.0
            selected_plateau = "2h/h"
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
        zero_tol = max(10.0 * stability_atol, 10.0 * chain_length * np.finfo(float).eps / max(0.5 * e, 1e-12))
        out_col = len(cols_p)
        engine_vector = engine_cols[:, out_col] if engine_cols is not None and out_col < engine_cols.shape[1] else None
        column_report = classify_column(
            dof=dof,
            relevant_block="translation+rotation",
            fd_vector=np.concatenate([richardson_p, richardson_r]),
            engine_vector=engine_vector,
            error_estimate=error_estimate,
            finite=finite,
            zero_tolerance=zero_tol,
            stability_tolerance=max(translation_tolerance, rotation_tolerance),
        )
        classification = column_report.classification
        stable = classification in {"stable_nonzero", "numerically_zero", "unstable_roundoff"}
        if classification == "numerically_zero":
            richardson_p = np.zeros(3)
            richardson_r = np.zeros(3)
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
                "error_estimate": error_estimate,
                "selected_plateau": selected_plateau,
            }
        )
        if not stable:
            unstable.append(dof)
        classifications.append(
            {
                "dof": int(dof),
                "class": classification,
                "classification": classification,
                "relevant_block": column_report.relevant_block,
                "stable": bool(stable),
                "column_norm": col_norm,
                "noise_norm": noise,
                "engine_norm": column_report.engine_norm,
                "fd_norm": column_report.fd_norm,
                "error_estimate": column_report.error_estimate,
                "normalized_error": column_report.normalized_error,
                "selected_plateau": selected_plateau,
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


def _characteristic_length(adapter: MuJoCoRuntimeModelAdapter, q: np.ndarray, reference: SemanticSite, target: SemanticSite) -> float:
    try:
        state = adapter.forward_kinematics(q)
        body_path = adapter.body_path(reference.body_name, target.body_name)
        points = [adapter.site_transform(state, reference)[:3, 3]]
        for body_name in body_path:
            points.append(adapter.body_transform(state, body_name)[:3, 3])
        points.append(adapter.site_transform(state, target)[:3, 3])
    except Exception:
        return 1.0
    length = 0.0
    for a, b in zip(points, points[1:]):
        length += float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
    direct = float(np.linalg.norm(points[-1] - points[0])) if len(points) >= 2 else 0.0
    return max(length, direct, 1.0)


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
        "backend": engine_rel.backend,
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "frobenius_error": float(np.linalg.norm(diff)),
        "engine_translation": engine.tolist(),
        "note": "engine relative site translation compared against finite-difference uncertainty estimate",
    }
