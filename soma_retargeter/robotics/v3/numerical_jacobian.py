"""Central-difference relative Jacobians in velocity coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

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

    def to_json(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "epsilons": self.epsilons.tolist(),
            "unstable_columns": self.unstable_columns,
            "active_coordinates": self.active_coordinates,
            "epsilon_discrepancies": self.epsilon_discrepancies,
        }


def coordinate_epsilon(adapter: MuJoCoRuntimeModelAdapter, dof_index: int) -> float:
    info = adapter.coordinate(dof_index)
    if np.isfinite(info.lower) and np.isfinite(info.upper):
        span = max(abs(info.upper - info.lower), 1.0)
    else:
        span = 1.0
    if info.joint_type == "prismatic":
        return float(np.clip(1e-4 * span, 1e-6, 1e-3))
    return float(np.clip(1e-4 * span, 1e-6, 1e-3))


def numerical_relative_jacobian(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    *,
    stability_rtol: float = 5e-2,
    stability_atol: float = 1e-6,
) -> RelativeJacobian:
    cols_p = []
    cols_r = []
    eps = []
    unstable = []
    discrepancies: list[dict[str, float | int | bool]] = []
    for dof in active_coordinates:
        e = coordinate_epsilon(adapter, dof)
        jp, jr = _column(adapter, q, reference, target, dof, e)
        jp_half, jr_half = _column(adapter, q, reference, target, dof, e * 0.5)
        translation_discrepancy = float(np.max(np.abs(jp - jp_half))) if jp.size else 0.0
        rotation_discrepancy = float(np.max(np.abs(jr - jr_half))) if jr.size else 0.0
        translation_norm_discrepancy = float(np.linalg.norm(jp - jp_half))
        rotation_norm_discrepancy = float(np.linalg.norm(jr - jr_half))
        translation_tolerance = float(stability_atol + stability_rtol * max(np.max(np.abs(jp_half)), np.max(np.abs(jp))))
        rotation_tolerance = float(stability_atol + stability_rtol * max(np.max(np.abs(jr_half)), np.max(np.abs(jr))))
        stable = translation_discrepancy <= translation_tolerance and rotation_discrepancy <= rotation_tolerance
        discrepancies.append(
            {
                "dof": int(dof),
                "epsilon": float(e * 0.5),
                "translation_max_abs": translation_discrepancy,
                "rotation_max_abs": rotation_discrepancy,
                "translation_l2": translation_norm_discrepancy,
                "rotation_l2": rotation_norm_discrepancy,
                "translation_tolerance": translation_tolerance,
                "rotation_tolerance": rotation_tolerance,
                "stable": stable,
            }
        )
        if not stable:
            unstable.append(dof)
        cols_p.append(jp_half)
        cols_r.append(jr_half)
        eps.append(e * 0.5)
    if cols_p:
        jp_mat = np.column_stack(cols_p)
        jr_mat = np.column_stack(cols_r)
    else:
        jp_mat = np.zeros((3, 0))
        jr_mat = np.zeros((3, 0))
    if not np.all(np.isfinite(jp_mat)) or not np.all(np.isfinite(jr_mat)):
        raise FloatingPointError("non-finite numerical Jacobian")
    return RelativeJacobian(jp_mat, jr_mat, np.asarray(eps), unstable, list(active_coordinates), discrepancies)


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
    if adapter.__class__.__name__ == "NewtonRuntimeModelAdapter":
        return _newton_translation_jacobian_crosscheck(
            adapter, q, reference, target, active_coordinates, finite_difference_translation
        )
    if adapter.__class__.__name__ != "MuJoCoRuntimeModelAdapter":
        return {"available": False, "note": "engine Jacobian cross-check unavailable for this backend"}
    if not active_coordinates:
        return {"available": True, "max_abs_error": 0.0, "frobenius_error": 0.0, "note": "empty active set"}
    lca = adapter.lca_body(reference.body_name, target.body_name)
    if lca != reference.body_name:
        return {"available": False, "note": "reference is not the LCA; cross-branch relative Jacobian not checked"}
    data = mujoco.MjData(adapter.model)
    data.qpos[:] = np.asarray(q, dtype=float)
    mujoco.mj_forward(adapter.model, data)
    state = adapter.forward_kinematics(q)
    target_world = adapter.site_transform(state, target)
    reference_world = adapter.site_transform(state, reference)
    jacp = np.zeros((3, adapter.nv))
    jacr = np.zeros((3, adapter.nv))
    mujoco.mj_jac(adapter.model, data, jacp, jacr, target_world[:3, 3], adapter.body_id(target.body_name))
    engine = reference_world[:3, :3].T @ jacp[:, active_coordinates]
    diff = engine - finite_difference_translation
    return {
        "available": True,
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "frobenius_error": float(np.linalg.norm(diff)),
        "engine_translation": engine.tolist(),
        "note": "MuJoCo mj_jac target-site translation expressed in semantic reference frame",
    }


def _newton_translation_jacobian_crosscheck(
    adapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
    finite_difference_translation: np.ndarray,
) -> dict:
    if not active_coordinates:
        return {"available": True, "max_abs_error": 0.0, "frobenius_error": 0.0, "note": "empty active set"}
    lca = adapter.lca_body(reference.body_name, target.body_name)
    if lca != reference.body_name:
        return {"available": False, "note": "reference is not the LCA; cross-branch relative Jacobian not checked"}
    try:
        import warp as wp

        state = adapter.model.state()
        q_wp = wp.array(np.asarray(q, dtype=np.float32), dtype=wp.float32, device="cpu")
        qd_wp = wp.array(np.zeros(adapter.nv, dtype=np.float32), dtype=wp.float32, device="cpu")
        adapter._newton.eval_fk(adapter.model, q_wp, qd_wp, state)
        spatial = adapter._newton.eval_jacobian(adapter.model, state).numpy()[0]
    except Exception as exc:
        return {"available": False, "note": f"Newton eval_jacobian failed: {type(exc).__name__}: {exc}"}
    state_np = adapter.forward_kinematics(q)
    reference_world = adapter.site_transform(state_np, reference)
    target_world = adapter.site_transform(state_np, target)
    target_body_id = adapter.body_id(target.body_name)
    block = spatial[target_body_id * 6 : (target_body_id + 1) * 6, :][:, active_coordinates]
    angular = block[3:6, :]
    origin_linear = block[0:3, :]
    target_point = target_world[:3, 3]
    point_world = np.column_stack(
        [origin_linear[:, col] + np.cross(angular[:, col], target_point) for col in range(origin_linear.shape[1])]
    )
    engine = reference_world[:3, :3].T @ point_world
    diff = engine - finite_difference_translation
    return {
        "available": True,
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "frobenius_error": float(np.linalg.norm(diff)),
        "engine_translation": engine.tolist(),
        "note": "Newton eval_jacobian body spatial Jacobian converted to target-site translation in semantic reference frame",
    }
