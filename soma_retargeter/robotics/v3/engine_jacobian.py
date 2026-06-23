"""Engine-backed relative semantic-site Jacobians."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model_adapter import MuJoCoRuntimeModelAdapter, RobotKinematicState, SemanticSite
from .spatial import relative_site_jacobian_from_world


@dataclass(frozen=True)
class EngineRelativeJacobian:
    translation: np.ndarray
    rotation: np.ndarray
    backend: str
    scalar_dtype: str
    source: str
    finite: bool
    convention: str

    def to_json(self) -> dict:
        return {
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "backend": self.backend,
            "scalar_dtype": self.scalar_dtype,
            "source": self.source,
            "finite": self.finite,
            "convention": self.convention,
        }


def engine_relative_jacobian(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> EngineRelativeJacobian:
    if adapter.__class__.__name__ == "MuJoCoRuntimeModelAdapter":
        return _mujoco_engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
    if adapter.__class__.__name__ == "NewtonRuntimeModelAdapter":
        return _newton_engine_relative_jacobian(adapter, q, reference, target, active_coordinates)
    raise TypeError(f"engine relative Jacobian unavailable for {adapter.__class__.__name__}")


def _mujoco_engine_relative_jacobian(
    adapter: MuJoCoRuntimeModelAdapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> EngineRelativeJacobian:
    idx = list(active_coordinates)
    if not idx:
        return _empty("mujoco", "float64", "mujoco.mj_jac")
    data = getattr(adapter, "_data", None)
    if data is None:
        data = mujoco.MjData(adapter.model)
        adapter._data = data
    data.qpos[:] = np.asarray(q, dtype=float)
    mujoco.mj_forward(adapter.model, data)
    state = RobotKinematicState(
        q=np.asarray(q, dtype=float).copy(),
        body_xpos=np.asarray(data.xpos, dtype=float).copy(),
        body_xmat=np.asarray(data.xmat, dtype=float).reshape(adapter.model.nbody, 3, 3).copy(),
    )
    ref_world = adapter.site_transform(state, reference)
    tgt_world = adapter.site_transform(state, target)
    ja_p = np.zeros((3, adapter.nv))
    ja_w = np.zeros((3, adapter.nv))
    jb_p = np.zeros((3, adapter.nv))
    jb_w = np.zeros((3, adapter.nv))
    mujoco.mj_jac(adapter.model, data, ja_p, ja_w, ref_world[:3, 3], adapter.body_id(reference.body_name))
    mujoco.mj_jac(adapter.model, data, jb_p, jb_w, tgt_world[:3, 3], adapter.body_id(target.body_name))
    translation, rotation = relative_site_jacobian_from_world(
        ref_world,
        tgt_world,
        ja_p[:, idx],
        ja_w[:, idx],
        jb_p[:, idx],
        jb_w[:, idx],
    )
    return _result(translation, rotation, "mujoco", "float64", "mujoco.mj_jac")


def _newton_engine_relative_jacobian(
    adapter,
    q: np.ndarray,
    reference: SemanticSite,
    target: SemanticSite,
    active_coordinates: list[int],
) -> EngineRelativeJacobian:
    idx = list(active_coordinates)
    if not idx:
        return _empty("newton", "float32", "newton.eval_jacobian")
    try:
        import warp as wp

        state = adapter.model.state()
        q_wp = wp.array(np.asarray(q, dtype=np.float32), dtype=wp.float32, device="cpu")
        qd_wp = wp.array(np.zeros(adapter.nv, dtype=np.float32), dtype=wp.float32, device="cpu")
        adapter._newton.eval_fk(adapter.model, q_wp, qd_wp, state)
        spatial = adapter._newton.eval_jacobian(adapter.model, state)
    except Exception as exc:  # pragma: no cover - backend import/runtime dependent
        raise RuntimeError(f"Newton eval_jacobian failed: {type(exc).__name__}: {exc}") from exc
    if spatial is None:
        raise RuntimeError("Newton model has no articulations")
    spatial_np = spatial.numpy()
    state_np = adapter.forward_kinematics(q)
    ref_world = adapter.site_transform(state_np, reference)
    tgt_world = adapter.site_transform(state_np, target)
    ref_p, ref_w = _newton_site_world_jacobian(adapter, spatial_np, reference, ref_world[:3, 3], idx, state_np)
    tgt_p, tgt_w = _newton_site_world_jacobian(adapter, spatial_np, target, tgt_world[:3, 3], idx, state_np)
    translation, rotation = relative_site_jacobian_from_world(ref_world, tgt_world, ref_p, ref_w, tgt_p, tgt_w)
    return _result(translation, rotation, "newton", "float32", "newton.eval_jacobian")


def _newton_site_world_jacobian(adapter, spatial: np.ndarray, site: SemanticSite, point_world: np.ndarray, active: list[int], state):
    body_id = adapter.body_id(site.body_name)
    if body_id < 0:
        return np.zeros((3, len(active))), np.zeros((3, len(active)))
    joint_id = _newton_joint_for_body(adapter, body_id)
    art_idx, joint_start, dof_start = _newton_articulation_for_joint(adapter, joint_id)
    row = (joint_id - joint_start) * 6
    linear = np.zeros((3, len(active)))
    angular = np.zeros((3, len(active)))
    for out_col, dof in enumerate(active):
        local_col = int(dof) - dof_start
        col_count = spatial.shape[2] if spatial.ndim == 3 else spatial.shape[1]
        if local_col < 0 or local_col >= col_count:
            continue
        block = spatial[art_idx, row : row + 6, local_col] if spatial.ndim == 3 else spatial[row : row + 6, local_col]
        linear[:, out_col] = block[:3]
        angular[:, out_col] = block[3:6]
    del state
    point = np.asarray(point_world, dtype=float).reshape(3)
    return linear + np.cross(angular, point[:, None], axis=0), angular


def _newton_joint_for_body(adapter, body_id: int) -> int:
    hits = [jid for jid, child in enumerate(adapter._joint_child) if int(child) == int(body_id)]
    if not hits:
        raise RuntimeError(f"Newton body {body_id} has no owning joint")
    return int(hits[0])


def _newton_articulation_for_joint(adapter, joint_id: int) -> tuple[int, int, int]:
    starts = np.asarray(adapter.model.articulation_start.numpy(), dtype=int)
    for art_idx in range(len(starts) - 1):
        joint_start = int(starts[art_idx])
        joint_end = int(starts[art_idx + 1])
        if joint_start <= joint_id < joint_end:
            return art_idx, joint_start, int(adapter._joint_qd_start[joint_start])
    raise RuntimeError(f"Newton joint {joint_id} is not in an articulation")


def _empty(backend: str, dtype: str, source: str) -> EngineRelativeJacobian:
    return _result(np.zeros((3, 0)), np.zeros((3, 0)), backend, dtype, source)


def _result(translation: np.ndarray, rotation: np.ndarray, backend: str, dtype: str, source: str) -> EngineRelativeJacobian:
    finite = bool(np.all(np.isfinite(translation)) and np.all(np.isfinite(rotation)))
    return EngineRelativeJacobian(
        np.asarray(translation, dtype=float),
        np.asarray(rotation, dtype=float),
        backend,
        dtype,
        source,
        finite,
        "relative_site_jacobian: p_AB=R_A^T(p_B-p_A); Jp=R_A^T(Jp_B-Jp_A)+[p_AB]x R_A^T Jw_A; Jw=R_A^T(Jw_B-Jw_A)",
    )
