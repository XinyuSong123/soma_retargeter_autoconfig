# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType


def range_normalized_joint_limit_barrier(q, lower, upper, margin_fraction=0.08):
    """Return piecewise-linear residual and derivative for one finite joint coordinate."""
    q = float(q)
    lower = float(lower)
    upper = float(upper)
    margin_fraction = float(margin_fraction)
    span = upper - lower
    if (
        not np.isfinite(q)
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or span <= 1.0e-8
        or span > 1.0e5
        or margin_fraction <= 0.0
    ):
        return 0.0, 0.0
    margin_width = max(span * min(margin_fraction, 0.49), 1.0e-8)
    safe_lower = lower + margin_width
    safe_upper = upper - margin_width
    if q < safe_lower:
        return (q - safe_lower) / margin_width, 1.0 / margin_width
    if q > safe_upper:
        return (q - safe_upper) / margin_width, 1.0 / margin_width
    return 0.0, 0.0


@wp.func
def _wp_smooth_joint_filter_func(
    x            : wp.float32,
    lower_limit  : wp.float32,
    upper_limit  : wp.float32,
    padding_limit: wp.float32,
    m            : wp.float32,
    p            : wp.float32
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += (padding_limit - c)
    upper_limit -= (padding_limit + c)
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit-x, x-upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),           # (n_batch, n_coords)
    dof_to_coord: wp.array1d(dtype=wp.int32),        # (n_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32), # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32), # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),       # (n_coords)
    weight: wp.array1d(dtype=wp.float32),            # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),     # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    if mask > 0.0:
        lower = joint_limit_lower[dof_idx]
        upper = joint_limit_upper[dof_idx]
        c = (lower + upper) * 0.5

        q = joint_q[problem, coord_idx]
        error = (q - c)

        smoother = _wp_smooth_joint_filter_func(error, lower, upper, 1.02, 1.0, 6.5)
        residuals[problem, start_idx + dof_idx] = error * smoother * weight[0] * mask
    else:
        residuals[problem, start_idx + dof_idx] = 0.0


@wp.kernel
def _range_normalized_joint_limit_barrier_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    joint_limit_lower: wp.array1d(dtype=wp.float32),
    joint_limit_upper: wp.array1d(dtype=wp.float32),
    coord_masks: wp.array1d(dtype=wp.float32),
    finite_limit_masks: wp.array1d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    margin_fraction: wp.float32,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx] * finite_limit_masks[dof_idx]
    if mask <= 0.0:
        residuals[problem, start_idx + dof_idx] = 0.0
        return

    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]
    span = upper - lower
    margin = wp.min(margin_fraction, 0.49)
    margin_width = wp.max(span * margin, 1.0e-8)
    safe_lower = lower + margin_width
    safe_upper = upper - margin_width
    q = joint_q[problem, coord_idx]

    residual = 0.0
    if q < safe_lower:
        residual = (q - safe_lower) / margin_width
    elif q > safe_upper:
        residual = (q - safe_upper) / margin_width
    residuals[problem, start_idx + dof_idx] = residual * weight[0] * mask


@wp.kernel
def _range_normalized_joint_limit_barrier_jac_analytic(
    joint_q: wp.array2d(dtype=wp.float32),
    dof_to_coord: wp.array1d(dtype=wp.int32),
    joint_limit_lower: wp.array1d(dtype=wp.float32),
    joint_limit_upper: wp.array1d(dtype=wp.float32),
    coord_masks: wp.array1d(dtype=wp.float32),
    finite_limit_masks: wp.array1d(dtype=wp.float32),
    margin_fraction: wp.float32,
    weight: wp.array1d(dtype=wp.float32),
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return

    mask = coord_masks[coord_idx] * finite_limit_masks[dof_idx]
    if mask <= 0.0:
        return

    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]
    span = upper - lower
    margin = wp.min(margin_fraction, 0.49)
    margin_width = wp.max(span * margin, 1.0e-8)
    safe_lower = lower + margin_width
    safe_upper = upper - margin_width
    q = joint_q[problem, coord_idx]

    if q < safe_lower or q > safe_upper:
        jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask / margin_width


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),  # (1)
):
    out_weight[0] = in_value


@wp.kernel
def _update_weight_at_index(
    in_index: wp.int32,
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),
):
    out_weight[in_index] = in_value


@wp.kernel
def _update_position_target_at_index(
    in_index: wp.int32,
    in_value: wp.vec3,
    out_target: wp.array1d(dtype=wp.vec3),
):
    out_target[in_index] = in_value


@wp.kernel
def _direction_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_dirs: wp.array1d(dtype=wp.vec3),
    parent_link_index: int,
    child_link_index: int,
    weight: wp.float32,
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    base = problem_idx_map[row]

    parent_tf = body_q[row, parent_link_index]
    child_tf = body_q[row, child_link_index]
    parent_pos = wp.vec3(parent_tf[0], parent_tf[1], parent_tf[2])
    child_pos = wp.vec3(child_tf[0], child_tf[1], child_tf[2])
    delta = child_pos - parent_pos
    length = wp.length(delta)
    current = wp.vec3(0.0, 0.0, 0.0)
    if length > 1.0e-6:
        current = delta / length
    target = target_dirs[base]
    error = target - current
    residuals[row, start_idx + 0] = weight * error[0]
    residuals[row, start_idx + 1] = weight * error[1]
    residuals[row, start_idx + 2] = weight * error[2]


@wp.kernel
def _pole_vector_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_normals: wp.array1d(dtype=wp.vec3),
    parent_link_index: int,
    middle_link_index: int,
    child_link_index: int,
    weight: wp.float32,
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    base = problem_idx_map[row]

    parent_tf = body_q[row, parent_link_index]
    middle_tf = body_q[row, middle_link_index]
    child_tf = body_q[row, child_link_index]
    parent_pos = wp.vec3(parent_tf[0], parent_tf[1], parent_tf[2])
    middle_pos = wp.vec3(middle_tf[0], middle_tf[1], middle_tf[2])
    child_pos = wp.vec3(child_tf[0], child_tf[1], child_tf[2])

    normal = wp.cross(middle_pos - parent_pos, child_pos - middle_pos)
    length = wp.length(normal)
    current = wp.vec3(0.0, 0.0, 0.0)
    if length > 1.0e-6:
        current = normal / length

    target = target_normals[base]
    error = target - current
    residuals[row, start_idx + 0] = weight * error[0]
    residuals[row, start_idx + 1] = weight * error[1]
    residuals[row, start_idx + 2] = weight * error[2]


@wp.kernel
def _autodiff_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    start_idx: int,
    component: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, dof_idx = wp.tid()
    if dof_idx < n_dofs:
        jacobian[problem_idx, start_idx + component, dof_idx] = q_grad[problem_idx, dof_idx]


@wp.kernel
def _autodiff_diag_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, dof_idx = wp.tid()
    if dof_idx < n_dofs:
        jacobian[problem_idx, start_idx + dof_idx, dof_idx] = q_grad[problem_idx, dof_idx]


@wp.kernel
def _per_env_position_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_pos: wp.array1d(dtype=wp.vec3),
    weights: wp.array1d(dtype=wp.float32),
    link_index: int,
    link_offset: wp.vec3,
    start_idx: int,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    row = wp.tid()
    base = problem_idx_map[row]

    body_tf = body_q[row, link_index]
    ee_pos = wp.transform_point(body_tf, link_offset)
    error = target_pos[base] - ee_pos
    weight = weights[base]
    residuals[row, start_idx + 0] = weight * error[0]
    residuals[row, start_idx + 1] = weight * error[1]
    residuals[row, start_idx + 2] = weight * error[2]


@wp.kernel
def _per_env_position_jac_analytic(
    link_index: int,
    link_offset: wp.vec3,
    affects_dof: wp.array1d(dtype=wp.uint8),
    weights: wp.array1d(dtype=wp.float32),
    body_q: wp.array2d(dtype=wp.transform),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    start_idx: int,
    n_dofs: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, dof_idx = wp.tid()

    if affects_dof[dof_idx] == 0:
        return

    body_tf = body_q[problem_idx, link_index]
    rot_w = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    pos_w = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    ee_pos_world = pos_w + wp.quat_rotate(rot_w, link_offset)

    S = joint_S_s[problem_idx, dof_idx]
    v_orig = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])
    v_ee = v_orig + wp.cross(omega, ee_pos_world)
    weight = weights[problem_idx]

    jacobian[problem_idx, start_idx + 0, dof_idx] = -weight * v_ee[0]
    jacobian[problem_idx, start_idx + 1, dof_idx] = -weight * v_ee[1]
    jacobian[problem_idx, start_idx + 2, dof_idx] = -weight * v_ee[2]


class IKObjectivePerEnvWeightedPosition(ik.IKObjective):
    """Position objective with independently updateable target and weight per environment."""

    def __init__(self, link_index, link_offset, target_positions, weights):
        super().__init__()
        self.link_index = link_index
        self.link_offset = link_offset
        self.target_positions = target_positions
        self.weights = weights
        self.affects_dof = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != IKJacobianType.ANALYTIC:
            raise NotImplementedError("IKObjectivePerEnvWeightedPosition currently supports analytic Jacobian mode only")

        joint_qd_start_np = model.joint_qd_start.numpy()
        dof_to_joint_np = np.empty(joint_qd_start_np[-1], dtype=np.int32)
        for j in range(len(joint_qd_start_np) - 1):
            dof_to_joint_np[joint_qd_start_np[j]:joint_qd_start_np[j + 1]] = j

        links_per_problem = model.body_count
        joint_child_np = model.joint_child.numpy()
        body_to_joint_np = np.full(links_per_problem, -1, np.int32)
        for j in range(model.joint_count):
            child = joint_child_np[j]
            if child != -1:
                body_to_joint_np[child] = j

        joint_q_start_np = model.joint_q_start.numpy()
        ancestors = np.zeros(len(joint_q_start_np) - 1, dtype=bool)
        joint_parent_np = model.joint_parent.numpy()
        body = self.link_index
        while body != -1:
            j = body_to_joint_np[body]
            if j != -1:
                ancestors[j] = True
            body = joint_parent_np[j] if j != -1 else -1
        self.affects_dof = wp.array(ancestors[dof_to_joint_np].astype(np.uint8), device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return 3

    def set_target_position(self, problem_idx, new_position):
        self._require_batch_layout()
        wp.launch(
            _update_position_target_at_index,
            dim=1,
            inputs=[problem_idx, new_position],
            outputs=[self.target_positions],
            device=self.device,
        )

    def set_weight(self, problem_idx, value):
        self._require_batch_layout()
        wp.launch(
            _update_weight_at_index,
            dim=1,
            inputs=[problem_idx, value],
            outputs=[self.weights],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _per_env_position_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self.target_positions,
                self.weights,
                self.link_index,
                self.link_offset,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        n_dofs = model.joint_dof_count
        wp.launch(
            _per_env_position_jac_analytic,
            dim=[body_q.shape[0], n_dofs],
            inputs=[
                self.link_index,
                self.link_offset,
                self.affects_dof,
                self.weights,
                body_q,
                joint_S_s,
                start_idx,
                n_dofs,
            ],
            outputs=[jacobian],
            device=self.device,
        )


class IKObjectiveDirection(ik.IKObjective):
    """Unit-vector objective between two robot bodies, driven by per-environment target directions."""

    def __init__(self, parent_link_index, child_link_index, target_dirs, weight=1.0):
        super().__init__()
        self.parent_link_index = parent_link_index
        self.child_link_index = child_link_index
        self.target_dirs = target_dirs
        self.weight = float(weight)
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != IKJacobianType.AUTODIFF:
            raise NotImplementedError("IKObjectiveDirection requires autodiff Jacobian mode")
        self.e_arrays = []
        for component in range(3):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                e[prob_idx, self.residual_offset + component] = 1.0
            self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def residual_dim(self):
        return 3

    def set_target_direction(self, problem_idx, new_direction):
        self._require_batch_layout()
        wp.launch(
            _update_position_target_at_index,
            dim=1,
            inputs=[problem_idx, new_direction],
            outputs=[self.target_dirs],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _direction_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self.target_dirs,
                self.parent_link_index,
                self.child_link_index,
                self.weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self.e_arrays is None:
            raise RuntimeError("IKObjectiveDirection buffers are not initialized")
        n_dofs = model.joint_dof_count
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _autodiff_jac_fill,
                dim=[self.n_batch, n_dofs],
                inputs=[q_grad, n_dofs, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


class IKObjectivePoleVector(ik.IKObjective):
    """Bend-plane normal objective for parent-middle-child robot body triplets."""

    def __init__(self, parent_link_index, middle_link_index, child_link_index, target_normals, weight=1.0):
        super().__init__()
        self.parent_link_index = parent_link_index
        self.middle_link_index = middle_link_index
        self.child_link_index = child_link_index
        self.target_normals = target_normals
        self.weight = float(weight)
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode != IKJacobianType.AUTODIFF:
            raise NotImplementedError("IKObjectivePoleVector requires autodiff Jacobian mode")
        self.e_arrays = []
        for component in range(3):
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                e[prob_idx, self.residual_offset + component] = 1.0
            self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def residual_dim(self):
        return 3

    def set_target_normal(self, problem_idx, new_normal):
        self._require_batch_layout()
        wp.launch(
            _update_position_target_at_index,
            dim=1,
            inputs=[problem_idx, new_normal],
            outputs=[self.target_normals],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _pole_vector_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self.target_normals,
                self.parent_link_index,
                self.middle_link_index,
                self.child_link_index,
                self.weight,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self.e_arrays is None:
            raise RuntimeError("IKObjectivePoleVector buffers are not initialized")
        n_dofs = model.joint_dof_count
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _autodiff_jac_fill,
                dim=[self.n_batch, n_dofs],
                inputs=[q_grad, n_dofs, start_idx, component],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()


class IKRangeNormalizedJointLimitBarrier(ik.IKObjective):
    """Range-normalized joint-limit margin barrier with analytic diagonal Jacobian."""

    def __init__(
        self,
        joint_limit_lower,
        joint_limit_upper,
        weight=0.01,
        coord_masks=None,
        margin_fraction=0.08,
    ):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.finite_limit_masks = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)
        self.margin_fraction = float(margin_fraction)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k
        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        lower = self.joint_limit_lower.numpy().astype(np.float32)
        upper = self.joint_limit_upper.numpy().astype(np.float32)
        span = upper - lower
        finite = np.isfinite(lower) & np.isfinite(upper) & (span > 1.0e-8) & (span < 1.0e5)
        self.finite_limit_masks = wp.array(finite.astype(np.float32), dtype=wp.float32, device=self.device)

        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return
        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device,
        )

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _range_normalized_joint_limit_barrier_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self.finite_limit_masks,
                self._weight,
                self.margin_fraction,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})
        q_grad = tape.gradients[dq_dof]
        wp.launch(
            _autodiff_diag_jac_fill,
            dim=[self.n_batch, self.n_dofs],
            inputs=[q_grad, self.n_dofs, start_idx],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _range_normalized_joint_limit_barrier_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self.finite_limit_masks,
                self.margin_fraction,
                self._weight,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    dof_to_coord: wp.array1d(dtype=wp.int32),    # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),   # (n_coords)
    n_dofs: int,
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32), # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),      # (n_batch, n_residuals, n_dofs)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    if coord_idx < 0:
        return

    # Jacobian is diagonal: dr[dof]/dq[dof] = weight
    jacobian[problem, start_idx + dof_idx, dof_idx] = weight[0] * mask


class IKSmoothJointFilter(ik.IKObjective):
    """
    An IK objective that applies a smooth penalty to joint coordinates that approach or exceed specified limits
    using an inverse gaussian filter.

    Args:
        joint_limit_lower (wp.array1d): An array of shape (n_dofs,) containing the lower limits for each joint degree of freedom.
        joint_limit_upper (wp.array1d): An array of shape (n_dofs,) containing the upper limits for each joint degree of freedom.
        weight (float, optional): A scalar weight that controls the strength of the joint limit penalty. Defaults to 0.01.
        coord_masks (wp.array1d, optional): An array of shape (n_coords,) containing mask values for each joint coordinate.
            Mask values should be in the range [0, 1], where 0 means the coordinate is ignored by this objective and 1 means it is fully considered.
            All coords are used by default if no masks are specified.
    """
    def __init__(self, joint_limit_lower, joint_limit_upper, weight=0.01, coord_masks=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([weight], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
                self.coord_masks = None
            elif isinstance(coord_masks, wp.array):
                self.coord_masks = coord_masks
                self.coord_masks_np = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None and len(self.coord_masks_np) == model.joint_coord_count:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)

        # All coords are considered if no coord masks have been declared
        if self.coord_masks is None:
            self.coord_masks = wp.ones(shape=model.joint_coord_count, dtype=wp.float32, device=self.device)

        # Build DOF to coordinate mapping
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k

        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        # For autodiff mode
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return

        wp.launch(
            _update_weight,
            dim=1,
            inputs=[value],
            outputs=[self._weight],
            device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[dq_dof]

        # Use the analytic Jacobian fill since it's simple
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                self.dof_to_coord,
                self.coord_masks,
                self.n_dofs,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
