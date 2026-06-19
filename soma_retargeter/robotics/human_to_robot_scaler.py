# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import warp as wp

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.pose_utils as pose_utils

from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer


@dataclass(frozen=True)
class SegmentLocalTargetBuilder:
    """Build v2 target positions by recursively following source segment directions."""

    joint_names: list[str]
    parent_indices: list[int]
    segment_lengths: np.ndarray
    root_scale: np.ndarray | None = None

    def __post_init__(self):
        if len(self.joint_names) != len(self.parent_indices):
            raise ValueError("joint_names and parent_indices must have the same length")
        lengths = np.asarray(self.segment_lengths, dtype=np.float64)
        if lengths.shape != (len(self.joint_names),):
            raise ValueError("segment_lengths must have shape [num_joints]")
        for idx, parent in enumerate(self.parent_indices):
            if parent >= idx:
                raise ValueError("parent_indices must be topologically sorted, with parent < child")
            if parent < -1:
                raise ValueError("parent index must be -1 or a valid parent index")
            if parent == -1:
                continue
            if not np.isfinite(lengths[idx]) or lengths[idx] <= 0.0:
                raise ValueError(f"segment length for {self.joint_names[idx]!r} must be positive")
        if self.root_scale is not None:
            root_scale = np.asarray(self.root_scale, dtype=np.float64)
            if root_scale.shape != (3,) or not np.all(np.isfinite(root_scale)):
                raise ValueError("root_scale must have shape [3] with finite values")

    def compute_positions(self, source_positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(source_positions, dtype=np.float64)
        if positions.shape != (len(self.joint_names), 3):
            raise ValueError("source_positions must have shape [num_joints, 3]")

        out = np.zeros_like(positions)
        root_scale = np.ones(3, dtype=np.float64) if self.root_scale is None else np.asarray(self.root_scale, dtype=np.float64)
        for idx, parent in enumerate(self.parent_indices):
            if parent == -1:
                out[idx] = positions[idx] * root_scale
                continue
            direction = positions[idx] - positions[parent]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise ValueError(f"source segment {self.joint_names[parent]!r}->{self.joint_names[idx]!r} has near-zero length")
            out[idx] = out[parent] + direction / norm * float(self.segment_lengths[idx])
        return out

    def compute_positions_batch(self, source_positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(source_positions, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (len(self.joint_names), 3):
            raise ValueError("source_positions must have shape [num_frames, num_joints, 3]")
        return np.stack([self.compute_positions(frame) for frame in positions], axis=0)

    def compute_transforms(self, source_transforms: np.ndarray) -> np.ndarray:
        transforms = np.asarray(source_transforms, dtype=np.float64)
        if transforms.shape != (len(self.joint_names), 7):
            raise ValueError("source_transforms must have shape [num_joints, 7]")
        out = np.array(transforms, copy=True)
        out[:, 0:3] = self.compute_positions(transforms[:, 0:3])
        return out

    def compute_transforms_batch(self, source_transforms: np.ndarray) -> np.ndarray:
        transforms = np.asarray(source_transforms, dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (len(self.joint_names), 7):
            raise ValueError("source_transforms must have shape [num_frames, num_joints, 7]")
        out = np.array(transforms, copy=True)
        out[:, :, 0:3] = self.compute_positions_batch(transforms[:, :, 0:3])
        return out


class HumanToRobotScaler:
    """
    Scale and map human motion to robot-aligned effectors.
    """
    def __init__(self, skeleton: Skeleton, human_height, config_file):
        config = io_utils.load_json(config_file)
        self.robot_type = config['robot_type']
        self.skeleton = skeleton

        ratio = human_height / config['human_height_assumption']
        joint_scales = {
            key: float(value) * ratio
            for key, value in config['joint_scales'].items()
        }

        joint_offsets = {}
        joint_offset_data = config['joint_offsets']
        for joint_name, entry in joint_offset_data.items():
            t_offset, q_offset = entry
            joint_offsets[joint_name] = wp.transform(
                wp.vec3(*t_offset),
                wp.normalize(wp.quat(*q_offset)))

        joint_offsets["LeftToeBase"] = joint_offsets["LeftToe"]
        joint_offsets["RightToeBase"] = joint_offsets["RightToe"]

        self.mapped_joints = [name for name in self.skeleton.joint_names if name in joint_scales.keys()]
        self.mapped_joint_indices = wp.array([self.skeleton.joint_index(name) for name in self.mapped_joints], dtype=wp.int32)
        self.mapped_joint_scales = wp.array([joint_scales[name] for name in self.mapped_joints], dtype=wp.float32)
        self.mapped_joint_offsets = wp.array([joint_offsets[name] for name in self.mapped_joints], dtype=wp.transform)
        self.mapped_joint_offsets_np = self.mapped_joint_offsets.numpy()

        joint_parents = config['joint_parents']
        self.mapped_joint_parents = [
            -1 if joint_parents[name] == "" else self.mapped_joints.index(joint_parents[name])
            for name in self.mapped_joints]
        self.mode = str(config.get("mode", "legacy"))
        self.segment_local_builder: SegmentLocalTargetBuilder | None = None
        self.source_reference_segment_lengths = self._compute_source_reference_segment_lengths()

    def _compute_source_reference_segment_lengths(self) -> np.ndarray:
        global_reference = pose_utils.compute_global_pose(
            self.skeleton,
            self.skeleton.reference_local_transforms,
            wp.transform_identity(),
        )
        mapped_positions = np.asarray(
            [global_reference[self.skeleton.joint_index(name)][0:3] for name in self.mapped_joints],
            dtype=np.float64,
        )
        lengths = np.zeros(len(self.mapped_joints), dtype=np.float64)
        for idx, parent in enumerate(self.mapped_joint_parents):
            if parent == -1:
                continue
            lengths[idx] = max(float(np.linalg.norm(mapped_positions[idx] - mapped_positions[parent])), 1e-6)
        return lengths

    def enable_segment_local_from_profile(self, compiled_profile_file):
        profile = io_utils.load_json(compiled_profile_file)
        if profile.get("schema_version") != 2:
            raise ValueError(f"Compiled retarget profile must have schema_version=2: {compiled_profile_file}")

        chains = profile.get("chains", {})
        segment_lengths = np.zeros(len(self.mapped_joints), dtype=np.float64)
        for idx, joint_name in enumerate(self.mapped_joints):
            parent = self.mapped_joint_parents[idx]
            if parent == -1:
                continue
            chain = chains.get(joint_name, {})
            total_length = chain.get("total_length") if isinstance(chain, dict) else None
            try:
                length = float(total_length)
            except (TypeError, ValueError):
                length = self.source_reference_segment_lengths[idx]
            if not np.isfinite(length) or length <= 1e-6:
                length = self.source_reference_segment_lengths[idx]
            segment_lengths[idx] = max(length, 1e-6)

        self.segment_local_builder = SegmentLocalTargetBuilder(
            joint_names=list(self.mapped_joints),
            parent_indices=list(self.mapped_joint_parents),
            segment_lengths=segment_lengths,
        )
        self.mode = "segment_local"

    def effector_names(self):
        """
        Return the list of mapped joint names used as effectors.

        Returns:
            list[str]: Names of joints for which effectors are computed.
        """
        return self.mapped_joints

    def compute_effectors_from_skeleton(self, skeleton_instance: SkeletonInstance, scale_animation: bool):
        """
        Compute scaled effectors from a single skeleton instance.

        The method computes global joint transforms from the skeleton instance,
        then applies per-joint scaling and offsets to produce effector
        transforms in world space.

        Args:
            skeleton_instance: SkeletonInstance whose skeleton must match the scaler's ``skeleton``.
            scale_animation: Whether to apply per-joint scaling when computing
                effectors. If False, only height scaling is applied.

        Returns:
            np.ndarray: Array of effector transforms (one per mapped joint) in the
            layout ``(num_mapped_joints, wp.transform)``.

        Raises:
            ValueError: If ``skeleton_instance.skeleton`` does not match the scaler's ``skeleton``.
        """
        if skeleton_instance.skeleton != self.skeleton:
            raise ValueError("[ERROR]: SkeletonInstance.skeleton is not equal to self.skeleton.")

        @wp.kernel
        def compute_global_pose_kernel(
            in_num_joints     : wp.int32,
            in_root_tx        : wp.transform,
            in_parent_indices : wp.array(dtype=wp.int32),
            in_local_pose     : wp.array(dtype=wp.transform),
            out_result        : wp.array(dtype=wp.transform)
        ):
            pose_utils.wp_compute_global_pose(in_num_joints, in_root_tx, in_parent_indices, in_local_pose, out_result)

        @wp.kernel
        def compute_scaled_effectors_kernel(
            in_num_mapped_joints    : wp.int32,
            in_global_pose          : wp.array(dtype=wp.transform),
            in_mapped_joint_indices : wp.array(dtype=wp.int32),
            in_mapped_joint_scales  : wp.array(dtype=wp.float32),
            in_mapped_joint_offsets : wp.array(dtype=wp.transform),
            in_scale_animation      : wp.bool,
            out_result              : wp.array(dtype=wp.transform)
        ):
            HumanToRobotScaler.wp_compute_scaled_effectors(
                in_num_mapped_joints, in_global_pose, in_mapped_joint_indices,
                in_mapped_joint_scales, in_mapped_joint_offsets, in_scale_animation, out_result)

        wp_global_pose = wp.array([wp.transform_identity()] * skeleton_instance.num_joints, dtype=wp.transform)
        wp.launch(
            compute_global_pose_kernel,
            dim=1,
            inputs=[
                skeleton_instance.num_joints,
                skeleton_instance.xform,
                wp.array(skeleton_instance.parent_indices, dtype=wp.int32),
                wp.array(skeleton_instance.local_transforms, dtype=wp.transform)],
                outputs=[wp_global_pose])

        if self.mode == "segment_local":
            return self._compute_segment_local_effectors(wp_global_pose.numpy())

        wp_effectors = wp.array([wp.transform_identity()] * len(self.mapped_joint_indices), dtype=wp.transform)
        wp.launch(
            compute_scaled_effectors_kernel,
            dim=1,
            inputs=[
                len(self.mapped_joint_indices),
                wp_global_pose,
                self.mapped_joint_indices,
                self.mapped_joint_scales,
                self.mapped_joint_offsets,
                scale_animation
            ],
            outputs=[wp_effectors])

        return wp_effectors.numpy()

    def compute_effectors_from_buffer(self, animation_buffer: AnimationBuffer, scale_animation: bool, xform: wp.transform = wp.transform_identity()):
        """
        Compute scaled effectors for all frames in an animation buffer.

        This is a batched variant of ``compute_effectors_from_skeleton`` that
        operates over all frames in an AnimationBuffer.

        Args:
            animation_buffer: AnimationBuffer whose skeleton must match the scaler's ``skeleton``.
            scale_animation: Whether to apply per-joint scaling when computing
                effectors. If False, only height scaling is applied.
            xform: Optional root transform applied to all frames before global
                pose computation.

        Returns:
            np.ndarray: Array of transforms of shape ``(num_frames, num_mapped_joints, wp.transform)``.

        Raises:
            ValueError: If ``animation_buffer.skeleton`` does not match the scaler's ``skeleton``.
        """
        if animation_buffer.skeleton != self.skeleton:
            raise ValueError("[ERROR]: AnimationBuffer.skeleton is not equal to self.skeleton.")

        @wp.kernel
        def batched_compute_global_pose_kernel(
            in_num_joints     : wp.int32,
            in_root_tx        : wp.transform,
            in_parent_indices : wp.array(dtype=wp.int32),
            in_local_pose     : wp.array2d(dtype=wp.transform),
            out_result        : wp.array2d(dtype=wp.transform)
        ):
            frame_idx = wp.tid()
            pose_utils.wp_compute_global_pose(
                in_num_joints, in_root_tx, in_parent_indices, in_local_pose[frame_idx], out_result[frame_idx])

        @wp.kernel
        def batched_compute_scaled_effectors_2d_kernel(
            in_num_mapped_joints    : wp.int32,
            in_global_pose          : wp.array2d(dtype=wp.transform),
            in_mapped_joint_indices : wp.array(dtype=wp.int32),
            in_mapped_joint_scales  : wp.array(dtype=wp.float32),
            in_mapped_joint_offsets : wp.array(dtype=wp.transform),
            in_scale_animation      : wp.bool,
            out_result              : wp.array2d(dtype=wp.transform)
        ):
            frame_idx = wp.tid()
            HumanToRobotScaler.wp_compute_scaled_effectors(
               in_num_mapped_joints, in_global_pose[frame_idx], in_mapped_joint_indices,
               in_mapped_joint_scales, in_mapped_joint_offsets, in_scale_animation, out_result[frame_idx])

        wp_global_poses = wp.empty(shape=(animation_buffer.num_frames, self.skeleton.num_joints), dtype=wp.transform)
        wp.launch(
            batched_compute_global_pose_kernel,
            dim=animation_buffer.num_frames,
            inputs=[
                self.skeleton.num_joints,
                xform,
                wp.array(self.skeleton.parent_indices, dtype=wp.int32),
                wp.array2d(animation_buffer.local_transforms, dtype=wp.transform)],
                outputs=[wp_global_poses])

        if self.mode == "segment_local":
            return self._compute_segment_local_effectors_batch(wp_global_poses.numpy())

        wp_effectors = wp.empty(shape=(animation_buffer.num_frames, len(self.mapped_joint_indices)), dtype=wp.transform)
        wp.launch(
            batched_compute_scaled_effectors_2d_kernel,
            dim=animation_buffer.num_frames,
            inputs=[
                len(self.mapped_joint_indices),
                wp_global_poses,
                self.mapped_joint_indices,
                self.mapped_joint_scales,
                self.mapped_joint_offsets,
                scale_animation
            ],
            outputs=[wp_effectors])

        return wp_effectors.numpy()

    @staticmethod
    def _quat_mul_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = lhs
        x2, y2, z2, w2 = rhs
        out = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def _quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        q_vec = np.asarray(quat[0:3], dtype=np.float64)
        w = float(quat[3])
        vec = np.asarray(vec, dtype=np.float64)
        uv = np.cross(q_vec, vec)
        uuv = np.cross(q_vec, uv)
        return vec + 2.0 * (w * uv + uuv)

    def _apply_offsets_np(self, mapped_transforms: np.ndarray) -> np.ndarray:
        out = np.array(mapped_transforms, copy=True, dtype=np.float64)
        for idx, offset in enumerate(self.mapped_joint_offsets_np):
            q = self._quat_mul_xyzw(out[idx, 3:7], offset[3:7])
            out[idx, 3:7] = q
            out[idx, 0:3] = out[idx, 0:3] + self._quat_rotate_xyzw(q, offset[0:3])
        return out.astype(np.float32)

    def _compute_segment_local_effectors(self, global_pose: np.ndarray) -> np.ndarray:
        if self.segment_local_builder is None:
            raise RuntimeError("segment_local mode requires segment_local_builder")
        mapped = np.asarray([global_pose[idx] for idx in self.mapped_joint_indices.numpy()], dtype=np.float64)
        mapped = self.segment_local_builder.compute_transforms(mapped)
        return self._apply_offsets_np(mapped)

    def _compute_segment_local_effectors_batch(self, global_poses: np.ndarray) -> np.ndarray:
        if self.segment_local_builder is None:
            raise RuntimeError("segment_local mode requires segment_local_builder")
        indices = self.mapped_joint_indices.numpy()
        mapped = np.asarray(global_poses[:, indices, :], dtype=np.float64)
        mapped = self.segment_local_builder.compute_transforms_batch(mapped)
        return np.stack([self._apply_offsets_np(frame) for frame in mapped], axis=0)

    def create_scaled_skeleton(self, skeleton_instance: SkeletonInstance):
        """
        Create a scaled Skeleton from a skeleton instance.

        This method computes scaled global effectors from the input skeleton
        instance, converts them to local transforms based on the mapped joint
        hierarchy, and returns a new Skeleton containing only the mapped joints.

        Args:
            skeleton_instance: SkeletonInstance to be converted into a scaled skeleton.

        Returns:
            Skeleton: A new skeleton with joints, parents, and local transforms
            derived from the mapped joints and their scaled effectors.
        """
        global_tx = self.compute_effectors_from_skeleton(skeleton_instance, True)

        num_joints = len(self.mapped_joints)
        wp_local_tx = wp.array([wp.transform_identity()] * num_joints, dtype=wp.transform)

        wp.launch(
            pose_utils.compute_local_pose_kernel,
            dim=1,
            inputs=[
                num_joints,
                skeleton_instance.xform,
                wp.array(self.mapped_joint_parents, dtype=wp.int32),
                wp.array(global_tx, dtype=wp.transform)],
            outputs=[wp_local_tx])

        return Skeleton(
            num_joints,
            self.mapped_joints,
            self.mapped_joint_parents,
            wp_local_tx.numpy())

    @wp.func
    def wp_compute_scaled_effectors(
        in_num_mapped_joints    : wp.int32,
        in_global_pose          : wp.array(dtype=wp.transform),
        in_mapped_joint_indices : wp.array(dtype=wp.int32),
        in_mapped_joint_scales  : wp.array(dtype=wp.float32),
        in_mapped_joint_offsets : wp.array(dtype=wp.transform),
        in_scale_animation      : wp.bool,
        out_result              : wp.array(dtype=wp.transform)
    ):
        root_t = in_global_pose[in_mapped_joint_indices[0]].p

        scale = wp.where(in_scale_animation, wp.vec3(in_mapped_joint_scales[0]), wp.vec3(1.0, 1.0, in_mapped_joint_scales[0]))
        scaled_root_t = wp.cw_mul(root_t, scale)

        for i in range(in_num_mapped_joints):
            idx = in_mapped_joint_indices[i]
            pose_tx = in_global_pose[idx]
            offset_tx = in_mapped_joint_offsets[i]

            scale = wp.where(in_scale_animation, wp.vec3(in_mapped_joint_scales[i]), wp.vec3(1.0, 1.0, in_mapped_joint_scales[i]))
            geocentric_scaled_t = wp.cw_mul((pose_tx.p - root_t), scale)

            q = wp.mul(pose_tx.q, offset_tx.q)
            t = geocentric_scaled_t + scaled_root_t + wp.quat_rotate(q, offset_tx.p)
            out_result[i] = wp.transform(t, q)


LegacyHumanToRobotScaler = HumanToRobotScaler
