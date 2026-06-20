# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np
import newton
import newton.ik as ik
from tqdm import trange

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.pipelines.ik_objectives import (
    IKObjectiveDirection,
    IKObjectivePoleVector,
    IKObjectivePerEnvGroundHeightBarrier,
    IKObjectivePerEnvWeightedPosition,
    IKObjectiveProjectedPosition,
    IKObjectiveSphereCollisionBarrier,
    IKRangeNormalizedJointLimitBarrier,
    IKTemporalJointRegularizer,
    range_normalized_joint_limit_barrier,
)
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.robotics.reachability import project_relative_rotation_quat_xyzw
from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer
from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper
from soma_retargeter.pipelines.motion_grounding import apply_virtual_foot_grounding_to_frames
from soma_retargeter.pipelines.foot_contact_inference import (
    infer_contacts_from_animation_buffer,
    contacts_from_npz_foot_contacts,
)

_DEFAULT_IK_SOLVER_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT = 5.5
_DEFAULT_TEMPORAL_VELOCITY_WEIGHT = 0.0
_DEFAULT_TEMPORAL_ACCELERATION_WEIGHT = 0.0
_DEFAULT_JOINT_MOTION_LIMIT_ENABLED = False
_DEFAULT_JOINT_VELOCITY_LIMIT_FRACTION_PER_SECOND = 2.0
_DEFAULT_JOINT_ACCELERATION_LIMIT_FRACTION_PER_SECOND2 = 40.0
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5


class NewtonPipeline:
    """
    Newton-based motion retargeting pipeline.

    This pipeline retargets human motion captured on a common skeleton
    to a target robot using inverse kinematics (IK),
    custom objectives, and optional post-processing filters such as
    joint limit clamping and feet stabilization.
    """
    def __init__(self, skeleton: Skeleton, source_type='soma', robot_type='unitree_g1', retarget_config: dict = None):
        """
        Initialize the Newton retargeting pipeline.

        Args:
            skeleton: Common skeleton definition used by the input clips to be retargeted.
            source_type: Source skeleton type name. Currently only "soma" is supported.
            robot_type: Target robot type name.
            retarget_config: Optional configuration dictionary. If None, a
                configuration is loaded from disk based on the source/target
                types.

        Raises:
            ValueError: If the target robot type is not supported.
        """
        self.source_type = pipeline_utils.get_source_type_from_str(source_type)
        self.target_type = pipeline_utils.get_target_type_from_str(robot_type)
        self.input_targets = []
        self.input_sample_rates = []
        self.max_frames = -1
        self.input_contact_scores = []

        if retarget_config is None:
            retargeter_config = pipeline_utils.get_retargeter_config(self.source_type, self.target_type)
        else:
            retargeter_config = retarget_config

        self.ik_iterations = retargeter_config.get('ik_iterations', _DEFAULT_IK_SOLVER_ITERATIONS)
        self.joint_limit_weight = retargeter_config.get('joint_limit_weight', _DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT)
        self.smooth_joint_filter_weight = retargeter_config.get('smooth_joint_filter_weight', _DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT)
        self.temporal_velocity_weight = retargeter_config.get('temporal_velocity_weight', _DEFAULT_TEMPORAL_VELOCITY_WEIGHT)
        self.temporal_acceleration_weight = retargeter_config.get('temporal_acceleration_weight', _DEFAULT_TEMPORAL_ACCELERATION_WEIGHT)
        self.joint_motion_limit_enabled = bool(
            retargeter_config.get('joint_motion_limit_enabled', _DEFAULT_JOINT_MOTION_LIMIT_ENABLED)
        )
        self.joint_velocity_limit_fraction_per_second = float(
            retargeter_config.get(
                'joint_velocity_limit_fraction_per_second',
                _DEFAULT_JOINT_VELOCITY_LIMIT_FRACTION_PER_SECOND,
            )
        )
        self.joint_acceleration_limit_fraction_per_second2 = float(
            retargeter_config.get(
                'joint_acceleration_limit_fraction_per_second2',
                _DEFAULT_JOINT_ACCELERATION_LIMIT_FRACTION_PER_SECOND2,
            )
        )
        self.collision_weight = float(retargeter_config.get("collision_weight", 0.0))
        self.priority_residual_guard_enabled = bool(
            retargeter_config.get("priority_residual_guard_enabled", bool(retargeter_config.get("compiled_retarget_profile")))
        )
        self.priority_residual_guard_tolerance = float(retargeter_config.get("priority_residual_guard_tolerance", 0.05))
        self.priority_residual_guard_absolute_tolerance = float(retargeter_config.get("priority_residual_guard_absolute_tolerance", 1.0e-5))
        self.priority_residual_guard_margin_fraction = float(retargeter_config.get("priority_residual_guard_margin_fraction", 0.0))
        self.post_processing_enabled = retargeter_config.get('enable_post_processing', True)
        self.virtual_foot_grounding_enabled = retargeter_config.get('enable_virtual_foot_grounding', True)
        self.virtual_foot_grounding_smooth_window = max(
            1,
            int(retargeter_config.get('virtual_foot_grounding_smooth_window', 5)),
        )
        self.enable_self_penetration = False
        self.contact_aware_foot_ik = retargeter_config.get("contact_aware_foot_ik", {})
        self.contact_aware_foot_ik_enabled = bool(self.contact_aware_foot_ik.get("enabled", False))
        self.contact_source = str(self.contact_aware_foot_ik.get("contact_source", "auto")).lower()
        self.ground_barrier_config = retargeter_config.get("ground_barrier", {})
        self.ground_barrier_enabled = bool(self.ground_barrier_config.get("enabled", False))
        self.smooth_joint_filter_coord_masks = None
        self.joint_limit_clamper = None

        self.robot_builder = newton.ModelBuilder()
        self.robot_builder.add_mjcf(str(pipeline_utils.get_robot_mjcf_path(self.target_type)))

        self.human_robot_scaler = HumanToRobotScaler(
            skeleton, retargeter_config['model_height'], io_utils.get_config_file(retargeter_config['human_robot_scaler_config']))
        compiled_profile = retargeter_config.get("compiled_retarget_profile")
        self.compiled_collision_config = {}
        if compiled_profile:
            compiled_profile_path = io_utils.get_config_file(compiled_profile)
            self.human_robot_scaler.enable_segment_local_from_profile(compiled_profile_path)
            try:
                self.compiled_collision_config = io_utils.load_json(compiled_profile_path).get("collision", {})
            except Exception as exc:
                print(f"[WARN] Failed to load compiled collision config from {compiled_profile_path}: {exc}")
                self.compiled_collision_config = {}

        self.num_body_count = self.robot_builder.body_count
        self.num_dofs = self.robot_builder.joint_dof_count
        self.ik_model = self._build_model(1)

        (
            self.mapped_joints,
            self.mapped_joint_indices,
            self.mapped_body_link_pos_data,
            self.mapped_body_link_rot_data,
            self.mapped_body_link_by_joint,
        ) = self._build_target_mapping(
            self.ik_model,
            self.human_robot_scaler.skeleton,
            retargeter_config)

        smooth_joint_filter_objective_body_masks = retargeter_config.get('smooth_joint_filter_objective_body_masks', None)
        if smooth_joint_filter_objective_body_masks is not None:
            self.smooth_joint_filter_coord_masks = newton_utils.create_joint_coord_masks(
                self.ik_model, smooth_joint_filter_objective_body_masks, 0.0)

        self.output_default_pose_blend_frames = max(0, retargeter_config.get('output_default_pose_blend_frames', 0))
        self.output_default_pose_blend_coord_mask = None
        output_default_pose_blend_bodies = retargeter_config.get('output_default_pose_blend_bodies', None)
        if output_default_pose_blend_bodies is not None:
            self.output_default_pose_blend_coord_mask = newton_utils.create_joint_coord_masks(
                self.ik_model,
                {body_name: 1.0 for body_name in output_default_pose_blend_bodies},
                0.0,
            )
        elif self.output_default_pose_blend_frames > 0:
            self.output_default_pose_blend_coord_mask = np.ones(self.ik_model.joint_coord_count, dtype=np.float32)
            self.output_default_pose_blend_coord_mask[:7] = 0.0

        effector_names = self.human_robot_scaler.effector_names()
        self.target_effector_indices = [effector_names.index(name) for name in self.mapped_joints]
        self.feet_effector_indices = []

        self.feet_stabilizer = None
        if self.post_processing_enabled:
            self.feet_effector_indices = [
                self.mapped_joints.index("LeftFoot"),
                self.mapped_joints.index("RightFoot")]
            feet_stabilizer_config = retargeter_config.get('feet_stabilizer_config', None)
            if feet_stabilizer_config is None:
                print("[INFO] Post-processing disabled: feet_stabilizer_config is not configured.")
                self.post_processing_enabled = False
            else:
                feet_stabilizer_path = io_utils.get_config_file(feet_stabilizer_config)
                if not feet_stabilizer_path.exists():
                    print(f"[INFO] Post-processing disabled: feet stabilizer config not found: {feet_stabilizer_path}")
                    self.post_processing_enabled = False
                else:
                    self.feet_stabilizer = FeetStabilizer(feet_stabilizer_path)

        self.joint_limit_clamper = JointLimitClamper(self.ik_model)

        self.initialization_pose = None
        self.num_initialization_frames = 0
        self.num_stabilization_frames = 0
        initialization_pose = retargeter_config.get('initialization_pose', None)
        if initialization_pose:
            init_skel, init_anim = bvh_utils.load_bvh(io_utils.get_config_file(initialization_pose))
            self.initialization_pose = SkeletonInstance(init_skel, [0, 0, 0], wp.transform_identity())
            self.initialization_pose.set_local_transforms(init_anim.get_local_transforms(0))
            self.num_initialization_frames = retargeter_config.get('num_initialization_frames', _DEFAULT_NUM_INITIALIZATION_FRAMES)
            self.num_stabilization_frames = retargeter_config.get('num_stabilization_frames', _DEFAULT_NUM_STABILIZATION_FRAMES)

    def clear(self):
        """
        Clear all accumulated input motions and reset internal state.

        This removes all previously added motions set for retargeting.
        It does not modify static configuration such as the robot model or IK settings.
        """
        self.input_targets = []
        self.input_sample_rates = []
        self.max_frames = -1
        self.input_contact_scores = []

    def add_input_motions(self, buffers: list[AnimationBuffer], offsets: list[wp.transform], scale_animation: bool):
        """
        Add input motions to be retargeted.
        Each buffer is converted into IK targets using the human-to-robot scaler.

        Args:
            buffers: List of input animation buffers defined on the common skeleton.
            offsets: List of root transforms applied to each buffer. If the
                length does not match `buffers`, identity transforms are used
                for all.
            scale_animation: Whether to rescale the source motion using the
                configured HumanToRobotScaler.
        """
        offsets = offsets if len(offsets) == len(buffers) else [wp.transform_identity()] * len(buffers)
        for i in trange(len(buffers), desc="[INFO] Converting Motions for Newton"):
            raw_buffer = buffers[i]
            explicit_scores = self._read_explicit_contact_scores(raw_buffer) if self.contact_aware_foot_ik_enabled else None
            buffer = raw_buffer
            if self.initialization_pose and self.num_initialization_frames > 0:
                buffer = newton_utils.create_buffer_with_initialization_frames(
                    self.initialization_pose, raw_buffer, self.num_initialization_frames, self.num_stabilization_frames)
                explicit_scores = self._prepend_contact_scores_for_warmup(explicit_scores)

            self.max_frames = max(self.max_frames, buffer.num_frames)
            buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(buffer, scale_animation, offsets[i])

            self.input_targets.append(buffer_effectors[:, self.target_effector_indices, :])
            self.input_sample_rates.append(raw_buffer.sample_rate)
            if self.contact_aware_foot_ik_enabled:
                try:
                    window = int(self.contact_aware_foot_ik.get("contact_score_smoothing_window", 5))
                    scores = explicit_scores
                    if scores is None and self.contact_source in ("auto", "soma_heuristic"):
                        scores = infer_contacts_from_animation_buffer(buffer, offsets[i], window)
                    self.input_contact_scores.append(scores)
                except Exception as exc:
                    print(f"[WARN] Contact inference failed, disabling lock for this clip: {exc}")
                    self.input_contact_scores.append(None)
            else:
                self.input_contact_scores.append(None)

    def _read_explicit_contact_scores(self, buffer):
        if self.contact_source not in ("auto", "npz_foot_contacts"):
            return None
        foot_contacts = getattr(buffer, "foot_contacts", None)
        if foot_contacts is None:
            foot_contacts = getattr(buffer, "contacts", None)
        if foot_contacts is None:
            return None
        window = int(self.contact_aware_foot_ik.get("contact_score_smoothing_window", 5))
        return contacts_from_npz_foot_contacts(np.asarray(foot_contacts), window)

    def _prepend_contact_scores_for_warmup(self, scores):
        if scores is None:
            return None
        prepend_count = self.num_initialization_frames + self.num_stabilization_frames
        if prepend_count <= 0:
            return scores
        out = {}
        for key, values in scores.items():
            values = np.asarray(values, dtype=np.float32)
            if len(values) == 0:
                out[key] = values
                continue
            out[key] = np.concatenate([np.full(prepend_count, values[0], dtype=np.float32), values])
        return out

    def execute(self):
        """
        Run the retargeting pipeline on all added input motions.

        Returns:
            list[CSVAnimationBuffer]: A list of retargeted robot motions, one per input motion.
        """
        num_envs = len(self.input_targets)
        if num_envs == 0:
            self.retargeted_motions = []
            return

        self.ik_iterations = max(1, self.ik_iterations)
        self.joint_limit_weight = max(0.0, self.joint_limit_weight)
        self.smooth_joint_filter_weight = max(0.0, self.smooth_joint_filter_weight)
        self.temporal_velocity_weight = max(0.0, float(self.temporal_velocity_weight))
        self.temporal_acceleration_weight = max(0.0, float(self.temporal_acceleration_weight))
        self.joint_velocity_limit_fraction_per_second = max(
            0.0,
            float(self.joint_velocity_limit_fraction_per_second),
        )
        self.joint_acceleration_limit_fraction_per_second2 = max(
            0.0,
            float(self.joint_acceleration_limit_fraction_per_second2),
        )
        self.joint_motion_limit_enabled = bool(
            self.joint_motion_limit_enabled
            and self.joint_velocity_limit_fraction_per_second > 0.0
            and self.joint_acceleration_limit_fraction_per_second2 > 0.0
        )
        self.priority_residual_guard_tolerance = max(0.0, float(self.priority_residual_guard_tolerance))
        self.priority_residual_guard_absolute_tolerance = max(0.0, float(self.priority_residual_guard_absolute_tolerance))
        self.priority_residual_guard_margin_fraction = max(0.0, float(self.priority_residual_guard_margin_fraction))

        print("[INFO] Newton Retargeter Settings: ")
        print(f"[INFO]\t  Source Skeleton Type: {pipeline_utils.get_source_str_from_type(self.source_type)}")
        print(f"[INFO]\t  Target Robot Type: {pipeline_utils.get_target_str_from_type(self.target_type)}")
        print(f"[INFO]\t  Post-Processing Enabled: {self.post_processing_enabled}")
        print(f"[INFO]\t  Initialization Pose: {self.initialization_pose is not None}")
        print(f"[INFO]\t  Initialization Frame Count: {self.num_initialization_frames}")
        print(f"[INFO]\t  Constraint Stabilization Frame Count: {self.num_stabilization_frames}")
        print(f"[INFO]\t  IK Solver Iterations: {self.ik_iterations}")
        print(f"[INFO]\t  Joint Limit Objective Weight: {self.joint_limit_weight}")
        print(f"[INFO]\t  Smooth Joint Filter Objective Weight: {self.smooth_joint_filter_weight}")
        print(f"[INFO]\t  Temporal Velocity Weight: {self.temporal_velocity_weight}")
        print(f"[INFO]\t  Temporal Acceleration Weight: {self.temporal_acceleration_weight}")
        print(f"[INFO]\t  Joint Motion Limit Enabled: {self.joint_motion_limit_enabled}")
        print(f"[INFO]\t  Priority Residual Guard Enabled: {self.priority_residual_guard_enabled}")

        restore_contact_aware = self.contact_aware_foot_ik_enabled
        try:
            model = self._build_model(num_envs)
            state = model.state()
            default_joint_q_raw = model.joint_q.numpy().astype(np.float32)
            coord_count = self.ik_model.joint_coord_count
            if default_joint_q_raw.size == num_envs * coord_count:
                default_joint_q = default_joint_q_raw.reshape(num_envs, coord_count)
            else:
                default_joint_q = np.tile(default_joint_q_raw[:coord_count], (num_envs, 1))
            temporal_velocity_scales, temporal_acceleration_scales, temporal_coord_masks = self._build_temporal_scale_matrices(
                model,
                num_envs,
            )
            self.temporal_prev_q = np.array(default_joint_q, copy=True)
            self.temporal_prevprev_q = np.array(default_joint_q, copy=True)
            self.temporal_velocity_scales = temporal_velocity_scales
            self.temporal_acceleration_scales = temporal_acceleration_scales
            self.temporal_coord_masks = temporal_coord_masks
            self.joint_motion_limit_ranges, self.joint_motion_limit_masks = self._joint_coord_ranges_and_temporal_mask(
                self.ik_model,
            )
            self.priority_guard_report = {
                "enabled": bool(self.priority_residual_guard_enabled),
                "protected_priority": 0,
                "protected_residual": "joint_limit_margin" if self.priority_residual_guard_margin_fraction > 0.0 else "joint_limit_penetration",
                "rollback_count": 0,
                "checked_frames": 0,
                "max_allowed": 0.0,
                "max_after": 0.0,
            }

            if self.post_processing_enabled:
                self.feet_stabilizer.setup_num_envs(num_envs)
                env_feet_tx = np.empty((num_envs, len(self.feet_effector_indices), 7), dtype=np.float32)

            (
                position_objectives,
                rotation_objectives,
                direction_objectives,
                pole_vector_objectives,
                joint_limit_objective,
                smooth_joint_filter_objective,
                temporal_velocity_objective,
                temporal_acceleration_objective,
                ground_barrier_objectives,
                collision_objectives,
                contact_objectives,
            ) = self._create_ik_objectives(num_envs, model, state)

            ik_solver_active_objectives = [
                *position_objectives,
                *rotation_objectives,
                *direction_objectives,
                *pole_vector_objectives,
                *ground_barrier_objectives,
                *collision_objectives,
                *contact_objectives,
            ]
            if self.joint_limit_weight > 0.0:
                ik_solver_active_objectives.append(joint_limit_objective)
            if self.smooth_joint_filter_weight > 0.0:
                ik_solver_active_objectives.append(smooth_joint_filter_objective)
            if self.temporal_velocity_weight > 0.0:
                ik_solver_active_objectives.append(temporal_velocity_objective)
            if self.temporal_acceleration_weight > 0.0:
                ik_solver_active_objectives.append(temporal_acceleration_objective)

            direction_analytic = sum(1 for *_, analytic in self.mapped_body_link_direction_data if analytic)
            pole_analytic = sum(1 for *_, analytic in self.mapped_body_link_pole_vector_data if analytic)
            jacobian_mode = self._select_ik_jacobian_mode(ik_solver_active_objectives)
            sparse_residual_dim = sum(
                int(objective.residual_dim())
                for objective in [
                    *position_objectives,
                    *rotation_objectives,
                    *direction_objectives,
                    *pole_vector_objectives,
                ]
            )
            autodiff_sparse_residual_dim = sum(
                int(objective.residual_dim())
                for objective in [
                    *direction_objectives,
                    *pole_vector_objectives,
                ]
                if not objective.supports_analytic()
            )
            self.ik_objective_summary = {
                "batch_size": int(num_envs),
                "ik_iterations": int(self.ik_iterations),
                "jacobian_mode": str(jacobian_mode.value),
                "position": int(len(position_objectives)),
                "rotation": int(len(rotation_objectives)),
                "direction": int(len(direction_objectives)),
                "direction_analytic": int(direction_analytic),
                "direction_autodiff": int(len(direction_objectives) - direction_analytic),
                "pole_vector": int(len(pole_vector_objectives)),
                "pole_vector_analytic": int(pole_analytic),
                "pole_vector_autodiff": int(len(pole_vector_objectives) - pole_analytic),
                "ground_barrier": int(len(ground_barrier_objectives)),
                "collision": int(len(collision_objectives)),
                "contact": int(len(contact_objectives)),
                "joint_limit": int(self.joint_limit_weight > 0.0),
                "smooth_joint_filter": int(self.smooth_joint_filter_weight > 0.0),
                "temporal_velocity": int(self.temporal_velocity_weight > 0.0),
                "temporal_acceleration": int(self.temporal_acceleration_weight > 0.0),
                "active_objectives": int(len(ik_solver_active_objectives)),
                "sparse_residual_dim": int(sparse_residual_dim),
                "autodiff_sparse_residual_dim": int(autodiff_sparse_residual_dim),
            }

            ik_solver = ik.IKSolver(
                model=self.ik_model,
                n_problems=num_envs,
                objectives=ik_solver_active_objectives,
                lambda_initial=0.1,
                jacobian_mode=jacobian_mode,
            )

            joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
            wp.copy(joint_q, model.joint_q)
            ik_solver.reset()

            graph_capture = None
            priority_guard_cpu_check = self.priority_residual_guard_enabled and self._priority_guard_requires_cpu_check(
                self.priority_residual_guard_margin_fraction,
            )
            priority_guard_fast_hard_limit = self.priority_residual_guard_enabled and not priority_guard_cpu_check

            def single_step():
                ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

            if wp.get_device().is_cuda and (not self.priority_residual_guard_enabled or priority_guard_fast_hard_limit):
                with wp.ScopedCapture() as cap:
                    single_step()
                graph_capture = cap.graph
            else:
                ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

            num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames
            joint_q_data = [
                np.zeros((len(self.input_targets[i]), self.ik_model.joint_coord_count), dtype=np.float32)
                for i in range(num_envs)
            ]
            motion_limit_prev_q = np.zeros((num_envs, self.ik_model.joint_coord_count), dtype=np.float32)
            motion_limit_prev_delta = np.zeros((num_envs, self.ik_model.joint_coord_count), dtype=np.float32)
            motion_limit_has_prev = np.zeros(num_envs, dtype=bool)

            for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions"):
                if num_frames_to_remove > 0 and frame <= num_frames_to_remove:
                    smooth_joint_filter_objective.set_weight(
                        self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove))
                    )
                if self.temporal_velocity_weight > 0.0:
                    temporal_velocity_objective.set_references(self.temporal_prev_q)
                if self.temporal_acceleration_weight > 0.0:
                    temporal_acceleration_objective.set_references(self.temporal_prev_q, self.temporal_prevprev_q)

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    frame_targets = self.input_targets[env][frame]
                    for i, (effector_idx, _, _, _, _) in enumerate(self.mapped_body_link_pos_data):
                        target = frame_targets[effector_idx]
                        position_objectives[i].set_target_position(env, wp.vec3(*target[0:3]))
                    for i, (effector_idx, _, _, basis) in enumerate(self.mapped_body_link_rot_data):
                        target = frame_targets[effector_idx]
                        target_rotation = self._project_rotation_target(target[3:7], basis)
                        rotation_objectives[i].set_target_rotation(env, wp.quat(*target_rotation))
                    for i, (reference_idx, target_idx, _, _, _, _, _) in enumerate(self.mapped_body_link_direction_data):
                        target_direction = self._direction_between_targets(frame_targets[reference_idx], frame_targets[target_idx])
                        direction_objectives[i].set_target_direction(env, wp.vec3(*target_direction))
                    for i, (reference_idx, middle_idx, target_idx, _, _, _, _, _, _) in enumerate(self.mapped_body_link_pole_vector_data):
                        target_normal, used_fallback = self._pole_normal_between_targets(
                            frame_targets[reference_idx],
                            frame_targets[middle_idx],
                            frame_targets[target_idx],
                            self.pole_vector_last_normals[env, i],
                        )
                        if used_fallback:
                            self.pole_vector_fallback_counts[i] += 1
                        else:
                            self.pole_vector_last_normals[env, i] = target_normal
                        pole_vector_objectives[i].set_target_normal(env, wp.vec3(*target_normal))

                    if self.contact_aware_foot_ik_enabled and env < len(self.input_contact_scores):
                        self._update_contact_objectives_for_frame(env, frame, frame_targets, contact_objectives)
                    if self.ground_barrier_enabled:
                        self._update_ground_barrier_objectives_for_frame(env, frame, ground_barrier_objectives)

                guard_before_q = None
                guard_before_cost = None
                if priority_guard_cpu_check:
                    guard_before_q = joint_q.numpy()
                    guard_before_cost = self._joint_limit_guard_costs(
                        self.ik_model,
                        guard_before_q,
                        self.priority_residual_guard_margin_fraction,
                    )

                if graph_capture is not None:
                    wp.capture_launch(graph_capture)
                else:
                    single_step()

                if priority_guard_fast_hard_limit:
                    self.priority_guard_report["checked_frames"] += 1
                    self.priority_guard_report["max_allowed"] = max(
                        float(self.priority_guard_report["max_allowed"]),
                        self.priority_residual_guard_absolute_tolerance,
                    )
                    self.priority_guard_report["max_after"] = max(
                        float(self.priority_guard_report["max_after"]),
                        0.0,
                    )
                elif priority_guard_cpu_check:
                    # Match the guarded state to the state that will be emitted and
                    # used as the next solve seed. The IK step can overshoot limits
                    # transiently; final joint-limit clamping is the runtime safety
                    # boundary.
                    self.joint_limit_clamper.apply(joint_q)
                    guard_after_q = joint_q.numpy()
                    guard_after_cost = self._joint_limit_guard_costs(
                        self.ik_model,
                        guard_after_q,
                        self.priority_residual_guard_margin_fraction,
                    )
                    should_rollback = self._priority_guard_should_rollback(
                        guard_before_cost,
                        guard_after_cost,
                        self.priority_residual_guard_tolerance,
                        self.priority_residual_guard_absolute_tolerance,
                    )
                    allowed = guard_before_cost * (1.0 + self.priority_residual_guard_tolerance) + self.priority_residual_guard_absolute_tolerance
                    self.priority_guard_report["checked_frames"] += 1
                    self.priority_guard_report["max_allowed"] = max(
                        float(self.priority_guard_report["max_allowed"]),
                        float(np.max(allowed)) if len(allowed) else 0.0,
                    )
                    self.priority_guard_report["max_after"] = max(
                        float(self.priority_guard_report["max_after"]),
                        float(np.max(guard_after_cost)) if len(guard_after_cost) else 0.0,
                    )
                    if bool(np.any(should_rollback)):
                        rollback_q = np.array(guard_after_q, copy=True)
                        rollback_q[should_rollback] = guard_before_q[should_rollback]
                        wp.copy(joint_q, wp.array(rollback_q.astype(np.float32), dtype=wp.float32, device=self.ik_model.device))
                        self.priority_guard_report["rollback_count"] += int(np.count_nonzero(should_rollback))

                if self.post_processing_enabled:
                    self.feet_stabilizer.reset_state(joint_q)
                    for env in range(num_envs):
                        if frame > (len(self.input_targets[env]) - 1):
                            env_feet_tx[env] = np.asarray(self.input_targets[env][-1][self.feet_effector_indices])
                        else:
                            env_feet_tx[env] = np.asarray(self.input_targets[env][frame][self.feet_effector_indices])
                    self.feet_stabilizer.solve(env_feet_tx)
                    data = self.joint_limit_clamper.apply(self.feet_stabilizer.current_state()).numpy()
                else:
                    data = self.joint_limit_clamper.apply(joint_q).numpy()

                if self.joint_motion_limit_enabled:
                    limited_data = np.array(data, dtype=np.float32, copy=True)
                    for env in range(num_envs):
                        if frame > (len(self.input_targets[env]) - 1):
                            continue
                        if motion_limit_has_prev[env]:
                            limited_frame, limited_delta = self._apply_joint_motion_limits_to_frame(
                                limited_data[env],
                                motion_limit_prev_q[env],
                                motion_limit_prev_delta[env],
                                self.joint_motion_limit_ranges,
                                self.joint_motion_limit_masks,
                                1.0 / max(float(self.input_sample_rates[env]), 1.0e-6),
                                self.joint_velocity_limit_fraction_per_second,
                                self.joint_acceleration_limit_fraction_per_second2,
                            )
                            limited_data[env] = limited_frame
                            motion_limit_prev_delta[env] = limited_delta
                        else:
                            motion_limit_prev_delta[env] = 0.0
                            motion_limit_has_prev[env] = True
                        motion_limit_prev_q[env] = limited_data[env]
                    data = self.joint_limit_clamper.apply(
                        wp.array(limited_data, dtype=wp.float32, device=self.ik_model.device)
                    ).numpy()

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    joint_q_data[env][frame] = np.array(data[env], copy=True)
                    self.temporal_prevprev_q[env] = self.temporal_prev_q[env]
                    self.temporal_prev_q[env] = np.asarray(data[env], dtype=np.float32)

            output_buffers = []
            for i in range(num_envs):
                raw_data = joint_q_data[i][num_frames_to_remove:].astype(np.float32)
                raw_data = self._apply_output_default_pose_blend(raw_data)
                if self.virtual_foot_grounding_enabled:
                    raw_data, grounding_stats = apply_virtual_foot_grounding_to_frames(
                        raw_data,
                        model=self.ik_model,
                        robot_builder=self.robot_builder,
                        robot_name=pipeline_utils.get_target_str_from_type(self.target_type),
                        smooth_window=self.virtual_foot_grounding_smooth_window,
                    )
                    if grounding_stats.applied:
                        print(
                            "[INFO] Virtual foot grounding: "
                            f"lifted {grounding_stats.lifted_frames}/{grounding_stats.frames} frames, "
                            f"max lift {grounding_stats.max_lift_m:.4f} m, "
                            f"min support z {grounding_stats.min_support_z_before_m:.4f} -> "
                            f"{grounding_stats.min_support_z_after_m:.4f} m"
                        )
                    elif grounding_stats.reason != "ok":
                        print(f"[INFO] Virtual foot grounding skipped: {grounding_stats.reason}")
                output_buffers.append(CSVAnimationBuffer.create_from_raw_data(raw_data, self.input_sample_rates[i]))
            return output_buffers
        finally:
            self.contact_aware_foot_ik_enabled = restore_contact_aware

    def _apply_output_default_pose_blend(self, joint_q_frames: np.ndarray) -> np.ndarray:
        if self.output_default_pose_blend_frames <= 0 or len(joint_q_frames) == 0:
            return joint_q_frames

        if self.output_default_pose_blend_coord_mask is None:
            return joint_q_frames

        coord_mask = self.output_default_pose_blend_coord_mask > 0.5
        if not np.any(coord_mask):
            return joint_q_frames

        num_blend_frames = min(self.output_default_pose_blend_frames, len(joint_q_frames))
        blended_frames = np.array(joint_q_frames, copy=True)
        default_pose = self.ik_model.joint_q.numpy().astype(np.float32)

        for frame in range(num_blend_frames):
            blend = 0.0 if num_blend_frames == 1 else float(frame) / float(num_blend_frames - 1)
            blended_frames[frame, coord_mask] = (
                default_pose[coord_mask] * (1.0 - blend) +
                joint_q_frames[frame, coord_mask] * blend
            )

        return blended_frames

    def _build_model(self, num_envs: int):
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self.robot_builder, xform=wp.transform_identity())

        builder.add_ground_plane()
        model = builder.finalize(requires_grad=True)

        return model

    def _build_target_mapping(self, model, skeleton, retargeter_config):
        mapped_joints = []
        mapped_joint_indices = []
        mapped_body_link_pos_data = []
        mapped_body_link_rot_data = []
        mapped_body_link_direction_data = []
        mapped_body_link_pole_vector_data = []
        mapped_body_link_by_joint = {}
        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        for joint, mapping_data in retargeter_config["ik_map"].items():
            effector_idx = len(mapped_joints)
            mapped_joints.append(joint)
            mapped_joint_indices.append(skeleton.joint_index(joint))
            t_link_idx = body_names.index(mapping_data['t_body'])
            r_link_idx = body_names.index(mapping_data['r_body'])
            mapped_body_link_by_joint[joint] = t_link_idx
            if float(mapping_data.get('t_weight', 0.0)) > 0.0:
                link_offset = np.asarray(mapping_data.get("v2_position_link_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
                if link_offset.shape != (3,) or not np.all(np.isfinite(link_offset)):
                    link_offset = np.zeros(3, dtype=np.float32)
                position_basis = self._normalize_position_basis(mapping_data.get("v2_position_basis"))
                mapped_body_link_pos_data.append((effector_idx, t_link_idx, float(mapping_data['t_weight']), link_offset, position_basis))
            if float(mapping_data.get('r_weight', 0.0)) > 0.0:
                mapped_body_link_rot_data.append((
                    effector_idx,
                    r_link_idx,
                    float(mapping_data['r_weight']),
                    self._normalize_rotation_basis(mapping_data.get("v2_rotation_basis")),
                ))

        for task in retargeter_config.get("direction_tasks", []):
            if not isinstance(task, dict):
                continue
            reference_site = task.get("reference_site")
            target_site = task.get("target_site")
            if reference_site not in mapped_body_link_by_joint or target_site not in mapped_body_link_by_joint:
                continue
            try:
                weight = float(task.get("weight", 0.0))
            except (TypeError, ValueError):
                continue
            if weight <= 0.0:
                continue
            mapped_body_link_direction_data.append(
                (
                    mapped_joints.index(reference_site),
                    mapped_joints.index(target_site),
                    mapped_body_link_by_joint[reference_site],
                    mapped_body_link_by_joint[target_site],
                    weight,
                    str(task.get("name") or f"{target_site}_direction"),
                    bool(task.get("analytic_jacobian", False)),
                )
            )
        self.mapped_body_link_direction_data = mapped_body_link_direction_data

        for task in retargeter_config.get("pole_vector_tasks", []):
            if not isinstance(task, dict):
                continue
            reference_site = task.get("reference_site")
            middle_site = task.get("source_semantic")
            target_site = task.get("target_site")
            if (
                reference_site not in mapped_body_link_by_joint
                or middle_site not in mapped_body_link_by_joint
                or target_site not in mapped_body_link_by_joint
            ):
                continue
            try:
                weight = float(task.get("weight", 0.0))
            except (TypeError, ValueError):
                continue
            if weight <= 0.0:
                continue
            mapped_body_link_pole_vector_data.append(
                (
                    mapped_joints.index(reference_site),
                    mapped_joints.index(middle_site),
                    mapped_joints.index(target_site),
                    mapped_body_link_by_joint[reference_site],
                    mapped_body_link_by_joint[middle_site],
                    mapped_body_link_by_joint[target_site],
                    weight,
                    str(task.get("name") or f"{middle_site}_pole_vector"),
                    bool(task.get("analytic_jacobian", False)),
                )
            )
        self.mapped_body_link_pole_vector_data = mapped_body_link_pole_vector_data

        return (
            mapped_joints,
            mapped_joint_indices,
            mapped_body_link_pos_data,
            mapped_body_link_rot_data,
            mapped_body_link_by_joint)

    def _create_ik_objectives(self, num_envs, model, state):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # Gather default body position and rotation based on model state to initialize
        # position and rotation objectives
        num_body_link_pos = len(self.mapped_body_link_pos_data)
        num_body_link_rot = len(self.mapped_body_link_rot_data)
        num_body_link_dir = len(self.mapped_body_link_direction_data)
        num_body_link_pole = len(self.mapped_body_link_pole_vector_data)
        pos_targets = np.zeros((num_envs, num_body_link_pos), dtype=wp.vec3)
        rot_targets = np.zeros((num_envs, num_body_link_rot), dtype=wp.quat)
        dir_targets = np.zeros((num_envs, num_body_link_dir), dtype=wp.vec3)
        pole_targets = np.zeros((num_envs, num_body_link_pole), dtype=wp.vec3)

        body_q = state.body_q.numpy()
        for env in range(num_envs):
            base = env * self.num_body_count
            for ee_idx, (_, link_idx, _, link_offset, _) in enumerate(self.mapped_body_link_pos_data):
                body_transform = body_q[base + link_idx]
                body_pos = np.asarray(body_transform[0:3], dtype=np.float32)
                body_rot = wp.quat(*body_transform[3:7])
                pos_targets[env, ee_idx] = body_pos + np.asarray(wp.quat_rotate(body_rot, wp.vec3(*link_offset)), dtype=np.float32)

            for ee_idx, (_, link_idx, _, _) in enumerate(self.mapped_body_link_rot_data):
                rot_wp = wp.quat(body_q[base + link_idx][3:7])
                rot_targets[env, ee_idx] = wp.normalize(rot_wp)
            for ee_idx, (_, _, parent_link_idx, child_link_idx, _, _, _) in enumerate(self.mapped_body_link_direction_data):
                dir_targets[env, ee_idx] = wp.vec3(
                    *self._direction_between_positions(
                        body_q[base + parent_link_idx][0:3],
                        body_q[base + child_link_idx][0:3],
                    )
                )
            for ee_idx, (_, _, _, parent_link_idx, middle_link_idx, child_link_idx, _, _, _) in enumerate(self.mapped_body_link_pole_vector_data):
                pole_targets[env, ee_idx] = wp.vec3(
                    *self._pole_normal_between_positions(
                        body_q[base + parent_link_idx][0:3],
                        body_q[base + middle_link_idx][0:3],
                        body_q[base + child_link_idx][0:3],
                    )[0]
                )

        pos_num_ees = len(self.mapped_body_link_pos_data)
        rot_num_ees = len(self.mapped_body_link_rot_data)
        dir_num_ees = len(self.mapped_body_link_direction_data)
        pole_num_ees = len(self.mapped_body_link_pole_vector_data)
        pos_target_arrays, rot_target_arrays, dir_target_arrays, pole_target_arrays = [], [], [], []
        for ee_idx in range(pos_num_ees):
            pos_wp = wp.array(pos_targets[:, ee_idx], dtype=wp.vec3)
            pos_target_arrays.append(pos_wp)

        for ee_idx in range(rot_num_ees):
            rot_wp = wp.array(rot_targets[:, ee_idx], dtype=wp.vec4)
            rot_target_arrays.append(rot_wp)

        for ee_idx in range(dir_num_ees):
            dir_wp = wp.array(dir_targets[:, ee_idx], dtype=wp.vec3)
            dir_target_arrays.append(dir_wp)

        for ee_idx in range(pole_num_ees):
            pole_wp = wp.array(pole_targets[:, ee_idx], dtype=wp.vec3)
            pole_target_arrays.append(pole_wp)

        self.pole_vector_last_normals = pole_targets.astype(np.float32)
        self.pole_vector_fallback_counts = np.zeros(pole_num_ees, dtype=np.int64)

        position_objectives = []
        for i, (_, link_idx, w, link_offset, position_basis) in enumerate(self.mapped_body_link_pos_data):
            if position_basis is None:
                objective = ik.IKObjectivePosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(*link_offset),
                    target_positions=pos_target_arrays[i],
                    weight=w)
            else:
                objective = IKObjectiveProjectedPosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(*link_offset),
                    target_positions=pos_target_arrays[i],
                    basis_vectors=position_basis,
                    weight=w)
            position_objectives.append(objective)

        rotation_objectives = []
        for i, (_, link_idx, w, _) in enumerate(self.mapped_body_link_rot_data):
            objective = ik.IKObjectiveRotation(
                link_index=link_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rot_target_arrays[i],
                weight=w)
            rotation_objectives.append(objective)

        direction_objectives = []
        for i, (_, _, parent_link_idx, child_link_idx, w, _, analytic_jacobian) in enumerate(self.mapped_body_link_direction_data):
            objective = IKObjectiveDirection(
                parent_link_index=parent_link_idx,
                child_link_index=child_link_idx,
                target_dirs=dir_target_arrays[i],
                weight=w,
                analytic_jacobian=analytic_jacobian)
            direction_objectives.append(objective)

        pole_vector_objectives = []
        for i, (_, _, _, parent_link_idx, middle_link_idx, child_link_idx, w, _, analytic_jacobian) in enumerate(self.mapped_body_link_pole_vector_data):
            objective = IKObjectivePoleVector(
                parent_link_index=parent_link_idx,
                middle_link_index=middle_link_idx,
                child_link_index=child_link_idx,
                target_normals=pole_target_arrays[i],
                weight=w,
                analytic_jacobian=analytic_jacobian)
            pole_vector_objectives.append(objective)

        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        # Weight is set to desired value once initialization frames have been processed.
        # The legacy config name is kept, but v2 uses a range-normalized margin barrier.
        smooth_joint_limiter_objective = IKRangeNormalizedJointLimitBarrier(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=0.0,
            coord_masks=self.smooth_joint_filter_coord_masks)

        temporal_prev_q = getattr(
            self,
            "temporal_prev_q",
            self.ik_model.joint_q.numpy().astype(np.float32),
        )
        temporal_prevprev_q = getattr(self, "temporal_prevprev_q", temporal_prev_q)
        temporal_velocity_scales = getattr(
            self,
            "temporal_velocity_scales",
            np.zeros_like(temporal_prev_q, dtype=np.float32),
        )
        temporal_acceleration_scales = getattr(
            self,
            "temporal_acceleration_scales",
            np.zeros_like(temporal_prev_q, dtype=np.float32),
        )
        temporal_coord_masks = getattr(
            self,
            "temporal_coord_masks",
            np.zeros(self.ik_model.joint_coord_count, dtype=np.float32),
        )
        temporal_velocity_objective = IKTemporalJointRegularizer(
            reference_q=wp.array(np.asarray(temporal_prev_q, dtype=np.float32), dtype=wp.float32),
            scales=wp.array(np.asarray(temporal_velocity_scales, dtype=np.float32), dtype=wp.float32),
            weight=self.temporal_velocity_weight,
            coord_masks=temporal_coord_masks,
            mode=IKTemporalJointRegularizer.MODE_VELOCITY,
        )
        temporal_acceleration_objective = IKTemporalJointRegularizer(
            reference_q=wp.array(np.asarray(temporal_prev_q, dtype=np.float32), dtype=wp.float32),
            reference_q2=wp.array(np.asarray(temporal_prevprev_q, dtype=np.float32), dtype=wp.float32),
            scales=wp.array(np.asarray(temporal_acceleration_scales, dtype=np.float32), dtype=wp.float32),
            weight=self.temporal_acceleration_weight,
            coord_masks=temporal_coord_masks,
            mode=IKTemporalJointRegularizer.MODE_ACCELERATION,
        )

        ground_barrier_objectives = self._create_ground_barrier_objectives(num_envs)
        collision_objectives = self._create_collision_objectives()
        contact_objectives = self._create_contact_aware_objectives(num_envs, pos_target_arrays)
        return (
            position_objectives,
            rotation_objectives,
            direction_objectives,
            pole_vector_objectives,
            joint_limit_objective,
            smooth_joint_limiter_objective,
            temporal_velocity_objective,
            temporal_acceleration_objective,
            ground_barrier_objectives,
            collision_objectives,
            contact_objectives,
        )

    def _build_temporal_scale_matrices(self, model, num_envs):
        coord_ranges, coord_masks = self._joint_coord_ranges_and_temporal_mask(model)
        sample_rates = np.asarray(
            [
                float(rate) if float(rate) > 0.0 else 60.0
                for rate in (self.input_sample_rates[:num_envs] or [60.0] * num_envs)
            ],
            dtype=np.float32,
        )
        if len(sample_rates) < num_envs:
            sample_rates = np.pad(sample_rates, (0, num_envs - len(sample_rates)), constant_values=60.0)
        sample_rates = sample_rates[:num_envs]

        inv_ranges = np.zeros_like(coord_ranges, dtype=np.float32)
        valid = coord_ranges > 1.0e-8
        inv_ranges[valid] = 1.0 / coord_ranges[valid]
        velocity_scales = sample_rates[:, None] * inv_ranges[None, :]
        acceleration_scales = (sample_rates[:, None] ** 2) * inv_ranges[None, :]
        velocity_scales *= coord_masks[None, :]
        acceleration_scales *= coord_masks[None, :]
        return velocity_scales.astype(np.float32), acceleration_scales.astype(np.float32), coord_masks.astype(np.float32)

    @staticmethod
    def _apply_joint_motion_limits_to_frame(
        frame_q,
        previous_q,
        previous_delta,
        coord_ranges,
        coord_masks,
        dt,
        velocity_fraction_per_second,
        acceleration_fraction_per_second2,
    ):
        frame_q = np.asarray(frame_q, dtype=np.float32)
        previous_q = np.asarray(previous_q, dtype=np.float32)
        previous_delta = np.asarray(previous_delta, dtype=np.float32)
        coord_ranges = np.asarray(coord_ranges, dtype=np.float32)
        coord_masks = np.asarray(coord_masks, dtype=np.float32) > 0.5
        dt = max(float(dt), 1.0e-6)

        limited = np.array(frame_q, dtype=np.float32, copy=True)
        delta = limited - previous_q
        active = coord_masks & (coord_ranges > 1.0e-8)
        if not np.any(active):
            return limited, delta.astype(np.float32)

        max_delta = coord_ranges * max(float(velocity_fraction_per_second), 0.0) * dt
        max_delta2 = coord_ranges * max(float(acceleration_fraction_per_second2), 0.0) * dt * dt
        delta[active] = np.clip(delta[active], -max_delta[active], max_delta[active])
        delta_change = delta - previous_delta
        delta[active] = previous_delta[active] + np.clip(
            delta_change[active],
            -max_delta2[active],
            max_delta2[active],
        )
        delta[active] = np.clip(delta[active], -max_delta[active], max_delta[active])
        limited[active] = previous_q[active] + delta[active]
        return limited.astype(np.float32), delta.astype(np.float32)

    @staticmethod
    def _joint_coord_ranges_and_temporal_mask(model):
        n_coords = model.joint_coord_count
        coord_ranges = np.zeros(n_coords, dtype=np.float32)
        coord_masks = np.zeros(n_coords, dtype=np.float32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        lower = model.joint_limit_lower.numpy().astype(np.float32)
        upper = model.joint_limit_upper.numpy().astype(np.float32)

        for joint_idx in range(model.joint_count):
            coord0 = q_start_np[joint_idx]
            dof0 = qd_start_np[joint_idx]
            lin, ang = joint_dof_dim_np[joint_idx]
            dof_count = int(lin + ang)
            for k in range(dof_count):
                coord_idx = coord0 + k
                dof_idx = dof0 + k
                if coord_idx >= n_coords or dof_idx >= len(lower):
                    continue
                span = float(upper[dof_idx] - lower[dof_idx])
                if np.isfinite(span) and 1.0e-8 < span < 1.0e5:
                    coord_ranges[coord_idx] = span
                    coord_masks[coord_idx] = 1.0
        coord_masks[: min(7, n_coords)] = 0.0
        return coord_ranges, coord_masks

    @staticmethod
    def _joint_limit_guard_residuals(model, joint_q_frames, margin_fraction=0.08):
        joint_q_frames = np.asarray(joint_q_frames, dtype=np.float32)
        if joint_q_frames.ndim == 1:
            joint_q_frames = joint_q_frames[None, :]
        n_envs, n_coords = joint_q_frames.shape
        residuals = np.zeros((n_envs, n_coords), dtype=np.float32)

        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        lower = model.joint_limit_lower.numpy().astype(np.float32)
        upper = model.joint_limit_upper.numpy().astype(np.float32)

        for joint_idx in range(model.joint_count):
            coord0 = q_start_np[joint_idx]
            dof0 = qd_start_np[joint_idx]
            lin, ang = joint_dof_dim_np[joint_idx]
            for k in range(int(lin + ang)):
                coord_idx = coord0 + k
                dof_idx = dof0 + k
                if coord_idx >= n_coords or dof_idx >= len(lower):
                    continue
                for env in range(n_envs):
                    if margin_fraction <= 0.0:
                        span = float(upper[dof_idx] - lower[dof_idx])
                        if not np.isfinite(span) or span <= 1.0e-8 or span > 1.0e5:
                            continue
                        q = float(joint_q_frames[env, coord_idx])
                        if q < float(lower[dof_idx]):
                            residuals[env, coord_idx] = (q - float(lower[dof_idx])) / span
                        elif q > float(upper[dof_idx]):
                            residuals[env, coord_idx] = (q - float(upper[dof_idx])) / span
                    else:
                        residuals[env, coord_idx] = range_normalized_joint_limit_barrier(
                            joint_q_frames[env, coord_idx],
                            lower[dof_idx],
                            upper[dof_idx],
                            margin_fraction,
                        )[0]
        residuals[:, : min(7, n_coords)] = 0.0
        return residuals

    @staticmethod
    def _joint_limit_guard_costs(model, joint_q_frames, margin_fraction=0.08):
        residuals = NewtonPipeline._joint_limit_guard_residuals(model, joint_q_frames, margin_fraction)
        return np.sum(residuals * residuals, axis=1).astype(np.float32)

    @staticmethod
    def _priority_guard_should_rollback(before_costs, after_costs, tolerance=0.05, absolute_tolerance=1.0e-5):
        before = np.asarray(before_costs, dtype=np.float32)
        after = np.asarray(after_costs, dtype=np.float32)
        allowed = before * (1.0 + float(tolerance)) + float(absolute_tolerance)
        return after > allowed

    @staticmethod
    def _priority_guard_requires_cpu_check(margin_fraction=0.0):
        return float(margin_fraction) > 0.0

    @staticmethod
    def _normalize_rotation_basis(raw_basis):
        if raw_basis is None:
            return None
        basis = np.asarray(raw_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != 3:
            return None
        if basis.shape[1] >= 3:
            return None
        return basis

    @staticmethod
    def _normalize_position_basis(raw_basis):
        if raw_basis is None:
            return None
        basis = np.asarray(raw_basis, dtype=np.float64)
        if basis.ndim != 2:
            return None
        if basis.shape[1] == 3:
            vectors = basis
        elif basis.shape[0] == 3:
            vectors = basis.T
        else:
            return None
        if vectors.shape[0] >= 3 or vectors.shape[0] <= 0:
            return None
        if not np.all(np.isfinite(vectors)):
            return None
        normalized = []
        for vector in vectors:
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-8:
                return None
            normalized.append((vector / norm).astype(np.float32))
        return np.asarray(normalized, dtype=np.float32)

    @staticmethod
    def _project_rotation_target(quat_xyzw, basis):
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        if basis is None:
            return quat_xyzw.astype(np.float32)
        return project_relative_rotation_quat_xyzw(quat_xyzw, basis).astype(np.float32)

    @staticmethod
    def _select_ik_jacobian_mode(objectives):
        if any(not objective.supports_analytic() for objective in objectives):
            return ik.IKJacobianType.MIXED
        return ik.IKJacobianType.ANALYTIC

    @staticmethod
    def _direction_between_positions(reference_pos, target_pos):
        delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(reference_pos, dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-6:
            return np.zeros(3, dtype=np.float32)
        return (delta / length).astype(np.float32)

    @staticmethod
    def _direction_between_targets(reference_target, target_target):
        return NewtonPipeline._direction_between_positions(reference_target[0:3], target_target[0:3])

    @staticmethod
    def _pole_normal_between_positions(reference_pos, middle_pos, target_pos, fallback=None):
        a = np.asarray(middle_pos, dtype=np.float64) - np.asarray(reference_pos, dtype=np.float64)
        b = np.asarray(target_pos, dtype=np.float64) - np.asarray(middle_pos, dtype=np.float64)
        normal = np.cross(a, b)
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-6:
            if fallback is None:
                return np.zeros(3, dtype=np.float32), True
            return np.asarray(fallback, dtype=np.float32), True
        return (normal / length).astype(np.float32), False

    @staticmethod
    def _pole_normal_between_targets(reference_target, middle_target, target_target, fallback=None):
        return NewtonPipeline._pole_normal_between_positions(
            reference_target[0:3],
            middle_target[0:3],
            target_target[0:3],
            fallback,
        )

    def _create_contact_aware_objectives(self, num_envs, pos_target_arrays):
        if not self.contact_aware_foot_ik_enabled:
            self.contact_objective_map = {}
            return []
        anchors = self.contact_aware_foot_ik.get("anchor_offsets", {})
        left = anchors.get("left", {})
        right = anchors.get("right", {})
        if not left or not right:
            print("[WARN] contact_aware_foot_ik enabled but anchor_offsets not configured; skipping.")
            self.contact_objective_map = {}
            return []
        link_lookup = getattr(self, "mapped_body_link_by_joint", None)
        if not link_lookup:
            link_lookup = {}
            for name, entry in zip(self.mapped_joints, self.mapped_body_link_pos_data):
                link_lookup[name] = entry[1]
        if "LeftFoot" not in link_lookup or "RightFoot" not in link_lookup:
            print("[WARN] contact_aware_foot_ik enabled but LeftFoot/RightFoot are missing in ik_map; skipping.")
            self.contact_objective_map = {}
            return []
        mapping = {
            "left_toe": ("LeftFoot", left.get("toe"), "left_toe_contact_score", self.contact_aware_foot_ik.get("toe_weight_stance", 0.8), self.contact_aware_foot_ik.get("toe_weight_swing", 0.1)),
            "left_heel": ("LeftFoot", left.get("heel"), "left_heel_contact_score", self.contact_aware_foot_ik.get("heel_weight_stance", 0.8), self.contact_aware_foot_ik.get("heel_weight_swing", 0.1)),
            "right_toe": ("RightFoot", right.get("toe"), "right_toe_contact_score", self.contact_aware_foot_ik.get("toe_weight_stance", 0.8), self.contact_aware_foot_ik.get("toe_weight_swing", 0.1)),
            "right_heel": ("RightFoot", right.get("heel"), "right_heel_contact_score", self.contact_aware_foot_ik.get("heel_weight_stance", 0.8), self.contact_aware_foot_ik.get("heel_weight_swing", 0.1)),
        }
        out=[]; self.contact_objective_map={}
        for key, (joint, offset, score_key, w_stance, w_swing) in mapping.items():
            if offset is None or joint not in link_lookup:
                continue
            targets = wp.array(np.zeros((num_envs,3), dtype=np.float32), dtype=wp.vec3)
            weights = wp.array(np.full(num_envs, float(w_swing), dtype=np.float32), dtype=wp.float32)
            obj = IKObjectivePerEnvWeightedPosition(
                link_index=link_lookup[joint],
                link_offset=wp.vec3(*offset),
                target_positions=targets,
                weights=weights)
            out.append(obj)
            self.contact_objective_map[key] = {
                "objective": obj,
                "score_key": score_key,
                "stance": float(w_stance),
                "swing": float(w_swing),
                "active": [False] * num_envs,
                "locked": [None] * num_envs,
                "age": [0] * num_envs,
                "release_remaining": [0] * num_envs,
                "release_total": [0] * num_envs,
                "release_start": [None] * num_envs,
            }
        return out

    def _update_contact_objectives_for_frame(self, env, frame, frame_targets, contact_objectives):
        if not hasattr(self, "contact_objective_map"):
            return
        scores = self.input_contact_scores[env] if env < len(self.input_contact_scores) else None
        on_t = float(self.contact_aware_foot_ik.get("contact_on_threshold", 0.6))
        off_t = float(self.contact_aware_foot_ik.get("contact_off_threshold", 0.3))
        min_contact_frames = max(1, int(self.contact_aware_foot_ik.get("min_contact_frames", 1)))
        release_blend_frames = max(0, int(self.contact_aware_foot_ik.get("release_blend_frames", 0)))
        if "LeftFoot" not in self.mapped_joints or "RightFoot" not in self.mapped_joints:
            return
        joint_idx = {"LeftFoot": self.mapped_joints.index("LeftFoot"), "RightFoot": self.mapped_joints.index("RightFoot")}
        for key, cfg in self.contact_objective_map.items():
            n_env_state = max(env + 1, len(cfg.get("active", [])))
            cfg.setdefault("age", [0] * n_env_state)
            cfg.setdefault("release_remaining", [0] * n_env_state)
            cfg.setdefault("release_total", [0] * n_env_state)
            cfg.setdefault("release_start", [None] * n_env_state)
            side_joint = "LeftFoot" if key.startswith("left") else "RightFoot"
            foot = frame_targets[joint_idx[side_joint]]
            pos=np.array(foot[0:3],dtype=np.float32); q=wp.quat(*foot[3:7]); off=cfg["objective"].link_offset
            world = pos + np.array(wp.quat_rotate(q, off), dtype=np.float32)
            score = 0.0
            if scores is not None and cfg["score_key"] in scores and frame < len(scores[cfg["score_key"]]):
                score = float(scores[cfg["score_key"]][frame])
            active = cfg["active"][env]
            if (not active) and score >= on_t:
                cfg["active"][env] = True
                cfg["locked"][env] = world.copy()
                cfg["age"][env] = 0
                cfg["release_remaining"][env] = 0
                cfg["release_total"][env] = 0
                cfg["release_start"][env] = None
            elif active and score <= off_t and cfg["age"][env] >= min_contact_frames:
                cfg["active"][env] = False
                if release_blend_frames > 0 and cfg["locked"][env] is not None:
                    cfg["release_remaining"][env] = release_blend_frames
                    cfg["release_total"][env] = release_blend_frames
                    cfg["release_start"][env] = np.array(cfg["locked"][env], dtype=np.float32)
                cfg["locked"][env] = None
                cfg["age"][env] = 0

            if cfg["active"][env]:
                cfg["age"][env] += 1
                target = cfg["locked"][env] if cfg["locked"][env] is not None else world
                weight = cfg["stance"]
            elif cfg["release_remaining"][env] > 0 and cfg["release_start"][env] is not None:
                alpha = float(cfg["release_remaining"][env]) / float(max(cfg["release_total"][env], 1))
                target = cfg["release_start"][env] * alpha + world * (1.0 - alpha)
                weight = cfg["swing"] + (cfg["stance"] - cfg["swing"]) * alpha
                cfg["release_remaining"][env] -= 1
                if cfg["release_remaining"][env] <= 0:
                    cfg["release_total"][env] = 0
                    cfg["release_start"][env] = None
            else:
                target = world
                weight = cfg["swing"]
            cfg["objective"].set_target_position(env, wp.vec3(*target))
            if hasattr(cfg["objective"], "set_weight"):
                cfg["objective"].set_weight(env, weight)

    def _create_collision_objectives(self):
        self.collision_objective_report = {
            "enabled": bool(self.compiled_collision_config.get("enabled", False)) and self.collision_weight > 0.0,
            "weight": float(self.collision_weight),
            "created": 0,
            "skipped": 0,
            "runtime_barrier": self.compiled_collision_config.get("runtime_barrier"),
        }
        if self.collision_weight <= 0.0 or not self.compiled_collision_config.get("enabled", False):
            return []

        link_lookup = getattr(self, "mapped_body_link_by_joint", None) or {}
        proxies = {
            str(proxy.get("semantic")): proxy
            for proxy in self.compiled_collision_config.get("proxies", [])
            if isinstance(proxy, dict) and proxy.get("semantic")
        }
        out = []
        default_margin = float(self.compiled_collision_config.get("margin", 0.03))
        for pair in self.compiled_collision_config.get("pairs", []):
            if not isinstance(pair, dict):
                self.collision_objective_report["skipped"] += 1
                continue
            semantic_a = str(pair.get("a"))
            semantic_b = str(pair.get("b"))
            proxy_a = proxies.get(semantic_a)
            proxy_b = proxies.get(semantic_b)
            if proxy_a is None or proxy_b is None or semantic_a not in link_lookup or semantic_b not in link_lookup:
                self.collision_objective_report["skipped"] += 1
                continue
            try:
                center_a = proxy_a.get("local_center", [0.0, 0.0, 0.0])
                center_b = proxy_b.get("local_center", [0.0, 0.0, 0.0])
                radius_a = float(proxy_a.get("radius", 0.0))
                radius_b = float(proxy_b.get("radius", 0.0))
                margin = float(pair.get("margin", default_margin))
            except (TypeError, ValueError):
                self.collision_objective_report["skipped"] += 1
                continue
            if radius_a <= 0.0 or radius_b <= 0.0 or margin <= 0.0:
                self.collision_objective_report["skipped"] += 1
                continue
            out.append(
                IKObjectiveSphereCollisionBarrier(
                    link_index_a=link_lookup[semantic_a],
                    link_offset_a=wp.vec3(*center_a),
                    radius_a=radius_a,
                    link_index_b=link_lookup[semantic_b],
                    link_offset_b=wp.vec3(*center_b),
                    radius_b=radius_b,
                    margin=margin,
                    weight=self.collision_weight,
                )
            )
            self.collision_objective_report["created"] += 1
        return out

    def _create_ground_barrier_objectives(self, num_envs):
        if not self.ground_barrier_enabled:
            self.ground_barrier_objective_map = {}
            return []

        anchors = self.contact_aware_foot_ik.get("anchor_offsets", {})
        left = anchors.get("left", {})
        right = anchors.get("right", {})
        if not left or not right:
            print("[WARN] ground_barrier enabled but contact_aware_foot_ik anchor_offsets are missing; skipping.")
            self.ground_barrier_objective_map = {}
            return []

        link_lookup = getattr(self, "mapped_body_link_by_joint", None)
        if not link_lookup:
            link_lookup = {}
            for name, entry in zip(self.mapped_joints, self.mapped_body_link_pos_data):
                link_lookup[name] = entry[1]
        if "LeftFoot" not in link_lookup or "RightFoot" not in link_lookup:
            print("[WARN] ground_barrier enabled but LeftFoot/RightFoot are missing in ik_map; skipping.")
            self.ground_barrier_objective_map = {}
            return []

        ground_height = float(self.ground_barrier_config.get("ground_height", 0.0))
        margin = float(self.ground_barrier_config.get("margin", 0.03))
        stance_weight = float(self.ground_barrier_config.get("stance_weight", 1.0))
        swing_weight = float(self.ground_barrier_config.get("swing_weight", 0.1))
        mapping = {
            "left_toe": ("LeftFoot", left.get("toe"), "left_toe_contact_score"),
            "left_heel": ("LeftFoot", left.get("heel"), "left_heel_contact_score"),
            "right_toe": ("RightFoot", right.get("toe"), "right_toe_contact_score"),
            "right_heel": ("RightFoot", right.get("heel"), "right_heel_contact_score"),
        }

        out = []
        self.ground_barrier_objective_map = {}
        for key, (joint, offset, score_key) in mapping.items():
            if offset is None or joint not in link_lookup:
                continue
            weights = wp.array(np.full(num_envs, swing_weight, dtype=np.float32), dtype=wp.float32)
            obj = IKObjectivePerEnvGroundHeightBarrier(
                link_index=link_lookup[joint],
                link_offset=wp.vec3(*offset),
                weights=weights,
                ground_height=ground_height,
                margin=margin,
            )
            out.append(obj)
            self.ground_barrier_objective_map[key] = {
                "objective": obj,
                "score_key": score_key,
                "stance": stance_weight,
                "swing": swing_weight,
            }
        return out

    def _update_ground_barrier_objectives_for_frame(self, env, frame, ground_barrier_objectives):
        if not hasattr(self, "ground_barrier_objective_map"):
            return
        scores = self.input_contact_scores[env] if env < len(self.input_contact_scores) else None
        on_t = float(self.contact_aware_foot_ik.get("contact_on_threshold", 0.6))
        for cfg in self.ground_barrier_objective_map.values():
            score = 0.0
            if scores is not None and cfg["score_key"] in scores and frame < len(scores[cfg["score_key"]]):
                score = float(scores[cfg["score_key"]][frame])
            weight = cfg["stance"] if score >= on_t else cfg["swing"]
            cfg["objective"].set_weight(env, weight)
