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
    IKObjectivePerEnvWeightedPosition,
    IKRangeNormalizedJointLimitBarrier,
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
        self.smooth_joint_filter_coord_masks = None
        self.joint_limit_clamper = None

        self.robot_builder = newton.ModelBuilder()
        self.robot_builder.add_mjcf(str(pipeline_utils.get_robot_mjcf_path(self.target_type)))

        self.human_robot_scaler = HumanToRobotScaler(
            skeleton, retargeter_config['model_height'], io_utils.get_config_file(retargeter_config['human_robot_scaler_config']))
        compiled_profile = retargeter_config.get("compiled_retarget_profile")
        if compiled_profile:
            self.human_robot_scaler.enable_segment_local_from_profile(io_utils.get_config_file(compiled_profile))

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

        restore_contact_aware = self.contact_aware_foot_ik_enabled
        try:
            model = self._build_model(num_envs)
            state = model.state()

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
                contact_objectives,
            ) = self._create_ik_objectives(num_envs, model, state)

            ik_solver_active_objectives = [
                *position_objectives,
                *rotation_objectives,
                *direction_objectives,
                *pole_vector_objectives,
                *contact_objectives,
            ]
            if self.joint_limit_weight > 0.0:
                ik_solver_active_objectives.append(joint_limit_objective)
            if self.smooth_joint_filter_weight > 0.0:
                ik_solver_active_objectives.append(smooth_joint_filter_objective)

            ik_solver = ik.IKSolver(
                model=self.ik_model,
                n_problems=num_envs,
                objectives=ik_solver_active_objectives,
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.MIXED if direction_objectives or pole_vector_objectives else ik.IKJacobianType.ANALYTIC,
            )

            joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
            wp.copy(joint_q, model.joint_q)
            ik_solver.reset()

            graph_capture = None

            def single_step():
                ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

            if wp.get_device().is_cuda:
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

            for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions"):
                if num_frames_to_remove > 0 and frame <= num_frames_to_remove:
                    smooth_joint_filter_objective.set_weight(
                        self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove))
                    )

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    frame_targets = self.input_targets[env][frame]
                    for i, (effector_idx, _, _) in enumerate(self.mapped_body_link_pos_data):
                        target = frame_targets[effector_idx]
                        position_objectives[i].set_target_position(env, wp.vec3(*target[0:3]))
                    for i, (effector_idx, _, _, basis) in enumerate(self.mapped_body_link_rot_data):
                        target = frame_targets[effector_idx]
                        target_rotation = self._project_rotation_target(target[3:7], basis)
                        rotation_objectives[i].set_target_rotation(env, wp.quat(*target_rotation))
                    for i, (reference_idx, target_idx, _, _, _, _) in enumerate(self.mapped_body_link_direction_data):
                        target_direction = self._direction_between_targets(frame_targets[reference_idx], frame_targets[target_idx])
                        direction_objectives[i].set_target_direction(env, wp.vec3(*target_direction))
                    for i, (reference_idx, middle_idx, target_idx, _, _, _, _, _) in enumerate(self.mapped_body_link_pole_vector_data):
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

                if graph_capture is not None:
                    wp.capture_launch(graph_capture)
                else:
                    single_step()

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

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env]) - 1):
                        continue
                    joint_q_data[env][frame] = np.array(data[env], copy=True)

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
                mapped_body_link_pos_data.append((effector_idx, t_link_idx, float(mapping_data['t_weight'])))
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
            for ee_idx, (_, link_idx, _) in enumerate(self.mapped_body_link_pos_data):
                pos_targets[env, ee_idx] = body_q[base + link_idx][0:3]

            for ee_idx, (_, link_idx, _, _) in enumerate(self.mapped_body_link_rot_data):
                rot_wp = wp.quat(body_q[base + link_idx][3:7])
                rot_targets[env, ee_idx] = wp.normalize(rot_wp)
            for ee_idx, (_, _, parent_link_idx, child_link_idx, _, _) in enumerate(self.mapped_body_link_direction_data):
                dir_targets[env, ee_idx] = wp.vec3(
                    *self._direction_between_positions(
                        body_q[base + parent_link_idx][0:3],
                        body_q[base + child_link_idx][0:3],
                    )
                )
            for ee_idx, (_, _, _, parent_link_idx, middle_link_idx, child_link_idx, _, _) in enumerate(self.mapped_body_link_pole_vector_data):
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
        for i, (_, link_idx, w) in enumerate(self.mapped_body_link_pos_data):
            objective = ik.IKObjectivePosition(
                link_index=link_idx,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=pos_target_arrays[i],
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
        for i, (_, _, parent_link_idx, child_link_idx, w, _) in enumerate(self.mapped_body_link_direction_data):
            objective = IKObjectiveDirection(
                parent_link_index=parent_link_idx,
                child_link_index=child_link_idx,
                target_dirs=dir_target_arrays[i],
                weight=w)
            direction_objectives.append(objective)

        pole_vector_objectives = []
        for i, (_, _, _, parent_link_idx, middle_link_idx, child_link_idx, w, _) in enumerate(self.mapped_body_link_pole_vector_data):
            objective = IKObjectivePoleVector(
                parent_link_index=parent_link_idx,
                middle_link_index=middle_link_idx,
                child_link_index=child_link_idx,
                target_normals=pole_target_arrays[i],
                weight=w)
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

        contact_objectives = self._create_contact_aware_objectives(num_envs, pos_target_arrays)
        return (
            position_objectives,
            rotation_objectives,
            direction_objectives,
            pole_vector_objectives,
            joint_limit_objective,
            smooth_joint_limiter_objective,
            contact_objectives,
        )

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
    def _project_rotation_target(quat_xyzw, basis):
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        if basis is None:
            return quat_xyzw.astype(np.float32)
        return project_relative_rotation_quat_xyzw(quat_xyzw, basis).astype(np.float32)

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
                link_lookup[name] = entry[1] if len(entry) == 3 else entry[0]
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
            self.contact_objective_map[key] = {"objective": obj, "score_key": score_key, "stance": float(w_stance), "swing": float(w_swing), "active": [False]*num_envs, "locked": [None]*num_envs}
        return out

    def _update_contact_objectives_for_frame(self, env, frame, frame_targets, contact_objectives):
        if not hasattr(self, "contact_objective_map"):
            return
        scores = self.input_contact_scores[env] if env < len(self.input_contact_scores) else None
        on_t = float(self.contact_aware_foot_ik.get("contact_on_threshold", 0.6)); off_t = float(self.contact_aware_foot_ik.get("contact_off_threshold", 0.3))
        if "LeftFoot" not in self.mapped_joints or "RightFoot" not in self.mapped_joints:
            return
        joint_idx = {"LeftFoot": self.mapped_joints.index("LeftFoot"), "RightFoot": self.mapped_joints.index("RightFoot")}
        for key, cfg in self.contact_objective_map.items():
            side_joint = "LeftFoot" if key.startswith("left") else "RightFoot"
            foot = frame_targets[joint_idx[side_joint]]
            pos=np.array(foot[0:3],dtype=np.float32); q=wp.quat(*foot[3:7]); off=cfg["objective"].link_offset
            world = pos + np.array(wp.quat_rotate(q, off), dtype=np.float32)
            score = float(scores[cfg["score_key"]][frame]) if scores is not None and frame < len(scores[cfg["score_key"]]) else 0.0
            active = cfg["active"][env]
            if (not active) and score >= on_t:
                cfg["active"][env]=True; cfg["locked"][env]=world.copy()
            elif active and score <= off_t:
                cfg["active"][env]=False; cfg["locked"][env]=None
            target = cfg["locked"][env] if cfg["active"][env] and cfg["locked"][env] is not None else world
            cfg["objective"].set_target_position(env, wp.vec3(*target))
            weight = cfg["stance"] if cfg["active"][env] else cfg["swing"]
            if hasattr(cfg["objective"], "set_weight"):
                cfg["objective"].set_weight(env, weight)
