# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path

import warp as wp
import numpy as np
import newton
import newton.ik as ik
from tqdm import trange

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.pipelines.utils as pipeline_utils
from soma_retargeter.pipelines.v3_runtime_config import parse_v3_runtime_profile_config
from soma_retargeter.pipelines.ik_objectives import IKSmoothJointFilter
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.pipelines.feet_stabilizer import FeetStabilizer
from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper
from soma_retargeter.pipelines.motion_grounding import apply_virtual_foot_grounding_to_frames
from soma_retargeter.pipelines.foot_contact_inference import (
    infer_contacts_from_animation_buffer,
    contacts_from_npz_foot_contacts,
)
from soma_retargeter.pipelines.stance_width_diagnostics import (
    DEBUG_CONFIG_KEY,
    apply_debug_options_to_ik_map,
    build_stance_width_report,
    joint_stats_from_raw_data,
    normalize_debug_config,
)

_DEFAULT_IK_SOLVER_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT = 5.5
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5


def _anchor_offsets_from_virtual_sole_anchors(anchors):
    if not isinstance(anchors, dict) or not anchors.get("enabled"):
        return None
    left = anchors.get("left", {}) if isinstance(anchors.get("left"), dict) else {}
    right = anchors.get("right", {}) if isinstance(anchors.get("right"), dict) else {}
    if not all(name in left for name in ("toe", "heel")) or not all(name in right for name in ("toe", "heel")):
        return None
    offsets = {
        "left": {"toe": left["toe"], "heel": left["heel"]},
        "right": {"toe": right["toe"], "heel": right["heel"]},
    }
    for edge_name in ("inner_edge", "outer_edge"):
        if edge_name in left:
            offsets["left"][edge_name] = left[edge_name]
        if edge_name in right:
            offsets["right"][edge_name] = right[edge_name]
    return offsets


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
        self.stance_width_report_inputs = []
        self.stance_width_reports = []

        if retarget_config is None:
            retargeter_config = pipeline_utils.get_retargeter_config(self.source_type, self.target_type)
        else:
            retargeter_config = retarget_config
        retargeter_config = copy.deepcopy(retargeter_config)
        self.v3_runtime_config = parse_v3_runtime_profile_config(retargeter_config)
        self.v3_runtime_profile = None
        self.v3_runtime_profile_resolution = None
        self.v3_runtime_adapter = None
        self.v3_runtime_diagnostics = []
        self.stance_width_debug_config = normalize_debug_config(retargeter_config.get(DEBUG_CONFIG_KEY, {}))
        retargeter_config["ik_map"], self.stance_width_debug_summary = apply_debug_options_to_ik_map(
            retargeter_config.get("ik_map", {}),
            self.stance_width_debug_config,
        )
        self.stance_width_diagnostics_enabled = bool(self.stance_width_debug_config.get("enabled", False))
        self.stance_width_report_inputs = []
        self.stance_width_reports = []
        self.retargeter_config = retargeter_config

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
        raw_contact_aware_foot_ik = retargeter_config.get("contact_aware_foot_ik", {})
        self.contact_aware_foot_ik = dict(raw_contact_aware_foot_ik) if isinstance(raw_contact_aware_foot_ik, dict) else {}
        if "anchor_offsets" not in self.contact_aware_foot_ik:
            anchor_offsets = _anchor_offsets_from_virtual_sole_anchors(retargeter_config.get("virtual_sole_anchors"))
            if anchor_offsets is not None:
                self.contact_aware_foot_ik["anchor_offsets"] = anchor_offsets
        self.contact_source = str(self.contact_aware_foot_ik.get("contact_source", "auto")).lower()
        self.contact_aware_foot_ik_enabled = (
            bool(self.contact_aware_foot_ik.get("enabled", False))
            and self.contact_source not in {"none", "disabled", "false", "null"}
        )
        self.smooth_joint_filter_coord_masks = None
        self.joint_limit_clamper = None

        self.robot_builder = newton.ModelBuilder()
        self.robot_mjcf_path = pipeline_utils.get_robot_mjcf_path(self.target_type)
        self.robot_builder.add_mjcf(str(self.robot_mjcf_path))

        self.human_robot_scaler = HumanToRobotScaler(
            skeleton, retargeter_config['model_height'], io_utils.get_config_file(retargeter_config['human_robot_scaler_config']))

        self.num_body_count = self.robot_builder.body_count
        self.num_dofs = self.robot_builder.joint_dof_count
        self.ik_model = self._build_model(1)

        (
            self.mapped_joints,
            self.mapped_joint_indices,
            self.mapped_body_link_pos_data,
            self.mapped_body_link_rot_data
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

        self._initialize_v3_runtime()

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
        if hasattr(self, "v3_runtime_diagnostics"):
            self.v3_runtime_diagnostics = []

    def _initialize_v3_runtime(self):
        cfg = getattr(self, "v3_runtime_config", None)
        if cfg is None or not cfg.should_compute_targets:
            return

        try:
            from soma_retargeter.runtime.v3.profile_loader import load_runtime_v3_profile
        except Exception as exc:
            raise RuntimeError(
                "v3_runtime_profile is enabled but the runtime profile loader is unavailable"
            ) from exc

        try:
            profile, resolution = load_runtime_v3_profile(self.target_type, self.robot_mjcf_path, cfg)
        except Exception as exc:
            raise RuntimeError(f"v3_runtime_profile initialization failed: {exc}") from exc

        self.v3_runtime_profile = profile
        self.v3_runtime_profile_resolution = resolution
        if profile is None:
            return

        try:
            from soma_retargeter.runtime.v3.target_adapter import RuntimeV3TargetAdapter
        except Exception as exc:
            raise RuntimeError(
                "v3_runtime_profile is enabled but the runtime target adapter is unavailable"
            ) from exc

        try:
            self.v3_runtime_adapter = RuntimeV3TargetAdapter(profile=profile, config=cfg)
        except TypeError:
            self.v3_runtime_adapter = RuntimeV3TargetAdapter(profile)

    def _apply_v3_runtime_targets(
        self,
        *,
        buffer,
        offset,
        scale_animation,
        legacy_buffer_effectors,
        clip_index,
    ):
        cfg = getattr(self, "v3_runtime_config", None)
        if cfg is None or not cfg.should_compute_targets:
            return legacy_buffer_effectors

        adapter = getattr(self, "v3_runtime_adapter", None)
        if adapter is None:
            self._record_v3_runtime_diagnostics(
                clip_index,
                {
                    "skipped_reason": "runtime_target_adapter_unavailable",
                    "profile_resolution": self._v3_profile_resolution_json(),
                },
            )
            if cfg.shadow_enabled:
                return legacy_buffer_effectors
            raise RuntimeError("v3 override_experimental requires an initialized runtime target adapter")

        legacy_for_adapter = np.array(legacy_buffer_effectors, copy=True)
        kwargs = {
            "buffer": buffer,
            "offset": offset,
            "scale_animation": scale_animation,
            "legacy_buffer_effectors": legacy_for_adapter,
            "legacy_effector_names": list(self.human_robot_scaler.effector_names()),
            "target_effector_indices": list(self.target_effector_indices),
            "mapped_joints": list(self.mapped_joints),
            "semantic_tasks": list(cfg.semantic_tasks),
            "override_tasks": list(cfg.override_tasks),
            "mode": cfg.mode,
            "robot_type": (
                pipeline_utils.get_target_str_from_type(self.target_type)
                if hasattr(self, "target_type")
                else ""
            ),
            "config": cfg,
            "profile": getattr(self, "v3_runtime_profile", None),
            "profile_resolution": getattr(self, "v3_runtime_profile_resolution", None),
            "clip_index": clip_index,
            "diagnostics_max_frames": cfg.diagnostics_max_frames,
        }
        result = self._call_v3_runtime_adapter(adapter, kwargs)
        target_effectors, diagnostics = self._normalize_v3_runtime_adapter_result(result)
        self._record_v3_runtime_diagnostics(clip_index, diagnostics)

        if cfg.shadow_enabled:
            return legacy_buffer_effectors
        if cfg.override_enabled:
            if target_effectors is None:
                raise RuntimeError("v3 override_experimental adapter did not return target_effectors")
            return np.asarray(target_effectors, dtype=np.float32)
        return legacy_buffer_effectors

    def _call_v3_runtime_adapter(self, adapter, kwargs):
        if hasattr(adapter, "compute_targets"):
            return adapter.compute_targets(**kwargs)
        if callable(adapter):
            return adapter(**kwargs)
        raise TypeError("v3 runtime target adapter must be callable or expose compute_targets(**kwargs)")

    def _normalize_v3_runtime_adapter_result(self, result):
        if result is None:
            return None, {}
        if isinstance(result, dict):
            return result.get("target_effectors"), dict(result.get("diagnostics", {}))
        target_effectors = getattr(result, "target_effectors", None)
        diagnostics = getattr(result, "diagnostics", {})
        if hasattr(diagnostics, "to_json"):
            diagnostics = diagnostics.to_json()
        return target_effectors, dict(diagnostics or {})

    def _record_v3_runtime_diagnostics(self, clip_index, diagnostics):
        cfg = getattr(self, "v3_runtime_config", None)
        if cfg is None or not cfg.should_collect_diagnostics:
            return
        self.v3_runtime_diagnostics.append(
            {
                "mode": cfg.mode,
                "clip_index": int(clip_index),
                "semantic_tasks": list(cfg.semantic_tasks),
                "diagnostics": self._json_safe(diagnostics),
            }
        )

    def _v3_profile_resolution_json(self):
        resolution = getattr(self, "v3_runtime_profile_resolution", None)
        if resolution is None:
            return None
        if hasattr(resolution, "to_json"):
            return resolution.to_json()
        if isinstance(resolution, dict):
            return dict(resolution)
        return {"value": str(resolution)}

    def _write_v3_runtime_diagnostics_summary(self, output_buffers=None):
        cfg = getattr(self, "v3_runtime_config", None)
        if cfg is None or not cfg.should_collect_diagnostics:
            return
        diagnostics = getattr(self, "v3_runtime_diagnostics", [])
        if not diagnostics:
            return

        output_dir = Path(cfg.diagnostics_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.shadow_enabled:
            filename = "shadow_summary.json"
        elif cfg.override_enabled:
            filename = "override_smoke_summary.json"
        else:
            filename = "runtime_summary.json"
        payload = {
            "schema_version": 1,
            "mode": cfg.mode,
            "clip_count": len(diagnostics),
            "output_buffer_count": len(output_buffers or []),
            "profile_resolution": self._v3_profile_resolution_json(),
            "diagnostics": self._json_safe(diagnostics),
        }
        (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _json_safe(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value

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
            buffer = buffers[i]
            if self.initialization_pose and self.num_initialization_frames > 0:
                buffer = newton_utils.create_buffer_with_initialization_frames(
                    self.initialization_pose, buffers[i], self.num_initialization_frames, self.num_stabilization_frames)

            self.max_frames = max(self.max_frames, buffer.num_frames)
            legacy_buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(buffer, scale_animation, offsets[i])
            buffer_effectors = self._apply_v3_runtime_targets(
                buffer=buffer,
                offset=offsets[i],
                scale_animation=scale_animation,
                legacy_buffer_effectors=legacy_buffer_effectors,
                clip_index=len(self.input_targets),
            )
            if getattr(self, "stance_width_diagnostics_enabled", False):
                try:
                    self.stance_width_report_inputs.append({
                        "source_soma_transforms": self._compute_source_global_transforms_for_diagnostics(
                            buffer,
                            offsets[i],
                        ),
                        "source_soma_names": list(buffer.skeleton.joint_names),
                        "scaled_target_transforms": np.asarray(legacy_buffer_effectors, dtype=np.float32),
                        "scaled_target_names": list(self.human_robot_scaler.effector_names()),
                    })
                except Exception as exc:
                    self.stance_width_report_inputs.append({"error": str(exc)})

            self.input_targets.append(buffer_effectors[:, self.target_effector_indices, :])
            self.input_sample_rates.append(buffers[i].sample_rate)
            if self.contact_aware_foot_ik_enabled:
                try:
                    window = int(self.contact_aware_foot_ik.get("contact_score_smoothing_window", 5))
                    scores = None
                    if self.contact_source in ("auto", "npz_foot_contacts"):
                        foot_contacts = getattr(buffer, "foot_contacts", None)
                        if foot_contacts is None:
                            foot_contacts = getattr(buffer, "contacts", None)
                        if foot_contacts is not None:
                            scores = contacts_from_npz_foot_contacts(
                                np.asarray(foot_contacts),
                                window,
                                contact_order=self.contact_aware_foot_ik.get("contact_order"),
                            )
                    if scores is None and self.contact_source in ("auto", "soma_heuristic"):
                        scores = infer_contacts_from_animation_buffer(
                            buffer,
                            offsets[i],
                            window,
                            source_foot_joint_aliases=self.contact_aware_foot_ik.get("source_foot_joint_aliases"),
                            contact_height_scale=self.contact_aware_foot_ik.get("contact_height_scale"),
                            contact_velocity_scale=self.contact_aware_foot_ik.get("contact_velocity_scale"),
                            ground_height_m=self.contact_aware_foot_ik.get("ground_height_m"),
                        )
                    self.input_contact_scores.append(scores)
                except Exception as exc:
                    print(f"[WARN] Contact inference failed, disabling lock for this clip: {exc}")
                    self.input_contact_scores.append(None)
            else:
                self.input_contact_scores.append(None)

    def _compute_source_global_transforms_for_diagnostics(self, buffer, offset):
        transforms = np.zeros((buffer.num_frames, buffer.skeleton.num_joints, 7), dtype=np.float32)
        for frame in range(buffer.num_frames):
            transforms[frame] = np.asarray(buffer.compute_global_transforms(frame, offset), dtype=np.float32)
        return transforms

    def execute(self):
        """
        Run the retargeting pipeline on all added input motions.

        This method builds a multi-environment Newton model, sets up IK
        objectives, and performs frame-by-frame IK solving.

        Returns:
            list[CSVAnimationBuffer]: A list of retargeted robot motions, one per input motion.
        """
        num_envs = len(self.input_targets)
        if num_envs == 0:
            self.retargeted_motions = []
            return
        self.stance_width_reports = []

        # Clamp objective weights to valid values
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
        if self.contact_aware_foot_ik_enabled and num_envs > 1:
            raise ValueError(
                "contact_aware_foot_ik currently requires batch_size=1 because IK "
                "objective weights are global, not per-env. Run with batch_size=1 "
                "or disable contact_aware_foot_ik."
            )

        model = self._build_model(num_envs)
        state = model.state()

        if self.post_processing_enabled:
            self.feet_stabilizer.setup_num_envs(num_envs)
            env_feet_tx = np.empty((num_envs, len(self.feet_effector_indices), 7), dtype=np.float32)

        (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_filter_objective,
            contact_objectives
        ) = self._create_ik_objectives(num_envs, model, state)

        # Add optional objectives
        ik_solver_active_objectives = [*position_objectives, *rotation_objectives, *contact_objectives]
        if self.joint_limit_weight > 0.0:
            ik_solver_active_objectives.append(joint_limit_objective)
        if self.smooth_joint_filter_weight > 0.0:
            ik_solver_active_objectives.append(smooth_joint_filter_objective)

        ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=num_envs,
            objectives=ik_solver_active_objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC)

        joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
        wp.copy(joint_q, model.joint_q)

        # Solver initialization
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

        #import time
        num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames
        joint_q_data = [
            np.zeros((len(self.input_targets[i]), self.ik_model.joint_coord_count), dtype=np.float32)
            for i in range(num_envs)
        ]
        for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions"):
            if num_frames_to_remove > 0 and frame <= num_frames_to_remove:
                smooth_joint_filter_objective.set_weight(self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove)))

            #start_time = time.time()
            for env in range(num_envs):
                if frame > (len(self.input_targets[env])-1):
                    continue
                frame_targets = self.input_targets[env][frame]
                for i, target in enumerate(frame_targets):
                    position_objectives[i].set_target_position(env, wp.vec3(*target[0:3]))
                    rotation_objectives[i].set_target_rotation(env, wp.quat(*target[3:7]))

                if self.contact_aware_foot_ik_enabled and env < len(self.input_contact_scores):
                    self._update_contact_objectives_for_frame(env, frame, frame_targets, contact_objectives)

            if graph_capture is not None:
                wp.capture_launch(graph_capture)
            else:
                single_step()

            data = None
            if self.post_processing_enabled:
                self.feet_stabilizer.reset_state(joint_q)

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env])-1):
                        env_feet_tx[env] = np.asarray(self.input_targets[env][-1][self.feet_effector_indices])
                    else:
                        env_feet_tx[env] = np.asarray(self.input_targets[env][frame][self.feet_effector_indices])

                self.feet_stabilizer.solve(env_feet_tx)
                data = self.joint_limit_clamper.apply(self.feet_stabilizer.current_state()).numpy()
            else:
                data = self.joint_limit_clamper.apply(joint_q).numpy()

            for env in range(num_envs):
                if frame > (len(self.input_targets[env])-1):
                    continue

                joint_q_data[env][frame] = np.array(data[env], copy=True)

            #end_time = time.time()
            #print(f"Time taken for frame {frame}: {end_time - start_time} seconds")

        output_buffers = []
        for i in range(num_envs):
            raw_ik_data = joint_q_data[i][num_frames_to_remove:].astype(np.float32)
            if getattr(self, "stance_width_diagnostics_enabled", False):
                self.stance_width_reports.append(
                    self._build_stance_width_report_for_output(i, raw_ik_data, num_frames_to_remove)
                )
            raw_data = np.array(raw_ik_data, copy=True)
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
        self._write_v3_runtime_diagnostics_summary(output_buffers)
        return output_buffers

    def _build_stance_width_report_for_output(self, env, raw_ik_data, num_frames_to_remove):
        record = (
            self.stance_width_report_inputs[env]
            if env < len(getattr(self, "stance_width_report_inputs", []))
            else None
        )
        if not isinstance(record, dict):
            return {"diagnostic": "stance_width", "error": "source/scaled diagnostic input was not recorded"}
        if "error" in record:
            return {"diagnostic": "stance_width", "error": record["error"]}

        source_transforms = np.asarray(record["source_soma_transforms"], dtype=np.float32)[num_frames_to_remove:]
        scaled_transforms = np.asarray(record["scaled_target_transforms"], dtype=np.float32)[num_frames_to_remove:]
        n = min(len(source_transforms), len(scaled_transforms), len(raw_ik_data))
        source_transforms = source_transforms[:n]
        scaled_transforms = scaled_transforms[:n]
        raw_ik_data = raw_ik_data[:n]

        robot_fk_names = self._stance_width_robot_semantic_names()
        robot_fk_transforms = self._compute_robot_semantic_fk_transforms(raw_ik_data, robot_fk_names)
        return build_stance_width_report(
            source_soma_transforms=source_transforms,
            source_soma_names=record["source_soma_names"],
            scaled_target_transforms=scaled_transforms,
            scaled_target_names=record["scaled_target_names"],
            robot_fk_transforms=robot_fk_transforms,
            robot_fk_names=robot_fk_names,
            ik_map=self.retargeter_config.get("ik_map", {}),
            debug_summary=getattr(self, "stance_width_debug_summary", {"enabled": False}),
            hip_thigh_joint_stats=joint_stats_from_raw_data(
                raw_ik_data,
                self._scalar_joint_q_index_map(),
            ),
        )

    def _stance_width_robot_semantic_names(self):
        ik_map = self.retargeter_config.get("ik_map", {})
        body_lookup = set(getattr(self, "body_names", []))
        semantic_names = []
        for joint_name in ("Hips", "Pelvis", "LeftFoot", "RightFoot"):
            entry = ik_map.get(joint_name)
            if not isinstance(entry, dict):
                continue
            t_body = entry.get("t_body")
            r_body = entry.get("r_body") or t_body
            if t_body in body_lookup and r_body in body_lookup:
                semantic_names.append(joint_name)
        return semantic_names

    def _compute_robot_semantic_fk_transforms(self, raw_ik_data, semantic_names):
        raw_ik_data = np.asarray(raw_ik_data, dtype=np.float32)
        out = np.zeros((len(raw_ik_data), len(semantic_names), 7), dtype=np.float32)
        if len(raw_ik_data) == 0 or not semantic_names:
            return out

        body_lookup = {name: idx for idx, name in enumerate(getattr(self, "body_names", []))}
        ik_map = self.retargeter_config.get("ik_map", {})
        semantic_body_indices = []
        for joint_name in semantic_names:
            entry = ik_map.get(joint_name, {})
            t_body = entry.get("t_body") if isinstance(entry, dict) else None
            r_body = entry.get("r_body") if isinstance(entry, dict) else None
            r_body = r_body or t_body
            semantic_body_indices.append((body_lookup[t_body], body_lookup[r_body]))

        model = self.ik_model
        state = model.state()
        original_q = np.array(model.joint_q.numpy(), copy=True)
        try:
            for frame, row in enumerate(raw_ik_data):
                wp.copy(model.joint_q, wp.array(row, dtype=wp.float32))
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                body_q = state.body_q.numpy()
                for semantic_idx, (t_body_idx, r_body_idx) in enumerate(semantic_body_indices):
                    out[frame, semantic_idx, :3] = body_q[t_body_idx, :3]
                    out[frame, semantic_idx, 3:7] = body_q[r_body_idx, 3:7]
        finally:
            wp.copy(model.joint_q, wp.array(original_q, dtype=wp.float32))
        return out

    def _scalar_joint_q_index_map(self):
        starts = self.ik_model.joint_q_start.numpy()
        out = {}
        for joint_index, label in enumerate(self.robot_builder.joint_label):
            start = int(starts[joint_index])
            end = (
                int(starts[joint_index + 1])
                if joint_index + 1 < len(starts)
                else int(self.ik_model.joint_coord_count)
            )
            if end - start == 1:
                out[newton_utils.get_name_from_label(label)] = start
        return out

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
        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        self.body_names = body_names
        for joint, mapping_data in retargeter_config["ik_map"].items():
            mapped_joints.append(joint)
            mapped_joint_indices.append(skeleton.joint_index(joint))
            mapped_body_link_pos_data.append((body_names.index(mapping_data['t_body']), mapping_data['t_weight']))
            mapped_body_link_rot_data.append((body_names.index(mapping_data['r_body']), mapping_data['r_weight']))

        return (
            mapped_joints,
            mapped_joint_indices,
            mapped_body_link_pos_data,
            mapped_body_link_rot_data)

    def _create_ik_objectives(self, num_envs, model, state):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # Gather default body position and rotation based on model state to initialize
        # position and rotation objectives
        num_body_link_pos = len(self.mapped_body_link_pos_data)
        num_body_link_rot = len(self.mapped_body_link_rot_data)
        pos_targets = np.zeros((num_envs, num_body_link_pos), dtype=wp.vec3)
        rot_targets = np.zeros((num_envs, num_body_link_rot), dtype=wp.quat)

        body_q = state.body_q.numpy()
        for env in range(num_envs):
            base = env * self.num_body_count
            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_pos_data):
                pos_targets[env, ee_idx] = body_q[base + link_idx][0:3]

            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_rot_data):
                rot_wp = wp.quat(body_q[base + link_idx][3:7])
                rot_targets[env, ee_idx] = wp.normalize(rot_wp)

        pos_num_ees = len(self.mapped_body_link_pos_data)
        rot_num_ees = len(self.mapped_body_link_rot_data)
        pos_target_arrays, rot_target_arrays = [], []
        for ee_idx in range(pos_num_ees):
            pos_wp = wp.array(pos_targets[:, ee_idx], dtype=wp.vec3)
            pos_target_arrays.append(pos_wp)

        for ee_idx in range(rot_num_ees):
            rot_wp = wp.array(rot_targets[:, ee_idx], dtype=wp.vec4)
            rot_target_arrays.append(rot_wp)

        position_objectives = []
        for i, (link_idx, w) in enumerate(self.mapped_body_link_pos_data):
            objective = ik.IKObjectivePosition(
                link_index=link_idx,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=pos_target_arrays[i],
                weight=w)
            position_objectives.append(objective)

        rotation_objectives = []
        for i, (link_idx, w) in enumerate(self.mapped_body_link_rot_data):
            objective = ik.IKObjectiveRotation(
                link_index=link_idx,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rot_target_arrays[i],
                weight=w)
            rotation_objectives.append(objective)

        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        # Weight is set to desired value once initialization frames have been processed
        smooth_joint_limiter_objective = IKSmoothJointFilter(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=0.0,
            coord_masks=self.smooth_joint_filter_coord_masks)

        contact_objectives = self._create_contact_aware_objectives(num_envs, pos_target_arrays)
        return position_objectives, rotation_objectives, joint_limit_objective, smooth_joint_limiter_objective, contact_objectives


    def _create_contact_aware_objectives(self, num_envs, pos_target_arrays):
        if not self.contact_aware_foot_ik_enabled or getattr(self, "contact_source", "auto") in {"none", "disabled", "false", "null"}:
            self.contact_objective_map = {}
            return []
        anchors = self.contact_aware_foot_ik.get("anchor_offsets", {})
        left = anchors.get("left", {})
        right = anchors.get("right", {})
        if not left or not right:
            print("[WARN] contact_aware_foot_ik enabled but anchor_offsets not configured; skipping.")
            self.contact_objective_map = {}
            return []
        link_lookup = {name: link for name, (link, _) in zip(self.mapped_joints, self.mapped_body_link_pos_data)}
        if "LeftFoot" not in link_lookup or "RightFoot" not in link_lookup:
            print("[WARN] contact_aware_foot_ik enabled but LeftFoot/RightFoot are missing in ik_map; skipping.")
            self.contact_objective_map = {}
            return []
        edge_stance = self.contact_aware_foot_ik.get("edge_weight_stance", 0.5)
        edge_swing = self.contact_aware_foot_ik.get("edge_weight_swing", 0.05)
        mapping = {
            "left_toe": (
                "LeftFoot",
                left.get("toe"),
                ("left_toe_contact_score",),
                "single",
                self.contact_aware_foot_ik.get("toe_weight_stance", 0.8),
                self.contact_aware_foot_ik.get("toe_weight_swing", 0.1),
            ),
            "left_heel": (
                "LeftFoot",
                left.get("heel"),
                ("left_heel_contact_score",),
                "single",
                self.contact_aware_foot_ik.get("heel_weight_stance", 0.8),
                self.contact_aware_foot_ik.get("heel_weight_swing", 0.1),
            ),
            "left_inner_edge": (
                "LeftFoot",
                left.get("inner_edge"),
                ("left_toe_contact_score", "left_heel_contact_score"),
                "flat_foot_min",
                edge_stance,
                edge_swing,
            ),
            "left_outer_edge": (
                "LeftFoot",
                left.get("outer_edge"),
                ("left_toe_contact_score", "left_heel_contact_score"),
                "flat_foot_min",
                edge_stance,
                edge_swing,
            ),
            "right_toe": (
                "RightFoot",
                right.get("toe"),
                ("right_toe_contact_score",),
                "single",
                self.contact_aware_foot_ik.get("toe_weight_stance", 0.8),
                self.contact_aware_foot_ik.get("toe_weight_swing", 0.1),
            ),
            "right_heel": (
                "RightFoot",
                right.get("heel"),
                ("right_heel_contact_score",),
                "single",
                self.contact_aware_foot_ik.get("heel_weight_stance", 0.8),
                self.contact_aware_foot_ik.get("heel_weight_swing", 0.1),
            ),
            "right_inner_edge": (
                "RightFoot",
                right.get("inner_edge"),
                ("right_toe_contact_score", "right_heel_contact_score"),
                "flat_foot_min",
                edge_stance,
                edge_swing,
            ),
            "right_outer_edge": (
                "RightFoot",
                right.get("outer_edge"),
                ("right_toe_contact_score", "right_heel_contact_score"),
                "flat_foot_min",
                edge_stance,
                edge_swing,
            ),
        }
        out=[]; self.contact_objective_map={}
        for key, (joint, offset, score_keys, score_mode, w_stance, w_swing) in mapping.items():
            if offset is None or joint not in link_lookup:
                continue
            targets = wp.array(np.zeros((num_envs,3), dtype=np.float32), dtype=wp.vec3)
            obj = ik.IKObjectivePosition(link_index=link_lookup[joint], link_offset=wp.vec3(*offset), target_positions=targets, weight=w_swing)
            out.append(obj)
            self.contact_objective_map[key] = {
                "objective": obj,
                "score_key": score_keys[0],
                "score_keys": score_keys,
                "score_mode": score_mode,
                "stance": float(w_stance),
                "swing": float(w_swing),
                "active": [False]*num_envs,
                "locked": [None]*num_envs,
            }
        return out

    def _contact_objective_score(self, cfg, scores, frame):
        if scores is None:
            return 0.0

        score_keys = tuple(cfg.get("score_keys", (cfg["score_key"],)))
        values = []
        for score_key in score_keys:
            if score_key not in scores or frame >= len(scores[score_key]):
                if cfg.get("score_mode") == "flat_foot_min":
                    return 0.0
                continue
            values.append(float(scores[score_key][frame]))

        if not values:
            return 0.0
        if cfg.get("score_mode") == "flat_foot_min":
            return min(values) if len(values) == len(score_keys) else 0.0
        return values[0]

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
            score = self._contact_objective_score(cfg, scores, frame)
            active = cfg["active"][env]
            if (not active) and score >= on_t:
                cfg["active"][env]=True; cfg["locked"][env]=world.copy()
            elif active and score <= off_t:
                cfg["active"][env]=False; cfg["locked"][env]=None
            target = cfg["locked"][env] if cfg["active"][env] and cfg["locked"][env] is not None else world
            cfg["objective"].set_target_position(env, wp.vec3(*target))
            weight = cfg["stance"] if cfg["active"][env] else cfg["swing"]
            if hasattr(cfg["objective"], "set_weight"):
                cfg["objective"].set_weight(weight)
