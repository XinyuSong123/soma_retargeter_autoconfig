import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import warp as wp

import newton
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.robot_registry_parser as robot_registry_parser
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.tools.pose_io import (
    build_target_effectors_from_robot_pose,
    create_reference_soma_skeleton_instance,
    create_skeleton_instance_from_human_pose_record,
    load_human_pose_json,
    load_robot_pose_json,
    normalize_quaternion_xyzw,
)
from soma_retargeter.tools.pose_optimizer_core import (
    DEFAULT_SOURCE_REFERENCE_BVH,
    resolve_default_output_file,
    resolve_default_report_file,
)

wp.init()


def _get_model_height(config: dict) -> float:
    if "model_height" in config:
        return float(config["model_height"])
    return robot_registry_parser.get_robot_model_height(config.get("robot_type"))


@wp.kernel
def compute_bone_length_loss(
    predicted_effectors: wp.array(dtype=wp.transform),
    target_effectors: wp.array(dtype=wp.transform),
    mapped_ancestors: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    loss_weight: float,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return

    parent_id = mapped_ancestors[tid]
    if parent_id < 0:
        return

    pred_t = wp.transform_get_translation(predicted_effectors[tid])
    pred_parent_t = wp.transform_get_translation(predicted_effectors[parent_id])
    pred_dist = wp.length(pred_t - pred_parent_t)

    targ_t = wp.transform_get_translation(target_effectors[tid])
    targ_parent_t = wp.transform_get_translation(target_effectors[parent_id])
    targ_dist = wp.length(targ_t - targ_parent_t)

    diff = pred_dist - targ_dist
    wp.atomic_add(loss, 0, loss_weight * diff * diff)


@wp.kernel
def compute_position_loss(
    predicted_effectors: wp.array(dtype=wp.transform),
    target_effectors: wp.array(dtype=wp.transform),
    pred_root: wp.array(dtype=wp.vec3),
    targ_root: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.float32),
    loss_weight: float,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return
    pred_t = wp.transform_get_translation(predicted_effectors[tid]) - pred_root[0]
    targ_t = wp.transform_get_translation(target_effectors[tid]) - targ_root[0]
    diff = pred_t - targ_t
    wp.atomic_add(loss, 0, loss_weight * wp.dot(diff, diff))


@wp.kernel
def compute_pose_loss(
    predicted_effectors: wp.array(dtype=wp.transform),
    target_effectors: wp.array(dtype=wp.transform),
    pred_root: wp.array(dtype=wp.vec3),
    targ_root: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.float32),
    position_weight: float,
    rotation_weight: float,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return

    if position_weight > 0.0:
        pred_t = wp.transform_get_translation(predicted_effectors[tid]) - pred_root[0]
        targ_t = wp.transform_get_translation(target_effectors[tid]) - targ_root[0]
        pos_diff = pred_t - targ_t
        wp.atomic_add(loss, 0, position_weight * wp.dot(pos_diff, pos_diff))

    if rotation_weight > 0.0:
        pred_q = wp.transform_get_rotation(predicted_effectors[tid])
        targ_q = wp.transform_get_rotation(target_effectors[tid])
        quat_dot = wp.dot(pred_q, targ_q)
        rot_err = 1.0 - quat_dot * quat_dot
        wp.atomic_add(loss, 0, rotation_weight * rot_err)


@wp.kernel
def kernel_compute_scaled_effectors(
    in_num_mapped_joints: wp.int32,
    in_global_pose: wp.array(dtype=wp.transform),
    in_mapped_joint_indices: wp.array(dtype=wp.int32),
    in_mapped_joint_scales: wp.array(dtype=wp.float32),
    in_mapped_joint_offsets: wp.array(dtype=wp.transform),
    in_scale_animation: wp.bool,
    out_result: wp.array(dtype=wp.transform),
):
    HumanToRobotScaler.wp_compute_scaled_effectors(
        in_num_mapped_joints,
        in_global_pose,
        in_mapped_joint_indices,
        in_mapped_joint_scales,
        in_mapped_joint_offsets,
        in_scale_animation,
        out_result,
    )


@wp.kernel
def kernel_compute_global_pose(
    in_num_joints: wp.int32,
    in_root_tx: wp.transform,
    in_parent_indices: wp.array(dtype=wp.int32),
    in_local_pose: wp.array(dtype=wp.transform),
    out_result: wp.array(dtype=wp.transform),
):
    pose_utils.wp_compute_global_pose(
        in_num_joints,
        in_root_tx,
        in_parent_indices,
        in_local_pose,
        out_result,
    )


@wp.kernel
def step_kernel_float(
    param: wp.array(dtype=wp.float32),
    grad: wp.array(dtype=wp.float32),
    mask: wp.array(dtype=wp.float32),
    lr: float,
):
    tid = wp.tid()
    param[tid] = param[tid] - lr * grad[tid] * mask[tid]


@wp.kernel
def step_kernel_vec3_offsets(
    offsets: wp.array(dtype=wp.transform),
    grad_offsets: wp.array(dtype=wp.transform),
    mask: wp.array(dtype=wp.float32),
    lr: float,
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return
    grad_t = wp.transform_get_translation(grad_offsets[tid])
    cur_t = wp.transform_get_translation(offsets[tid])
    cur_q = wp.transform_get_rotation(offsets[tid])
    offsets[tid] = wp.transform(cur_t - lr * grad_t, cur_q)


@wp.kernel
def step_kernel_quat_offsets(
    offsets: wp.array(dtype=wp.transform),
    grad_offsets: wp.array(dtype=wp.transform),
    mask: wp.array(dtype=wp.float32),
    lr: float,
):
    tid = wp.tid()
    if mask[tid] < 0.5:
        return
    cur_t = wp.transform_get_translation(offsets[tid])
    cur_q = wp.transform_get_rotation(offsets[tid])
    grad_q = wp.transform_get_rotation(grad_offsets[tid])
    new_q = wp.normalize(cur_q - lr * grad_q)
    offsets[tid] = wp.transform(cur_t, new_q)


@wp.kernel
def kernel_get_root(eff: wp.array(dtype=wp.transform), out: wp.array(dtype=wp.vec3)):
    out[0] = wp.transform_get_translation(eff[0])


def build_wp_global_pose_from_skeleton_instance(skeleton_instance: SkeletonInstance):
    num_joints = skeleton_instance.num_joints
    wp_global_pose = wp.array(
        [wp.transform_identity()] * num_joints,
        dtype=wp.transform,
        requires_grad=True,
    )
    wp.launch(
        kernel_compute_global_pose,
        dim=1,
        inputs=[
            num_joints,
            skeleton_instance.xform,
            wp.array(skeleton_instance.skeleton.parent_indices, dtype=wp.int32),
            wp.array(skeleton_instance.local_transforms, dtype=wp.transform),
        ],
        outputs=[wp_global_pose],
    )
    return wp_global_pose


def get_mapped_parent(idx, parents, mask_arr):
    parent_idx = parents[idx]
    while parent_idx >= 0 and mask_arr[parent_idx] < 0.5:
        parent_idx = parents[parent_idx]
    return parent_idx


def build_mapped_ancestors(scaler, mask_np):
    mapped_ancestors_np = np.zeros(len(scaler.mapped_joints), dtype=np.int32)
    for idx in range(len(scaler.mapped_joints)):
        mapped_ancestors_np[idx] = get_mapped_parent(
            idx,
            scaler.mapped_joint_parents,
            mask_np,
        )
    return mapped_ancestors_np


def prepare_optimizer_masks(
    scaler,
    retargeter_config_file=None,
    optimize_offset_rotation=True,
):
    mask_np = np.ones(len(scaler.mapped_joints), dtype=np.float32)
    retargeter_config = None
    ik_map_joints = set()

    if retargeter_config_file:
        print(f"[INFO] Using retargeter config to mask non-mapped joints: {retargeter_config_file}")
        retargeter_config = robot_registry_parser.load_retargeter_config(
            retargeter_config_file,
            robot_name=getattr(scaler, "robot_type", None),
        )
        ik_map_joints = set(retargeter_config.get("ik_map", {}).keys())
        for idx, joint_name in enumerate(scaler.mapped_joints):
            if joint_name not in ik_map_joints:
                mask_np[idx] = 0.0
                print(f"[INFO] Ignored optimization for unmapped joint: {joint_name}")

    update_mask = wp.array(mask_np, dtype=wp.float32)
    rotation_mask_np = mask_np.copy() if optimize_offset_rotation else np.zeros_like(mask_np)
    rotation_update_mask = wp.array(rotation_mask_np, dtype=wp.float32)
    return (
        mask_np,
        update_mask,
        rotation_mask_np,
        rotation_update_mask,
        retargeter_config,
        ik_map_joints,
    )


def optimize_scaler(
    config_file,
    bvh_file,
    retargeter_config_file=None,
    iterations=100,
    learning_rate=0.01,
    output_file=None,
    optimize_offset_rotation=True,
    rotation_iterations=None,
    rotation_learning_rate=None,
    rotation_position_weight=1.0,
    rotation_loss_weight=1.0,
):
    print(f"[INFO] Loading scaler config: {config_file}")
    config = io_utils.load_json(config_file)

    print(f"[INFO] Loading target BVH: {bvh_file}")
    skeleton, anim = bvh_utils.load_bvh(bvh_file)

    import math

    rx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
    rz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
    root_xform = wp.transform([0.0, 0.0, 0.0], wp.mul(rz, rx))
    skeleton_instance = SkeletonInstance(skeleton, [0, 0, 0], root_xform)
    skeleton_instance.set_local_transforms(anim.get_local_transforms(0))

    scaler = HumanToRobotScaler(
        skeleton,
        _get_model_height(config),
        config_file,
    )

    (
        mask_np,
        update_mask,
        rotation_mask_np,
        rotation_update_mask,
        retargeter_config,
        ik_map_joints,
    ) = prepare_optimizer_masks(
        scaler,
        retargeter_config_file,
        optimize_offset_rotation,
    )
    optimized_rotation_joint_names = [
        joint_name
        for idx, joint_name in enumerate(scaler.mapped_joints)
        if rotation_mask_np[idx] > 0.5
    ]

    mapped_joint_scales = wp.array(
        scaler.mapped_joint_scales.numpy(),
        dtype=wp.float32,
        requires_grad=True,
    )
    mapped_joint_offsets = wp.array(
        scaler.mapped_joint_offsets.numpy(),
        dtype=wp.transform,
        requires_grad=True,
    )

    wp_global_pose = build_wp_global_pose_from_skeleton_instance(skeleton_instance)

    target_effectors_np = scaler.compute_effectors_from_skeleton(
        skeleton_instance,
        scale_animation=True,
    )

    if retargeter_config_file:
        print("[INFO] Extracting actual robot rest pose to use as optimization targets...")
        pipeline = NewtonPipeline(
            skeleton,
            robot_type=config.get("robot_type", "unitree_g1"),
            retarget_config=retargeter_config,
        )
        model = pipeline.ik_model
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()

        body_names = [
            newton_utils.get_name_from_label(label)
            for label in pipeline.robot_builder.body_label
        ]

        for idx, joint_name in enumerate(scaler.mapped_joints):
            if joint_name not in ik_map_joints:
                continue

            t_body = retargeter_config["ik_map"][joint_name]["t_body"]
            r_body = retargeter_config["ik_map"][joint_name]["r_body"]
            if t_body not in body_names or r_body not in body_names:
                print(
                    f"[WARNING] Could not find body link '{t_body}' or '{r_body}' "
                    f"in robot model for {joint_name}."
                )
                continue

            t_idx = body_names.index(t_body)
            r_idx = body_names.index(r_body)
            target_effectors_np[idx][0:3] = body_q[t_idx][0:3]
            target_effectors_np[idx][3:7] = body_q[r_idx][3:7]
    else:
        print("[WARNING] No retargeter config provided. Using dummy 5cm perturbation as target.")
        for idx in range(len(target_effectors_np)):
            target_effectors_np[idx][1] += 0.05

    target_effectors = wp.array(target_effectors_np, dtype=wp.transform)

    mapped_ancestors_np = build_mapped_ancestors(scaler, mask_np)
    wp_mapped_ancestors = wp.array(mapped_ancestors_np, dtype=wp.int32)
    wp_targ_root = wp.array([target_effectors_np[0][0:3]], dtype=wp.vec3)

    print(f"[INFO] Phase 1 – Scale optimization for {iterations} iterations...")
    for i in range(iterations):
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        wp_effectors = wp.zeros(
            len(scaler.mapped_joint_indices),
            dtype=wp.transform,
            requires_grad=True,
        )
        with tape:
            wp.launch(
                kernel_compute_scaled_effectors,
                dim=1,
                inputs=[
                    len(scaler.mapped_joint_indices),
                    wp_global_pose,
                    scaler.mapped_joint_indices,
                    mapped_joint_scales,
                    mapped_joint_offsets,
                    True,
                ],
                outputs=[wp_effectors],
            )
            wp.launch(
                compute_bone_length_loss,
                dim=len(scaler.mapped_joint_indices),
                inputs=[
                    wp_effectors,
                    target_effectors,
                    wp_mapped_ancestors,
                    update_mask,
                    1.0,
                    loss,
                ],
            )
        tape.backward(loss)
        if i % 10 == 0 or i == iterations - 1:
            print(f"[Phase1] Iter {i:4d} | BoneLen Loss: {loss.numpy()[0]:10.6f}")
        wp.launch(
            step_kernel_float,
            dim=len(scaler.mapped_joint_indices),
            inputs=[
                mapped_joint_scales,
                tape.gradients[mapped_joint_scales],
                update_mask,
                learning_rate,
            ],
        )
        tape.zero()

    print(f"[INFO] Phase 2 – Offset translation optimization for {iterations} iterations...")
    lr_offset = learning_rate * 0.1
    for i in range(iterations):
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        wp_effectors = wp.zeros(
            len(scaler.mapped_joint_indices),
            dtype=wp.transform,
            requires_grad=True,
        )
        wp_pred_root = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
        with tape:
            wp.launch(
                kernel_compute_scaled_effectors,
                dim=1,
                inputs=[
                    len(scaler.mapped_joint_indices),
                    wp_global_pose,
                    scaler.mapped_joint_indices,
                    mapped_joint_scales,
                    mapped_joint_offsets,
                    True,
                ],
                outputs=[wp_effectors],
            )
            wp.launch(kernel_get_root, dim=1, inputs=[wp_effectors], outputs=[wp_pred_root])
            wp.launch(
                compute_position_loss,
                dim=len(scaler.mapped_joint_indices),
                inputs=[
                    wp_effectors,
                    target_effectors,
                    wp_pred_root,
                    wp_targ_root,
                    update_mask,
                    1.0,
                    loss,
                ],
            )
        tape.backward(loss)
        if i % 10 == 0 or i == iterations - 1:
            print(f"[Phase2] Iter {i:4d} | Position Loss: {loss.numpy()[0]:10.6f}")
        wp.launch(
            step_kernel_vec3_offsets,
            dim=len(scaler.mapped_joint_indices),
            inputs=[
                mapped_joint_offsets,
                tape.gradients[mapped_joint_offsets],
                update_mask,
                lr_offset,
            ],
        )
        tape.zero()

    if optimize_offset_rotation:
        if rotation_iterations is None:
            rotation_iterations = iterations
        if rotation_learning_rate is None:
            rotation_learning_rate = learning_rate * 0.1

        print(
            f"[INFO] Phase 3 – Offset rotation optimization for {rotation_iterations} iterations..."
        )
        print(
            "[INFO] Phase 3 joints: "
            + ", ".join(optimized_rotation_joint_names)
        )
        print(
            "[INFO] Phase 3 weights: "
            f"position={rotation_position_weight}, rotation={rotation_loss_weight}, "
            f"lr={rotation_learning_rate}"
        )

        for i in range(rotation_iterations):
            tape = wp.Tape()
            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            wp_effectors = wp.zeros(
                len(scaler.mapped_joint_indices),
                dtype=wp.transform,
                requires_grad=True,
            )
            wp_pred_root = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
            with tape:
                wp.launch(
                    kernel_compute_scaled_effectors,
                    dim=1,
                    inputs=[
                        len(scaler.mapped_joint_indices),
                        wp_global_pose,
                        scaler.mapped_joint_indices,
                        mapped_joint_scales,
                        mapped_joint_offsets,
                        True,
                    ],
                    outputs=[wp_effectors],
                )
                wp.launch(kernel_get_root, dim=1, inputs=[wp_effectors], outputs=[wp_pred_root])
                wp.launch(
                    compute_pose_loss,
                    dim=len(scaler.mapped_joint_indices),
                    inputs=[
                        wp_effectors,
                        target_effectors,
                        wp_pred_root,
                        wp_targ_root,
                        rotation_update_mask,
                        float(rotation_position_weight),
                        float(rotation_loss_weight),
                        loss,
                    ],
                )
            tape.backward(loss)
            if i % 10 == 0 or i == rotation_iterations - 1:
                print(f"[Phase3] Iter {i:4d} | Pose Loss: {loss.numpy()[0]:10.6f}")
            wp.launch(
                step_kernel_quat_offsets,
                dim=len(scaler.mapped_joint_indices),
                inputs=[
                    mapped_joint_offsets,
                    tape.gradients[mapped_joint_offsets],
                    rotation_update_mask,
                    float(rotation_learning_rate),
                ],
            )
            tape.zero()

    print("[INFO] Optimization complete.")

    model_height = _get_model_height(config)
    human_height_assump = config.get("human_height_assumption", 1.80)
    ratio = model_height / human_height_assump

    optimized_scales = mapped_joint_scales.numpy()
    for idx, joint_name in enumerate(scaler.mapped_joints):
        config["joint_scales"][joint_name] = float(optimized_scales[idx]) / ratio

    optimized_offsets = mapped_joint_offsets.numpy()
    for idx, joint_name in enumerate(scaler.mapped_joints):
        config_joint_name = joint_name
        if config_joint_name in ("LeftToeBase", "RightToeBase"):
            config_joint_name = config_joint_name.replace("ToeBase", "Toe")
        if config_joint_name not in config.get("joint_offsets", {}):
            continue
        translation = optimized_offsets[idx][0:3]
        if optimize_offset_rotation:
            rotation = normalize_quaternion_xyzw(optimized_offsets[idx][3:7])
            if rotation[3] < 0.0:
                rotation = -rotation
            rotation = [float(v) for v in rotation]
        else:
            rotation = list(config["joint_offsets"][config_joint_name][1])
        config["joint_offsets"][config_joint_name] = [
            [float(translation[0]), float(translation[1]), float(translation[2])],
            rotation,
        ]

    if output_file is None:
        config_path = pathlib.Path(config_file)
        output_file = str(config_path.with_name(f"{config_path.stem}_optimized.json"))
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
    print(f"[INFO] Saved optimized scaler config to: {output_file}")

    final_effectors = wp.zeros(len(scaler.mapped_joint_indices), dtype=wp.transform)
    wp.launch(
        kernel_compute_scaled_effectors,
        dim=1,
        inputs=[
            len(scaler.mapped_joint_indices),
            wp_global_pose,
            scaler.mapped_joint_indices,
            mapped_joint_scales,
            mapped_joint_offsets,
            True,
        ],
        outputs=[final_effectors],
    )
    final_effectors_np = final_effectors.numpy()

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        targ_root = target_effectors_np[0][0:3].copy()
        pred_root = final_effectors_np[0][0:3].copy()

        targ_x, targ_y, targ_z = [], [], []
        pred_x, pred_y, pred_z = [], [], []

        for idx, _ in enumerate(scaler.mapped_joints):
            if mask_np[idx] <= 0.5:
                continue
            targ_p = target_effectors_np[idx][0:3] - targ_root
            pred_p = final_effectors_np[idx][0:3] - pred_root
            targ_x.append(targ_p[0])
            targ_y.append(targ_p[1])
            targ_z.append(targ_p[2])
            pred_x.append(pred_p[0])
            pred_y.append(pred_p[1])
            pred_z.append(pred_p[2])

        for idx in range(len(scaler.mapped_joints)):
            if mask_np[idx] <= 0.5:
                continue
            parent_idx = mapped_ancestors_np[idx]
            if parent_idx < 0:
                continue

            targ_parent = target_effectors_np[parent_idx][0:3] - targ_root
            targ_child = target_effectors_np[idx][0:3] - targ_root
            ax.plot(
                [targ_parent[0], targ_child[0]],
                [targ_parent[1], targ_child[1]],
                [targ_parent[2], targ_child[2]],
                "g-",
                alpha=0.6,
            )

            pred_parent = final_effectors_np[parent_idx][0:3] - pred_root
            pred_child = final_effectors_np[idx][0:3] - pred_root
            ax.plot(
                [pred_parent[0], pred_child[0]],
                [pred_parent[1], pred_child[1]],
                [pred_parent[2], pred_child[2]],
                "r-",
                alpha=0.6,
            )

        ax.scatter(targ_x, targ_y, targ_z, c="green", marker="s", s=45, label="Robot Target (MJCF)")
        ax.scatter(pred_x, pred_y, pred_z, c="red", marker="o", s=45, label="Soma Scaled (Optimized T-Pose)")
        ax.set_title("Structural Alignment (Robot Rest Pose vs SOMA T-Pose)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        all_x = targ_x + pred_x
        all_y = targ_y + pred_y
        all_z = targ_z + pred_z
        if all_x and all_y and all_z:
            mid_x = (max(all_x) + min(all_x)) / 2
            mid_y = (max(all_y) + min(all_y)) / 2
            mid_z = (max(all_z) + min(all_z)) / 2
            half_range = max(
                max(all_x) - min(all_x),
                max(all_y) - min(all_y),
                max(all_z) - min(all_z),
            ) / 2
            ax.set_xlim(mid_x - half_range, mid_x + half_range)
            ax.set_ylim(mid_y - half_range, mid_y + half_range)
            ax.set_zlim(mid_z - half_range, mid_z + half_range)
        ax.legend()

        plt.tight_layout()
        plot_path = "optimization_alignment.png"
        plt.savefig(plot_path)
        print(f"[INFO] Optimization alignment plot saved to: {plot_path}")
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping visualization.")


def optimize_scaler_from_pose_pairs(
    config_file,
    human_pose_files,
    robot_pose_json_files,
    retargeter_config_file,
    source_reference_bvh=DEFAULT_SOURCE_REFERENCE_BVH,
    iterations=100,
    learning_rate=0.01,
    output_file=None,
    optimize_offset_rotation=True,
    rotation_iterations=None,
    rotation_learning_rate=None,
    rotation_position_weight=1.0,
    rotation_loss_weight=1.0,
    report_file=None,
):
    if not human_pose_files:
        raise ValueError("paired_pose_optimization requires at least one --human-pose.")
    if len(human_pose_files) != len(robot_pose_json_files):
        raise ValueError(
            "paired_pose_optimization requires the same number of --human-pose and "
            "--robot-target-pose-json inputs."
        )
    if not retargeter_config_file:
        raise ValueError("paired_pose_optimization requires --retargeter_config.")

    print(f"[INFO] Loading scaler config: {config_file}")
    config = io_utils.load_json(config_file)
    print(f"[INFO] Loading retargeter config: {retargeter_config_file}")
    retargeter_config = robot_registry_parser.load_retargeter_config(
        retargeter_config_file,
        robot_name=config.get("robot_type"),
    )
    print(f"[INFO] Loading source reference skeleton: {source_reference_bvh}")
    skeleton, _ = create_reference_soma_skeleton_instance(source_reference_bvh)

    scaler = HumanToRobotScaler(
        skeleton,
        _get_model_height(config),
        config_file,
    )
    (
        mask_np,
        update_mask,
        rotation_mask_np,
        rotation_update_mask,
        _retargeter_config_unused,
        _ik_map_joints_unused,
    ) = prepare_optimizer_masks(
        scaler,
        retargeter_config_file,
        optimize_offset_rotation,
    )

    mapped_joint_scales = wp.array(
        scaler.mapped_joint_scales.numpy(),
        dtype=wp.float32,
        requires_grad=True,
    )
    mapped_joint_offsets = wp.array(
        scaler.mapped_joint_offsets.numpy(),
        dtype=wp.transform,
        requires_grad=True,
    )

    pipeline = NewtonPipeline(
        skeleton,
        robot_type=config.get("robot_type", "unitree_g1"),
        retarget_config=retargeter_config,
    )

    sample_global_poses = []
    sample_target_effectors_np = []
    sample_target_effectors = []
    sample_target_roots = []
    sample_metadata = []

    for human_pose_file, robot_pose_file in zip(human_pose_files, robot_pose_json_files):
        print(f"[INFO] Loading paired human pose: {human_pose_file}")
        human_pose_record = load_human_pose_json(human_pose_file, skeleton)
        human_skeleton_instance = create_skeleton_instance_from_human_pose_record(
            skeleton,
            human_pose_record,
        )
        sample_global_poses.append(build_wp_global_pose_from_skeleton_instance(human_skeleton_instance))

        print(f"[INFO] Loading paired robot pose: {robot_pose_file}")
        robot_pose_record = load_robot_pose_json(robot_pose_file)
        target_effectors_np, ignored_joints = build_target_effectors_from_robot_pose(
            scaler,
            retargeter_config,
            pipeline,
            robot_pose_record,
        )
        sample_target_effectors_np.append(target_effectors_np)
        sample_target_effectors.append(wp.array(target_effectors_np, dtype=wp.transform))
        sample_target_roots.append(wp.array([target_effectors_np[0][0:3]], dtype=wp.vec3))
        sample_metadata.append(
            {
                "human_pose_file": str(human_pose_file),
                "human_pose_name": human_pose_record.pose_name,
                "robot_pose_file": str(robot_pose_file),
                "robot_pose_name": robot_pose_record.pose_name,
                "ignored_robot_joints": ignored_joints,
            }
        )

    mapped_ancestors_np = build_mapped_ancestors(scaler, mask_np)
    wp_mapped_ancestors = wp.array(mapped_ancestors_np, dtype=wp.int32)
    sample_loss_weight = 1.0 / float(len(sample_global_poses))

    print(
        f"[INFO] Paired pose optimization with {len(sample_global_poses)} samples "
        f"for {iterations} iterations."
    )

    print(f"[INFO] Phase 1 – Scale optimization for {iterations} iterations...")
    for i in range(iterations):
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        with tape:
            for wp_global_pose, target_effectors in zip(sample_global_poses, sample_target_effectors):
                wp_effectors = wp.zeros(
                    len(scaler.mapped_joint_indices),
                    dtype=wp.transform,
                    requires_grad=True,
                )
                wp.launch(
                    kernel_compute_scaled_effectors,
                    dim=1,
                    inputs=[
                        len(scaler.mapped_joint_indices),
                        wp_global_pose,
                        scaler.mapped_joint_indices,
                        mapped_joint_scales,
                        mapped_joint_offsets,
                        True,
                    ],
                    outputs=[wp_effectors],
                )
                wp.launch(
                    compute_bone_length_loss,
                    dim=len(scaler.mapped_joint_indices),
                    inputs=[
                        wp_effectors,
                        target_effectors,
                        wp_mapped_ancestors,
                        update_mask,
                        sample_loss_weight,
                        loss,
                    ],
                )
        tape.backward(loss)
        if i % 10 == 0 or i == iterations - 1:
            print(f"[Paired-Phase1] Iter {i:4d} | BoneLen Loss: {loss.numpy()[0]:10.6f}")
        wp.launch(
            step_kernel_float,
            dim=len(scaler.mapped_joint_indices),
            inputs=[
                mapped_joint_scales,
                tape.gradients[mapped_joint_scales],
                update_mask,
                learning_rate,
            ],
        )
        tape.zero()

    print(f"[INFO] Phase 2 – Offset translation optimization for {iterations} iterations...")
    lr_offset = learning_rate * 0.1
    for i in range(iterations):
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        with tape:
            for wp_global_pose, target_effectors, target_root in zip(
                sample_global_poses,
                sample_target_effectors,
                sample_target_roots,
            ):
                wp_effectors = wp.zeros(
                    len(scaler.mapped_joint_indices),
                    dtype=wp.transform,
                    requires_grad=True,
                )
                wp_pred_root = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
                wp.launch(
                    kernel_compute_scaled_effectors,
                    dim=1,
                    inputs=[
                        len(scaler.mapped_joint_indices),
                        wp_global_pose,
                        scaler.mapped_joint_indices,
                        mapped_joint_scales,
                        mapped_joint_offsets,
                        True,
                    ],
                    outputs=[wp_effectors],
                )
                wp.launch(kernel_get_root, dim=1, inputs=[wp_effectors], outputs=[wp_pred_root])
                wp.launch(
                    compute_position_loss,
                    dim=len(scaler.mapped_joint_indices),
                    inputs=[
                        wp_effectors,
                        target_effectors,
                        wp_pred_root,
                        target_root,
                        update_mask,
                        sample_loss_weight,
                        loss,
                    ],
                )
        tape.backward(loss)
        if i % 10 == 0 or i == iterations - 1:
            print(f"[Paired-Phase2] Iter {i:4d} | Position Loss: {loss.numpy()[0]:10.6f}")
        wp.launch(
            step_kernel_vec3_offsets,
            dim=len(scaler.mapped_joint_indices),
            inputs=[
                mapped_joint_offsets,
                tape.gradients[mapped_joint_offsets],
                update_mask,
                lr_offset,
            ],
        )
        tape.zero()

    if optimize_offset_rotation:
        if rotation_iterations is None:
            rotation_iterations = iterations
        if rotation_learning_rate is None:
            rotation_learning_rate = learning_rate * 0.1

        optimized_rotation_joint_names = [
            joint_name
            for idx, joint_name in enumerate(scaler.mapped_joints)
            if rotation_mask_np[idx] > 0.5
        ]
        print(
            f"[INFO] Phase 3 – Offset rotation optimization for {rotation_iterations} iterations..."
        )
        print("[INFO] Phase 3 joints: " + ", ".join(optimized_rotation_joint_names))
        print(
            "[INFO] Phase 3 weights: "
            f"position={rotation_position_weight}, rotation={rotation_loss_weight}, "
            f"lr={rotation_learning_rate}"
        )

        for i in range(rotation_iterations):
            tape = wp.Tape()
            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            with tape:
                for wp_global_pose, target_effectors, target_root in zip(
                    sample_global_poses,
                    sample_target_effectors,
                    sample_target_roots,
                ):
                    wp_effectors = wp.zeros(
                        len(scaler.mapped_joint_indices),
                        dtype=wp.transform,
                        requires_grad=True,
                    )
                    wp_pred_root = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
                    wp.launch(
                        kernel_compute_scaled_effectors,
                        dim=1,
                        inputs=[
                            len(scaler.mapped_joint_indices),
                            wp_global_pose,
                            scaler.mapped_joint_indices,
                            mapped_joint_scales,
                            mapped_joint_offsets,
                            True,
                        ],
                        outputs=[wp_effectors],
                    )
                    wp.launch(kernel_get_root, dim=1, inputs=[wp_effectors], outputs=[wp_pred_root])
                    wp.launch(
                        compute_pose_loss,
                        dim=len(scaler.mapped_joint_indices),
                        inputs=[
                            wp_effectors,
                            target_effectors,
                            wp_pred_root,
                            target_root,
                            rotation_update_mask,
                            float(rotation_position_weight * sample_loss_weight),
                            float(rotation_loss_weight * sample_loss_weight),
                            loss,
                        ],
                    )
            tape.backward(loss)
            if i % 10 == 0 or i == rotation_iterations - 1:
                print(f"[Paired-Phase3] Iter {i:4d} | Pose Loss: {loss.numpy()[0]:10.6f}")
            wp.launch(
                step_kernel_quat_offsets,
                dim=len(scaler.mapped_joint_indices),
                inputs=[
                    mapped_joint_offsets,
                    tape.gradients[mapped_joint_offsets],
                    rotation_update_mask,
                    float(rotation_learning_rate),
                ],
            )
            tape.zero()

    print("[INFO] Paired pose optimization complete.")

    model_height = _get_model_height(config)
    human_height_assump = config.get("human_height_assumption", 1.80)
    ratio = model_height / human_height_assump

    optimized_scales = mapped_joint_scales.numpy()
    for idx, joint_name in enumerate(scaler.mapped_joints):
        config["joint_scales"][joint_name] = float(optimized_scales[idx]) / ratio

    optimized_offsets = mapped_joint_offsets.numpy()
    for idx, joint_name in enumerate(scaler.mapped_joints):
        config_joint_name = joint_name
        if config_joint_name in ("LeftToeBase", "RightToeBase"):
            config_joint_name = config_joint_name.replace("ToeBase", "Toe")
        if config_joint_name not in config.get("joint_offsets", {}):
            continue
        translation = optimized_offsets[idx][0:3]
        if optimize_offset_rotation:
            rotation = normalize_quaternion_xyzw(optimized_offsets[idx][3:7])
            if rotation[3] < 0.0:
                rotation = -rotation
            rotation = [float(v) for v in rotation]
        else:
            rotation = list(config["joint_offsets"][config_joint_name][1])
        config["joint_offsets"][config_joint_name] = [
            [float(translation[0]), float(translation[1]), float(translation[2])],
            rotation,
        ]

    if output_file is None:
        output_file = resolve_default_output_file(config_file)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
    print(f"[INFO] Saved optimized scaler config to: {output_file}")

    if report_file is None:
        report_file = resolve_default_report_file(output_file)
    report_payload = {
        "mode": "paired_pose_optimization",
        "config_file": str(config_file),
        "retargeter_config_file": str(retargeter_config_file),
        "source_reference_bvh": str(source_reference_bvh),
        "human_pose_files": [str(path) for path in human_pose_files],
        "robot_pose_files": [str(path) for path in robot_pose_json_files],
        "output_file": str(output_file),
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "optimize_offset_rotation": bool(optimize_offset_rotation),
        "rotation_iterations": None if rotation_iterations is None else int(rotation_iterations),
        "rotation_learning_rate": None
        if rotation_learning_rate is None
        else float(rotation_learning_rate),
        "rotation_position_weight": float(rotation_position_weight),
        "rotation_loss_weight": float(rotation_loss_weight),
        "rotation_joint_scope": "all_mapped_joints",
        "rotation_joints": optimized_rotation_joint_names if optimize_offset_rotation else [],
        "sample_count": len(sample_metadata),
        "samples": sample_metadata,
    }
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved paired pose run report to: {report_file}")

    final_effectors = wp.zeros(len(scaler.mapped_joint_indices), dtype=wp.transform)
    wp.launch(
        kernel_compute_scaled_effectors,
        dim=1,
        inputs=[
            len(scaler.mapped_joint_indices),
            sample_global_poses[0],
            scaler.mapped_joint_indices,
            mapped_joint_scales,
            mapped_joint_offsets,
            True,
        ],
        outputs=[final_effectors],
    )
    final_effectors_np = final_effectors.numpy()
    first_target_effectors_np = sample_target_effectors_np[0]

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        targ_root = first_target_effectors_np[0][0:3].copy()
        pred_root = final_effectors_np[0][0:3].copy()

        targ_x, targ_y, targ_z = [], [], []
        pred_x, pred_y, pred_z = [], [], []

        for idx, _ in enumerate(scaler.mapped_joints):
            if mask_np[idx] <= 0.5:
                continue
            targ_p = first_target_effectors_np[idx][0:3] - targ_root
            pred_p = final_effectors_np[idx][0:3] - pred_root
            targ_x.append(targ_p[0])
            targ_y.append(targ_p[1])
            targ_z.append(targ_p[2])
            pred_x.append(pred_p[0])
            pred_y.append(pred_p[1])
            pred_z.append(pred_p[2])

        for idx in range(len(scaler.mapped_joints)):
            if mask_np[idx] <= 0.5:
                continue
            parent_idx = mapped_ancestors_np[idx]
            if parent_idx < 0:
                continue

            targ_parent = first_target_effectors_np[parent_idx][0:3] - targ_root
            targ_child = first_target_effectors_np[idx][0:3] - targ_root
            ax.plot(
                [targ_parent[0], targ_child[0]],
                [targ_parent[1], targ_child[1]],
                [targ_parent[2], targ_child[2]],
                "g-",
                alpha=0.6,
            )

            pred_parent = final_effectors_np[parent_idx][0:3] - pred_root
            pred_child = final_effectors_np[idx][0:3] - pred_root
            ax.plot(
                [pred_parent[0], pred_child[0]],
                [pred_parent[1], pred_child[1]],
                [pred_parent[2], pred_child[2]],
                "r-",
                alpha=0.6,
            )

        ax.scatter(targ_x, targ_y, targ_z, c="green", marker="s", s=45, label="Robot Target")
        ax.scatter(pred_x, pred_y, pred_z, c="red", marker="o", s=45, label="Soma Scaled")
        ax.set_title("Paired Pose Alignment (First Sample)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plot_path = str(pathlib.Path(output_file).with_suffix(".paired_pose_alignment.png"))
        plt.savefig(plot_path)
        print(f"[INFO] Paired pose alignment plot saved to: {plot_path}")
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping visualization.")

    return {
        "output_file": str(output_file),
        "report_file": str(report_file),
        "sample_count": len(sample_metadata),
        "samples": sample_metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize scaler config parameters with learning-based scale, translation, and rotation updates."
    )
    parser.add_argument(
        "--mode",
        choices=["scale_translation", "paired_pose_optimization"],
        default="scale_translation",
        help="Optimization mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the scaler config JSON file.",
    )
    parser.add_argument(
        "--bvh",
        action="append",
        default=[],
        help="Path to a BVH file used as reference human motion. Repeat to pass multiple motions.",
    )
    parser.add_argument(
        "--human-pose",
        action="append",
        dest="human_pose_files",
        default=[],
        help="Path to a saved human static pose JSON. Repeat to provide paired samples.",
    )
    parser.add_argument(
        "--robot-target-pose-json",
        action="append",
        dest="robot_target_pose_json_files",
        default=[],
        help="Path to a robot pose JSON paired with --human-pose. Repeat to add more pairs.",
    )
    parser.add_argument(
        "--source-reference-bvh",
        type=str,
        default=str(DEFAULT_SOURCE_REFERENCE_BVH),
        help="Reference SOMA BVH used to reconstruct saved human poses.",
    )
    parser.add_argument(
        "--retargeter_config",
        "--retarget-config",
        dest="retargeter_config",
        type=str,
        default=None,
        help="Path to retargeter config JSON file to mask joints not mapped in IK.",
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of gradient descent iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--output", type=str, default=None, help="Output scaler config path.")
    parser.add_argument(
        "--rotation-iters",
        type=int,
        default=None,
        help="Number of iterations used by offset rotation optimization. Defaults to --iters.",
    )
    parser.add_argument(
        "--rotation-lr",
        type=float,
        default=None,
        help="Learning rate used by offset rotation optimization. Defaults to 0.1 * --lr.",
    )
    parser.add_argument(
        "--rotation-position-weight",
        type=float,
        default=1.0,
        help="Position-loss weight used during offset rotation optimization.",
    )
    parser.add_argument(
        "--rotation-loss-weight",
        type=float,
        default=1.0,
        help="Rotation-loss weight used during offset rotation optimization.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional JSON report path for paired_pose_optimization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "scale_translation":
        if not args.bvh:
            raise ValueError("scale_translation requires at least one --bvh input.")
        if len(args.bvh) > 1:
            print(
                "[WARNING] scale_translation currently uses only the first BVH file. "
                "Extra motions will be ignored."
            )
        optimize_scaler(
            config_file=args.config,
            bvh_file=args.bvh[0],
            retargeter_config_file=args.retargeter_config,
            iterations=args.iters,
            learning_rate=args.lr,
            output_file=args.output,
            optimize_offset_rotation=True,
            rotation_iterations=args.rotation_iters,
            rotation_learning_rate=args.rotation_lr,
            rotation_position_weight=args.rotation_position_weight,
            rotation_loss_weight=args.rotation_loss_weight,
        )
        return

    if args.mode == "paired_pose_optimization":
        if not args.human_pose_files:
            raise ValueError("paired_pose_optimization requires --human-pose inputs.")
        if not args.robot_target_pose_json_files:
            raise ValueError("paired_pose_optimization requires --robot-target-pose-json inputs.")
        optimize_scaler_from_pose_pairs(
            config_file=args.config,
            human_pose_files=args.human_pose_files,
            robot_pose_json_files=args.robot_target_pose_json_files,
            retargeter_config_file=args.retargeter_config,
            source_reference_bvh=args.source_reference_bvh,
            iterations=args.iters,
            learning_rate=args.lr,
            output_file=args.output,
            optimize_offset_rotation=True,
            rotation_iterations=args.rotation_iters,
            rotation_learning_rate=args.rotation_lr,
            rotation_position_weight=args.rotation_position_weight,
            rotation_loss_weight=args.rotation_loss_weight,
            report_file=args.report,
        )
        return


if __name__ == "__main__":
    main()
