# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
from scipy.spatial.transform import Rotation as R
import warp as wp

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.utils.newton_utils as newton_utils
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.tools.pose_optimizer_core import DEFAULT_SOURCE_REFERENCE_BVH


@dataclass
class HumanPoseRecord:
    pose_name: str
    path: str | None
    source_type: str
    source_reference_bvh: str | None
    root_translation_m: np.ndarray
    root_rotation_xyzw: np.ndarray
    local_transforms: np.ndarray
    payload: dict


@dataclass
class RobotPoseRecord:
    pose_name: str
    path: str | None
    joint_positions_rad: dict[str, float]
    payload: dict


def build_default_soma_root_xform() -> wp.transform:
    rx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
    rz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
    return wp.transform([0.0, 0.0, 0.0], wp.mul(rz, rx))


def normalize_quaternion_xyzw(quaternion_xyzw, eps: float = 1e-8) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    if norm < eps:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quaternion / norm


def euler_xyz_deg_to_quaternion_xyzw(euler_deg) -> np.ndarray:
    return R.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)


def quaternion_xyzw_to_euler_xyz_deg(quaternion_xyzw) -> np.ndarray:
    quaternion_xyzw = normalize_quaternion_xyzw(quaternion_xyzw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return R.from_quat(quaternion_xyzw).as_euler("xyz", degrees=True).astype(np.float32)


def load_reference_soma_skeleton(reference_bvh_path: str | Path = DEFAULT_SOURCE_REFERENCE_BVH):
    return bvh_utils.load_bvh(str(reference_bvh_path))


def create_reference_soma_skeleton_instance(
    reference_bvh_path: str | Path = DEFAULT_SOURCE_REFERENCE_BVH,
) -> tuple[Skeleton, SkeletonInstance]:
    skeleton, animation = load_reference_soma_skeleton(reference_bvh_path)
    skeleton_instance = SkeletonInstance(
        skeleton,
        [0.0, 0.0, 0.0],
        build_default_soma_root_xform(),
    )
    skeleton_instance.set_local_transforms(np.array(animation.get_local_transforms(0), copy=True))
    return skeleton, skeleton_instance


def _transform_to_serializable_dict(transform_array) -> dict:
    transform_array = np.asarray(transform_array, dtype=np.float32)
    return {
        "translation_m": [float(v) for v in transform_array[0:3]],
        "rotation_xyzw": [float(v) for v in normalize_quaternion_xyzw(transform_array[3:7])],
    }


def serialize_human_pose(
    skeleton_instance: SkeletonInstance,
    pose_name: str,
    source_reference_bvh: str | Path | None = DEFAULT_SOURCE_REFERENCE_BVH,
) -> dict:
    payload = {
        "pose_name": pose_name,
        "source_type": "soma",
        "source_reference_bvh": str(source_reference_bvh) if source_reference_bvh else None,
        "units": {
            "translation": "meters",
            "rotation": "quaternion_xyzw",
        },
        "root_transform": _transform_to_serializable_dict(np.array(skeleton_instance.xform, dtype=np.float32)),
        "local_joint_transforms": {},
    }

    local_transforms = np.asarray(skeleton_instance.local_transforms, dtype=np.float32)
    for joint_index, joint_name in enumerate(skeleton_instance.skeleton.joint_names):
        payload["local_joint_transforms"][joint_name] = _transform_to_serializable_dict(
            local_transforms[joint_index]
        )
    return payload


def save_human_pose_json(
    path: str | Path,
    skeleton_instance: SkeletonInstance,
    pose_name: str,
    source_reference_bvh: str | Path | None = DEFAULT_SOURCE_REFERENCE_BVH,
) -> dict:
    payload = serialize_human_pose(skeleton_instance, pose_name, source_reference_bvh)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def load_human_pose_json(
    path: str | Path,
    skeleton: Skeleton,
) -> HumanPoseRecord:
    payload = io_utils.load_json(path)
    return load_human_pose_payload(payload, skeleton, path)


def load_human_pose_payload(
    payload: dict,
    skeleton: Skeleton,
    path: str | Path | None = None,
) -> HumanPoseRecord:
    local_joint_transforms = payload.get("local_joint_transforms", {})

    missing_joints = [
        joint_name
        for joint_name in skeleton.joint_names
        if joint_name not in local_joint_transforms
    ]
    extra_joints = sorted(set(local_joint_transforms.keys()) - set(skeleton.joint_names))
    if missing_joints:
        raise ValueError(
            "Human pose is missing source joints: " + ", ".join(missing_joints[:10])
        )
    if extra_joints:
        raise ValueError(
            "Human pose contains unexpected source joints: " + ", ".join(extra_joints[:10])
        )

    local_transforms = np.array(skeleton.reference_local_transforms, copy=True)
    for joint_index, joint_name in enumerate(skeleton.joint_names):
        joint_payload = local_joint_transforms[joint_name]
        translation = np.asarray(joint_payload["translation_m"], dtype=np.float32)
        rotation = normalize_quaternion_xyzw(joint_payload["rotation_xyzw"])
        local_transforms[joint_index, 0:3] = translation
        local_transforms[joint_index, 3:7] = rotation

    root_payload = payload.get("root_transform", {})
    root_translation = np.asarray(root_payload.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float32)
    root_rotation = normalize_quaternion_xyzw(root_payload.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))

    return HumanPoseRecord(
        pose_name=payload.get("pose_name", Path(path).stem if path else "human_pose"),
        path=str(path) if path else None,
        source_type=payload.get("source_type", "soma"),
        source_reference_bvh=payload.get("source_reference_bvh"),
        root_translation_m=root_translation,
        root_rotation_xyzw=root_rotation,
        local_transforms=local_transforms,
        payload=payload,
    )


def apply_human_pose_record_to_skeleton_instance(
    skeleton_instance: SkeletonInstance,
    pose_record: HumanPoseRecord,
) -> None:
    skeleton_instance.xform = wp.transform(
        wp.vec3(*pose_record.root_translation_m.tolist()),
        wp.quat(*pose_record.root_rotation_xyzw.tolist()),
    )
    skeleton_instance.set_local_transforms(np.array(pose_record.local_transforms, copy=True))


def create_skeleton_instance_from_human_pose_record(
    skeleton: Skeleton,
    pose_record: HumanPoseRecord,
    color=(0.0, 0.0, 0.0),
) -> SkeletonInstance:
    skeleton_instance = SkeletonInstance(
        skeleton,
        color,
        wp.transform(
            wp.vec3(*pose_record.root_translation_m.tolist()),
            wp.quat(*pose_record.root_rotation_xyzw.tolist()),
        ),
    )
    skeleton_instance.set_local_transforms(np.array(pose_record.local_transforms, copy=True))
    return skeleton_instance


def load_robot_pose_json(path: str | Path) -> RobotPoseRecord:
    payload = io_utils.load_json(path)
    return load_robot_pose_payload(payload, path)


def load_robot_pose_payload(
    payload: dict,
    path: str | Path | None = None,
) -> RobotPoseRecord:
    joint_positions_rad = payload.get("joint_positions_rad")
    if not isinstance(joint_positions_rad, dict):
        raise ValueError(f"Robot pose JSON has no valid joint_positions_rad: {path}")
    for joint_name, value in joint_positions_rad.items():
        if value is None:
            raise ValueError(f"Robot pose joint [{joint_name}] is null in [{path}]")
    return RobotPoseRecord(
        pose_name=payload.get("pose_name", Path(path).stem if path else "robot_pose"),
        path=str(path) if path else None,
        joint_positions_rad={key: float(value) for key, value in joint_positions_rad.items()},
        payload=payload,
    )


def create_joint_name_to_q_index_map(model, robot_builder) -> dict[str, int]:
    return {
        newton_utils.get_name_from_label(label): index
        for index, label in enumerate(robot_builder.joint_label)
    }


def apply_robot_pose_to_model(
    model,
    robot_builder,
    robot_pose: RobotPoseRecord | dict,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(robot_pose, RobotPoseRecord):
        joint_positions_rad = robot_pose.joint_positions_rad
    else:
        joint_positions_rad = robot_pose

    joint_name_to_index = create_joint_name_to_q_index_map(model, robot_builder)
    joint_q_start = model.joint_q_start.numpy()
    default_joint_q = np.array(model.joint_q.numpy(), copy=True)
    ignored_joints = []

    for joint_name, joint_value in joint_positions_rad.items():
        if joint_name not in joint_name_to_index:
            if joint_name == "joint_head_yaw":
                ignored_joints.append(joint_name)
                continue
            raise ValueError(f"Robot pose joint [{joint_name}] does not exist in runtime model.")

        joint_index = joint_name_to_index[joint_name]
        start = int(joint_q_start[joint_index])
        end = int(joint_q_start[joint_index + 1]) if joint_index + 1 < len(joint_q_start) else int(model.joint_coord_count)
        if end - start != 1:
            raise ValueError(
                f"Robot pose joint [{joint_name}] maps to [{end - start}] coordinates, expected 1."
            )
        default_joint_q[start] = float(joint_value)

    wp.copy(model.joint_q, wp.array(default_joint_q, dtype=wp.float32))
    return default_joint_q, ignored_joints


def compute_robot_body_q_from_pose(
    pipeline: NewtonPipeline,
    robot_pose: RobotPoseRecord,
) -> tuple[np.ndarray, list[str]]:
    model = pipeline.ik_model
    state = model.state()
    _, ignored_joints = apply_robot_pose_to_model(model, pipeline.robot_builder, robot_pose)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state.body_q.numpy(), ignored_joints


def build_target_effectors_from_robot_pose(
    scaler,
    retargeter_config: dict,
    pipeline: NewtonPipeline,
    robot_pose: RobotPoseRecord,
) -> tuple[np.ndarray, list[str]]:
    body_q, ignored_joints = compute_robot_body_q_from_pose(pipeline, robot_pose)
    body_names = [
        newton_utils.get_name_from_label(label)
        for label in pipeline.robot_builder.body_label
    ]
    target_effectors_np = scaler.mapped_joint_offsets.numpy().copy()

    for idx, joint_name in enumerate(scaler.mapped_joints):
        ik_entry = retargeter_config.get("ik_map", {}).get(joint_name)
        if ik_entry is None:
            continue

        t_body = ik_entry["t_body"]
        r_body = ik_entry["r_body"]
        if t_body not in body_names or r_body not in body_names:
            raise ValueError(
                f"Robot pose target body [{t_body}] or [{r_body}] is missing for mapped joint [{joint_name}]."
            )
        t_idx = body_names.index(t_body)
        r_idx = body_names.index(r_body)
        target_effectors_np[idx][0:3] = body_q[t_idx][0:3]
        target_effectors_np[idx][3:7] = body_q[r_idx][3:7]

    return target_effectors_np, ignored_joints
