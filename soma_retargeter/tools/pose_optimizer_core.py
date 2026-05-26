# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

import soma_retargeter.robot_registry_parser as robot_registry_parser


POSE_SLOT_ORDER = (
    "t_pose",
    "natural_down",
    "both_arms_forward",
    "both_elbows_forward_90",
    "arms_forward_squat_hip_yaw_out_45",
)
POSE_SLOT_LABELS = {
    "t_pose": "T Pose",
    "natural_down": "Natural Down",
    "both_arms_forward": "Both Arms Forward",
    "both_elbows_forward_90": "Both Elbows Forward 90",
    "arms_forward_squat_hip_yaw_out_45": "Arms Forward Squat Hip Yaw Out 45",
}


def get_default_source_reference_bvh(robot_name: str | None = None) -> Path | None:
    del robot_name
    return robot_registry_parser.get_hardcoded_source_reference_bvh()


def get_default_scaler_config(robot_name: str | None = None) -> Path | None:
    return robot_registry_parser.get_profile_path(robot_name, "scaler_config")


def get_default_retargeter_config(robot_name: str | None = None) -> Path | None:
    return robot_registry_parser.get_profile_path(robot_name, "retargeter_config")


def get_default_output_config(robot_name: str | None = None, scaler_config: str | Path | None = None) -> Path | None:
    if scaler_config is not None and str(scaler_config):
        return Path(scaler_config)
    return get_default_scaler_config(robot_name)


def get_default_report_path(robot_name: str | None = None, output_path: str | Path | None = None) -> Path:
    if output_path is None or not str(output_path):
        path = robot_registry_parser.get_profile_path(robot_name, "paired_pose_report")
        return path or Path("")
    return Path(resolve_default_report_file(output_path))


def get_default_pose_artifact_path(robot_name: str | None = None, output_path: str | Path | None = None) -> Path:
    if output_path is None or not str(output_path):
        path = robot_registry_parser.get_profile_path(robot_name, "paired_pose_artifact")
        return path or Path("")
    return Path(resolve_default_pose_artifact_file(output_path))


def get_default_human_pose_files(robot_name: str | None = None) -> dict[str, Path]:
    del robot_name
    return robot_registry_parser.get_hardcoded_standard_human_pose_files()


def get_default_robot_pose_files(robot_name: str | None = None) -> dict[str, Path]:
    return robot_registry_parser.get_profile_pose_files(robot_name, "robot_pose_files")


DEFAULT_SOURCE_REFERENCE_BVH = get_default_source_reference_bvh()
DEFAULT_SCALER_CONFIG = get_default_scaler_config()
DEFAULT_RETARGETER_CONFIG = get_default_retargeter_config()
DEFAULT_HUMAN_POSE_FILES = get_default_human_pose_files()
DEFAULT_ROBOT_POSE_FILES = get_default_robot_pose_files()


def resolve_default_output_file(config_path: str | Path) -> str:
    return str(Path(config_path))


def resolve_default_report_file(output_path: str | Path) -> str:
    output_path = Path(output_path)
    return str(output_path.with_name(f"{output_path.stem}_run_report.json"))


def resolve_default_pose_artifact_file(output_path: str | Path) -> str:
    output_path = Path(output_path)
    return str(output_path.with_name(f"{output_path.stem}_paired_pose_artifact.json"))


@dataclass
class PairedPoseOptimizationDefaults:
    iterations: int = 100
    learning_rate: float = 0.01
    optimize_offset_rotation: bool = True
    rotation_iterations: int | None = None
    rotation_learning_rate: float | None = None
    rotation_position_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    device: str = "cpu"


@dataclass
class PoseSlotState:
    slot_id: str
    label: str
    status: str = "Empty"
    path: str | None = None
    pose_name: str | None = None
    summary: str = ""
    payload: dict | None = None
