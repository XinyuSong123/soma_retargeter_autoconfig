# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import dataclass
from typing import Protocol, ClassVar, List

import numpy as np
import warp as wp

from scipy.spatial.transform import Rotation as R
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


class RobotCSVConfig(Protocol):
    name: str
    csv_header: List[str]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        ...
    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        ...


class RobotNPZCSVConfig(Protocol):
    """Protocol for CSV configs compatible with csv_to_npz.py format."""
    name: str
    
    def to_csv_row(self, anim_row: np.ndarray) -> List[float]:
        """Convert animation buffer row to CSV row (no frame index, no header)."""
        ...


@dataclass
class UnitreeG129DOF_CSVConfig:
    name: str = "unitree_g1_29dof"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        """
        Convert one CSV row (including frame index) into one anim buffer frame.
        """
        # csv_row layout: [frame index, tx, ty, tz, rx, ry, rz, dof0, ...]
        num_joint_dofs = csv_row.shape[0] - 1 # Remove frame index
        anim_row = np.zeros(
            num_joint_dofs + 1, # euler rotate xyz values converted to quat
            dtype=np.float32)

        # translation (cm -> m)
        anim_row[0:3] = csv_row[1:4] * 0.01

        # rotation (euler deg -> quat)
        euler = np.deg2rad(csv_row[4:7])
        quat = wp.quat_rpy(euler[0], euler[1], euler[2])
        anim_row[3:7] = quat

        # remaining joints (deg -> rad)
        anim_row[7:] = np.deg2rad(csv_row[7:])

        return anim_row

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        """
        Convert one anim buffer row into a CSV row with this config's layout.
        """
        # translation (m -> cm)
        t = wp.vec3(*anim_row[0:3]) * 100.0
        # root rotation (quat -> euler deg)
        q = wp.quat(*anim_row[3:7])
        euler = R.from_quat([q[0], q[1], q[2], q[3]]).as_euler("xyz", degrees=True)

        row = [frame_idx, t[0], t[1], t[2], euler[0], euler[1], euler[2]]

        # joints (rad -> deg)
        row.extend(np.rad2deg(anim_row[7:]))

        return row


@dataclass
class GenericNPZCSVConfig:
    """
    Headerless CSV configuration for dynamically registered robots.

    The animation buffer already stores rows as:
    [tx, ty, tz, qx, qy, qz, qw, joint_0, ...].
    """
    name: str = "generic_npz"

    def to_csv_row(self, anim_row: np.ndarray) -> List[float]:
        return anim_row.tolist()


def load_csv(file_path: str, fps: float = 120.0, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> CSVAnimationBuffer:
    """
    Load a robot motion CSV file into a ``CSVAnimationBuffer``.
    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.
        csv_config (RobotCSVConfig, optional): Configuration object that defines how to parse
            CSV rows into animation frames. Defaults to ``UnitreeG129DOF_CSVConfig``.
    Returns:
        CSVAnimationBuffer: An animation buffer containing the loaded and converted animation data.
    Raises:
        FileNotFoundError: If the CSV file at file_path does not exist.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"[INFO]: Loading CSV [{file_path}] for robot [{csv_config.name}]")
        csv_data = np.loadtxt(f, delimiter=",", skiprows=1)
        if csv_data.ndim == 1:
            csv_data = csv_data[np.newaxis, :]
        num_frames = csv_data.shape[0]

        # Each anim row is derived by config, so infer size from first row
        first_row_anim = csv_config.to_anim_frame(csv_data[0])
        anim_data = np.zeros((num_frames, first_row_anim.shape[0]), dtype=np.float32)
        anim_data[0, :] = first_row_anim

        for i in range(1, num_frames):
            anim_data[i, :] = csv_config.to_anim_frame(csv_data[i])

        return CSVAnimationBuffer.create_from_raw_data(anim_data, fps)


def load_csv_npz_compatible(file_path: str, fps: float = 120.0) -> CSVAnimationBuffer:
    """
    Load a headerless CSV file written by ``save_csv_npz_compatible``.

    The stored layout already matches ``CSVAnimationBuffer`` rows:
    ``[tx, ty, tz, qx, qy, qz, qw, joint_0, ...]``.

    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.

    Returns:
        CSVAnimationBuffer: Animation buffer containing the raw motion rows.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"[INFO]: Loading NPZ-compatible CSV [{file_path}]")
        csv_data = np.loadtxt(f, delimiter=",")
        if csv_data.ndim == 1:
            csv_data = csv_data[np.newaxis, :]
        return CSVAnimationBuffer.create_from_raw_data(csv_data.astype(np.float32), fps)


def load_csv_auto(file_path: str, robot_type: str = "unitree_g1", fps: float = 120.0) -> CSVAnimationBuffer:
    """
    Load either the legacy headered CSV format or the headerless NPZ-compatible format.

    Args:
        file_path (str): Path to the CSV file to load.
        robot_type (str, optional): Robot type used when parsing a headered CSV.
            Defaults to "unitree_g1".
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.

    Returns:
        CSVAnimationBuffer: Parsed animation buffer.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    if first_line.startswith("Frame"):
        return load_csv(file_path, fps=fps, csv_config=get_csv_config(robot_type))
    return load_csv_npz_compatible(file_path, fps=fps)


def save_csv(file_path: str, buffer: CSVAnimationBuffer, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> None:
    """
    Save a ``CSVAnimationBuffer`` to a robot motion CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.
        csv_config (RobotCSVConfig, optional): Configuration object that defines CSV format and headers.
            Defaults to ``UnitreeG129DOF_CSVConfig``.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_config.csv_header)

        for i in range(buffer.num_frames):
            data = buffer.get_data(i)
            row = csv_config.to_csv_row(i, data)
            writer.writerow(row)


@dataclass
class UnitreeG129DOF_NPZCSVConfig:
    """
    CSV configuration compatible with csv_to_npz.py format.
    
    Output format (no header, no frame index):
    - Columns 0-2: root position (x, y, z) in meters
    - Columns 3-6: root rotation as quaternion (x, y, z, w)
    - Columns 7+: joint angles in radians
    """
    name: str = "unitree_g1_29dof_npz"
    
    def to_csv_row(self, anim_row: np.ndarray) -> List[float]:
        """
        Convert one anim buffer row into a CSV row compatible with csv_to_npz.py.
        
        anim_row format: [tx, ty, tz, qx, qy, qz, qw, joint_0, joint_1, ...]
        output format: [tx, ty, tz, qx, qy, qz, qw, joint_0, joint_1, ...]
        """
        # Position is already in meters, quaternion already in xyzw, joints already in radians
        return anim_row.tolist()


@dataclass
class RoboPartyRPO_CSVConfig:
    """
    CSV configuration for RoboParty RPO (Atom01) robot.
    
    Joint order (23 DOF):
    - Left leg (6): left_thigh_yaw, left_thigh_roll, left_thigh_pitch, left_knee, left_ankle_pitch, left_ankle_roll
    - Right leg (6): right_thigh_yaw, right_thigh_roll, right_thigh_pitch, right_knee, right_ankle_pitch, right_ankle_roll
    - Torso (1): torso_joint
    - Left arm (5): left_arm_pitch, left_arm_roll, left_arm_yaw, left_elbow_pitch, left_elbow_yaw
    - Right arm (5): right_arm_pitch, right_arm_roll, right_arm_yaw, right_elbow_pitch, right_elbow_yaw
    """
    name: str = "roboparty_rpo_23dof"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_thigh_yaw_joint_dof", "left_thigh_roll_joint_dof", "left_thigh_pitch_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_thigh_yaw_joint_dof", "right_thigh_roll_joint_dof", "right_thigh_pitch_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "torso_joint_dof",
        "left_arm_pitch_joint_dof", "left_arm_roll_joint_dof", "left_arm_yaw_joint_dof",
        "left_elbow_pitch_joint_dof", "left_elbow_yaw_joint_dof",
        "right_arm_pitch_joint_dof", "right_arm_roll_joint_dof", "right_arm_yaw_joint_dof",
        "right_elbow_pitch_joint_dof", "right_elbow_yaw_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        """
        Convert one CSV row (including frame index) into one anim buffer frame.
        """
        num_joint_dofs = csv_row.shape[0] - 1
        anim_row = np.zeros(num_joint_dofs + 1, dtype=np.float32)

        # translation (cm -> m)
        anim_row[0:3] = csv_row[1:4] * 0.01

        # rotation (euler deg -> quat)
        euler = np.deg2rad(csv_row[4:7])
        quat = wp.quat_rpy(euler[0], euler[1], euler[2])
        anim_row[3:7] = quat

        # remaining joints (deg -> rad)
        anim_row[7:] = np.deg2rad(csv_row[7:])

        return anim_row

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        """
        Convert one anim buffer row into a CSV row with this config's layout.
        """
        # translation (m -> cm)
        t = wp.vec3(*anim_row[0:3]) * 100.0
        # root rotation (quat -> euler deg)
        q = wp.quat(*anim_row[3:7])
        euler = R.from_quat([q[0], q[1], q[2], q[3]]).as_euler("xyz", degrees=True)

        row = [frame_idx, t[0], t[1], t[2], euler[0], euler[1], euler[2]]

        # joints (rad -> deg)
        row.extend(np.rad2deg(anim_row[7:]))

        return row


@dataclass
class RoboPartyRPO_NPZCSVConfig:
    """
    CSV configuration for RoboParty RPO (Atom01) robot compatible with csv_to_npz.py format.
    
    Output format (no header, no frame index):
    - Columns 0-2: root position (x, y, z) in meters
    - Columns 3-6: root rotation as quaternion (x, y, z, w)
    - Columns 7+: joint angles in radians (23 DOF)
    """
    name: str = "roboparty_rpo_23dof_npz"
    
    def to_csv_row(self, anim_row: np.ndarray) -> List[float]:
        """
        Convert one anim buffer row into a CSV row compatible with csv_to_npz.py.
        """
        return anim_row.tolist()


def get_csv_config(robot_type: str = "unitree_g1") -> RobotCSVConfig:
    """
    Return the headered CSV configuration for a robot type.

    Args:
        robot_type (str): Robot type name.

    Returns:
        RobotCSVConfig: CSV configuration instance for the requested robot.

    Raises:
        ValueError: If the robot type is unknown.
    """
    if robot_type == "unitree_g1":
        return UnitreeG129DOF_CSVConfig()
    if robot_type == "roboparty_rpo":
        return RoboPartyRPO_CSVConfig()

    raise ValueError(
        f"[ERROR]: Unsupported robot type: {robot_type}. "
        "Supported types: unitree_g1, roboparty_rpo"
    )


def get_npz_csv_config(robot_type: str = "unitree_g1") -> RobotNPZCSVConfig:
    """
    Return the headerless NPZ-compatible CSV configuration for a robot type.

    Args:
        robot_type (str): Robot type name.

    Returns:
        RobotNPZCSVConfig: NPZ-compatible CSV configuration instance.

    Raises:
        ValueError: If the robot type is unknown.
    """
    if robot_type == "unitree_g1":
        return UnitreeG129DOF_NPZCSVConfig()
    if robot_type == "roboparty_rpo":
        return RoboPartyRPO_NPZCSVConfig()

    return GenericNPZCSVConfig(name=f"{robot_type}_npz")


def save_csv_npz_compatible(file_path: str, buffer: CSVAnimationBuffer, robot_type: str = "unitree_g1") -> None:
    """
    Save a ``CSVAnimationBuffer`` to a CSV file compatible with csv_to_npz.py.
    
    This format has no header and no frame index column:
    - Columns 0-2: root position (x, y, z) in meters
    - Columns 3-6: root rotation as quaternion (x, y, z, w)
    - Columns 7+: joint angles in radians

    Args:
        file_path (str): The path where the CSV file will be saved.
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.
        robot_type (str): The robot type. Supported: "unitree_g1" and "roboparty_rpo".
            Custom robots use a generic NPZ-compatible output layout.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
        ValueError: If the robot type is not supported.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")
    
    config = get_npz_csv_config(robot_type)
    
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        for i in range(buffer.num_frames):
            data = buffer.get_data(i)
            row = config.to_csv_row(data)
            writer.writerow(row)
