# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
from pathlib import Path
from typing import Union

import newton

import soma_retargeter.robot_registry_parser as robot_registry_parser
import soma_retargeter.utils.io_utils as io_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()


class TargetType(IntEnum):
    """Enumeration of supported target model types."""
    UNITREE_G1 = auto()
    ROBOPARTY_RPO = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1 : "unitree_g1",
    TargetType.ROBOPARTY_RPO : "roboparty_rpo",
}
_STR_TO_TARGET_TYPE = {s : t for t, s in _TARGET_TYPE_TO_STR.items()}
_DEFAULT_RUNTIME_V3_PROFILE_IDS = {
    "roboparty_rpo": "roboparty_rpo_local",
    "unitree_g1": "unitree_g1_mjcf",
}


def get_source_str_from_type(source: SourceType) -> str:
    """
    Get the string name associated with a given source type.

    Args:
        source (SourceType): The source type enum value.

    Returns:
        str: The string representation of the source type.
    """
    return _SOURCE_TYPE_TO_STR[source]


def get_source_type_from_str(source: str) -> SourceType:
    """
    Convert a string to its corresponding SourceType enum value.

    Args:
        source (str): The string representation of a source.

    Returns:
        SourceType: The corresponding source type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid source type.
    """
    try:
        return _STR_TO_SOURCE_TYPE[source]
    except KeyError:
        allowed = ", ".join(_STR_TO_SOURCE_TYPE.keys())
        raise ValueError(f"Unknown source type: [{source}]. Allowed values: {allowed}") from None


def get_target_str_from_type(target: Union[str, TargetType]) -> str:
    """
    Get the string name associated with a given target type.

    Args:
        target (TargetType): The target type enum value.

    Returns:
        str: The string representation of the target type.
    """
    if isinstance(target, str):
        return target
    return _TARGET_TYPE_TO_STR[target]


def get_target_type_from_str(target: str) -> Union[str, TargetType]:
    """
    Convert a string to its corresponding TargetType enum value.

    Args:
        target (str): The string representation of a target.

    Returns:
        TargetType: The corresponding target type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid target type.
    """
    target = robot_registry_parser.resolve_robot_name(target)
    if target in robot_registry_parser.get_robot_profiles():
        return target
    try:
        return _STR_TO_TARGET_TYPE[target]
    except KeyError:
        allowed = ", ".join(get_supported_target_names())
        raise ValueError(f"Unknown target type: [{target}]. Allowed values: {allowed}") from None


def get_supported_target_names() -> list[str]:
    """Return built-in target names plus robots declared in params.py."""
    return sorted(set(_STR_TO_TARGET_TYPE.keys()) | set(robot_registry_parser.get_supported_robot_names()))


def get_default_runtime_v3_profile_id(target: Union[str, TargetType]) -> str | None:
    """Return the default Step 2 capability profile id for a runtime target."""
    target_name = get_target_str_from_type(target)
    target_name = robot_registry_parser.resolve_robot_name(target_name)
    return _DEFAULT_RUNTIME_V3_PROFILE_IDS.get(target_name)


def get_source_model_mesh(source: SourceType, skeleton) -> dict:
    """
    Retrieve model mesh for a given source type.

    Args:
        source (SourceType): The source type for which properties should be retrieved.
        skeleton: The skeleton associated with the source model, used for loading the mesh.

    Returns:
        SkeletalMesh: The skeleton mesh for the given source type.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        import soma_retargeter.assets.usd as usd_utils

        return usd_utils.load_skeletal_mesh_from_usd(
            str(io_utils.get_config_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")


def get_repo_root() -> Path:
    """
    Return the repository root directory.

    Returns:
        Path: Absolute path to the repository root.
    """
    return io_utils.get_repo_root()


def get_robot_mjcf_path(target: Union[str, TargetType]) -> Path:
    """
    Resolve the runtime MJCF path for a target robot.

    Args:
        target: Target robot name or enum value.

    Returns:
        Path: Filesystem path to the MJCF entry file for the robot.

    Raises:
        ValueError: If the target robot type is unknown.
    """
    if isinstance(target, str):
        target = robot_registry_parser.resolve_robot_name(target)
        profile_path = robot_registry_parser.get_profile_path(target, "mjcf_path")
        if profile_path is not None:
            return profile_path
        target = (
            _STR_TO_TARGET_TYPE[target]
            if target in _STR_TO_TARGET_TYPE
            else get_target_type_from_str(target)
        )

    if target == TargetType.UNITREE_G1:
        return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
    if target == TargetType.ROBOPARTY_RPO:
        return get_repo_root() / "assets/robots/atom01/mjcf/atom01.xml"
    raise ValueError(f"Unknown target type [{target}].")


def get_retargeter_config(source: SourceType, target: Union[str, TargetType]) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType): The source type.
        target (TargetType): The target type.

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    if isinstance(target, str):
        target = robot_registry_parser.resolve_robot_name(target)
        retargeter_config_path = robot_registry_parser.get_profile_path(target, "retargeter_config")
        if retargeter_config_path is not None:
            robot_registry_parser.ensure_generated_runtime_configs(target)
            return robot_registry_parser.load_retargeter_config(retargeter_config_path, robot_name=target)
        target = (
            _STR_TO_TARGET_TYPE[target]
            if target in _STR_TO_TARGET_TYPE
            else get_target_type_from_str(target)
        )

    if target == TargetType.UNITREE_G1:
        if source == SourceType.SOMA:
            filename = 'soma_to_g1_retargeter_config.json'
        else:
            raise ValueError(f"Unknown source type [{source}] for target [{target}].")
        return robot_registry_parser.load_retargeter_config(
            io_utils.get_config_file('unitree_g1', filename),
            robot_name="unitree_g1",
        )
    elif target == TargetType.ROBOPARTY_RPO:
        if source == SourceType.SOMA:
            filename = 'soma_to_rpo_retargeter_config.json'
        else:
            raise ValueError(f"Unknown source type [{source}] for target [{target}].")
        return robot_registry_parser.load_retargeter_config(
            io_utils.get_config_file('roboparty_rpo', filename),
            robot_name="roboparty_rpo",
        )
    else:
        raise ValueError(f"Unknown target type [{target}].")
