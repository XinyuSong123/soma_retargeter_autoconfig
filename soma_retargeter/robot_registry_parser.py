# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import math
import struct
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import soma_retargeter.utils.io_utils as io_utils


POSE_SLOT_ORDER = (
    "t_pose",
    "natural_down",
    "both_arms_forward",
    "both_elbows_forward_90",
    "arms_forward_squat_hip_yaw_out_45",
)

_BUILTIN_ROBOT_NAMES = {"unitree_g1", "roboparty_rpo"}
_BUILTIN_ALIASES = {
    "g1": "unitree_g1",
    "unitree": "unitree_g1",
    "unitree-g1": "unitree_g1",
    "rpo": "roboparty_rpo",
    "atom01": "roboparty_rpo",
}

_REPO_ROOT = io_utils.get_repo_root()
_STANDARD_HUMAN_POSE_DIR = _REPO_ROOT / "soma_retargeter/assets/standard_human_pos"
_STANDARD_HUMAN_POSE_FILES = {
    "t_pose": _STANDARD_HUMAN_POSE_DIR / "human_t_pose.json",
    "natural_down": _STANDARD_HUMAN_POSE_DIR / "human_natural_down.json",
    "both_arms_forward": _STANDARD_HUMAN_POSE_DIR / "human_both_arms_forward.json",
    "both_elbows_forward_90": _STANDARD_HUMAN_POSE_DIR / "human_t_pose_both_elbows_forward_90.json",
    "arms_forward_squat_hip_yaw_out_45": _STANDARD_HUMAN_POSE_DIR / "human_arms_forward_squat_hip_yaw_out_45.json",
}
_HARDCODED_SOURCE_REFERENCE_BVH = _REPO_ROOT / "soma_retargeter/configs/soma/soma_zero_frame0.bvh"

_DEFAULT_CONVERTER_IMPORT_CANDIDATES = (
    "assets/motions/bvh",
    "assets/motions/soma",
)
_DEFAULT_CONVERTER_EXPORT_ROOT = "assets/motions"
_DEFAULT_CONVERTER_BATCH_SIZE = 100
_DEFAULT_RETARGETER_NAME = "Newton"
_DEFAULT_RETARGET_SOURCE = "soma"
_DEFAULT_FACING_DIRECTION = "Mujoco"
_DEFAULT_MODEL_HEIGHT_M = 1.70
_DEFAULT_HUMAN_HEIGHT_ASSUMPTION_M = 1.8
_DEFAULT_HUMAN_ROOT_NAME = "Hips"
_DEFAULT_IK_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_WEIGHT = 5.5
_DEFAULT_COLLISION_WEIGHT = 0.0
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5

_DEFAULT_SCALER_JOINT_PARENTS = {
    "Hips": "",
    "Chest": "Hips",
    "Neck1": "Chest",
    "LeftLeg": "Hips",
    "RightLeg": "Hips",
    "LeftShin": "LeftLeg",
    "RightShin": "RightLeg",
    "LeftFoot": "LeftShin",
    "RightFoot": "RightShin",
    "LeftToe": "LeftFoot",
    "RightToe": "RightFoot",
    "LeftToeBase": "LeftFoot",
    "RightToeBase": "RightFoot",
    "LeftArm": "Chest",
    "RightArm": "Chest",
    "LeftForeArm": "LeftArm",
    "RightForeArm": "RightArm",
    "LeftHand": "LeftForeArm",
    "RightHand": "RightForeArm",
}

_DEFAULT_IK_WEIGHTS = {
    "Hips": (10.0, 2.0),
    "Chest": (0.5, 0.5),
    "LeftArm": (0.5, 0.1),
    "RightArm": (0.5, 0.1),
    "LeftForeArm": (1.0, 0.5),
    "RightForeArm": (1.0, 0.5),
    "LeftHand": (1.0, 0.5),
    "RightHand": (1.0, 0.5),
    "LeftLeg": (0.5, 0.2),
    "RightLeg": (0.5, 0.2),
    "LeftShin": (1.0, 0.5),
    "RightShin": (1.0, 0.5),
    "LeftFoot": (2.0, 1.0),
    "RightFoot": (2.0, 1.0),
}
_DEFAULT_PRIORITY_WEIGHT_BANDS = {"0": 10000.0, "1": 1000.0, "2": 100.0, "3": 10.0, "4": 1.0}


def get_params_path() -> Path:
    return io_utils.get_repo_root() / "params.py"


def _load_params_module() -> ModuleType | None:
    params_path = get_params_path()
    if not params_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("_soma_retargeter_workspace_params", params_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_dict(module: ModuleType, name: str) -> dict:
    value = getattr(module, name, {})
    return value if isinstance(value, dict) else {}


def _normalize_path(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    if not value:
        return None
    return str(io_utils.resolve_path(value))


def _normalize_pose_pairs(value: Any) -> dict[str, dict[str, str | None]]:
    if not isinstance(value, dict):
        return {}

    normalized = {}
    for slot_id, pair in value.items():
        robot_path = _normalize_path(pair.get("robot")) if isinstance(pair, dict) else _normalize_path(pair)
        hardcoded_human_path = _STANDARD_HUMAN_POSE_FILES.get(str(slot_id))
        normalized[str(slot_id)] = {
            "human": str(hardcoded_human_path) if hardcoded_human_path is not None else None,
            "robot": robot_path,
        }
    return normalized


def _profiles_from_flat_registries(module: ModuleType) -> dict[str, dict[str, Any]]:
    robot_xml = _as_dict(module, "ROBOT_XML_DICT")
    robot_urdf = _as_dict(module, "ROBOT_URDF_DICT")
    retargeter_configs = _as_dict(module, "RETARGETER_CONFIG_DICT")
    compiled_profiles = _as_dict(module, "COMPILED_RETARGET_PROFILE_DICT")
    pose_pairs_by_robot = _as_dict(module, "POSE_PAIR_JSON_DICT")

    robot_names = set()
    for registry in (robot_xml, robot_urdf, retargeter_configs, compiled_profiles, pose_pairs_by_robot):
        robot_names.update(str(name) for name in registry.keys())

    profiles = {}
    for robot_name in robot_names:
        pose_pairs = _normalize_pose_pairs(pose_pairs_by_robot.get(robot_name, {}))
        profiles[robot_name] = {
            "profile_name": robot_name,
            "robot_type": robot_name,
            "mjcf_path": _normalize_path(robot_xml.get(robot_name)),
            "urdf_path": _normalize_path(robot_urdf.get(robot_name)),
            "retargeter_config": _normalize_path(retargeter_configs.get(robot_name)),
            "compiled_retarget_profile": _normalize_path(compiled_profiles.get(robot_name)),
            "source_reference_bvh": str(_HARDCODED_SOURCE_REFERENCE_BVH),
            "pose_pairs": pose_pairs,
            "human_pose_files": {
                slot_id: pair["human"]
                for slot_id, pair in pose_pairs.items()
                if pair.get("human")
            },
            "robot_pose_files": {
                slot_id: pair["robot"]
                for slot_id, pair in pose_pairs.items()
                if pair.get("robot")
            },
        }
    return profiles


@lru_cache(maxsize=1)
def get_robot_profiles() -> dict[str, dict[str, Any]]:
    module = _load_params_module()
    if module is None:
        return {}
    return _profiles_from_flat_registries(module)


@lru_cache(maxsize=1)
def get_robot_aliases() -> dict[str, str]:
    aliases = dict(_BUILTIN_ALIASES)
    module = _load_params_module()
    if module is None:
        return aliases

    for attr_name in ("ROBOT_ALIAS_DICT", "ROBOT_ALIASES"):
        raw_aliases = getattr(module, attr_name, {})
        if isinstance(raw_aliases, dict):
            aliases.update({str(alias): str(target) for alias, target in raw_aliases.items()})
            aliases.update({str(alias).lower(): str(target) for alias, target in raw_aliases.items()})
    return aliases


def resolve_robot_name(robot_name: str | None = None) -> str:
    if robot_name is None:
        robot_name = get_active_robot_name(resolve_alias=False)
    robot_name = str(robot_name)
    aliases = get_robot_aliases()
    resolved = aliases.get(robot_name, aliases.get(robot_name.lower(), robot_name))
    if resolved != robot_name:
        return resolve_robot_name(resolved)
    return resolved


def get_active_robot_name(default: str = "roboparty_rpo", *, resolve_alias: bool = True) -> str:
    module = _load_params_module()
    if module is not None and hasattr(module, "ACTIVE_ROBOT"):
        robot_name = str(getattr(module, "ACTIVE_ROBOT", default))
    else:
        profiles = sorted(get_robot_profiles().keys())
        robot_name = profiles[0] if profiles else default
    return resolve_robot_name(robot_name) if resolve_alias else robot_name


def get_robot_profile(robot_name: str | None = None) -> dict[str, Any] | None:
    profiles = get_robot_profiles()
    if robot_name is None:
        robot_name = get_active_robot_name()
    robot_name = resolve_robot_name(robot_name)
    return profiles.get(robot_name)


def get_supported_robot_names() -> list[str]:
    return sorted(_BUILTIN_ROBOT_NAMES | set(get_robot_profiles().keys()))


def _get_raw_profile_value(robot_name: str | None, key: str) -> Any:
    profile = get_robot_profile(robot_name)
    if profile is None:
        return None
    return profile.get(key)


def _get_raw_profile_path(robot_name: str | None, key: str) -> Path | None:
    value = _get_raw_profile_value(robot_name, key)
    if value is None:
        return None
    return Path(value)


def get_robot_config_dir(robot_name: str | None) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    retargeter_path = _get_raw_profile_path(robot_name, "retargeter_config")
    if retargeter_path is not None:
        return retargeter_path.parent
    if robot_name in _BUILTIN_ROBOT_NAMES:
        return io_utils.get_configs_dir() / robot_name
    return None


def _default_import_folder_reference() -> str:
    for candidate in _DEFAULT_CONVERTER_IMPORT_CANDIDATES:
        if (_REPO_ROOT / candidate).exists():
            return candidate
    return _DEFAULT_CONVERTER_IMPORT_CANDIDATES[0]


def _default_export_folder_reference(robot_name: str) -> str:
    return f"{_DEFAULT_CONVERTER_EXPORT_ROOT}/{robot_name}-export"


def _derive_scaler_filename(robot_name: str, retargeter_path: Path | None) -> str:
    if retargeter_path is not None:
        stem = retargeter_path.stem
        if stem.endswith("_retargeter_config"):
            return f"{stem[:-len('_retargeter_config')]}_scaler_config{retargeter_path.suffix}"
    return f"{robot_name}_scaler_config.json"


def _derive_converter_filename(robot_name: str) -> str:
    return f"{robot_name}_bvh_to_csv_converter_config.json"


def _derive_compiled_profile_filename(robot_name: str) -> str:
    return f"{robot_name}_compiled_retarget_profile_v2.json"


def get_generated_scaler_config_path(robot_name: str | None) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    config_dir = get_robot_config_dir(robot_name)
    if config_dir is None:
        return None
    retargeter_path = _get_raw_profile_path(robot_name, "retargeter_config")
    return config_dir / _derive_scaler_filename(robot_name, retargeter_path)


def get_generated_converter_config_path(robot_name: str | None) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    config_dir = get_robot_config_dir(robot_name)
    if config_dir is None:
        return None
    return config_dir / _derive_converter_filename(robot_name)


def get_generated_compiled_profile_path(robot_name: str | None) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    explicit_path = _get_raw_profile_path(robot_name, "compiled_retarget_profile")
    if explicit_path is not None:
        return explicit_path
    config_dir = get_robot_config_dir(robot_name)
    if config_dir is None:
        return None
    return config_dir / _derive_compiled_profile_filename(robot_name)


def get_generated_report_path(robot_name: str | None) -> Path | None:
    scaler_path = get_generated_scaler_config_path(robot_name)
    if scaler_path is None:
        return None
    return scaler_path.with_name(f"{scaler_path.stem}_run_report.json")


def get_generated_pose_artifact_path(robot_name: str | None) -> Path | None:
    scaler_path = get_generated_scaler_config_path(robot_name)
    if scaler_path is None:
        return None
    return scaler_path.with_name(f"{scaler_path.stem}_paired_pose_artifact.json")


def _build_default_scaler_joint_scales() -> dict[str, float]:
    return {joint_name: 1.0 for joint_name in _DEFAULT_SCALER_JOINT_PARENTS}


def _build_default_scaler_joint_offsets() -> dict[str, list[list[float]]]:
    return {
        joint_name: [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        for joint_name in _DEFAULT_SCALER_JOINT_PARENTS
    }


def _as_vec3(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = value.split()
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return default


def _as_quat(value: str | None) -> tuple[float, float, float, float]:
    if not value:
        return (1.0, 0.0, 0.0, 0.0)
    parts = value.split()
    if len(parts) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return (1.0, 0.0, 0.0, 0.0)


def _vec_scale(vec: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _vec_sub(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _vec_length(vec: tuple[float, float, float]) -> float:
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def _vec_normalize(
    vec: tuple[float, float, float],
    default: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float]:
    length = _vec_length(vec)
    if length <= 1e-12:
        return default
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _quat_normalize(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    length = math.sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
    if length <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (quat[0] / length, quat[1] / length, quat[2] / length, quat[3] / length)


def _quat_mul(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _axis_angle_to_quat(
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    axis = _vec_normalize(axis)
    half_angle = angle_rad * 0.5
    sin_half = math.sin(half_angle)
    return _quat_normalize((
        math.cos(half_angle),
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
    ))


def _quat_conjugate(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (quat[0], -quat[1], -quat[2], -quat[3])


def _quat_rotate(
    quat: tuple[float, float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quat_mul(_quat_mul(quat, (0.0, point[0], point[1], point[2])), _quat_conjugate(quat))
    return (rotated[1], rotated[2], rotated[3])


def _vec_add(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _vec_mul(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2])


@lru_cache(maxsize=256)
def _read_stl_vertices(path: Path) -> list[tuple[float, float, float]]:
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        triangle_count = struct.unpack("<I", data[80:84])[0]
        if 84 + triangle_count * 50 == len(data):
            offset = 84
            for _ in range(triangle_count):
                offset += 12
                for _ in range(3):
                    vertices.append(struct.unpack("<fff", data[offset:offset + 12]))
                    offset += 12
                offset += 2
            return vertices

    for line in data.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("vertex "):
            continue
        parts = line.split()
        if len(parts) == 4:
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return vertices


def _resolve_mjcf_mesh_file(xml_path: Path, mesh_file: str, mesh_dir: str | None) -> Path:
    mesh_path = Path(mesh_file)
    if mesh_path.is_absolute():
        return mesh_path
    if mesh_dir:
        return (xml_path.parent / mesh_dir / mesh_path).resolve()
    return (xml_path.parent / mesh_path).resolve()


def _get_robot_t_pose_path(robot_name: str) -> Path | None:
    profile = get_robot_profile(robot_name)
    if profile is None:
        return None
    pose_pair = profile.get("pose_pairs", {}).get("t_pose")
    if not isinstance(pose_pair, dict):
        return None
    robot_path = pose_pair.get("robot")
    return Path(robot_path) if robot_path else None


def _load_robot_t_pose_joint_positions(robot_name: str) -> dict[str, float]:
    t_pose_path = _get_robot_t_pose_path(robot_name)
    if t_pose_path is None or not t_pose_path.exists():
        return {}
    payload = io_utils.load_json(t_pose_path)
    joint_positions = payload.get("joint_positions_rad")
    if not isinstance(joint_positions, dict):
        return {}
    normalized = {}
    for joint_name, value in joint_positions.items():
        if value is None:
            continue
        try:
            normalized[str(joint_name)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _apply_mjcf_joint_pose(
    body: ET.Element,
    body_pos: tuple[float, float, float],
    body_quat: tuple[float, float, float, float],
    joint_positions_rad: dict[str, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    for joint in body.findall("joint"):
        joint_name = joint.attrib.get("name")
        joint_value = joint_positions_rad.get(joint_name, 0.0) if joint_name else 0.0
        if abs(joint_value) <= 1e-12:
            continue

        joint_type = joint.attrib.get("type", "hinge")
        joint_axis = _vec_normalize(_as_vec3(joint.attrib.get("axis"), (0.0, 0.0, 1.0)))

        if joint_type == "slide":
            body_pos = _vec_add(body_pos, _quat_rotate(body_quat, _vec_scale(joint_axis, joint_value)))
        elif joint_type == "hinge":
            joint_pos = _as_vec3(joint.attrib.get("pos"), (0.0, 0.0, 0.0))
            anchor_world = _vec_add(body_pos, _quat_rotate(body_quat, joint_pos))
            joint_quat = _axis_angle_to_quat(joint_axis, joint_value)
            body_quat = _quat_normalize(_quat_mul(body_quat, joint_quat))
            body_pos = _vec_sub(anchor_world, _quat_rotate(body_quat, joint_pos))

    return body_pos, body_quat


def _infer_robot_model_height(robot_name: str) -> float:
    mjcf_path = _get_raw_profile_path(robot_name, "mjcf_path")
    if mjcf_path is None or not mjcf_path.exists():
        return _DEFAULT_MODEL_HEIGHT_M

    try:
        root = ET.parse(mjcf_path).getroot()
    except ET.ParseError:
        return _DEFAULT_MODEL_HEIGHT_M

    compiler = root.find("compiler")
    mesh_dir = compiler.attrib.get("meshdir") if compiler is not None else None
    mesh_assets: dict[str, tuple[Path, tuple[float, float, float]]] = {}
    for mesh in root.findall("./asset/mesh"):
        name = mesh.attrib.get("name")
        filename = mesh.attrib.get("file")
        if not name or not filename:
            continue
        scale = _as_vec3(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
        mesh_assets[name] = (_resolve_mjcf_mesh_file(mjcf_path, filename, mesh_dir), scale)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return _DEFAULT_MODEL_HEIGHT_M

    try:
        joint_positions_rad = _load_robot_t_pose_joint_positions(robot_name)
    except Exception:
        joint_positions_rad = {}

    min_z = math.inf
    max_z = -math.inf

    def walk_body(
        body: ET.Element,
        parent_pos: tuple[float, float, float],
        parent_quat: tuple[float, float, float, float],
    ) -> None:
        nonlocal min_z, max_z
        local_pos = _as_vec3(body.attrib.get("pos"), (0.0, 0.0, 0.0))
        local_quat = _as_quat(body.attrib.get("quat"))
        body_pos = _vec_add(parent_pos, _quat_rotate(parent_quat, local_pos))
        body_quat = _quat_mul(parent_quat, local_quat)
        body_pos, body_quat = _apply_mjcf_joint_pose(body, body_pos, body_quat, joint_positions_rad)

        for geom in body.findall("geom"):
            mesh_name = geom.attrib.get("mesh")
            geom_type = geom.attrib.get("type")
            if (geom_type not in (None, "mesh")) or mesh_name not in mesh_assets:
                continue
            mesh_path, mesh_scale = mesh_assets[mesh_name]
            if not mesh_path.exists():
                continue
            geom_pos = _as_vec3(geom.attrib.get("pos"), (0.0, 0.0, 0.0))
            geom_quat = _as_quat(geom.attrib.get("quat"))
            world_geom_pos = _vec_add(body_pos, _quat_rotate(body_quat, geom_pos))
            world_geom_quat = _quat_mul(body_quat, geom_quat)
            for vertex in _read_stl_vertices(mesh_path):
                scaled_vertex = _vec_mul(vertex, mesh_scale)
                world_vertex = _vec_add(world_geom_pos, _quat_rotate(world_geom_quat, scaled_vertex))
                min_z = min(min_z, world_vertex[2])
                max_z = max(max_z, world_vertex[2])

        for child in body.findall("body"):
            walk_body(child, body_pos, body_quat)

    for body in worldbody.findall("body"):
        walk_body(body, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    if not math.isfinite(min_z) or not math.isfinite(max_z) or max_z <= min_z:
        return _DEFAULT_MODEL_HEIGHT_M
    return float(max_z - min_z)


def get_robot_model_height(robot_name: str | None) -> float:
    robot_name = resolve_robot_name(robot_name)
    if get_robot_profile(robot_name) is None:
        return _DEFAULT_MODEL_HEIGHT_M
    return _infer_robot_model_height(robot_name)


def build_default_scaler_config(robot_name: str | None) -> dict[str, Any]:
    robot_name = resolve_robot_name(robot_name)
    return {
        "robot_type": robot_name,
        "human_root_name": _DEFAULT_HUMAN_ROOT_NAME,
        "human_height_assumption": _DEFAULT_HUMAN_HEIGHT_ASSUMPTION_M,
        "joint_scales": _build_default_scaler_joint_scales(),
        "joint_parents": dict(_DEFAULT_SCALER_JOINT_PARENTS),
        "joint_offsets": _build_default_scaler_joint_offsets(),
    }


def build_default_converter_config(robot_name: str | None) -> dict[str, Any]:
    robot_name = resolve_robot_name(robot_name)
    return {
        "import_folder": _default_import_folder_reference(),
        "export_folder": _default_export_folder_reference(robot_name),
        "batch_size": _DEFAULT_CONVERTER_BATCH_SIZE,
        "retargeter": _DEFAULT_RETARGETER_NAME,
        "retarget_source": _DEFAULT_RETARGET_SOURCE,
        "retarget_target": robot_name,
        "retarget_source_facing_direction": _DEFAULT_FACING_DIRECTION,
    }


def _default_ik_weights(joint_name: str) -> tuple[float, float]:
    return _DEFAULT_IK_WEIGHTS.get(joint_name, (1.0, 0.5))


def _as_weight(value: Any, default: float, *, label: str) -> float:
    if value is None:
        return default
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Retargeter {label} must be numeric, got {value!r}.") from None
    if weight < 0.0:
        raise ValueError(f"Retargeter {label} must be non-negative, got {weight}.")
    return weight


def _normalize_ik_mapping_entry(joint_name: str, entry: Any) -> dict[str, Any] | None:
    t_weight, r_weight = _default_ik_weights(joint_name)

    if isinstance(entry, dict):
        t_body = entry.get("t_body") or entry.get("body") or entry.get("link")
        r_body = entry.get("r_body") or t_body
        if not isinstance(t_body, str) or not t_body.strip():
            raise ValueError(
                f"Retargeter ik_map[{joint_name!r}].t_body must be a non-empty robot link name."
            )
        if not isinstance(r_body, str) or not r_body.strip():
            raise ValueError(
                f"Retargeter ik_map[{joint_name!r}].r_body must be a non-empty robot link name."
            )
        return {
            "t_body": t_body.strip(),
            "r_body": r_body.strip(),
            "t_weight": _as_weight(
                entry.get("t_weight"),
                t_weight,
                label=f"ik_map[{joint_name!r}].t_weight",
            ),
            "r_weight": _as_weight(
                entry.get("r_weight"),
                r_weight,
                label=f"ik_map[{joint_name!r}].r_weight",
            ),
        }

    if not isinstance(entry, str):
        raise TypeError(
            f"Retargeter ik_map[{joint_name!r}] must be a robot link name string, "
            "or an object with t_body/r_body/t_weight/r_weight."
        )

    body_name = entry.strip()
    if not body_name:
        raise ValueError(f"Retargeter ik_map[{joint_name!r}] cannot be empty.")
    return {
        "t_body": body_name,
        "r_body": body_name,
        "t_weight": t_weight,
        "r_weight": r_weight,
    }


def _extract_user_ik_map(raw_config: dict[str, Any]) -> dict[str, Any]:
    value = raw_config.get("ik_map", {})
    return value if isinstance(value, dict) else {}


def _build_contact_aware_foot_ik_from_virtual_anchors(raw_config: dict[str, Any]) -> dict[str, Any] | None:
    anchors = raw_config.get("virtual_sole_anchors", {})
    if not isinstance(anchors, dict) or not anchors.get("enabled"):
        return None
    left = anchors.get("left", {}) if isinstance(anchors.get("left", {}), dict) else {}
    right = anchors.get("right", {}) if isinstance(anchors.get("right", {}), dict) else {}
    if "toe" not in left or "heel" not in left or "toe" not in right or "heel" not in right:
        return None
    anchor_offsets = {
        "left": {"toe": left["toe"], "heel": left["heel"]},
        "right": {"toe": right["toe"], "heel": right["heel"]},
    }
    for edge_name in ("inner_edge", "outer_edge"):
        if edge_name in left:
            anchor_offsets["left"][edge_name] = left[edge_name]
        if edge_name in right:
            anchor_offsets["right"][edge_name] = right[edge_name]
    return {
        "enabled": True,
        "contact_source": "auto",
        "anchor_offsets": anchor_offsets,
    }


def infer_robot_name_from_retargeter_config_path(path: str | Path) -> str | None:
    path = io_utils.resolve_path(path).resolve()
    for robot_name, profile in get_robot_profiles().items():
        registered_path = profile.get("retargeter_config")
        if not registered_path:
            continue
        if io_utils.resolve_path(registered_path).resolve() == path:
            return robot_name
    return None


def build_runtime_retargeter_config(robot_name: str | None, raw_config: dict[str, Any]) -> dict[str, Any]:
    robot_name = resolve_robot_name(robot_name or raw_config.get("robot_type") or get_active_robot_name())
    scaler_path = ensure_generated_scaler_config(robot_name)
    if scaler_path is None:
        raise FileNotFoundError(
            f"Scaler config cannot be generated for robot {robot_name!r}. "
            "Please register RETARGETER_CONFIG_DICT in params.py."
        )
    scaler_reference = make_config_reference(scaler_path)
    compiled_profile_path = ensure_compiled_retarget_profile(robot_name)

    runtime_config = {
        "initialization_pose": "soma/soma_zero_frame0.bvh",
        "num_initialization_frames": _DEFAULT_NUM_INITIALIZATION_FRAMES,
        "num_stabilization_frames": _DEFAULT_NUM_STABILIZATION_FRAMES,
        "human_robot_scaler_config": scaler_reference,
        "model_height": get_robot_model_height(robot_name),
        "ik_iterations": _DEFAULT_IK_ITERATIONS,
        "joint_limit_weight": _DEFAULT_JOINT_LIMIT_WEIGHT,
        "smooth_joint_filter_weight": _DEFAULT_SMOOTH_JOINT_FILTER_WEIGHT,
        "collision_weight": _DEFAULT_COLLISION_WEIGHT,
        "enable_post_processing": False,
    }
    if compiled_profile_path is not None:
        runtime_config["compiled_retarget_profile"] = make_config_reference(compiled_profile_path)
        runtime_config["compiled_retarget_profile_schema_version"] = 2
        if compiled_profile_path.exists():
            compiled_profile = io_utils.load_json(compiled_profile_path)
            priority_bands, priority_diagnostics = _resolve_priority_weight_bands(compiled_profile)
            runtime_config["priority_weight_bands"] = priority_bands
            runtime_config["priority_scheduler_diagnostics"] = priority_diagnostics
            runtime_config["direction_tasks"] = _extract_compiled_profile_direction_tasks(compiled_profile)
            runtime_config["pole_vector_tasks"] = _extract_compiled_profile_pole_vector_tasks(compiled_profile)

    ik_map = {}
    for joint_name, entry in _extract_user_ik_map(raw_config).items():
        normalized_entry = _normalize_ik_mapping_entry(str(joint_name), entry)
        if normalized_entry is not None:
            ik_map[str(joint_name)] = normalized_entry
    if compiled_profile_path is not None and compiled_profile_path.exists():
        ik_map = _apply_compiled_profile_tasks_to_ik_map(ik_map, io_utils.load_json(compiled_profile_path))
    runtime_config["ik_map"] = ik_map

    passthrough_keys = (
        "ik_iterations",
        "joint_limit_weight",
        "smooth_joint_filter_weight",
        "temporal_velocity_weight",
        "temporal_acceleration_weight",
        "priority_residual_guard_enabled",
        "priority_residual_guard_tolerance",
        "priority_residual_guard_absolute_tolerance",
        "priority_residual_guard_margin_fraction",
        "collision_weight",
        "enable_post_processing",
        "feet_stabilizer_config",
        "smooth_joint_filter_objective_body_masks",
        "output_default_pose_blend_frames",
        "output_default_pose_blend_bodies",
        "enable_virtual_foot_grounding",
        "virtual_foot_grounding_smooth_window",
        "contact_aware_foot_ik",
        "ground_barrier",
    )
    for key in passthrough_keys:
        if key in raw_config:
            runtime_config[key] = raw_config[key]
    runtime_config.pop("teacher_refinement", None)
    runtime_config.pop("g1_teacher_refined", None)
    if "contact_aware_foot_ik" not in runtime_config:
        auto_contact_cfg = _build_contact_aware_foot_ik_from_virtual_anchors(raw_config)
        if auto_contact_cfg is not None:
            runtime_config["contact_aware_foot_ik"] = auto_contact_cfg

    if "model_height" in raw_config:
        runtime_config["model_height"] = raw_config["model_height"]
    if "human_robot_scaler_config" in raw_config:
        runtime_config["human_robot_scaler_config"] = raw_config["human_robot_scaler_config"]
    return runtime_config


def _apply_compiled_profile_tasks_to_ik_map(
    ik_map: dict[str, dict[str, Any]],
    compiled_profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if compiled_profile.get("schema_version") != 2:
        return ik_map

    priority_bands, _ = _resolve_priority_weight_bands(compiled_profile)
    position_weight_by_semantic: dict[str, float] = {}
    rotation_task_by_semantic: dict[str, dict[str, Any]] = {}
    for task in compiled_profile.get("tasks", []):
        if not isinstance(task, dict) or not task.get("enabled", False):
            continue
        semantic = task.get("target_site") or task.get("source_semantic")
        if not isinstance(semantic, str):
            continue
        if task.get("position_mask_or_basis") is not None:
            position_weight_by_semantic[semantic] = _task_priority_weight(task, priority_bands)
        if task.get("rotation_mask_or_basis") is not None:
            rotation_task_by_semantic[semantic] = task

    out: dict[str, dict[str, Any]] = {}
    for joint_name, entry in ik_map.items():
        updated = dict(entry)
        if joint_name not in position_weight_by_semantic:
            updated["t_weight"] = 0.0
            updated["v2_position_disabled_reason"] = "compiled profile has no enabled position task"
        else:
            updated["t_weight"] = position_weight_by_semantic[joint_name]
            updated["v2_position_priority"] = _semantic_task_priority(compiled_profile, joint_name, "position")
            updated["v2_position_weight_source"] = "compiled priority band"
        if joint_name not in rotation_task_by_semantic:
            updated["r_weight"] = 0.0
            updated["v2_rotation_disabled_reason"] = "compiled profile has no enabled rotation task"
        else:
            task = rotation_task_by_semantic[joint_name]
            updated["r_weight"] = _task_priority_weight(task, priority_bands)
            updated["v2_rotation_basis"] = task.get("rotation_mask_or_basis")
            updated["v2_rotation_priority"] = int(task.get("priority", 3))
            updated["v2_rotation_weight_source"] = "compiled priority band"
        out[joint_name] = updated
    return out


def _extract_compiled_profile_direction_tasks(compiled_profile: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_compiled_profile_link_tasks(compiled_profile, "direction")


def _extract_compiled_profile_pole_vector_tasks(compiled_profile: dict[str, Any]) -> list[dict[str, Any]]:
    return _extract_compiled_profile_link_tasks(compiled_profile, "pole_vector")


def _extract_compiled_profile_link_tasks(compiled_profile: dict[str, Any], task_type: str) -> list[dict[str, Any]]:
    if compiled_profile.get("schema_version") != 2:
        return []

    priority_bands, _ = _resolve_priority_weight_bands(compiled_profile)
    out: list[dict[str, Any]] = []
    for task in compiled_profile.get("tasks", []):
        if not isinstance(task, dict) or not task.get("enabled", False):
            continue
        if task.get("task_type") != task_type:
            continue
        target_site = task.get("target_site") or task.get("source_semantic")
        reference_site = task.get("reference_site")
        if not isinstance(target_site, str) or not isinstance(reference_site, str):
            continue
        try:
            normalized_weight = float(task.get("normalized_weight", 0.0))
            characteristic_length = float(task.get("characteristic_length", 1.0))
            priority = int(task.get("priority", 3))
        except (TypeError, ValueError):
            continue
        if normalized_weight <= 0.0:
            continue
        priority_weight = _task_priority_weight(task, priority_bands)
        out.append(
            {
                "name": str(task.get("name") or f"{target_site}_{task_type}"),
                "reference_site": reference_site,
                "target_site": target_site,
                "source_semantic": str(task.get("source_semantic") or target_site),
                "weight": priority_weight,
                "normalized_weight": normalized_weight,
                "characteristic_length": characteristic_length,
                "priority": priority,
                "priority_weight_band": priority_bands.get(str(priority), normalized_weight),
                "weight_source": "compiled priority band",
            }
        )
    return out


def _resolve_priority_weight_bands(compiled_profile: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    raw_bands = compiled_profile.get("solver", {}).get("priority_weight_bands", {})
    bands: dict[str, float] = {}
    diagnostics: list[dict[str, Any]] = []
    for key, default_value in _DEFAULT_PRIORITY_WEIGHT_BANDS.items():
        raw_value = raw_bands.get(key, default_value) if isinstance(raw_bands, dict) else default_value
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            diagnostics.append({"code": "invalid_priority_weight_band", "priority": key, "value": raw_value})
            value = float(default_value)
        if value <= 0.0 or not math.isfinite(value):
            diagnostics.append({"code": "invalid_priority_weight_band", "priority": key, "value": raw_value})
            value = float(default_value)
        bands[key] = value

    for high, low in zip(range(0, 4), range(1, 5)):
        high_value = bands[str(high)]
        low_value = bands[str(low)]
        ratio = high_value / low_value if low_value > 0.0 else float("inf")
        if ratio < 10.0:
            diagnostics.append(
                {
                    "code": "priority_band_ratio_too_small",
                    "higher_priority": high,
                    "lower_priority": low,
                    "ratio": ratio,
                    "minimum_ratio": 10.0,
                }
            )
    return bands, diagnostics


def _task_priority_weight(task: dict[str, Any], priority_bands: dict[str, float]) -> float:
    try:
        priority = int(task.get("priority", 3))
    except (TypeError, ValueError):
        priority = 3
    try:
        fallback = float(task.get("normalized_weight", priority_bands.get("3", 10.0)))
    except (TypeError, ValueError):
        fallback = priority_bands.get("3", 10.0)
    return float(priority_bands.get(str(priority), fallback))


def _semantic_task_priority(compiled_profile: dict[str, Any], semantic: str, task_kind: str) -> int | None:
    for task in compiled_profile.get("tasks", []):
        if not isinstance(task, dict) or not task.get("enabled", False):
            continue
        task_semantic = task.get("target_site") or task.get("source_semantic")
        if task_semantic != semantic:
            continue
        if task_kind == "position" and task.get("position_mask_or_basis") is None:
            continue
        if task_kind == "rotation" and task.get("rotation_mask_or_basis") is None:
            continue
        try:
            return int(task.get("priority", 3))
        except (TypeError, ValueError):
            return None
    return None


def load_retargeter_config(
    retargeter_config_file: str | Path | None = None,
    *,
    robot_name: str | None = None,
) -> dict[str, Any]:
    if retargeter_config_file is None:
        retargeter_config_file = get_profile_path(robot_name, "retargeter_config")
    if retargeter_config_file is None:
        raise FileNotFoundError(f"Retargeter config is not registered for robot: {robot_name}")

    path = io_utils.resolve_path(retargeter_config_file)
    raw_config = io_utils.load_json(path)
    resolved_robot_name = robot_name or infer_robot_name_from_retargeter_config_path(path) or raw_config.get("robot_type")
    return build_runtime_retargeter_config(resolved_robot_name, raw_config)


def validate_compiled_retarget_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if profile.get("schema_version") != 2:
        diagnostics.append({"code": "invalid_schema_version", "expected": 2, "actual": profile.get("schema_version")})
    if profile.get("quaternion_order") != "xyzw":
        diagnostics.append({"code": "invalid_quaternion_order", "expected": "xyzw", "actual": profile.get("quaternion_order")})
    for key in ("robot_fingerprint", "source_skeleton_fingerprint", "semantic_sites", "chains", "tasks", "solver"):
        if key not in profile:
            diagnostics.append({"code": "missing_profile_key", "key": key})
    if not isinstance(profile.get("semantic_sites", {}), dict):
        diagnostics.append({"code": "invalid_semantic_sites"})
    if not isinstance(profile.get("chains", {}), dict):
        diagnostics.append({"code": "invalid_chains"})
    if not isinstance(profile.get("tasks", []), list):
        diagnostics.append({"code": "invalid_tasks"})
    return diagnostics


def load_compiled_retarget_profile(robot_name: str | None, *, require_valid: bool = True) -> dict[str, Any] | None:
    path = get_profile_path(robot_name, "compiled_retarget_profile")
    if path is None or not path.exists():
        return None
    profile = io_utils.load_json(path)
    diagnostics = validate_compiled_retarget_profile(profile)
    if require_valid and diagnostics:
        raise ValueError(f"Invalid compiled retarget profile {path}: {diagnostics}")
    return profile


def make_config_reference(path: str | Path) -> str:
    path = io_utils.resolve_path(path).resolve()
    configs_dir = io_utils.get_configs_dir().resolve()
    repo_root = io_utils.get_repo_root().resolve()
    try:
        return path.relative_to(configs_dir).as_posix()
    except ValueError:
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            return str(path)


def link_scaler_config_to_retargeter(retargeter_config_path: str | Path, scaler_config_path: str | Path) -> str:
    del retargeter_config_path
    return make_config_reference(scaler_config_path)


def ensure_generated_scaler_config(robot_name: str | None, *, force: bool = False) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    scaler_path = get_generated_scaler_config_path(robot_name)
    if scaler_path is None:
        return None

    if force or not scaler_path.exists():
        io_utils.save_json(scaler_path, build_default_scaler_config(robot_name), indent=4, ensure_ascii=False)
    else:
        data = io_utils.load_json(scaler_path)
        updated = False
        if data.get("robot_type") != robot_name:
            data["robot_type"] = robot_name
            updated = True
        if "model_height" in data:
            del data["model_height"]
            updated = True
        if "human_height_assumption" not in data:
            data["human_height_assumption"] = _DEFAULT_HUMAN_HEIGHT_ASSUMPTION_M
            updated = True
        if updated:
            io_utils.save_json(scaler_path, data, indent=4, ensure_ascii=False)

    retargeter_path = _get_raw_profile_path(robot_name, "retargeter_config")
    if retargeter_path is not None and retargeter_path.exists():
        link_scaler_config_to_retargeter(retargeter_path, scaler_path)
    return scaler_path


def ensure_generated_converter_config(robot_name: str | None, *, force: bool = False) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    converter_path = get_generated_converter_config_path(robot_name)
    if converter_path is None:
        return None
    if force or not converter_path.exists():
        io_utils.save_json(converter_path, build_default_converter_config(robot_name), indent=4, ensure_ascii=False)
    return converter_path


def ensure_compiled_retarget_profile(robot_name: str | None, *, force: bool = False) -> Path | None:
    robot_name = resolve_robot_name(robot_name)
    output_path = get_generated_compiled_profile_path(robot_name)
    if output_path is None:
        return None
    if output_path.exists() and not force:
        diagnostics = validate_compiled_retarget_profile(io_utils.load_json(output_path))
        if not diagnostics:
            return output_path

    profile = get_robot_profile(robot_name)
    if profile is None:
        return None
    raw_config_path = profile.get("retargeter_config")
    if not raw_config_path:
        return None

    from soma_retargeter.robotics.morphology import analyze_mjcf_morphology
    from soma_retargeter.robotics.retarget_profile import write_profile_json
    from soma_retargeter.robotics.task_compiler import compile_retarget_profile

    raw_config = io_utils.load_json(raw_config_path)
    morphology = analyze_mjcf_morphology(profile.get("mjcf_path"))
    compiled = compile_retarget_profile(
        robot_name=robot_name,
        raw_config=raw_config,
        morphology=morphology,
        source_config_path=raw_config_path,
    )
    write_profile_json(compiled, output_path)
    return output_path


def ensure_generated_runtime_configs(robot_name: str | None, *, force: bool = False) -> dict[str, Path | None]:
    return {
        "scaler_config": ensure_generated_scaler_config(robot_name, force=force),
        "bvh_converter_config": ensure_generated_converter_config(robot_name, force=force),
        "compiled_retarget_profile": ensure_compiled_retarget_profile(robot_name, force=force),
    }


def get_profile_path(robot_name: str | None, key: str) -> Path | None:
    robot_name = resolve_robot_name(robot_name)

    if key == "source_reference_bvh":
        return _HARDCODED_SOURCE_REFERENCE_BVH
    if key in {"scaler_config", "optimized_scaler_output"}:
        return ensure_generated_scaler_config(robot_name)
    if key == "bvh_converter_config":
        return ensure_generated_converter_config(robot_name)
    if key == "compiled_retarget_profile":
        return ensure_compiled_retarget_profile(robot_name)
    if key == "paired_pose_report":
        return get_generated_report_path(robot_name)
    if key == "paired_pose_artifact":
        return get_generated_pose_artifact_path(robot_name)
    if key == "link_mapping_config":
        return _get_raw_profile_path(robot_name, "retargeter_config")
    if key == "link_mapping_template":
        return None

    value = _get_raw_profile_value(robot_name, key)
    if value is None:
        return None
    return Path(value)


def get_profile_pose_files(robot_name: str | None, key: str) -> dict[str, Path]:
    profile = get_robot_profile(robot_name)
    if profile is None:
        return {}
    value = profile.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {slot_id: Path(path) for slot_id, path in value.items() if path}


def get_pose_pair_issues(robot_name: str | None) -> list[str]:
    profile = get_robot_profile(robot_name)
    if profile is None:
        return [f"未找到机器人注册信息: {robot_name}"]

    issues = []
    pose_pairs = profile.get("pose_pairs", {})
    if not pose_pairs:
        issues.append(
            f"{profile['robot_type']} 还没有在 params.py 的 POSE_PAIR_JSON_DICT 中注册任何机器人 pose JSON。"
        )
        return issues

    for slot_id in POSE_SLOT_ORDER:
        pair = pose_pairs.get(slot_id)
        if pair is None:
            continue
        human_path = pair.get("human")
        robot_path = pair.get("robot")
        if not human_path:
            issues.append(f"姿态组 [{slot_id}] 缺少硬编码标准 human pose JSON 路径。")
        elif not Path(human_path).exists():
            issues.append(f"姿态组 [{slot_id}] 的 human pose JSON 不存在: {human_path}")
        if not robot_path:
            issues.append(f"姿态组 [{slot_id}] 缺少 robot pose JSON 路径。")
        elif not Path(robot_path).exists():
            issues.append(f"姿态组 [{slot_id}] 的 robot pose JSON 不存在: {robot_path}")
    return issues


def get_registered_robot_names() -> list[str]:
    return sorted(get_robot_profiles().keys())


def _prepared_line(label: str, ready: bool, detail: str = "") -> str:
    state = "已准备" if ready else "缺少"
    return f"{label}: {state}{(' - ' + detail) if detail else ''}"


def get_robot_setup_status(robot_name: str | None) -> list[str]:
    robot_name = resolve_robot_name(robot_name)
    profile = get_robot_profile(robot_name)
    if profile is None:
        return [f"当前 robot: {robot_name}", "params.py 中没有这个 robot 的注册信息。"]

    ensure_generated_runtime_configs(robot_name)

    registered = ", ".join(get_registered_robot_names()) or "无"
    lines = [
        f"当前 robot: {robot_name}",
        f"params.py 已注册: {registered}",
    ]

    mjcf_path = get_profile_path(robot_name, "mjcf_path")
    urdf_path = get_profile_path(robot_name, "urdf_path")
    retargeter_path = get_profile_path(robot_name, "retargeter_config")
    scaler_path = get_generated_scaler_config_path(robot_name)
    converter_path = get_generated_converter_config_path(robot_name)
    compiled_profile_path = get_generated_compiled_profile_path(robot_name)

    lines.append(_prepared_line("MJCF/XML", bool(mjcf_path and mjcf_path.exists())))
    lines.append(_prepared_line("URDF", bool(urdf_path and urdf_path.exists()), "可选参考文件"))
    lines.append(_prepared_line("Retargeter link map", bool(retargeter_path and retargeter_path.exists())))
    lines.append(_prepared_line("Scaler config", bool(scaler_path and scaler_path.exists()), "自动生成"))
    lines.append(_prepared_line("BVH converter config", bool(converter_path and converter_path.exists()), "自动生成"))
    lines.append(_prepared_line("Compiled retarget profile v2", bool(compiled_profile_path and compiled_profile_path.exists()), "自动生成"))

    human_ready = sum(1 for path in _STANDARD_HUMAN_POSE_FILES.values() if path.exists())
    robot_pose_files = get_profile_pose_files(robot_name, "robot_pose_files")
    robot_ready = sum(1 for path in robot_pose_files.values() if path.exists())
    lines.append(f"标准 human pose: 已准备 {human_ready}/{len(_STANDARD_HUMAN_POSE_FILES)} 组")
    lines.append(f"机器人 pose pair: 已准备 {robot_ready}/{len(POSE_SLOT_ORDER)} 组")

    if retargeter_path and retargeter_path.exists():
        try:
            raw_config = io_utils.load_json(retargeter_path)
            ik_map = _extract_user_ik_map(raw_config)
            valid_count = 0
            for value in ik_map.values():
                if isinstance(value, str) and value.strip():
                    valid_count += 1
                elif isinstance(value, dict) and (value.get("t_body") or value.get("body") or value.get("link")):
                    valid_count += 1
            invalid_count = len(ik_map) - valid_count
            suffix = f"，{invalid_count} 个格式错误" if invalid_count else ""
            lines.append(f"Human/link 映射: 已填写 {valid_count} 个部位{suffix}")
        except Exception as exc:
            lines.append(f"Human/link 映射: 读取失败 - {exc}")
    return lines


def get_hardcoded_standard_human_pose_files() -> dict[str, Path]:
    return dict(_STANDARD_HUMAN_POSE_FILES)


def get_hardcoded_source_reference_bvh() -> Path:
    return _HARDCODED_SOURCE_REFERENCE_BVH


def apply_profile_to_converter_config(config: dict[str, Any], robot_name: str | None = None) -> dict[str, Any]:
    robot_name = resolve_robot_name(robot_name or config.get("retarget_target"))
    ensure_generated_runtime_configs(robot_name)

    config.setdefault("retargeter", _DEFAULT_RETARGETER_NAME)
    config.setdefault("retarget_source", _DEFAULT_RETARGET_SOURCE)
    config.setdefault("retarget_source_facing_direction", _DEFAULT_FACING_DIRECTION)
    config.setdefault("batch_size", _DEFAULT_CONVERTER_BATCH_SIZE)
    config["retarget_target"] = robot_name

    if not config.get("import_folder"):
        config["import_folder"] = _default_import_folder_reference()
    if not config.get("export_folder"):
        config["export_folder"] = _default_export_folder_reference(robot_name)
    return config
