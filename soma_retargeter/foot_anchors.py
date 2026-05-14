# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import soma_retargeter.robot_registry_parser as robot_registry_parser


ANCHOR_NAMES = ("sole_center", "toe", "heel", "inner_edge", "outer_edge")

_PROPORTIONAL_FOOT_LENGTH_RATIO = 0.12
_PROPORTIONAL_FOOT_WIDTH_RATIO = 0.055
_PROPORTIONAL_SOLE_Z_RATIO = -0.02


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() == "false")


def scalar_to_float(value: Any) -> float | None:
    if is_auto(value) or is_false(value) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_capability_profile_path(robot_name: str, explicit_path: str | Path | None = None) -> Path | None:
    from soma_retargeter.teacher_refinement.capability_loader import find_capability_profile_path as _impl

    return _impl(robot_name, explicit_path)


def load_capability_profile(
    robot_name: str,
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    from soma_retargeter.teacher_refinement.capability_loader import load_capability_profile as _impl

    return _impl(robot_name, explicit_path)


def resolve_capability_profile(
    robot_name: str,
    *,
    profile: dict[str, Any] | None = None,
    profile_path: str | None = None,
):
    from soma_retargeter.teacher_refinement.capability_resolver import resolve_capability_profile as _impl

    return _impl(robot_name, profile=profile, profile_path=profile_path)


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


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _quat_matrix(quat_wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat_wxyz
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if length <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / length, x / length, y / length, z / length
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, pos: tuple[float, float, float], rot: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return (points @ rot.T) + np.asarray(pos, dtype=np.float64)


def _normalize_vec(vec: np.ndarray, default: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1e-12:
        if default is not None:
            return _normalize_vec(default)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / length


def _axis_vector(axis: str, fallback: str) -> np.ndarray:
    idx, sign = _axis_index(axis, fallback)
    vec = np.zeros(3, dtype=np.float64)
    vec[idx] = sign
    return vec


def _orthogonalize(vec: np.ndarray, normal: np.ndarray, default: np.ndarray) -> np.ndarray:
    projected = np.asarray(vec, dtype=np.float64) - np.asarray(normal, dtype=np.float64) * float(np.dot(vec, normal))
    if np.linalg.norm(projected) <= 1e-12:
        projected = default - normal * float(np.dot(default, normal))
    return _normalize_vec(projected, default)


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


def _quat_rotate(quat_wxyz: tuple[float, float, float, float], point: tuple[float, float, float]) -> tuple[float, float, float]:
    rot = _quat_matrix(quat_wxyz)
    out = rot @ np.asarray(point, dtype=np.float64)
    return (float(out[0]), float(out[1]), float(out[2]))


def _as_quat(value: str | None) -> tuple[float, float, float, float]:
    if not value:
        return (1.0, 0.0, 0.0, 0.0)
    parts = value.split()
    if len(parts) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    try:
        quat = tuple(float(part) for part in parts)
    except ValueError:
        return (1.0, 0.0, 0.0, 0.0)
    length = math.sqrt(sum(value * value for value in quat))
    if length <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (quat[0] / length, quat[1] / length, quat[2] / length, quat[3] / length)


def _vec_add(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> tuple[float, float, float]:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


@lru_cache(maxsize=512)
def _read_stl_vertices(path: Path) -> tuple[tuple[float, float, float], ...]:
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
            return tuple(vertices)

    for line in data.decode(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("vertex "):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue
    return tuple(vertices)


def _resolve_mesh_path(base_path: Path, filename: str) -> Path | None:
    if filename.startswith("package://"):
        return None
    path = Path(filename)
    if path.is_absolute():
        return path
    return (base_path.parent / path).resolve()


def _box_points(size: tuple[float, float, float]) -> np.ndarray:
    x, y, z = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    return np.array(
        [
            [sx * x, sy * y, sz * z]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def _points_from_urdf_geometry(urdf_path: Path, geometry: ET.Element) -> np.ndarray:
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.attrib.get("filename")
        mesh_path = _resolve_mesh_path(urdf_path, filename) if filename else None
        if mesh_path is None or not mesh_path.exists():
            return np.zeros((0, 3), dtype=np.float64)
        points = np.asarray(_read_stl_vertices(mesh_path), dtype=np.float64)
        scale = _as_vec3(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
        return points * np.asarray(scale, dtype=np.float64)

    box = geometry.find("box")
    if box is not None:
        return _box_points(_as_vec3(box.attrib.get("size"), (0.0, 0.0, 0.0)))

    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.attrib.get("radius", 0.0))
        return _box_points((radius * 2.0, radius * 2.0, radius * 2.0))

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.attrib.get("radius", 0.0))
        length = float(cylinder.attrib.get("length", 0.0))
        return _box_points((radius * 2.0, radius * 2.0, length))

    return np.zeros((0, 3), dtype=np.float64)


def _urdf_link_bbox_points(urdf_path: Path, link_name: str) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        return np.zeros((0, 3), dtype=np.float64)

    all_points: list[np.ndarray] = []
    geometry_nodes = list(link.findall("collision")) or list(link.findall("visual"))
    for node in geometry_nodes:
        geometry = node.find("geometry")
        if geometry is None:
            continue
        points = _points_from_urdf_geometry(urdf_path, geometry)
        if len(points) == 0:
            continue
        origin = node.find("origin")
        pos = _as_vec3(origin.attrib.get("xyz"), (0.0, 0.0, 0.0)) if origin is not None else (0.0, 0.0, 0.0)
        rpy = _as_vec3(origin.attrib.get("rpy"), (0.0, 0.0, 0.0)) if origin is not None else (0.0, 0.0, 0.0)
        all_points.append(_transform_points(points, pos, _rpy_matrix(rpy)))

    if not all_points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(all_points, axis=0)


def _mjcf_mesh_assets(mjcf_path: Path, root: ET.Element) -> dict[str, tuple[Path, tuple[float, float, float]]]:
    compiler = root.find("compiler")
    mesh_dir = compiler.attrib.get("meshdir") if compiler is not None else None
    assets: dict[str, tuple[Path, tuple[float, float, float]]] = {}
    for mesh in root.findall("./asset/mesh"):
        name = mesh.attrib.get("name")
        filename = mesh.attrib.get("file")
        if not name or not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = (mjcf_path.parent / mesh_dir / path).resolve() if mesh_dir else (mjcf_path.parent / path).resolve()
        scale = _as_vec3(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
        assets[name] = (path, scale)
    return assets


def _mjcf_link_bbox_points(mjcf_path: Path, link_name: str) -> np.ndarray:
    root = ET.parse(mjcf_path).getroot()
    assets = _mjcf_mesh_assets(mjcf_path, root)
    all_points: list[np.ndarray] = []
    for body in root.findall(f".//body[@name='{link_name}']"):
        for geom in body.findall("geom"):
            points = np.zeros((0, 3), dtype=np.float64)
            if geom.attrib.get("type", "mesh") == "mesh" and geom.attrib.get("mesh") in assets:
                mesh_path, scale = assets[geom.attrib["mesh"]]
                if mesh_path.exists():
                    points = np.asarray(_read_stl_vertices(mesh_path), dtype=np.float64) * np.asarray(scale, dtype=np.float64)
            elif geom.attrib.get("type") == "box":
                size = _as_vec3(geom.attrib.get("size"), (0.0, 0.0, 0.0))
                points = _box_points((size[0] * 2.0, size[1] * 2.0, size[2] * 2.0))
            if len(points) == 0:
                continue
            pos = _as_vec3(geom.attrib.get("pos"), (0.0, 0.0, 0.0))
            quat_values = geom.attrib.get("quat", "1 0 0 0").split()
            rot = np.eye(3, dtype=np.float64)
            if len(quat_values) == 4:
                try:
                    rot = _quat_matrix(tuple(float(value) for value in quat_values))  # type: ignore[arg-type]
                except ValueError:
                    rot = np.eye(3, dtype=np.float64)
            all_points.append(_transform_points(points, pos, rot))
    if not all_points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(all_points, axis=0)


def _link_points(robot_name: str, link_name: str) -> tuple[np.ndarray, str | None]:
    urdf_path = robot_registry_parser.get_profile_path(robot_name, "urdf_path")
    if urdf_path is not None and Path(urdf_path).exists():
        points = _urdf_link_bbox_points(Path(urdf_path), link_name)
        if len(points) > 0:
            return points, "urdf_bbox"

    mjcf_path = robot_registry_parser.get_profile_path(robot_name, "mjcf_path")
    if mjcf_path is not None and Path(mjcf_path).exists():
        points = _mjcf_link_bbox_points(Path(mjcf_path), link_name)
        if len(points) > 0:
            return points, "mjcf_bbox"

    return np.zeros((0, 3), dtype=np.float64), None


def _urdf_link_world_poses(urdf_path: Path, link_names: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ET.parse(urdf_path).getroot()
    child_joints: dict[str | None, list[ET.Element]] = {}
    child_links = set()
    for joint in root.findall("./joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link") if parent is not None else None
        child_name = child.attrib.get("link") if child is not None else None
        if child_name:
            child_links.add(child_name)
        child_joints.setdefault(parent_name, []).append(joint)

    roots = [
        link.attrib["name"]
        for link in root.findall("./link")
        if link.attrib.get("name") and link.attrib["name"] not in child_links
    ]
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def walk(link_name: str, parent_pos: np.ndarray, parent_rot: np.ndarray) -> None:
        poses[link_name] = (parent_pos, parent_rot)
        for joint in child_joints.get(link_name, []):
            child = joint.find("child")
            child_name = child.attrib.get("link") if child is not None else None
            if not child_name:
                continue
            origin = joint.find("origin")
            xyz = _as_vec3(origin.attrib.get("xyz"), (0.0, 0.0, 0.0)) if origin is not None else (0.0, 0.0, 0.0)
            rpy = _as_vec3(origin.attrib.get("rpy"), (0.0, 0.0, 0.0)) if origin is not None else (0.0, 0.0, 0.0)
            child_pos = parent_pos + parent_rot @ np.asarray(xyz, dtype=np.float64)
            child_rot = parent_rot @ _rpy_matrix(rpy)
            walk(child_name, child_pos, child_rot)

    for root_link in roots:
        walk(root_link, np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))
    return {name: poses[name] for name in link_names if name in poses}


def _mjcf_link_world_poses(
    robot_name: str,
    mjcf_path: Path,
    link_names: set[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return {}
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        joint_positions_rad = robot_registry_parser._load_robot_t_pose_joint_positions(robot_name)
    except Exception:
        joint_positions_rad = {}

    def walk(
        body: ET.Element,
        parent_pos: tuple[float, float, float],
        parent_quat: tuple[float, float, float, float],
    ) -> None:
        body_name = body.attrib.get("name")
        local_pos = _as_vec3(body.attrib.get("pos"), (0.0, 0.0, 0.0))
        local_quat = _as_quat(body.attrib.get("quat"))
        body_pos = _vec_add(parent_pos, _quat_rotate(parent_quat, local_pos))
        body_quat = _quat_mul(parent_quat, local_quat)
        try:
            body_pos, body_quat = robot_registry_parser._apply_mjcf_joint_pose(
                body,
                body_pos,
                body_quat,
                joint_positions_rad,
            )
        except Exception:
            pass
        if body_name in link_names:
            poses[body_name] = (np.asarray(body_pos, dtype=np.float64), _quat_matrix(body_quat))
        for child in body.findall("body"):
            walk(child, body_pos, body_quat)

    for body in worldbody.findall("body"):
        walk(body, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    return poses


def _link_world_poses(robot_name: str, link_names: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    urdf_path = robot_registry_parser.get_profile_path(robot_name, "urdf_path")
    if urdf_path is not None and Path(urdf_path).exists():
        poses = _urdf_link_world_poses(Path(urdf_path), link_names)
        if poses:
            return poses
    mjcf_path = robot_registry_parser.get_profile_path(robot_name, "mjcf_path")
    if mjcf_path is not None and Path(mjcf_path).exists():
        return _mjcf_link_world_poses(robot_name, Path(mjcf_path), link_names)
    return {}


def _axis_index(axis: str, fallback: str) -> tuple[int, float]:
    axis = axis.strip().lower()
    if len(axis) == 2 and axis[0] in "+-" and axis[1] in "xyz":
        return {"x": 0, "y": 1, "z": 2}[axis[1]], 1.0 if axis[0] == "+" else -1.0
    fallback = fallback.strip().lower()
    return {"x": 0, "y": 1, "z": 2}[fallback[-1]], 1.0 if fallback[0] == "+" else -1.0


def _anchors_from_bbox(
    points: np.ndarray,
    *,
    side: str,
    forward_local: np.ndarray | None = None,
    lateral_local: np.ndarray | None = None,
    up_local: np.ndarray | None = None,
) -> dict[str, list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    extents = maxs - mins

    if up_local is None:
        up_idx = int(np.argmin(extents))
        up_local = np.eye(3, dtype=np.float64)[up_idx]
    up = _normalize_vec(up_local, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))

    if forward_local is None:
        candidates = [idx for idx in range(3) if abs(float(np.dot(np.eye(3)[idx], up))) < 0.75]
        if not candidates:
            candidates = [0, 1, 2]
        forward_idx = max(candidates, key=lambda idx: extents[idx])
        forward_local = np.eye(3, dtype=np.float64)[forward_idx]
    forward = _orthogonalize(forward_local, up, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))

    if lateral_local is None:
        lateral_local = np.cross(up, forward)
    lateral = _orthogonalize(lateral_local, up, np.cross(up, forward))
    lateral = _orthogonalize(lateral, forward, np.cross(up, forward))

    def support(direction: np.ndarray, sign: float) -> np.ndarray:
        direction = _normalize_vec(direction)
        dots = points @ direction
        value = float(np.max(dots) if sign >= 0.0 else np.min(dots))
        return center + direction * (value - float(center @ direction))

    sole_center = support(up, -1.0)

    def point(direction: np.ndarray, sign: float) -> list[float]:
        values = support(up, -1.0)
        axis_point = support(direction, sign)
        direction = _normalize_vec(direction)
        values += direction * float((axis_point - values) @ direction)
        return [float(f"{value:.8g}") for value in values.tolist()]

    inner_sign = 1.0 if side == "left" else -1.0
    outer_sign = -inner_sign

    return {
        "sole_center": [float(f"{value:.8g}") for value in sole_center.tolist()],
        "toe": point(forward, 1.0),
        "heel": point(forward, -1.0),
        "inner_edge": point(lateral, inner_sign),
        "outer_edge": point(lateral, outer_sign),
    }


def _proportional_dimensions(robot_name: str) -> tuple[float, float, float] | None:
    if robot_registry_parser.get_robot_profile(robot_name) is None:
        return None
    height = robot_registry_parser.get_robot_model_height(robot_name)
    if not math.isfinite(height) or height <= 0.0:
        return None
    return (
        height * _PROPORTIONAL_FOOT_LENGTH_RATIO,
        height * _PROPORTIONAL_FOOT_WIDTH_RATIO,
        height * _PROPORTIONAL_SOLE_Z_RATIO,
    )


def _manual_size_has_concrete(manual: dict[str, Any]) -> bool:
    return any(scalar_to_float(manual.get(field_name)) is not None for field_name in ("length", "width", "sole_z"))


def _anchors_from_manual_size(
    side: str,
    manual: dict[str, Any],
    dimensions: tuple[float, float, float],
) -> dict[str, list[float]]:
    length = scalar_to_float(manual.get("length")) or dimensions[0]
    width = scalar_to_float(manual.get("width")) or dimensions[1]
    sole_z = scalar_to_float(manual.get("sole_z")) or dimensions[2]
    toe_x = length * 0.55
    heel_x = -length * 0.45
    inner_y = -width * 0.5 if side == "left" else width * 0.5
    outer_y = width * 0.5 if side == "left" else -width * 0.5
    return {
        "sole_center": [0.0, 0.0, sole_z],
        "toe": [toe_x, 0.0, sole_z],
        "heel": [heel_x, 0.0, sole_z],
        "inner_edge": [0.0, inner_y, sole_z],
        "outer_edge": [0.0, outer_y, sole_z],
    }


def _anchors_from_manual_anchors(manual_anchors: dict[str, Any]) -> dict[str, list[float]] | None:
    anchors: dict[str, list[float]] = {}
    for anchor_name in ANCHOR_NAMES:
        value = manual_anchors.get(anchor_name)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            anchors[anchor_name] = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return anchors


def _is_disabled(value: Any) -> bool:
    return value is None or is_false(value) or (isinstance(value, str) and value.strip().lower() == "disabled")


def _is_missing_or_auto(mapping: dict[str, Any], key: str) -> bool:
    return key not in mapping or mapping.get(key) is None or is_auto(mapping.get(key))


def _bbox_extent_directions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extents = points.max(axis=0) - points.min(axis=0)
    up_idx = int(np.argmin(extents))
    forward_idx = int(np.argmax(extents))
    if forward_idx == up_idx:
        forward_idx = 0 if up_idx != 0 else 1
    lateral_idx = next(idx for idx in (0, 1, 2) if idx not in {up_idx, forward_idx})
    basis = np.eye(3, dtype=np.float64)
    return basis[forward_idx], basis[lateral_idx], basis[up_idx]


def _local_anchor_directions(
    *,
    robot_name: str,
    points: np.ndarray,
    side: str,
    link_name: str,
    links: dict[str, Any],
    foot_profile: dict[str, Any],
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox_forward, bbox_lateral, bbox_up = _bbox_extent_directions(points)
    forward_local = bbox_forward
    lateral_local = bbox_lateral
    up_local = bbox_up

    link_names = {
        name
        for name in (
            links.get("left_foot"),
            links.get("right_foot"),
            links.get("pelvis"),
            links.get("chest"),
        )
        if isinstance(name, str) and name
    }
    poses = _link_world_poses(robot_name, link_names)
    left_link = links.get("left_foot")
    right_link = links.get("right_foot")
    if isinstance(left_link, str) and isinstance(right_link, str) and left_link in poses and right_link in poses and link_name in poses:
        left_pos, _ = poses[left_link]
        right_pos, _ = poses[right_link]
        _, foot_rot = poses[link_name]
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        world_lateral = right_pos - left_pos
        world_lateral[2] = 0.0
        if np.linalg.norm(world_lateral) > 1e-6:
            world_lateral = _normalize_vec(world_lateral)
            world_forward = _normalize_vec(np.cross(world_lateral, world_up), np.asarray([1.0, 0.0, 0.0]))
            world_forward_hint = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            if float(np.dot(world_forward, world_forward_hint)) < 0.0:
                world_forward = -world_forward
            forward_local = foot_rot.T @ world_forward
            lateral_local = foot_rot.T @ world_lateral
            up_local = foot_rot.T @ world_up
        else:
            warnings.append(f"{side} foot axis inference used bbox extents; left/right foot separation was too small.")
    else:
        warnings.append(f"{side} foot axis inference used bbox extents; default foot world poses were unavailable.")

    if not _is_missing_or_auto(foot_profile, "up_axis"):
        up_local = _axis_vector(str(foot_profile["up_axis"]), "+z")
    if not _is_missing_or_auto(foot_profile, "forward_axis"):
        forward_local = _axis_vector(str(foot_profile["forward_axis"]), "+x")

    up_local = _normalize_vec(up_local, bbox_up)
    forward_local = _orthogonalize(forward_local, up_local, bbox_forward)
    lateral_local = _orthogonalize(lateral_local, up_local, bbox_lateral)
    lateral_local = _orthogonalize(lateral_local, forward_local, np.cross(up_local, forward_local))
    return forward_local, lateral_local, up_local


def generate_virtual_sole_anchors(
    robot_name: str,
    resolved_capability: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate soft sole-anchor offsets from foot link bboxes or manual size."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    if profile is None:
        profile, _ = load_capability_profile(robot_name)

    foot_profile = profile.get("foot", {}) if isinstance(profile.get("foot"), dict) else {}
    if "sole_anchor_mode" in foot_profile:
        mode_value = foot_profile.get("sole_anchor_mode")
    else:
        mode_value = resolved_capability.get("foot_anchor_mode", "auto")
    if _is_disabled(mode_value):
        return {"enabled": False, "source": "disabled", "anchors": list(ANCHOR_NAMES)}
    mode = str(mode_value).strip().lower() if mode_value is not None else "auto"
    if mode == "auto":
        mode = "bbox"
    generated_modes = {"bbox", "mesh", "manual_anchors", "manual", "manual_size"}

    manual_size = foot_profile.get("manual_size", {}) if isinstance(foot_profile.get("manual_size"), dict) else {}
    manual_anchors = (
        foot_profile.get("manual_anchors", {})
        if isinstance(foot_profile.get("manual_anchors"), dict)
        else {}
    )
    result: dict[str, Any] = {
        "enabled": True,
        "source": None,
        "anchors": list(ANCHOR_NAMES),
        "left": None,
        "right": None,
        "warnings": [],
    }

    for side in ("left", "right"):
        links = resolved_capability.get("links", {}) if isinstance(resolved_capability.get("links"), dict) else {}
        link_name = links.get(f"{side}_foot")
        side_manual = manual_size.get(side, {}) if isinstance(manual_size.get(side), dict) else {}
        side_manual_anchors = manual_anchors.get(side, {}) if isinstance(manual_anchors.get(side), dict) else {}
        anchors = None
        source = None

        if side_manual_anchors or mode == "manual_anchors":
            anchors = _anchors_from_manual_anchors(side_manual_anchors)
            if anchors is not None:
                source = "manual_anchors"
            elif mode == "manual_anchors":
                result["warnings"].append(
                    f"{side} foot manual_anchors are incomplete; falling back to generated anchors."
                )

        if anchors is None and _manual_size_has_concrete(side_manual):
            dimensions = _proportional_dimensions(robot_name)
            if dimensions is not None:
                anchors = _anchors_from_manual_size(side, side_manual, dimensions)
                source = "manual_size"
            else:
                result["warnings"].append(
                    f"{side} foot manual_size has overrides, but robot model height could not be estimated."
                )

        if anchors is None and mode in generated_modes and link_name:
            points, source = _link_points(robot_name, link_name)
            if len(points) > 0:
                forward_local, lateral_local, up_local = _local_anchor_directions(
                    robot_name=robot_name,
                    points=points,
                    side=side,
                    link_name=link_name,
                    links=links,
                    foot_profile=foot_profile,
                    warnings=result["warnings"],
                )
                anchors = _anchors_from_bbox(
                    points,
                    side=side,
                    forward_local=forward_local,
                    lateral_local=lateral_local,
                    up_local=up_local,
                )

        if anchors is None and mode in generated_modes:
            dimensions = _proportional_dimensions(robot_name)
            if dimensions is not None:
                anchors = _anchors_from_manual_size(side, side_manual, dimensions)
                source = "proportional_fallback"
                result["warnings"].append(
                    f"{side} foot anchors used proportional fallback from robot model height; "
                    f"no usable bbox was found for link {link_name!r}."
                )
            else:
                result["warnings"].append(
                    f"{side} foot anchors disabled; no manual anchors, usable bbox, or robot model scale was available."
                )
        result[side] = anchors
        if result["source"] is None:
            result["source"] = source

    if result["left"] is None or result["right"] is None:
        result["enabled"] = False
        result["source"] = "disabled"
    if result["source"] is None:
        result["source"] = "disabled" if not result["enabled"] else "unknown"
    return result


def generate_virtual_sole_anchors_for_robot(
    robot_name: str,
    *,
    capability_profile_path: str | Path | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve robot capability data and generate virtual sole anchors."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    loaded_profile_path: Path | None = None
    if profile is None:
        profile, loaded_profile_path = load_capability_profile(robot_name, capability_profile_path)
    profile_source = capability_profile_path if capability_profile_path is not None else loaded_profile_path
    resolution = resolve_capability_profile(
        robot_name,
        profile=profile,
        profile_path=str(profile_source) if profile_source is not None else None,
    )
    resolution.raise_for_errors()
    return generate_virtual_sole_anchors(robot_name, resolution.payload, profile=profile)


__all__ = [
    "ANCHOR_NAMES",
    "find_capability_profile_path",
    "generate_virtual_sole_anchors",
    "generate_virtual_sole_anchors_for_robot",
    "load_capability_profile",
    "resolve_capability_profile",
]
