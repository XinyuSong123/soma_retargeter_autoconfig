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
from soma_retargeter.teacher_refinement.capability_loader import load_capability_profile
from soma_retargeter.teacher_refinement.capability_schema import ANCHOR_NAMES, is_auto, is_false, scalar_to_float


_DEFAULT_MANUAL_FOOT_LENGTH = 0.20
_DEFAULT_MANUAL_FOOT_WIDTH = 0.09
_DEFAULT_MANUAL_SOLE_Z = -0.035


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
    forward_axis: str,
    up_axis: str,
) -> dict[str, list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    extents = maxs - mins

    if is_auto(forward_axis):
        horizontal = [(0, extents[0]), (1, extents[1])]
        forward_idx = max(horizontal, key=lambda item: item[1])[0]
        forward_sign = 1.0
    else:
        forward_idx, forward_sign = _axis_index(str(forward_axis), "+x")

    up_idx, up_sign = _axis_index(str(up_axis), "+z") if not is_auto(up_axis) else (2, 1.0)
    width_candidates = [idx for idx in (0, 1, 2) if idx not in {forward_idx, up_idx}]
    width_idx = width_candidates[0] if width_candidates else (1 if forward_idx != 1 else 0)

    sole_value = mins[up_idx] if up_sign > 0.0 else maxs[up_idx]
    toe_value = maxs[forward_idx] if forward_sign > 0.0 else mins[forward_idx]
    heel_value = mins[forward_idx] if forward_sign > 0.0 else maxs[forward_idx]
    inner_value = mins[width_idx] if side == "left" else maxs[width_idx]
    outer_value = maxs[width_idx] if side == "left" else mins[width_idx]

    def point(**overrides: float) -> list[float]:
        values = center.copy()
        values[up_idx] = sole_value
        for idx, value in overrides.items():
            values[int(idx)] = value
        return [float(f"{value:.8g}") for value in values.tolist()]

    return {
        "sole_center": point(),
        "toe": point(**{str(forward_idx): toe_value}),
        "heel": point(**{str(forward_idx): heel_value}),
        "inner_edge": point(**{str(width_idx): inner_value}),
        "outer_edge": point(**{str(width_idx): outer_value}),
    }


def _anchors_from_manual(side: str, manual: dict[str, Any]) -> dict[str, list[float]]:
    length = scalar_to_float(manual.get("length")) or _DEFAULT_MANUAL_FOOT_LENGTH
    width = scalar_to_float(manual.get("width")) or _DEFAULT_MANUAL_FOOT_WIDTH
    sole_z = scalar_to_float(manual.get("sole_z")) or _DEFAULT_MANUAL_SOLE_Z
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
    mode = str(foot_profile.get("sole_anchor_mode", resolved_capability.get("foot_anchor_mode", "auto")))
    if is_false(mode) or mode == "disabled":
        return {"enabled": False, "source": "disabled", "anchors": list(ANCHOR_NAMES)}

    forward_axis = str(foot_profile.get("forward_axis", "auto"))
    up_axis = str(foot_profile.get("up_axis", "auto"))
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
        link_name = resolved_capability.get("links", {}).get(f"{side}_foot")
        side_manual = manual_size.get(side, {}) if isinstance(manual_size.get(side), dict) else {}
        side_manual_anchors = manual_anchors.get(side, {}) if isinstance(manual_anchors.get(side), dict) else {}
        anchors = None
        source = None

        if mode == "manual_anchors" or side_manual_anchors:
            anchors = _anchors_from_manual_anchors(side_manual_anchors)
            if anchors is not None:
                source = "manual_anchors"
            elif mode == "manual_anchors":
                result["warnings"].append(
                    f"{side} foot manual_anchors are incomplete; falling back to generated anchors."
                )

        if anchors is None and mode in {"auto", "bbox", "mesh"} and link_name:
            points, source = _link_points(robot_name, link_name)
            if len(points) > 0:
                anchors = _anchors_from_bbox(points, side=side, forward_axis=forward_axis, up_axis=up_axis)
        if anchors is None and mode in {"auto", "manual", "bbox", "mesh", "manual_anchors"}:
            anchors = _anchors_from_manual(side, side_manual)
            source = "manual_fallback" if mode != "manual" else "manual"
            result["warnings"].append(
                f"{side} foot anchors used manual fallback; no usable bbox was found for link {link_name!r}."
            )
        result[side] = anchors
        if result["source"] is None:
            result["source"] = source

    if result["left"] is None or result["right"] is None:
        result["enabled"] = False
        result["source"] = "disabled"
    return result
