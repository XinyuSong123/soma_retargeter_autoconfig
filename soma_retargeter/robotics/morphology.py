# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct
from typing import Any

import numpy as np

from soma_retargeter.robotics.retarget_profile import JointDofInfo, file_sha256, stable_hash_payload


@dataclass(frozen=True)
class BodyInfo:
    body_name: str
    parent_body_name: str | None
    local_position: np.ndarray
    world_position: np.ndarray
    world_rotation_wxyz: np.ndarray


@dataclass(frozen=True)
class GeomInfo:
    geom_name: str
    body_name: str
    geom_type: str
    local_position: np.ndarray
    world_position: np.ndarray
    size: np.ndarray
    bounding_radius: float


@dataclass(frozen=True)
class SiteInfo:
    site_name: str
    body_name: str
    local_position: np.ndarray
    world_position: np.ndarray
    local_rotation_wxyz: np.ndarray
    world_rotation_wxyz: np.ndarray


@dataclass(frozen=True)
class MorphologyAnalysis:
    mjcf_path: str | None
    robot_fingerprint: str
    body_names: list[str]
    bodies: dict[str, BodyInfo]
    geoms_by_body: dict[str, list[GeomInfo]]
    sites_by_body: dict[str, list[SiteInfo]]
    joint_dofs: list[JointDofInfo]
    joints_by_body: dict[str, list[JointDofInfo]]
    warnings: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "mjcf_path": self.mjcf_path,
            "body_count": len(self.body_names),
            "geom_count": sum(len(items) for items in self.geoms_by_body.values()),
            "site_count": sum(len(items) for items in self.sites_by_body.values()),
            "movable_joint_count": len(self.joint_dofs),
            "joint_names": [j.joint_name for j in self.joint_dofs],
        }

    def body_path(self, root_body: str, tip_body: str) -> list[str]:
        if root_body not in self.bodies or tip_body not in self.bodies:
            return []

        tip_ancestors: list[str] = []
        body: str | None = tip_body
        while body is not None:
            tip_ancestors.append(body)
            if body == root_body:
                return list(reversed(tip_ancestors))
            body = self.bodies[body].parent_body_name
        return []

    def joints_on_path(self, body_path: list[str]) -> list[JointDofInfo]:
        out: list[JointDofInfo] = []
        for body_name in body_path[1:]:
            out.extend(self.joints_by_body.get(body_name, []))
        return out


def _as_vec3(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.array(default, dtype=float)
    parts = value.split()
    if len(parts) != 3:
        return np.array(default, dtype=float)
    try:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
    except ValueError:
        return np.array(default, dtype=float)


def _as_quat_wxyz(value: str | None) -> np.ndarray:
    if not value:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    parts = value.split()
    if len(parts) != 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    try:
        quat = np.array([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
    except ValueError:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    out = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    q_vec = np.array([x, y, z], dtype=float)
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2.0 * (w * uv + uuv)


def _as_float_array(value: str | None) -> np.ndarray:
    if not value:
        return np.zeros(0, dtype=float)
    out = []
    for item in value.split():
        try:
            out.append(float(item))
        except ValueError:
            return np.zeros(0, dtype=float)
    return np.array(out, dtype=float)


def _resolve_mesh_file(xml_path: Path, mesh_file: str, mesh_dir: str | None) -> Path:
    path = Path(mesh_file)
    if path.is_absolute():
        return path
    if mesh_dir:
        return (xml_path.parent / mesh_dir / path).resolve()
    return (xml_path.parent / path).resolve()


@lru_cache(maxsize=512)
def _read_stl_vertices(path: Path, mesh_digest: str | None) -> tuple[tuple[float, float, float], ...]:
    del mesh_digest
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


@lru_cache(maxsize=512)
def _mesh_bounds(
    path: Path,
    scale: tuple[float, float, float],
    mesh_digest: str | None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if not path.exists() or path.suffix.lower() != ".stl":
        return None
    vertices = _read_stl_vertices(path, mesh_digest)
    if not vertices:
        return None
    points = np.asarray(vertices, dtype=float) * np.asarray(scale, dtype=float)
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    center = (min_corner + max_corner) * 0.5
    half_extents = (max_corner - min_corner) * 0.5
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    return center, half_extents, radius


def _geom_bounding_radius(geom_type: str, size: np.ndarray) -> float:
    if len(size) == 0:
        return 0.0
    if geom_type == "sphere":
        return float(size[0])
    if geom_type in {"capsule", "cylinder"}:
        radius = float(size[0])
        half_length = float(size[1]) if len(size) > 1 else 0.0
        return float(np.sqrt(radius * radius + half_length * half_length))
    if geom_type == "box":
        return float(np.linalg.norm(size[:3]))
    if geom_type == "ellipsoid":
        return float(np.max(size[:3]))
    return float(np.max(np.abs(size)))


def _parse_range(value: str | None, joint_type: str) -> tuple[float, float, bool]:
    if joint_type == "free":
        return -float("inf"), float("inf"), True
    if not value:
        return -float("inf"), float("inf"), True
    parts = value.split()
    if len(parts) != 2:
        return -float("inf"), float("inf"), True
    try:
        lower = float(parts[0])
        upper = float(parts[1])
    except ValueError:
        return -float("inf"), float("inf"), True
    return lower, upper, False


def analyze_mjcf_morphology(mjcf_path: str | Path | None) -> MorphologyAnalysis:
    warnings: list[dict[str, Any]] = []
    if mjcf_path is None:
        return MorphologyAnalysis(None, "missing-mjcf", [], {}, {}, {}, [], {}, [{"code": "missing_mjcf_path"}])

    path = Path(mjcf_path)
    digest = file_sha256(path)
    if digest is None:
        return MorphologyAnalysis(str(path), "missing-mjcf", [], {}, {}, {}, [], {}, [{"code": "mjcf_not_found", "path": str(path)}])

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return MorphologyAnalysis(str(path), digest, [], {}, {}, {}, [], {}, [{"code": "mjcf_parse_error", "message": str(exc)}])

    body_names: list[str] = []
    bodies: dict[str, BodyInfo] = {}
    geoms_by_body: dict[str, list[GeomInfo]] = {}
    sites_by_body: dict[str, list[SiteInfo]] = {}
    joint_dofs: list[JointDofInfo] = []
    joints_by_body: dict[str, list[JointDofInfo]] = {}
    mesh_assets: dict[str, tuple[Path, tuple[float, float, float], str | None]] = {}
    mesh_hashes: dict[str, str] = {}

    compiler = root.find("compiler")
    mesh_dir = compiler.attrib.get("meshdir") if compiler is not None else None
    for mesh in root.findall("./asset/mesh"):
        name = mesh.attrib.get("name")
        filename = mesh.attrib.get("file")
        if not name or not filename:
            continue
        scale_vec = _as_vec3(mesh.attrib.get("scale"), (1.0, 1.0, 1.0))
        scale = (float(scale_vec[0]), float(scale_vec[1]), float(scale_vec[2]))
        mesh_path = _resolve_mesh_file(path, filename, mesh_dir)
        digest = file_sha256(mesh_path)
        mesh_assets[name] = (mesh_path, scale, digest)
        if digest is None:
            warnings.append({"code": "mesh_asset_not_found", "mesh": name, "path": str(mesh_path)})
        else:
            mesh_hashes[name] = digest

    robot_fingerprint = stable_hash_payload({"mjcf": digest, "meshes": mesh_hashes})

    def walk(
        body: ET.Element,
        parent_name: str | None,
        parent_world_pos: np.ndarray,
        parent_world_rot: np.ndarray,
    ) -> None:
        body_name = body.attrib.get("name", f"anonymous_body_{len(body_names)}")
        body_names.append(body_name)
        local_pos = _as_vec3(body.attrib.get("pos"), (0.0, 0.0, 0.0))
        local_rot = _as_quat_wxyz(body.attrib.get("quat"))
        world_pos = parent_world_pos + _quat_rotate_wxyz(parent_world_rot, local_pos)
        world_rot = _quat_mul_wxyz(parent_world_rot, local_rot)
        bodies[body_name] = BodyInfo(
            body_name=body_name,
            parent_body_name=parent_name,
            local_position=local_pos,
            world_position=world_pos,
            world_rotation_wxyz=world_rot,
        )
        joints_by_body[body_name] = []
        geoms_by_body[body_name] = []
        sites_by_body[body_name] = []
        for site_idx, site in enumerate(body.findall("site")):
            local_site_pos = _as_vec3(site.attrib.get("pos"), (0.0, 0.0, 0.0))
            local_site_rot = _as_quat_wxyz(site.attrib.get("quat"))
            world_site_pos = world_pos + _quat_rotate_wxyz(world_rot, local_site_pos)
            world_site_rot = _quat_mul_wxyz(world_rot, local_site_rot)
            sites_by_body[body_name].append(
                SiteInfo(
                    site_name=site.attrib.get("name", f"{body_name}_site_{site_idx}"),
                    body_name=body_name,
                    local_position=local_site_pos,
                    world_position=world_site_pos,
                    local_rotation_wxyz=local_site_rot,
                    world_rotation_wxyz=world_site_rot,
                )
            )
        for geom_idx, geom in enumerate(body.findall("geom")):
            geom_type = geom.attrib.get("type", "sphere")
            local_geom_pos = _as_vec3(geom.attrib.get("pos"), (0.0, 0.0, 0.0))
            size = _as_float_array(geom.attrib.get("size"))
            radius = _geom_bounding_radius(geom_type, size)
            if geom_type == "mesh":
                mesh_name = geom.attrib.get("mesh")
                mesh_asset = mesh_assets.get(mesh_name or "")
                bounds = _mesh_bounds(*mesh_asset) if mesh_asset is not None else None
                if bounds is None:
                    warnings.append({
                        "code": "unsupported_geom_for_collision_proxy",
                        "body": body_name,
                        "geom": geom.attrib.get("name", f"{body_name}_geom_{geom_idx}"),
                        "type": geom_type,
                        "mesh": mesh_name,
                    })
                    continue
                mesh_center, mesh_half_extents, mesh_radius = bounds
                local_geom_pos = local_geom_pos + mesh_center
                size = mesh_half_extents
                radius = mesh_radius
            world_geom_pos = world_pos + _quat_rotate_wxyz(world_rot, local_geom_pos)
            if radius <= 0.0:
                warnings.append({
                    "code": "unsupported_geom_for_collision_proxy",
                    "body": body_name,
                    "geom": geom.attrib.get("name", f"{body_name}_geom_{geom_idx}"),
                    "type": geom_type,
                })
                continue
            geoms_by_body[body_name].append(
                GeomInfo(
                    geom_name=geom.attrib.get("name", f"{body_name}_geom_{geom_idx}"),
                    body_name=body_name,
                    geom_type=geom_type,
                    local_position=local_geom_pos,
                    world_position=world_geom_pos,
                    size=size,
                    bounding_radius=radius,
                )
            )
        for joint in body.findall("joint"):
            joint_type = joint.attrib.get("type", "hinge")
            if joint_type == "free":
                continue
            joint_name = joint.attrib.get("name", f"{body_name}_joint_{len(joint_dofs)}")
            axis = _as_vec3(joint.attrib.get("axis"), (0.0, 0.0, 1.0))
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                warnings.append({"code": "degenerate_joint_axis", "joint": joint_name})
                axis = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                axis = axis / norm
            lower, upper, continuous = _parse_range(joint.attrib.get("range"), joint_type)
            neutral = 0.0 if continuous else (lower + upper) * 0.5
            info = JointDofInfo(
                joint_name=joint_name,
                body_name=body_name,
                parent_body_name=parent_name,
                joint_type=joint_type,
                axis_local=axis,
                axis_world_rest=_quat_rotate_wxyz(world_rot, axis),
                q_index=len(joint_dofs),
                dof_index=len(joint_dofs),
                lower=lower,
                upper=upper,
                neutral=neutral,
                continuous=continuous,
            )
            joint_dofs.append(info)
            joints_by_body[body_name].append(info)
        for child in body.findall("body"):
            walk(child, body_name, world_pos, world_rot)

    worldbody = root.find("worldbody")
    if worldbody is None:
        warnings.append({"code": "missing_worldbody"})
    else:
        for child in worldbody.findall("body"):
            walk(child, None, np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    return MorphologyAnalysis(
        str(path),
        robot_fingerprint,
        body_names,
        bodies,
        geoms_by_body,
        sites_by_body,
        joint_dofs,
        joints_by_body,
        warnings,
    )
