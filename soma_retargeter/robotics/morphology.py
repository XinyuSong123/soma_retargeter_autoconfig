# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soma_retargeter.robotics.retarget_profile import JointDofInfo, file_sha256


@dataclass(frozen=True)
class BodyInfo:
    body_name: str
    parent_body_name: str | None
    local_position: np.ndarray
    world_position: np.ndarray
    world_rotation_wxyz: np.ndarray


@dataclass(frozen=True)
class MorphologyAnalysis:
    mjcf_path: str | None
    robot_fingerprint: str
    body_names: list[str]
    bodies: dict[str, BodyInfo]
    joint_dofs: list[JointDofInfo]
    joints_by_body: dict[str, list[JointDofInfo]]
    warnings: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "mjcf_path": self.mjcf_path,
            "body_count": len(self.body_names),
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
        return MorphologyAnalysis(None, "missing-mjcf", [], {}, [], {}, [{"code": "missing_mjcf_path"}])

    path = Path(mjcf_path)
    digest = file_sha256(path)
    if digest is None:
        return MorphologyAnalysis(str(path), "missing-mjcf", [], {}, [], {}, [{"code": "mjcf_not_found", "path": str(path)}])

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return MorphologyAnalysis(str(path), digest, [], {}, [], {}, [{"code": "mjcf_parse_error", "message": str(exc)}])

    body_names: list[str] = []
    bodies: dict[str, BodyInfo] = {}
    joint_dofs: list[JointDofInfo] = []
    joints_by_body: dict[str, list[JointDofInfo]] = {}

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

    return MorphologyAnalysis(str(path), digest, body_names, bodies, joint_dofs, joints_by_body, warnings)
