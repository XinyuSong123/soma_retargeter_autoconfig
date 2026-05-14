# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soma_retargeter.robot_registry_parser as robot_registry_parser


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent_link: str | None = None
    child_link: str | None = None
    axis: tuple[float, float, float] | None = None


@dataclass
class RobotModelInfo:
    robot_name: str
    joints: dict[str, JointInfo] = field(default_factory=dict)
    links: set[str] = field(default_factory=set)
    source_paths: dict[str, str] = field(default_factory=dict)

    @property
    def actuated_dof_count(self) -> int:
        count = 0
        for joint in self.joints.values():
            if joint.joint_type in {"fixed", "free"}:
                continue
            if joint.joint_type == "ball":
                count += 3
            elif joint.joint_type == "freejoint":
                count += 0
            else:
                count += 1
        return count


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


def _vec_len(vec: tuple[float, float, float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _normalize_axis(vec: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if vec is None:
        return None
    length = _vec_len(vec)
    if length <= 1e-12:
        return None
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def _parse_urdf(path: Path, robot_name: str) -> RobotModelInfo:
    root = ET.parse(path).getroot()
    info = RobotModelInfo(robot_name=robot_name, source_paths={"urdf": str(path)})

    for link in root.findall("./link"):
        name = link.attrib.get("name")
        if name:
            info.links.add(name)

    for joint in root.findall("./joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        info.joints[name] = JointInfo(
            name=name,
            joint_type=joint.attrib.get("type", "revolute"),
            parent_link=parent.attrib.get("link") if parent is not None else None,
            child_link=child.attrib.get("link") if child is not None else None,
            axis=_normalize_axis(_as_vec3(axis.attrib.get("xyz"), (1.0, 0.0, 0.0))) if axis is not None else None,
        )
    return info


def _parse_mjcf(path: Path, robot_name: str) -> RobotModelInfo:
    root = ET.parse(path).getroot()
    info = RobotModelInfo(robot_name=robot_name, source_paths={"mjcf": str(path)})
    worldbody = root.find("worldbody")
    if worldbody is None:
        return info

    def walk_body(body: ET.Element, parent_link: str | None) -> None:
        body_name = body.attrib.get("name")
        if body_name:
            info.links.add(body_name)
        for joint in body.findall("joint"):
            name = joint.attrib.get("name")
            if not name:
                continue
            info.joints[name] = JointInfo(
                name=name,
                joint_type=joint.attrib.get("type", "hinge"),
                parent_link=parent_link,
                child_link=body_name,
                axis=_normalize_axis(_as_vec3(joint.attrib.get("axis"), (0.0, 0.0, 1.0))),
            )
        for child in body.findall("body"):
            walk_body(child, body_name)

    for body in worldbody.findall("body"):
        walk_body(body, None)
    return info


def load_robot_model_info(robot_name: str) -> RobotModelInfo:
    """Load joint/link names from a registered URDF or MJCF."""

    robot_name = robot_registry_parser.resolve_robot_name(robot_name)
    urdf_path = robot_registry_parser.get_profile_path(robot_name, "urdf_path")
    if urdf_path is not None and Path(urdf_path).exists():
        return _parse_urdf(Path(urdf_path), robot_name)

    mjcf_path = robot_registry_parser.get_profile_path(robot_name, "mjcf_path")
    if mjcf_path is not None and Path(mjcf_path).exists():
        return _parse_mjcf(Path(mjcf_path), robot_name)

    return RobotModelInfo(robot_name=robot_name)


def name_contains(name: str, *needles: str) -> bool:
    lowered = name.lower()
    return all(needle.lower() in lowered for needle in needles)


def axis_matches(axis: tuple[float, float, float] | None, axis_name: str, threshold: float = 0.7) -> bool:
    if axis is None:
        return False
    index = {"x": 0, "y": 1, "z": 2}[axis_name.lower()]
    return abs(axis[index]) >= threshold


def candidate_by_name(names: list[str], predicates: list[tuple[str, ...]]) -> str | None:
    for tokens in predicates:
        for name in names:
            if name_contains(name, *tokens):
                return name
    return None


def sorted_names(values: Any) -> list[str]:
    return sorted(str(value) for value in values)
