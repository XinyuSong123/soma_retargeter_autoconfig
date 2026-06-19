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
class MorphologyAnalysis:
    mjcf_path: str | None
    robot_fingerprint: str
    body_names: list[str]
    joint_dofs: list[JointDofInfo]
    warnings: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "mjcf_path": self.mjcf_path,
            "body_count": len(self.body_names),
            "movable_joint_count": len(self.joint_dofs),
            "joint_names": [j.joint_name for j in self.joint_dofs],
        }


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
        return MorphologyAnalysis(None, "missing-mjcf", [], [], [{"code": "missing_mjcf_path"}])

    path = Path(mjcf_path)
    digest = file_sha256(path)
    if digest is None:
        return MorphologyAnalysis(str(path), "missing-mjcf", [], [], [{"code": "mjcf_not_found", "path": str(path)}])

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return MorphologyAnalysis(str(path), digest, [], [], [{"code": "mjcf_parse_error", "message": str(exc)}])

    body_names: list[str] = []
    joint_dofs: list[JointDofInfo] = []

    def walk(body: ET.Element, parent_name: str | None) -> None:
        body_name = body.attrib.get("name", f"anonymous_body_{len(body_names)}")
        body_names.append(body_name)
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
            joint_dofs.append(
                JointDofInfo(
                    joint_name=joint_name,
                    body_name=body_name,
                    parent_body_name=parent_name,
                    joint_type=joint_type,
                    axis_local=axis,
                    axis_world_rest=axis,
                    q_index=len(joint_dofs),
                    dof_index=len(joint_dofs),
                    lower=lower,
                    upper=upper,
                    neutral=neutral,
                    continuous=continuous,
                )
            )
        for child in body.findall("body"):
            walk(child, body_name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        warnings.append({"code": "missing_worldbody"})
    else:
        for child in worldbody.findall("body"):
            walk(child, None)

    return MorphologyAnalysis(str(path), digest, body_names, joint_dofs, warnings)
