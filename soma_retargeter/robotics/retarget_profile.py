# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
COMPILER_VERSION = "2.1.2"
QUATERNION_ORDER = "xyzw"


@dataclass(frozen=True)
class JointDofInfo:
    joint_name: str
    body_name: str
    parent_body_name: str | None
    joint_type: str
    axis_local: np.ndarray
    axis_world_rest: np.ndarray
    q_index: int
    dof_index: int
    lower: float
    upper: float
    neutral: float
    continuous: bool


@dataclass(frozen=True)
class SemanticSite:
    semantic_name: str
    body_name: str
    local_position: np.ndarray
    local_rotation_xyzw: np.ndarray
    source: str
    confidence: float
    orientation_supported: bool


@dataclass(frozen=True)
class KinematicChainProfile:
    name: str
    root_body: str
    tip_site: str
    joint_names: list[str]
    segment_lengths: list[float]
    total_length: float
    translational_basis: np.ndarray
    rotational_basis: np.ndarray
    translational_rank: int
    rotational_rank: int
    singular_values_translation: list[float]
    singular_values_rotation: list[float]
    confidence: float


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    source_semantic: str
    target_site: str
    reference_site: str | None
    priority: int
    position_mask_or_basis: list[list[float]] | None
    rotation_mask_or_basis: list[list[float]] | None
    normalized_weight: float
    characteristic_length: float
    robust_loss: str
    enabled: bool
    reason: str


@dataclass(frozen=True)
class CompiledRetargetProfile:
    schema_version: int
    compiler_version: str
    quaternion_order: str
    robot_fingerprint: str
    source_skeleton_fingerprint: str
    morphology_summary: dict[str, Any]
    semantic_sites: dict[str, SemanticSite]
    chains: dict[str, KinematicChainProfile]
    rest_frame_alignment: dict[str, Any]
    segment_ratios: dict[str, float]
    tasks: list[TaskSpec]
    contact: dict[str, Any]
    collision: dict[str, Any]
    solver: dict[str, Any]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    source_config_hash: str | None = None


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot serialize non-finite float: {value!r}")
        return value
    return value


def profile_to_json_dict(profile: CompiledRetargetProfile) -> dict[str, Any]:
    payload = to_jsonable(profile)
    if not isinstance(payload, dict):
        raise TypeError("profile serialization produced a non-dict payload")
    return payload


def write_profile_json(profile: CompiledRetargetProfile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile_to_json_dict(profile), sort_keys=True, indent=2, ensure_ascii=True) + "\n")


def validate_legacy_scaler_for_v2(config: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    joint_scales = config.get("joint_scales", {})
    if not isinstance(joint_scales, dict):
        return [{"code": "invalid_joint_scales", "message": "joint_scales must be a mapping"}]

    for joint_name, raw_value in sorted(joint_scales.items()):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            warnings.append({"code": "invalid_scale", "joint": str(joint_name), "value": raw_value})
            continue
        if not math.isfinite(value) or value <= 0.0:
            warnings.append({"code": "non_positive_scale", "joint": str(joint_name), "value": value})
        elif value > 10.0:
            warnings.append({"code": "suspiciously_large_scale", "joint": str(joint_name), "value": value})
    return warnings
