# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from soma_retargeter.teacher_refinement.contact_detector import detect_foot_contacts
from soma_retargeter.utils import io_utils


def build_semantic_teacher_record(
    *,
    motion_name: str,
    fps: float,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the semantic teacher-data JSON shape used by refinement reports."""

    contacts = detect_foot_contacts(frames)
    enriched_frames = []
    for frame, frame_contacts in zip(frames, contacts, strict=False):
        enriched = dict(frame)
        enriched["contacts"] = {
            "left_foot": bool(frame_contacts.get("left_foot", False)),
            "right_foot": bool(frame_contacts.get("right_foot", False)),
        }
        enriched_frames.append(enriched)
    return {
        "schema_version": 1,
        "motion_name": motion_name,
        "fps": float(fps),
        "frames": enriched_frames,
        "data_semantics": "semantic_link_trajectories_not_joint_angles",
    }


def save_semantic_teacher_record(path: str | Path, record: dict[str, Any]) -> Path:
    return io_utils.save_json(path, record, indent=4, ensure_ascii=False)
