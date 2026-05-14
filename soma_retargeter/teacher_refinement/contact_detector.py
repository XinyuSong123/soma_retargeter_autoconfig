# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def detect_foot_contacts(
    frames: list[dict[str, Any]],
    *,
    foot_height_threshold_m: float = 0.035,
    foot_speed_threshold_mps: float = 0.12,
) -> list[dict[str, bool]]:
    """Detect contacts from semantic foot trajectories.

    This helper is intentionally independent from G1 joint angles; it expects
    frame dictionaries with semantic link poses such as ``links.left_foot.pos``.
    """

    contacts: list[dict[str, bool]] = []
    previous: dict[str, tuple[float, float, float]] = {}
    previous_t: float | None = None
    for frame in frames:
        t = float(frame.get("t", len(contacts)))
        frame_contacts: dict[str, bool] = {}
        links = frame.get("links", {})
        for foot_name in ("left_foot", "right_foot"):
            foot = links.get(foot_name, {})
            pos = foot.get("pos", [0.0, 0.0, 0.0])
            if len(pos) < 3:
                frame_contacts[foot_name] = False
                continue
            xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
            speed = 0.0
            if foot_name in previous and previous_t is not None and t > previous_t:
                prev = previous[foot_name]
                dx = xyz[0] - prev[0]
                dy = xyz[1] - prev[1]
                dz = xyz[2] - prev[2]
                speed = (dx * dx + dy * dy + dz * dz) ** 0.5 / (t - previous_t)
            frame_contacts[foot_name] = xyz[2] <= foot_height_threshold_m and speed <= foot_speed_threshold_mps
            previous[foot_name] = xyz
        previous_t = t
        contacts.append(frame_contacts)
    return contacts
