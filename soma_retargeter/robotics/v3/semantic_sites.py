"""Semantic site construction for explicit Step-2 maps."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite
from .spatial import matrix_to_quat_xyzw, quat_xyzw_to_matrix, transform


REQUIRED_SEMANTICS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")


def load_semantic_map(path: str | Path) -> dict[str, str | dict]:
    data = json.loads(Path(path).read_text())
    if "ik_map" in data:
        return data["ik_map"]
    if "semantics" in data:
        return data["semantics"]
    return data


def default_rpo_semantic_map() -> dict[str, str]:
    return {
        "Hips": "base_link",
        "Chest": "torso_link",
        "LeftHand": "left_elbow_yaw_link",
        "RightHand": "right_elbow_yaw_link",
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
    }


def build_semantic_sites(
    adapter: MuJoCoRuntimeModelAdapter,
    semantic_map: dict[str, str | dict],
    *,
    foot_offsets: dict[str, list[float]] | None = None,
    hand_offsets: dict[str, list[float]] | None = None,
) -> dict[str, SemanticSite]:
    sites: dict[str, SemanticSite] = {}
    foot_offsets = foot_offsets or {}
    hand_offsets = hand_offsets or {}
    for semantic, entry in semantic_map.items():
        if isinstance(entry, str):
            body = entry
            pos = [0.0, 0.0, 0.0]
            quat = [0.0, 0.0, 0.0, 1.0]
            reason = "explicit_body"
        elif "site" in entry or "model_site" in entry:
            site_name = str(entry.get("site", entry.get("model_site")))
            body, site_pos, site_quat = adapter.model_site_frame(site_name)
            offset_pos = np.asarray(entry.get("local_position", [0.0, 0.0, 0.0]), dtype=float)
            offset_quat = np.asarray(entry.get("local_rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
            local_t = transform(site_pos, quat_xyzw_to_matrix(site_quat)) @ transform(
                offset_pos,
                quat_xyzw_to_matrix(offset_quat),
            )
            pos = local_t[:3, 3]
            quat = matrix_to_quat_xyzw(local_t[:3, :3])
            reason = "explicit_model_site" if np.allclose(offset_pos, 0.0) else "explicit_model_site_offset"
        else:
            body = str(entry["body"])
            pos = entry.get("local_position", [0.0, 0.0, 0.0])
            quat = entry.get("local_rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            reason = "explicit_site"
        if semantic in foot_offsets:
            pos = foot_offsets[semantic]
            reason = "explicit_sole_offset"
        if semantic in hand_offsets:
            pos = hand_offsets[semantic]
            reason = "explicit_distal_hand_offset"
        sites[semantic] = SemanticSite(
            semantic_name=semantic,
            body_name=adapter.resolve_body_name(body),
            local_position=np.asarray(pos, dtype=float),
            local_rotation_xyzw=np.asarray(quat, dtype=float),
            source="explicit_semantic_override",
            confidence=1.0,
            reason=reason,
        )
    return sites


def missing_required_semantics(sites: dict[str, SemanticSite]) -> list[str]:
    return [name for name in REQUIRED_SEMANTICS if name not in sites]


def infer_semantic_map_from_body_names(adapter: MuJoCoRuntimeModelAdapter) -> dict[str, str]:
    names = adapter.body_names
    lowered = {name: name.lower() for name in names}
    out: dict[str, str] = {}
    hips = _first_match(lowered, [["pelvis"], ["base", "link"], ["body", "link"], ["torso"], ["trunk"]])
    chest = _first_match(lowered, [["torso"], ["chest"], ["trunk"], ["body", "link"], ["base", "link"]])
    left_hand = _deepest_match(adapter, lowered, [["left", "hand"], ["l_", "hand"], ["gripper", "left"], ["left", "wrist"], ["l_", "wrist"], ["left", "elbow"], ["l_", "el"], ["left", "arm"], ["arm_left"]])
    right_hand = _deepest_match(adapter, lowered, [["right", "hand"], ["r_", "hand"], ["gripper", "right"], ["right", "wrist"], ["r_", "wrist"], ["right", "elbow"], ["r_", "el"], ["right", "arm"], ["arm_right"]])
    left_foot = _deepest_match(adapter, lowered, [["left", "foot"], ["l_", "foot"], ["left", "ankle"], ["l_", "ank"], ["left", "sole"], ["leg_left"], ["ll_"]])
    right_foot = _deepest_match(adapter, lowered, [["right", "foot"], ["r_", "foot"], ["right", "ankle"], ["r_", "ank"], ["right", "sole"], ["leg_right"], ["lr_"]])
    if hips is None and {"ll_hr", "lr_hr"} & set(lowered):
        hips = "world"
    if chest is None and hips == "world":
        chest = "world"
    for semantic, value in [
        ("Hips", hips),
        ("Chest", chest),
        ("LeftHand", left_hand),
        ("RightHand", right_hand),
        ("LeftFoot", left_foot),
        ("RightFoot", right_foot),
    ]:
        if value:
            out[semantic] = value
    return out


def _first_match(lowered: dict[str, str], patterns: list[list[str]]) -> str | None:
    for pattern in patterns:
        for original, low in lowered.items():
            if all(part in low for part in pattern):
                return original
    return None


def _deepest_match(adapter: MuJoCoRuntimeModelAdapter, lowered: dict[str, str], patterns: list[list[str]]) -> str | None:
    best: tuple[int, str] | None = None
    for pattern in patterns:
        for original, low in lowered.items():
            if all(part in low for part in pattern):
                depth = len(adapter.body_path("world", original)) if "world" in adapter.body_names else len(original)
                if best is None or depth > best[0]:
                    best = (depth, original)
        if best is not None:
            return best[1]
    return None
