"""Source semantic rest-frame loading for SOMA pose JSON assets."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .spatial import quat_xyzw_to_matrix, transform


DEFAULT_SOMA_REST_POSE = Path("soma_retargeter/assets/standard_human_pos/human_t_pose.json")
DEFAULT_SOMA_SKELETON_SPEC = Path("soma_retargeter/configs/soma/soma_skeleton_spec.json")
DEFAULT_SEMANTIC_NAMES = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")


def load_soma_source_rest_frames(
    pose_path: str | Path = DEFAULT_SOMA_REST_POSE,
    skeleton_spec_path: str | Path = DEFAULT_SOMA_SKELETON_SPEC,
    semantic_names: tuple[str, ...] = DEFAULT_SEMANTIC_NAMES,
) -> tuple[dict[str, np.ndarray], str]:
    pose_path = Path(pose_path)
    skeleton_spec_path = Path(skeleton_spec_path)
    pose = json.loads(pose_path.read_text())
    spec = json.loads(skeleton_spec_path.read_text())
    parent = {joint["name"]: joint["parent"] for joint in spec["joints"]}
    locals_by_name = {
        name: transform(
            entry["translation_m"],
            quat_xyzw_to_matrix(entry["rotation_xyzw"]),
        )
        for name, entry in pose["local_joint_transforms"].items()
    }
    root_payload = pose.get("root_transform", {})
    root_world = transform(
        root_payload.get("translation_m", [0.0, 0.0, 0.0]),
        quat_xyzw_to_matrix(root_payload.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])),
    )
    world_cache: dict[str, np.ndarray] = {}
    visiting: set[str] = set()

    def world_transform(name: str) -> np.ndarray:
        if name in world_cache:
            return world_cache[name]
        if name in visiting:
            raise ValueError(f"cycle in source rest skeleton at joint {name!r}")
        if name not in locals_by_name:
            raise ValueError(f"missing local source rest transform for joint {name!r}")
        if name not in parent:
            raise ValueError(f"missing source skeleton parent entry for joint {name!r}")
        visiting.add(name)
        local = locals_by_name[name]
        parent_name = parent.get(name)
        if parent_name is None:
            out = root_world @ local
        else:
            out = world_transform(parent_name) @ local
        visiting.remove(name)
        world_cache[name] = out
        return out

    missing_semantics = [name for name in semantic_names if name not in locals_by_name]
    if missing_semantics:
        raise ValueError(f"missing source rest semantic transforms: {', '.join(missing_semantics)}")
    frames = {name: world_transform(name) for name in semantic_names}
    provenance = f"soma_pose_json:{pose_path};skeleton_spec:{skeleton_spec_path}"
    return frames, provenance
