"""Full-chain path discovery for semantic relative tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .model_adapter import MuJoCoRuntimeModelAdapter, SemanticSite


TASKS = {
    "torso": ("Hips", "Chest"),
    "left_hand": ("Chest", "LeftHand"),
    "right_hand": ("Chest", "RightHand"),
    "left_foot": ("Hips", "LeftFoot"),
    "right_foot": ("Hips", "RightFoot"),
}


@dataclass(frozen=True)
class KinematicPath:
    task: str
    reference: str
    target: str
    reference_body: str
    target_body: str
    lca_body: str
    body_path: list[str]
    active_velocity_coordinates: list[int]
    coordinate_labels: list[str]
    joint_types: list[str]
    limit_source: str = "runtime_model"

    def to_json(self) -> dict:
        return self.__dict__.copy()


def discover_paths(adapter: MuJoCoRuntimeModelAdapter, sites: dict[str, SemanticSite]) -> dict[str, KinematicPath]:
    paths: dict[str, KinematicPath] = {}
    for task, (ref_name, target_name) in TASKS.items():
        if ref_name not in sites or target_name not in sites:
            continue
        ref = sites[ref_name]
        target = sites[target_name]
        active = adapter.active_velocity_coordinates(ref, target)
        infos = adapter.coordinate_limits(active)
        paths[task] = KinematicPath(
            task=task,
            reference=ref_name,
            target=target_name,
            reference_body=ref.body_name,
            target_body=target.body_name,
            lca_body=adapter.lca_body(ref.body_name, target.body_name),
            body_path=adapter.body_path(ref.body_name, target.body_name),
            active_velocity_coordinates=active,
            coordinate_labels=[i.label for i in infos],
            joint_types=[i.joint_type for i in infos],
        )
    return paths

