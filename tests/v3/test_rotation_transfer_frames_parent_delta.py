from __future__ import annotations

import numpy as np

from soma_retargeter.robotics.v3.model_adapter import SemanticSite
from soma_retargeter.robotics.v3.rest_frames import calibrate_rest_frames
from soma_retargeter.robotics.v3.spatial import invert_transform, rotation_error, so3_exp, transform
from soma_retargeter.robotics.v3.target_builder import build_targets_from_source_semantic_frames


def _site(name: str) -> SemanticSite:
    return SemanticSite(name, name, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


class _RestAdapter:
    model_format = "test"
    nq = 1
    nv = 1

    def __init__(self, transforms: dict[str, np.ndarray]):
        self.transforms = transforms

    def neutral_q(self) -> np.ndarray:
        return np.zeros(1)

    def forward_kinematics(self, q: np.ndarray) -> object:
        return object()

    def site_transform(self, state: object, site: SemanticSite) -> np.ndarray:
        return self.transforms[site.semantic_name].copy()


def test_parent_frame_rotation_delta_handles_nonidentity_rest_frames():
    source = {
        "Hips": transform([0.0, 0.0, 0.0], so3_exp(np.array([0.2, 0.1, -0.1]))),
        "Chest": transform([0.0, 0.0, 1.0], so3_exp(np.array([0.4, -0.2, 0.3]))),
        "LeftHand": transform([0.0, 1.0, 1.0]),
        "RightHand": transform([0.0, -1.0, 1.0]),
        "LeftFoot": transform([0.0, 0.4, -1.0]),
        "RightFoot": transform([0.0, -0.4, -1.0]),
    }
    alignment = so3_exp(np.array([0.0, 0.0, 0.7]))
    robot = {name: transform(alignment @ frame[:3, 3], alignment @ frame[:3, :3]) for name, frame in source.items()}
    calibration = calibrate_rest_frames(
        _RestAdapter(robot),
        {name: _site(name) for name in robot},
        source_rest_transforms=source,
        source_provenance="nonidentity_test",
    )
    source_pose = {name: frame.copy() for name, frame in source.items()}
    delta_parent = so3_exp(np.array([0.31, -0.17, 0.09]))
    neutral_rel_source = invert_transform(source["Hips"]) @ source["Chest"]
    source_pose["Chest"][:3, :3] = source["Hips"][:3, :3] @ delta_parent @ neutral_rel_source[:3, :3]

    targets = build_targets_from_source_semantic_frames(calibration, source_pose, mode="nonidentity")
    target_rel = invert_transform(targets.transforms["Hips"]) @ targets.transforms["Chest"]
    neutral_rel_robot = invert_transform(robot["Hips"]) @ robot["Chest"]
    edge_alignment = calibration.edge_alignment_rotations["torso"]
    expected = edge_alignment @ delta_parent @ edge_alignment.T @ neutral_rel_robot[:3, :3]
    wrong_child_delta = neutral_rel_robot[:3, :3] @ (edge_alignment @ delta_parent @ edge_alignment.T)

    assert rotation_error(target_rel[:3, :3], expected) < 1e-12
    assert rotation_error(target_rel[:3, :3], wrong_child_delta) > 1e-3
