import numpy as np

from soma_retargeter.robotics.v3.model_adapter import SemanticSite
from soma_retargeter.robotics.v3.rest_frames import calibrate_rest_frames
from soma_retargeter.robotics.v3.spatial import invert_transform, rotation_error, so3_exp, transform
from soma_retargeter.robotics.v3.target_builder import build_targets_from_source_semantic_frames


def _site(name: str) -> SemanticSite:
    return SemanticSite(
        semantic_name=name,
        body_name=name,
        local_position=np.zeros(3),
        local_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )


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


def _sites(transforms: dict[str, np.ndarray]) -> dict[str, SemanticSite]:
    return {name: _site(name) for name in transforms}


def _source_rest() -> dict[str, np.ndarray]:
    return {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 2.0]),
        "LeftHand": transform([0.0, 2.0, 2.0]),
        "RightHand": transform([0.0, -2.0, 2.0]),
        "LeftFoot": transform([0.0, 1.0, -2.0]),
        "RightFoot": transform([0.0, -1.0, -2.0]),
    }


def _robot_rest() -> dict[str, np.ndarray]:
    align = so3_exp(np.array([0.0, 0.0, np.pi / 2.0]))
    return {name: transform(align @ frame[:3, 3] * 0.5, align @ frame[:3, :3]) for name, frame in _source_rest().items()}


def _calibration():
    robot = _robot_rest()
    return calibrate_rest_frames(
        _RestAdapter(robot),
        _sites(robot),
        source_rest_transforms=_source_rest(),
        source_provenance="test_source_rest",
    )


def test_relative_rotation_delta_is_mapped_by_calibrated_conjugation():
    calibration = _calibration()
    source_pose = {name: frame.copy() for name, frame in calibration.source_rest_semantic_frames.items()}
    delta_h = so3_exp(np.array([0.3, 0.0, 0.0]))
    source_pose["Chest"][:3, :3] = source_pose["Hips"][:3, :3] @ delta_h

    targets = build_targets_from_source_semantic_frames(calibration, source_pose, mode="torso_delta")
    target_rel = invert_transform(targets.transforms["Hips"]) @ targets.transforms["Chest"]
    robot_rel0 = (
        invert_transform(calibration.robot_neutral_site_transforms["Hips"])
        @ calibration.robot_neutral_site_transforms["Chest"]
    )
    alignment = calibration.edge_alignment_rotations["torso"]
    expected_delta = alignment @ delta_h @ alignment.T
    expected_rel_rotation = robot_rel0[:3, :3] @ expected_delta

    np.testing.assert_allclose(target_rel[:3, :3], expected_rel_rotation, atol=1e-12)


def test_root_horizontal_translation_uses_leg_length_ratio():
    calibration = _calibration()
    source_pose = {name: frame.copy() for name, frame in calibration.source_rest_semantic_frames.items()}
    source_pose["Hips"][:3, 3] += np.array([0.4, 0.0, 0.0])

    targets = build_targets_from_source_semantic_frames(calibration, source_pose, mode="root_translation")
    root_delta = (
        calibration.robot_neutral_site_transforms["Hips"][:3, :3].T
        @ (targets.transforms["Hips"][:3, 3] - calibration.robot_neutral_site_transforms["Hips"][:3, 3])
    )

    np.testing.assert_allclose(root_delta, [0.2, 0.0, 0.0], atol=1e-12)


def test_common_source_yaw_preserves_local_limb_articulation():
    calibration = _calibration()
    yaw = transform([0.0, 0.0, 0.0], so3_exp(np.array([0.0, 0.0, 0.4])))
    source_pose = {name: yaw @ frame for name, frame in calibration.source_rest_semantic_frames.items()}

    neutral = build_targets_from_source_semantic_frames(
        calibration,
        calibration.source_rest_semantic_frames,
        mode="neutral",
    )
    moved = build_targets_from_source_semantic_frames(calibration, source_pose, mode="global_yaw")

    for parent, child in (("Hips", "Chest"), ("Chest", "LeftHand"), ("Chest", "RightHand")):
        neutral_rel = invert_transform(neutral.transforms[parent]) @ neutral.transforms[child]
        moved_rel = invert_transform(moved.transforms[parent]) @ moved.transforms[child]
        np.testing.assert_allclose(moved_rel[:3, 3], neutral_rel[:3, 3], atol=1e-12)
        assert rotation_error(moved_rel[:3, :3], neutral_rel[:3, :3]) < 1e-12
