import numpy as np

from soma_retargeter.robotics.v3.calibration_validation import validate_neutral_reconstruction
from soma_retargeter.robotics.v3.model_adapter import SemanticSite
from soma_retargeter.robotics.v3.rest_frames import calibrate_rest_frames
from soma_retargeter.robotics.v3.spatial import so3_exp, transform


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


def _robot_rest() -> dict[str, np.ndarray]:
    return {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 1.0]),
        "LeftHand": transform([0.0, 1.0, 1.0]),
        "RightHand": transform([0.0, -1.0, 1.0]),
        "LeftFoot": transform([0.0, 0.5, -1.0]),
        "RightFoot": transform([0.0, -0.5, -1.0]),
    }


def _source_rest() -> dict[str, np.ndarray]:
    return {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 2.0]),
        "LeftHand": transform([0.0, 2.0, 2.0]),
        "RightHand": transform([0.0, -2.0, 2.0]),
        "LeftFoot": transform([0.0, 1.0, -2.0]),
        "RightFoot": transform([0.0, -1.0, -2.0]),
    }


def test_calibration_records_independent_neutral_errors_and_wahba_sources():
    robot = _robot_rest()
    calibration = calibrate_rest_frames(
        _RestAdapter(robot),
        _sites(robot),
        source_rest_transforms=_source_rest(),
        source_provenance="test_source_rest",
    )

    assert set(calibration.neutral_position_errors) == set(robot)
    assert calibration.max_position_error < 1e-12
    assert calibration.max_orientation_error < 1e-12
    assert calibration.root_horizontal_scale == 0.5
    assert calibration.vertical_root_scale == 0.5
    assert all(source.startswith("conditioned_edge_frame:") for source in calibration.edge_frame_sources.values())
    assert all(np.isfinite(value) for value in calibration.edge_conditioning.values())

    validation = validate_neutral_reconstruction(calibration)
    assert validation.passed
    assert validation.source == "independent_target_builder_vs_runtime_neutral_fk"
    assert validation.max_position_error < 1e-12
    assert validation.max_orientation_error < 1e-12


def test_degenerate_calibration_records_fallbacks_and_reduces_confidence():
    robot = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 0.0]),
        "LeftHand": transform([1.0, 0.0, 0.0]),
    }
    source = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 1.0]),
        "LeftHand": transform([0.0, 0.0, 1.0]),
    }

    calibration = calibrate_rest_frames(
        _RestAdapter(robot),
        _sites(robot),
        source_rest_transforms=source,
        source_provenance="test_source_rest",
    )

    assert "torso:zero_length_robot_edge" in calibration.fallbacks
    assert "left_arm:zero_length_source_edge" in calibration.fallbacks
    assert calibration.confidence < 1.0
    assert "torso" not in calibration.edge_alignment_rotations
    assert "left_arm" not in calibration.edge_alignment_rotations


def test_root_vertical_uses_pelvis_to_support_foot_height_not_global_source_height():
    robot = _robot_rest()
    source = _source_rest()
    calibration = calibrate_rest_frames(
        _RestAdapter(robot),
        _sites(robot),
        source_rest_transforms=source,
        source_provenance="test_source_rest",
    )
    moved = {name: frame.copy() for name, frame in source.items()}
    for frame in moved.values():
        frame[:3, 3] += np.array([0.0, 0.0, 3.0])

    validation = validate_neutral_reconstruction(calibration)
    assert validation.passed
    assert calibration.source_support_height == 2.0
    assert calibration.robot_support_height == 1.0


def test_frame_alignment_handles_rotated_robot_neutral_frames():
    source = _source_rest()
    rotation = so3_exp(np.array([0.0, 0.0, np.pi / 2.0]))
    robot = {name: transform(rotation @ frame[:3, 3], frame[:3, :3]) for name, frame in source.items()}

    calibration = calibrate_rest_frames(
        _RestAdapter(robot),
        _sites(robot),
        source_rest_transforms=source,
        source_provenance="test_source_rest",
    )

    np.testing.assert_allclose(calibration.edge_alignment_rotations["torso"], rotation, atol=1e-12)
    assert calibration.max_position_error < 1e-12
