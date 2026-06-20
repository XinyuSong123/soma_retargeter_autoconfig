import json
from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3.model_adapter import SemanticSite
from soma_retargeter.robotics.v3.rest_frames import calibrate_rest_frames
from soma_retargeter.robotics.v3.source_rest import load_soma_source_rest_frames
from soma_retargeter.robotics.v3.spatial import so3_exp, transform
from soma_retargeter.robotics.v3.target_builder import (
    build_targets_from_source_semantic_frames,
    canonical_motion_targets,
    validate_canonical_targets,
)


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


def _sites(*names: str) -> dict[str, SemanticSite]:
    return {name: _site(name) for name in names}


def _simple_robot_rest() -> dict[str, np.ndarray]:
    return {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 1.0]),
        "LeftHand": transform([0.0, 1.0, 1.0]),
        "RightHand": transform([0.0, -1.0, 1.0]),
        "LeftFoot": transform([0.0, 0.5, -1.0]),
        "RightFoot": transform([0.0, -0.5, -1.0]),
    }


def _simple_source_rest() -> dict[str, np.ndarray]:
    return {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 2.0]),
        "LeftHand": transform([0.0, 2.0, 2.0]),
        "RightHand": transform([0.0, -2.0, 2.0]),
        "LeftFoot": transform([0.0, 1.0, -2.0]),
        "RightFoot": transform([0.0, -1.0, -2.0]),
    }


def _calibration():
    robot = _simple_robot_rest()
    return calibrate_rest_frames(
        _RestAdapter(robot),
        _sites(*robot),
        source_rest_transforms=_simple_source_rest(),
        source_provenance="test_source_rest",
    )


def test_zero_length_source_and_robot_edges_record_fallbacks_and_reduce_confidence():
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
        _sites(*robot),
        source_rest_transforms=source,
        source_provenance="test_source_rest",
    )

    assert "torso:zero_length_robot_edge" in calibration.fallbacks
    assert "left_arm:zero_length_source_edge" in calibration.fallbacks
    assert calibration.confidence < 1.0
    assert "torso" not in calibration.edge_alignment_rotations
    assert "left_arm" not in calibration.edge_alignment_rotations


def test_neutral_target_reconstruction_and_calibration_errors_are_independently_checked():
    calibration = _calibration()
    targets = canonical_motion_targets(calibration)
    validation = validate_canonical_targets(calibration, targets)

    assert validation["neutral_reconstruction_reference"] == "robot_neutral_site_transforms"
    assert validation["neutral_reconstruction"]["max_position_error"] == 0.0
    assert validation["neutral_reconstruction"]["max_orientation_error"] == 0.0
    assert validation["calibration_error_check"]["matches_recorded_calibration_errors"]
    assert validation["target_builder_reference"] == (
        "source_rest_semantic_frames+edge_alignment_rotations+robot_segment_lengths"
    )
    assert validation["failures"] == []


def test_source_reference_target_reconstruction_is_global_transform_equivariant():
    calibration = _calibration()
    base = build_targets_from_source_semantic_frames(
        calibration,
        calibration.source_rest_semantic_frames,
        mode="neutral",
    )
    global_t = transform([0.3, -0.2, 0.4], so3_exp([0.0, 0.0, 0.5]))
    moved_source = {name: global_t @ frame for name, frame in calibration.source_rest_semantic_frames.items()}

    moved = build_targets_from_source_semantic_frames(calibration, moved_source, mode="global_test")

    for name, base_transform in base.transforms.items():
        np.testing.assert_allclose(moved.transforms[name], global_t @ base_transform, atol=1e-12)
    for error in moved.segment_length_errors.values():
        assert abs(error) < 1e-12


def test_source_rest_loader_ignores_legacy_offset_and_scaler_fields(tmp_path: Path):
    spec_path = tmp_path / "skeleton.json"
    spec_path.write_text(
        json.dumps(
            {
                "joint_scales": {"Hips": 999.0},
                "joint_offsets": {"Hips": [999.0, 999.0, 999.0]},
                "human_to_robot_scaler": 999.0,
                "scaler_config": {"optimized": True},
                "optimized": True,
                "joints": [
                    {"name": "Root", "parent": None},
                    {"name": "Hips", "parent": "Root"},
                    {"name": "Chest", "parent": "Hips"},
                    {"name": "LeftHand", "parent": "Chest"},
                    {"name": "RightHand", "parent": "Chest"},
                    {"name": "LeftFoot", "parent": "Hips"},
                    {"name": "RightFoot", "parent": "Hips"},
                ],
            }
        )
    )
    base_pose = {
        "root_transform": {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "local_joint_transforms": {
            name: {"translation_m": [idx, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
            for idx, name in enumerate(("Root", "Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"))
        },
    }
    clean_pose_path = tmp_path / "clean_pose.json"
    legacy_pose_path = tmp_path / "legacy_pose.json"
    clean_pose_path.write_text(json.dumps(base_pose))
    legacy_pose = dict(base_pose)
    legacy_pose.update(
        {
            "joint_scales": {"Hips": -123.0},
            "joint_offsets": {"Chest": [123.0, 123.0, 123.0]},
            "human_to_robot_scaler": -123.0,
            "scaler_config": {"optimized": True},
            "optimized": True,
        }
    )
    legacy_pose_path.write_text(json.dumps(legacy_pose))

    clean, _ = load_soma_source_rest_frames(clean_pose_path, spec_path)
    legacy, _ = load_soma_source_rest_frames(legacy_pose_path, spec_path)

    assert set(clean) == set(legacy)
    for name in clean:
        np.testing.assert_allclose(legacy[name], clean[name], atol=0.0)


def test_source_rest_loader_raises_clear_error_for_missing_ancestor_local_transform(tmp_path: Path):
    spec_path = tmp_path / "skeleton.json"
    pose_path = tmp_path / "pose.json"
    spec_path.write_text(
        json.dumps(
            {
                "joints": [
                    {"name": "Root", "parent": None},
                    {"name": "Hips", "parent": "Root"},
                    {"name": "Chest", "parent": "Hips"},
                ]
            }
        )
    )
    pose_path.write_text(
        json.dumps(
            {
                "local_joint_transforms": {
                    "Root": {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                    "Chest": {"translation_m": [0.0, 0.0, 1.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
                }
            }
        )
    )

    with pytest.raises(ValueError, match="missing local source rest transform for joint 'Hips'"):
        load_soma_source_rest_frames(pose_path, spec_path, semantic_names=("Chest",))
