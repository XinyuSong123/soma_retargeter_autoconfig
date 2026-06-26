import numpy as np
import pytest

from soma_retargeter.robotics.v3.rest_frames import RestCalibration
from soma_retargeter.robotics.v3.spatial import transform
from soma_retargeter.runtime.v3.source_frames import SourceSemanticFrameBatch
from soma_retargeter.runtime.v3 import target_adapter
from soma_retargeter.runtime.v3.target_adapter import (
    RuntimeSemanticTargetBatch,
    build_runtime_semantic_targets,
    semantic_targets_to_effector_order,
)


SEMANTICS = ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]


def _calibration() -> RestCalibration:
    source = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 1.0]),
        "LeftHand": transform([0.0, 1.0, 1.0]),
        "RightHand": transform([0.0, -1.0, 1.0]),
        "LeftFoot": transform([0.0, 0.5, -1.0]),
        "RightFoot": transform([0.0, -0.5, -1.0]),
    }
    robot = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 2.0]),
        "LeftHand": transform([0.0, 2.0, 2.0]),
        "RightHand": transform([0.0, -2.0, 2.0]),
        "LeftFoot": transform([0.0, 1.0, -2.0]),
        "RightFoot": transform([0.0, -1.0, -2.0]),
    }
    return RestCalibration(
        source_rest_semantic_frames=source,
        source_provenance="test_source",
        robot_neutral_site_transforms=robot,
        edge_alignment_rotations={
            "torso": np.eye(3),
            "left_arm": np.eye(3),
            "right_arm": np.eye(3),
            "left_leg": np.eye(3),
            "right_leg": np.eye(3),
        },
        edge_conditioning={},
        edge_frame_sources={},
        segment_lengths={
            "torso": 2.0,
            "left_arm": 2.0,
            "right_arm": 2.0,
            "left_leg": float(np.linalg.norm([0.0, 1.0, -2.0])),
            "right_leg": float(np.linalg.norm([0.0, -1.0, -2.0])),
        },
        root_horizontal_scale=2.0,
        vertical_root_scale=2.0,
        source_support_height=1.0,
        robot_support_height=2.0,
        neutral_position_errors={name: 0.0 for name in SEMANTICS},
        neutral_orientation_errors={name: 0.0 for name in SEMANTICS},
        max_position_error=0.0,
        max_orientation_error=0.0,
        bilateral_symmetry={},
        confidence=1.0,
        fallbacks=[],
    )


def _profile_payload() -> dict:
    return {
        "status": "passed",
        "capability_status": "full_humanoid_ready",
        "rest_calibration": _calibration().to_json(),
        "task_certificate_summary": {
            "per_task": {
                "torso": {"statuses": ["converged"], "target": "Chest"},
                "left_hand": {"statuses": ["converged"], "target": "LeftHand"},
                "right_hand": {"statuses": ["converged"], "target": "RightHand"},
                "left_foot": {"statuses": ["converged"], "target": "LeftFoot"},
                "right_foot": {"statuses": ["converged"], "target": "RightFoot"},
            }
        },
    }


def _source_batch() -> SourceSemanticFrameBatch:
    frame0 = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 1.0]),
        "LeftHand": transform([0.0, 1.0, 1.0]),
        "RightHand": transform([0.0, -1.0, 1.0]),
        "LeftFoot": transform([0.0, 0.5, -1.0]),
        "RightFoot": transform([0.0, -0.5, -1.0]),
    }
    frame1 = {name: value.copy() for name, value in frame0.items()}
    for value in frame1.values():
        value[:3, 3] += [0.25, 0.0, 0.0]
    return SourceSemanticFrameBatch(
        semantic_names=list(SEMANTICS),
        joint_names={name: name for name in SEMANTICS},
        transforms={name: np.stack([frame0[name], frame1[name]]).astype(np.float64) for name in SEMANTICS},
        frame_count=2,
        sample_rate=60.0,
        source="test",
        joint_evidence={name: {"match_type": "exact", "joint_name": name} for name in SEMANTICS},
    )


def test_target_adapter_reuses_step2_builder_for_each_frame(monkeypatch):
    calls = []
    original = target_adapter.target_builder.build_targets_from_source_semantic_frames

    def spy(calibration, source_pose, *, mode):
        calls.append((mode, tuple(sorted(source_pose))))
        return original(calibration, source_pose, mode=mode)

    monkeypatch.setattr(target_adapter.target_builder, "build_targets_from_source_semantic_frames", spy)

    targets = build_runtime_semantic_targets(_source_batch(), _profile_payload(), mode="shadow")

    assert isinstance(targets, RuntimeSemanticTargetBatch)
    assert len(calls) == 2
    assert all(mode == "shadow" for mode, _ in calls)
    assert all(names == tuple(sorted(SEMANTICS)) for _, names in calls)
    assert targets.target_source == "target_builder.build_targets_from_source_semantic_frames"
    assert targets.transforms["Hips"].shape == (2, 4, 4)
    np.testing.assert_allclose(targets.transforms["Hips"][1, :3, 3], [0.5, 0.0, 0.0])
    assert targets.capability_status["LeftHand"] == "converged"
    assert targets.capability_status["Hips"] == "root_reference"


def test_target_adapter_maps_semantic_targets_to_existing_effector_order_without_reordering():
    targets = build_runtime_semantic_targets(_source_batch(), _profile_payload(), mode="shadow")
    legacy = np.zeros((2, 4, 4, 4), dtype=np.float64)
    legacy[:] = np.eye(4)
    legacy[:, 2, :3, 3] = [99.0, 0.0, 0.0]

    ordered = semantic_targets_to_effector_order(
        targets,
        ["LeftHand", "Hips", "Neck1", "RightFoot"],
        fill_missing_with=legacy,
    )

    assert ordered.effector_names == ["LeftHand", "Hips", "Neck1", "RightFoot"]
    assert ordered.semantic_by_effector == ["LeftHand", "Hips", None, "RightFoot"]
    assert ordered.available_mask.tolist() == [True, True, False, True]
    np.testing.assert_allclose(ordered.transforms[:, 0], targets.transforms["LeftHand"])
    np.testing.assert_allclose(ordered.transforms[:, 1], targets.transforms["Hips"])
    np.testing.assert_allclose(ordered.transforms[:, 2], legacy[:, 2])
    np.testing.assert_allclose(ordered.transforms[:, 3], targets.transforms["RightFoot"])


def test_target_adapter_fail_closed_for_missing_or_unsupported_semantics():
    source = _source_batch()
    broken_source = SourceSemanticFrameBatch(
        semantic_names=["Hips"],
        joint_names={"Hips": "Hips"},
        transforms={"Hips": source.transforms["Hips"]},
        frame_count=2,
        sample_rate=60.0,
        source="test",
    )
    with pytest.raises(ValueError, match="Chest"):
        build_runtime_semantic_targets(broken_source, _profile_payload())

    with pytest.raises(ValueError, match="Head"):
        build_runtime_semantic_targets(source, _profile_payload(), semantic_names=["Hips", "Head"])
