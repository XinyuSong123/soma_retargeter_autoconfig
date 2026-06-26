from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.robotics.v3.rest_frames import RestCalibration
from soma_retargeter.robotics.v3.spatial import transform
from soma_retargeter.runtime.v3 import target_stream
from soma_retargeter.runtime.v3.source_frames import DEFAULT_SEMANTIC_NAMES


SEMANTICS = tuple(DEFAULT_SEMANTIC_NAMES)


def _wp_transform(position, quat=(0.0, 0.0, 0.0, 1.0)):
    return np.asarray([*position, *quat], dtype=np.float32)


def _synthetic_soma_buffer(num_frames: int = 4, *, missing: str | None = None) -> AnimationBuffer:
    names = ["Root", "Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
    parents = [-1, 0, 1, 2, 2, 1, 1]
    rest = np.asarray(
        [
            _wp_transform([0.0, 0.0, 0.0]),
            _wp_transform([0.0, 0.0, 0.0]),
            _wp_transform([0.0, 0.0, 1.0]),
            _wp_transform([0.0, 0.6, 0.2]),
            _wp_transform([0.0, -0.6, 0.2]),
            _wp_transform([0.0, 0.25, -0.9]),
            _wp_transform([0.0, -0.25, -0.9]),
        ],
        dtype=np.float32,
    )
    if missing is not None:
        index = names.index(missing)
        names.pop(index)
        parents.pop(index)
        rest = np.delete(rest, index, axis=0)
        parents = [parent - 1 if parent > index else parent for parent in parents]
    skeleton = Skeleton(len(names), names, parents, rest)
    frames = np.tile(rest[None, :, :], (num_frames, 1, 1))
    for frame in range(num_frames):
        frames[frame, names.index("Hips"), :3] = np.asarray([float(frame) * 0.1, 0.0, 0.0], dtype=np.float32)
    return AnimationBuffer(skeleton, num_frames, 120.0, frames)


def _calibration() -> RestCalibration:
    source = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 1.0]),
        "LeftHand": transform([0.0, 0.6, 1.2]),
        "RightHand": transform([0.0, -0.6, 1.2]),
        "LeftFoot": transform([0.0, 0.25, -0.9]),
        "RightFoot": transform([0.0, -0.25, -0.9]),
    }
    robot = {
        "Hips": transform([0.0, 0.0, 0.0]),
        "Chest": transform([0.0, 0.0, 2.0]),
        "LeftHand": transform([0.0, 1.2, 2.4]),
        "RightHand": transform([0.0, -1.2, 2.4]),
        "LeftFoot": transform([0.0, 0.5, -1.8]),
        "RightFoot": transform([0.0, -0.5, -1.8]),
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
            "left_arm": float(np.linalg.norm([0.0, 1.2, 0.4])),
            "right_arm": float(np.linalg.norm([0.0, -1.2, 0.4])),
            "left_leg": float(np.linalg.norm([0.0, 0.5, -1.8])),
            "right_leg": float(np.linalg.norm([0.0, -0.5, -1.8])),
        },
        root_horizontal_scale=2.0,
        vertical_root_scale=2.0,
        source_support_height=0.9,
        robot_support_height=1.8,
        neutral_position_errors={name: 0.0 for name in SEMANTICS},
        neutral_orientation_errors={name: 0.0 for name in SEMANTICS},
        max_position_error=0.0,
        max_orientation_error=0.0,
        bilateral_symmetry={},
        confidence=1.0,
        fallbacks=[],
    )


def _full_profile() -> dict:
    return {
        "status": "passed",
        "capability_status": "full_humanoid_ready",
        "model": {"id": "full_test_humanoid"},
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


def _write_bvh_metadata(path: Path, *, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT Hips",
                "{",
                "  OFFSET 0 0 0",
                "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
                "}",
                "MOTION",
                f"Frames: {frames}",
                "Frame Time: 0.008333",
                *("0 0 0 0 0 0" for _ in range(frames)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_full_profile_target_stream_uses_source_frames_and_target_adapter(monkeypatch, tmp_path: Path) -> None:
    clip = _write_bvh_metadata(tmp_path / "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh", frames=4)
    calls = {"source_frames": 0, "target_adapter": 0}
    original_extract = target_stream.extract_source_semantic_frames
    original_build = target_stream.build_runtime_semantic_targets

    def spy_extract(*args, **kwargs):
        calls["source_frames"] += 1
        return original_extract(*args, **kwargs)

    def spy_build(*args, **kwargs):
        calls["target_adapter"] += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(target_stream, "_load_bvh_animation", lambda path: _synthetic_soma_buffer(4))
    monkeypatch.setattr(target_stream, "extract_source_semantic_frames", spy_extract)
    monkeypatch.setattr(target_stream, "build_runtime_semantic_targets", spy_build)

    result = target_stream.generate_target_stream_for_clip(_full_profile(), clip, repo_root=tmp_path)

    assert result.status == "passed"
    assert result.target_stream_status == "generated"
    assert result.semantic_names == SEMANTICS
    assert result.supported_semantics == SEMANTICS
    assert result.target_source == "target_builder.build_targets_from_source_semantic_frames"
    assert result.frame_count == 4
    assert result.frame_budget is not None
    assert result.frame_budget.frame_indices == (0, 1, 2, 3)
    assert result.finite is True
    assert result.se3_valid is True
    assert result.target_batch is not None
    assert calls == {"source_frames": 1, "target_adapter": 1}
    for semantic in SEMANTICS:
        assert result.per_semantic[semantic]["status"] == "generated"
        assert result.per_semantic[semantic]["finite"] is True


def test_full_profile_missing_source_semantic_is_structured_failure(monkeypatch, tmp_path: Path) -> None:
    clip = _write_bvh_metadata(tmp_path / "assets/motions/bvh/wave_R_001__A428.bvh", frames=3)
    monkeypatch.setattr(target_stream, "_load_bvh_animation", lambda path: _synthetic_soma_buffer(3, missing="RightHand"))

    result = target_stream.generate_target_stream_for_clip(_full_profile(), clip, repo_root=tmp_path)

    assert result.status == "blocked"
    assert result.target_stream_status == "failed"
    assert result.failure is not None
    assert result.failure["code"] == "missing_source_semantics"
    assert "RightHand" in result.failure["message"]
    assert result.target_batch is None
