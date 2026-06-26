import numpy as np
import pytest

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.runtime.v3.source_frames import (
    DEFAULT_SEMANTIC_NAMES,
    SourceSemanticFrameBatch,
    extract_source_semantic_frames,
    resolve_soma_semantic_joints,
)


def _wp_transform(position, quat=(0.0, 0.0, 0.0, 1.0)):
    return np.asarray([*position, *quat], dtype=np.float32)


def _synthetic_soma_buffer(num_frames=4):
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
    skeleton = Skeleton(len(names), names, parents, rest)
    frames = np.tile(rest[None, :, :], (num_frames, 1, 1))
    for frame in range(num_frames):
        frames[frame, names.index("Hips"), :3] = np.asarray([float(frame), 0.0, 0.0], dtype=np.float32)
    return AnimationBuffer(skeleton, num_frames, 120.0, frames)


def test_source_semantic_frame_batch_extracts_sliced_finite_se3_with_joint_evidence():
    buffer = _synthetic_soma_buffer()
    offset = np.eye(4)
    offset[:3, 3] = [10.0, 0.0, 0.0]

    batch = extract_source_semantic_frames(buffer, frame_start=1, max_frames=2, offset_transform=offset)

    assert isinstance(batch, SourceSemanticFrameBatch)
    assert batch.semantic_names == list(DEFAULT_SEMANTIC_NAMES)
    assert batch.frame_count == 2
    assert batch.sample_rate == 120.0
    assert batch.source == "AnimationBuffer"
    assert batch.joint_names == {semantic: semantic for semantic in DEFAULT_SEMANTIC_NAMES}
    assert batch.joint_evidence["LeftHand"]["match_type"] == "exact"

    hips = batch.transforms["Hips"]
    chest = batch.transforms["Chest"]
    assert hips.dtype == np.float64
    assert hips.shape == (2, 4, 4)
    np.testing.assert_allclose(hips[:, :3, 3], [[11.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    np.testing.assert_allclose(chest[:, :3, 3], [[11.0, 0.0, 1.0], [12.0, 0.0, 1.0]])
    for transform in hips:
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)


def test_source_semantic_lookup_fail_closed_when_required_semantic_is_missing():
    buffer = _synthetic_soma_buffer()
    names = [name for name in buffer.skeleton.joint_names if name != "LeftFoot"]
    indices = [buffer.skeleton.joint_names.index(name) for name in names]
    parents = [-1, 0, 1, 2, 2, 1]
    skeleton = Skeleton(len(names), names, parents, buffer.skeleton.reference_local_transforms[indices])
    broken = AnimationBuffer(skeleton, buffer.num_frames, buffer.sample_rate)

    with pytest.raises(ValueError, match="LeftFoot"):
        resolve_soma_semantic_joints(broken.skeleton)

    with pytest.raises(ValueError, match="LeftFoot"):
        extract_source_semantic_frames(broken)


def test_source_semantic_frames_reject_nonfinite_transforms():
    buffer = _synthetic_soma_buffer()
    bad = np.array(buffer.local_transforms, copy=True)
    bad[0, buffer.skeleton.joint_index("RightHand"), 0] = np.nan
    broken = AnimationBuffer(buffer.skeleton, buffer.num_frames, buffer.sample_rate, bad)

    with pytest.raises(ValueError, match="non-finite"):
        extract_source_semantic_frames(broken)
