import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.pipelines.v3_runtime_config import parse_v3_runtime_profile_config


class _FakeScaler:
    def __init__(self, legacy_effectors, effector_names):
        self._legacy_effectors = np.asarray(legacy_effectors, dtype=np.float32)
        self._effector_names = list(effector_names)
        self.calls = []

    def compute_effectors_from_buffer(self, buffer, scale_animation, offset):
        self.calls.append(
            {
                "buffer": buffer,
                "scale_animation": scale_animation,
                "offset": offset,
            }
        )
        return np.array(self._legacy_effectors, copy=True)

    def effector_names(self):
        return list(self._effector_names)


class _FakeBuffer:
    def __init__(self, num_frames=3, sample_rate=60.0):
        self.num_frames = num_frames
        self.sample_rate = sample_rate


class _ShadowAdapter:
    def __init__(self, replacement_effectors):
        self.replacement_effectors = np.asarray(replacement_effectors, dtype=np.float32)
        self.calls = []

    def compute_targets(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "target_effectors": np.array(self.replacement_effectors, copy=True),
            "diagnostics": {
                "semantic_names": list(kwargs["semantic_tasks"]),
                "finite_count": int(np.isfinite(self.replacement_effectors).sum()),
            },
        }


class _AdapterMustNotRun:
    def compute_targets(self, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("disabled V3 runtime must not compute targets")


def _legacy_effectors():
    out = np.zeros((3, 4, 7), dtype=np.float32)
    out[:, :, 6] = 1.0
    for frame in range(out.shape[0]):
        for effector in range(out.shape[1]):
            out[frame, effector, :3] = [frame, effector, frame + effector]
    return out


def _make_pipe(config_payload, legacy_effectors=None, adapter=None):
    legacy_effectors = _legacy_effectors() if legacy_effectors is None else legacy_effectors
    pipe = NewtonPipeline.__new__(NewtonPipeline)
    pipe.initialization_pose = None
    pipe.num_initialization_frames = 0
    pipe.num_stabilization_frames = 0
    pipe.max_frames = -1
    pipe.input_targets = []
    pipe.input_sample_rates = []
    pipe.input_contact_scores = []
    pipe.stance_width_diagnostics_enabled = False
    pipe.contact_aware_foot_ik_enabled = False
    pipe.target_effector_indices = [0, 2, 3]
    pipe.mapped_joints = ["Hips", "LeftHand", "RightHand"]
    pipe.ik_iterations = 24
    pipe.joint_limit_weight = 10.0
    pipe.smooth_joint_filter_weight = 5.5
    pipe.human_robot_scaler = _FakeScaler(
        legacy_effectors,
        ["Hips", "Chest", "LeftHand", "RightHand"],
    )
    pipe.v3_runtime_config = parse_v3_runtime_profile_config(config_payload)
    pipe.v3_runtime_profile = None
    pipe.v3_runtime_adapter = adapter
    pipe.v3_runtime_diagnostics = []
    return pipe


def test_missing_v3_config_does_not_compute_or_change_input_targets():
    legacy = _legacy_effectors()
    pipe = _make_pipe({}, legacy_effectors=legacy, adapter=_AdapterMustNotRun())

    pipe.add_input_motions([_FakeBuffer(num_frames=len(legacy))], [], scale_animation=True)

    np.testing.assert_array_equal(pipe.input_targets[0], legacy[:, [0, 2, 3], :])
    assert pipe.v3_runtime_diagnostics == []
    assert len(pipe.human_robot_scaler.calls) == 1
    assert pipe.input_sample_rates == [60.0]


def test_shadow_mode_computes_targets_but_preserves_disabled_input_targets():
    legacy = _legacy_effectors()
    disabled = _make_pipe({}, legacy_effectors=legacy, adapter=_AdapterMustNotRun())
    shadow_adapter = _ShadowAdapter(np.full_like(legacy, 42.0))
    shadow = _make_pipe(
        {
            "v3_runtime_profile": {
                "enabled": True,
                "mode": "shadow",
                "diagnostics_enabled": True,
            }
        },
        legacy_effectors=legacy,
        adapter=shadow_adapter,
    )

    buffer = _FakeBuffer(num_frames=len(legacy))
    disabled.add_input_motions([buffer], [], scale_animation=False)
    shadow.add_input_motions([buffer], [], scale_animation=False)

    np.testing.assert_array_equal(shadow.input_targets[0], disabled.input_targets[0])
    np.testing.assert_array_equal(shadow.input_targets[0], legacy[:, [0, 2, 3], :])
    assert len(shadow_adapter.calls) == 1
    call = shadow_adapter.calls[0]
    assert call["buffer"] is buffer
    assert call["legacy_effector_names"] == ["Hips", "Chest", "LeftHand", "RightHand"]
    assert call["target_effector_indices"] == [0, 2, 3]
    assert call["mapped_joints"] == ["Hips", "LeftHand", "RightHand"]
    assert call["semantic_tasks"] == [
        "Hips",
        "Chest",
        "LeftHand",
        "RightHand",
        "LeftFoot",
        "RightFoot",
    ]
    assert shadow.v3_runtime_diagnostics[0]["mode"] == "shadow"
    assert shadow.v3_runtime_diagnostics[0]["clip_index"] == 0
    assert shadow.v3_runtime_diagnostics[0]["diagnostics"]["finite_count"] > 0


def test_shadow_add_input_motions_does_not_change_ik_weights_or_iterations():
    legacy = _legacy_effectors()
    pipe = _make_pipe(
        {"v3_runtime_profile": {"enabled": True, "mode": "shadow"}},
        legacy_effectors=legacy,
        adapter=_ShadowAdapter(np.full_like(legacy, 7.0)),
    )
    pipe.ik_iterations = 31
    pipe.joint_limit_weight = 3.5
    pipe.smooth_joint_filter_weight = 1.25

    pipe.add_input_motions([_FakeBuffer(num_frames=len(legacy))], [], scale_animation=False)

    assert pipe.ik_iterations == 31
    assert pipe.joint_limit_weight == 3.5
    assert pipe.smooth_joint_filter_weight == 1.25


def test_post_run_shadow_diagnostics_summary_is_written_by_helper(tmp_path):
    pipe = _make_pipe(
        {
            "v3_runtime_profile": {
                "enabled": True,
                "mode": "shadow",
                "diagnostics_output_dir": str(tmp_path),
            }
        },
        adapter=_ShadowAdapter(_legacy_effectors()),
    )
    pipe.v3_runtime_diagnostics = [
        {
            "mode": "shadow",
            "clip_index": 0,
            "diagnostics": {"finite_count": 3},
        }
    ]

    pipe._write_v3_runtime_diagnostics_summary(output_buffers=[object()])

    summary_path = Path(tmp_path) / "shadow_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["mode"] == "shadow"
    assert payload["clip_count"] == 1
    assert payload["output_buffer_count"] == 1
    assert payload["diagnostics"][0]["diagnostics"]["finite_count"] == 3


def test_disabled_mode_does_not_write_v3_diagnostics(tmp_path):
    pipe = _make_pipe(
        {
            "v3_runtime_profile": {
                "enabled": False,
                "diagnostics_output_dir": str(tmp_path),
            }
        },
        adapter=_AdapterMustNotRun(),
    )
    pipe.v3_runtime_diagnostics = [{"mode": "shadow", "diagnostics": {"finite_count": 3}}]

    pipe._write_v3_runtime_diagnostics_summary(output_buffers=[object()])

    assert list(tmp_path.iterdir()) == []
