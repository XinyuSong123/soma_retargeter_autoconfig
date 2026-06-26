from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_retargeter.runtime.v3 import target_stream


def _write_bvh_metadata(path: Path, *, frames: int = 5) -> Path:
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


def _partial_profile_from_morphology() -> dict:
    return {
        "status": "partial_passed",
        "capability_status": "partial_humanoid",
        "model": {"id": "partial_lower_body"},
        "morphology_classification": {
            "expected_capability": "partial_humanoid",
            "supported_semantics": ["Hips", "Chest", "LeftFoot", "RightFoot"],
            "missing_required_semantics": ["LeftHand", "RightHand"],
            "humanoid_profile_generated": False,
        },
    }


def test_partial_target_stream_reports_supported_semantics_without_fabricating_missing_ones(
    monkeypatch, tmp_path: Path
) -> None:
    clip = _write_bvh_metadata(tmp_path / "assets/motions/bvh/body_stretch_1_004__A069.bvh")

    def fail_if_adapter_is_called(*args, **kwargs):
        raise AssertionError("partial report-only path must not build missing semantic targets")

    monkeypatch.setattr(target_stream, "build_runtime_semantic_targets", fail_if_adapter_is_called)
    monkeypatch.setattr(
        target_stream,
        "_load_bvh_animation",
        lambda path: pytest.fail("partial report-only path must not load BVH for target generation"),
    )

    result = target_stream.generate_target_stream_for_clip(
        _partial_profile_from_morphology(),
        clip,
        repo_root=tmp_path,
    )

    assert result.status == "passed"
    assert result.profile_kind == "partial_humanoid"
    assert result.target_stream_status == "supported_semantics_reported"
    assert result.semantic_names == ("Hips", "Chest", "LeftFoot", "RightFoot")
    assert result.supported_semantics == ("Hips", "Chest", "LeftFoot", "RightFoot")
    assert result.missing_semantics == ("LeftHand", "RightHand")
    assert "LeftHand" not in result.per_semantic
    assert "RightHand" not in result.per_semantic
    assert result.target_batch is None
    assert result.frame_count == 5


def test_partial_supported_semantics_can_come_from_semantic_expectation_file(tmp_path: Path) -> None:
    expectation = tmp_path / "assets/robot_zoo/semantic_expectations/simple_humanoid_urdf.json"
    expectation.parent.mkdir(parents=True, exist_ok=True)
    expectation.write_text(
        json.dumps(
            {
                "expected_capability": "partial_humanoid",
                "supported_semantics": ["Chest", "Hips"],
                "missing_required_semantics": ["LeftHand", "RightHand", "LeftFoot", "RightFoot"],
            }
        ),
        encoding="utf-8",
    )
    profile = {
        "status": "partial_passed",
        "capability_status": "partial_humanoid",
        "model": {"id": "simple_humanoid_urdf"},
        "semantic_map_resolution": {
            "status": "structured_partial_expectation",
            "path": "assets/robot_zoo/semantic_expectations/simple_humanoid_urdf.json",
        },
    }

    supported = target_stream.supported_semantics_for_profile(profile, repo_root=tmp_path)

    assert supported == ("Hips", "Chest")


def test_partial_profile_without_supported_semantics_is_blocked(tmp_path: Path) -> None:
    clip = _write_bvh_metadata(tmp_path / "assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh")
    profile = {
        "status": "partial_passed",
        "capability_status": "partial_humanoid",
        "model": {"id": "undocumented_partial"},
    }

    result = target_stream.generate_target_stream_for_clip(profile, clip, repo_root=tmp_path)

    assert result.status == "blocked"
    assert result.failure is not None
    assert result.failure["code"] == "partial_supported_semantics_missing"
