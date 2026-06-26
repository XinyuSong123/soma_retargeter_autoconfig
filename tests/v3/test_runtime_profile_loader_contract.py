from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_retargeter.runtime.v3.profile_loader import (
    RuntimeV3Profile,
    RuntimeV3ProfileError,
    load_runtime_v3_profile,
)


REQUIRED_SEMANTICS = {"Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"}


def _minimal_profile(*, status: str = "passed", semantic_sites: dict | None = None) -> dict:
    sites = {
        name: {
            "semantic_name": name,
            "body_name": f"{name}_body",
            "local_position": [0.0, 0.0, 0.0],
            "local_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "source": "verified_semantic_map",
            "confidence": 1.0,
            "evidence": ["unit-test"],
        }
        for name in REQUIRED_SEMANTICS
    }
    if semantic_sites is not None:
        sites = semantic_sites
    return {
        "schema_version": 3,
        "status": status,
        "capability_status": "full_humanoid_ready",
        "model": {
            "id": "unit_test_robot",
            "format": "mjcf",
            "backend": "newton",
            "path": "unit_test.xml",
            "fingerprint": "a" * 64,
            "local_file_sha256": "b" * 64,
        },
        "runtime_adapter": {"backend": "newton", "nq": 7, "nv": 6},
        "semantic_map_resolution": {"status": "available", "source": "verified_semantic_map"},
        "semantic_sites": sites,
        "rest_calibration": {
            "confidence": 1.0,
            "root_horizontal_scale": 1.0,
            "vertical_root_scale": 1.0,
            "robot_support_height": 0.75,
            "source_support_height": 0.1,
            "robot_neutral_site_transforms": {},
        },
        "canonical_targets": {"neutral": {"transforms": {}}},
        "canonical_projection_reports": {"motion_order": ["neutral"], "motions": {"neutral": {"tasks": {}}}},
        "task_certificate_summary": {
            "schema_version": 1,
            "status": "available",
            "task_count": 1,
            "canonical_motion_count": 1,
            "per_task": {"torso": {"motion_count": 1}},
        },
    }


def _write_profile(root: Path, profile_id: str, payload: dict | str) -> Path:
    per_robot = root / "per_robot"
    per_robot.mkdir(parents=True)
    path = per_robot / f"{profile_id}.json"
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, sort_keys=True))
    return path


def test_loads_committed_step2_profile_with_required_runtime_fields():
    profile = load_runtime_v3_profile(
        "roboparty_rpo_local",
        profile_artifact_root=Path("artifacts/retargeting_v3_step2_capability"),
    )

    assert isinstance(profile, RuntimeV3Profile)
    assert profile.model_id == "roboparty_rpo_local"
    assert profile.schema_version == 3
    assert profile.status == "passed"
    assert profile.profile_fingerprint == "0208dff000ed332b8fa77527a1cc66ab4494e1fac167389da27f1b2f366f0bea"
    assert profile.source_sha256 == "d9c91b9cb9e14581975d6a76433c6cbaf5a00e81939e2fbb74c5b31196be1fa1"
    assert set(profile.semantic_sites) == REQUIRED_SEMANTICS
    assert profile.rest_calibration["confidence"] == 1.0
    assert len(profile.canonical_targets) == 15
    assert profile.task_certificate_summary["status"] == "available"
    assert profile.task_certificate_summary["task_count"] == 5


def test_profile_loader_rejects_lfs_pointer_json(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_profile(
        root,
        "pointer_robot",
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:" + "0" * 64,
                "size 12345",
            ]
        ),
    )

    with pytest.raises(RuntimeV3ProfileError) as excinfo:
        load_runtime_v3_profile("pointer_robot", profile_artifact_root=root)

    assert excinfo.value.code == "lfs_pointer"
    assert excinfo.value.details["path"].endswith("pointer_robot.json")


def test_profile_loader_reports_missing_required_semantic_site(tmp_path: Path):
    root = tmp_path / "artifacts"
    payload = _minimal_profile(semantic_sites={name: {} for name in REQUIRED_SEMANTICS - {"RightFoot"}})
    _write_profile(root, "missing_site_robot", payload)

    with pytest.raises(RuntimeV3ProfileError) as excinfo:
        load_runtime_v3_profile("missing_site_robot", profile_artifact_root=root)

    assert excinfo.value.code == "invalid_profile_schema"
    assert "missing semantic sites" in excinfo.value.message
    assert excinfo.value.details["missing"] == ["RightFoot"]


def test_profile_loader_rejects_non_terminal_profile_status(tmp_path: Path):
    root = tmp_path / "artifacts"
    _write_profile(root, "partial_robot", _minimal_profile(status="partial_passed"))

    with pytest.raises(RuntimeV3ProfileError) as excinfo:
        load_runtime_v3_profile("partial_robot", profile_artifact_root=root)

    assert excinfo.value.code == "invalid_profile_status"
    assert excinfo.value.details["status"] == "partial_passed"
