from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.runtime.v3 import profile_loader
from soma_retargeter.runtime.v3.profile_loader import (
    RuntimeModelIdentity,
    RuntimeV3ProfileError,
    resolve_runtime_v3_profile_id,
)


REQUIRED_SEMANTICS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")


def _profile_payload(profile_id: str, *, fingerprint: str = "f" * 64, source_sha256: str = "1" * 64) -> dict:
    return {
        "schema_version": 3,
        "status": "passed",
        "capability_status": "full_humanoid_ready",
        "model": {
            "id": profile_id,
            "format": "mjcf",
            "backend": "newton",
            "path": f"{profile_id}.xml",
            "fingerprint": fingerprint,
            "local_file_sha256": source_sha256,
        },
        "runtime_adapter": {"backend": "newton", "nq": 7, "nv": 6},
        "semantic_map_resolution": {"status": "available", "source": "verified_semantic_map"},
        "semantic_sites": {
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
        },
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


def _artifact_root(tmp_path: Path, profile_id: str, *, fingerprint: str = "f" * 64, source_sha256: str = "1" * 64) -> Path:
    root = tmp_path / "step2"
    per_robot = root / "per_robot"
    per_robot.mkdir(parents=True)
    (per_robot / f"{profile_id}.json").write_text(
        json.dumps(_profile_payload(profile_id, fingerprint=fingerprint, source_sha256=source_sha256), sort_keys=True)
    )
    return root


def _runtime_file(tmp_path: Path) -> Path:
    path = tmp_path / "runtime.xml"
    path.write_text("<mujoco model='runtime'><worldbody/></mujoco>\n")
    return path


def _identity(path: Path, *, fingerprint: str = "f" * 64, source_sha256: str = "1" * 64) -> RuntimeModelIdentity:
    return RuntimeModelIdentity(
        path=path,
        fingerprint=fingerprint,
        source_sha256=source_sha256,
        model_format="mjcf",
        backend="newton",
    )


def test_pipeline_utils_resolves_default_runtime_v3_profile_ids():
    assert pipeline_utils.get_default_runtime_v3_profile_id("roboparty_rpo") == "roboparty_rpo_local"
    assert pipeline_utils.get_default_runtime_v3_profile_id("unitree_g1") == "unitree_g1_mjcf"
    assert pipeline_utils.get_default_runtime_v3_profile_id(pipeline_utils.TargetType.ROBOPARTY_RPO) == "roboparty_rpo_local"


def test_rpo_requires_strict_runtime_fingerprint_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime_path = _runtime_file(tmp_path)
    root = _artifact_root(tmp_path, "roboparty_rpo_local", fingerprint="a" * 64, source_sha256="b" * 64)
    monkeypatch.setattr(
        profile_loader,
        "_compute_runtime_model_identity",
        lambda path: _identity(Path(path), fingerprint="c" * 64, source_sha256="b" * 64),
    )

    with pytest.raises(RuntimeV3ProfileError) as excinfo:
        resolve_runtime_v3_profile_id(
            "roboparty_rpo",
            runtime_path,
            {"mode": "shadow", "profile_artifact_root": str(root)},
        )

    assert excinfo.value.code == "fingerprint_mismatch"
    assert excinfo.value.details["robot_type"] == "roboparty_rpo"
    assert excinfo.value.details["strict_match_required"] is True


def test_g1_shadow_mismatch_returns_fail_closed_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime_path = _runtime_file(tmp_path)
    root = _artifact_root(tmp_path, "unitree_g1_mjcf", fingerprint="a" * 64, source_sha256="b" * 64)
    monkeypatch.setattr(
        profile_loader,
        "_compute_runtime_model_identity",
        lambda path: _identity(Path(path), fingerprint="c" * 64, source_sha256="d" * 64),
    )

    resolution = resolve_runtime_v3_profile_id(
        "unitree_g1",
        runtime_path,
        {
            "mode": "shadow",
            "target_policy": "shadow_only",
            "profile_artifact_root": str(root),
        },
    )

    assert resolution.profile_model_id == "unitree_g1_mjcf"
    assert resolution.resolution_status == "fingerprint_mismatch"
    assert resolution.fingerprint_match is False
    assert resolution.source_hash_match is False
    assert resolution.override_allowed is False
    assert resolution.errors[0]["code"] == "fingerprint_mismatch"


def test_g1_override_mismatch_fails_closed_without_runtime_local_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime_path = _runtime_file(tmp_path)
    root = _artifact_root(tmp_path, "unitree_g1_mjcf", fingerprint="a" * 64, source_sha256="b" * 64)
    monkeypatch.setattr(
        profile_loader,
        "_compute_runtime_model_identity",
        lambda path: _identity(Path(path), fingerprint="c" * 64, source_sha256="b" * 64),
    )

    with pytest.raises(RuntimeV3ProfileError) as excinfo:
        resolve_runtime_v3_profile_id(
            "unitree_g1",
            runtime_path,
            {
                "mode": "override_experimental",
                "target_policy": "replace_configured_semantics",
                "profile_artifact_root": str(root),
            },
        )

    assert excinfo.value.code == "fingerprint_mismatch"
    assert excinfo.value.details["robot_type"] == "unitree_g1"
    assert excinfo.value.details["runtime_local_allowed"] is False


def test_g1_override_mismatch_can_request_runtime_local_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime_path = _runtime_file(tmp_path)
    root = _artifact_root(tmp_path, "unitree_g1_mjcf", fingerprint="a" * 64, source_sha256="b" * 64)
    monkeypatch.setattr(
        profile_loader,
        "_compute_runtime_model_identity",
        lambda path: _identity(Path(path), fingerprint="c" * 64, source_sha256="b" * 64),
    )

    resolution = resolve_runtime_v3_profile_id(
        "unitree_g1",
        runtime_path,
        {
            "mode": "override_experimental",
            "target_policy": "replace_configured_semantics",
            "profile_artifact_root": str(root),
            "allow_runtime_recompile_on_mismatch": True,
        },
    )

    assert resolution.resolution_status == "runtime_local_profile_required"
    assert resolution.runtime_local_allowed is True
    assert resolution.override_allowed is False
    assert resolution.warnings[0]["code"] == "runtime_local_profile_required"


def test_runtime_mjcf_lfs_pointer_is_rejected_before_fingerprint(tmp_path: Path):
    root = _artifact_root(tmp_path, "roboparty_rpo_local")
    runtime_path = tmp_path / "runtime.xml"
    runtime_path.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:" + "0" * 64,
                "size 12345",
            ]
        )
    )

    with pytest.raises(RuntimeV3ProfileError) as excinfo:
        resolve_runtime_v3_profile_id(
            "roboparty_rpo",
            runtime_path,
            {"profile_artifact_root": str(root)},
        )

    assert excinfo.value.code == "lfs_pointer"
    assert excinfo.value.details["path"].endswith("runtime.xml")
