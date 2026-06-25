from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.robot_zoo import load_robot_zoo_manifest, resolve_robot_source
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites, load_semantic_map
from soma_retargeter.robotics.v3.semantic_validation import (
    validate_no_fake_semantic_evidence,
    validate_nonzero_distal_sites,
    validate_verified_semantic_map_payload,
)


MAP_ROOT = Path("assets/robot_zoo/semantic_maps")
EXPECTATION_ROOT = Path("assets/robot_zoo/semantic_expectations")

VERIFIED_UNITREE_MAPS = ("unitree_g1_urdf", "unitree_g1_mjcf", "unitree_h1_mjcf")

EXPECTED_PATHS = {
    "unitree_g1_urdf": {
        ("Hips", "Chest"): (
            ["world", "waist_yaw_link", "waist_roll_link", "torso_link"],
            ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        ),
        ("Chest", "LeftHand"): (
            [
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
            ],
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
        ),
        ("Hips", "LeftFoot"): (
            [
                "world",
                "left_hip_pitch_link",
                "left_hip_roll_link",
                "left_hip_yaw_link",
                "left_knee_link",
                "left_ankle_pitch_link",
                "left_ankle_roll_link",
            ],
            [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
        ),
    },
    "unitree_g1_mjcf": {
        ("Hips", "Chest"): (
            ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"],
            ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        ),
        ("Chest", "LeftHand"): (
            [
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
            ],
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
        ),
        ("Hips", "LeftFoot"): (
            [
                "pelvis",
                "left_hip_pitch_link",
                "left_hip_roll_link",
                "left_hip_yaw_link",
                "left_knee_link",
                "left_ankle_pitch_link",
                "left_ankle_roll_link",
            ],
            [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
            ],
        ),
    },
    "unitree_h1_mjcf": {
        ("Hips", "Chest"): (
            ["pelvis", "torso_link"],
            ["torso"],
        ),
        ("Chest", "LeftHand"): (
            [
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
            ],
            ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow"],
        ),
        ("Hips", "LeftFoot"): (
            [
                "pelvis",
                "left_hip_yaw_link",
                "left_hip_roll_link",
                "left_hip_pitch_link",
                "left_knee_link",
                "left_ankle_link",
            ],
            ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"],
        ),
    },
}


def _manifest_entries():
    manifest = load_robot_zoo_manifest()
    return {entry.id: entry for entry in manifest.entries}


def _adapter_for(model_id: str) -> MuJoCoRuntimeModelAdapter:
    entry = _manifest_entries()[model_id]
    resolved = resolve_robot_source(entry, allow_fetch=False)
    assert resolved.available, resolved.to_json(
        manifest_path=load_robot_zoo_manifest().path,
        manifest_sha256=load_robot_zoo_manifest().sha256,
    )
    return MuJoCoRuntimeModelAdapter(resolved.path, entry.model_format)


@pytest.mark.parametrize("model_id", VERIFIED_UNITREE_MAPS)
def test_unitree_verified_maps_match_compiled_fingerprint_and_build_with_distal_gate(model_id: str):
    payload = json.loads((MAP_ROOT / f"{model_id}.json").read_text())
    adapter = _adapter_for(model_id)
    try:
        assert payload["model_fingerprint"] == adapter.fingerprint
        assert validate_verified_semantic_map_payload(payload) == []

        sites = build_semantic_sites(
            adapter,
            load_semantic_map(MAP_ROOT / f"{model_id}.json", include_auxiliary=True),
            require_distal_site_offsets=True,
        )

        assert validate_no_fake_semantic_evidence(sites) == []
        assert validate_nonzero_distal_sites(sites) == []
        for semantic in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToe", "RightToe", "LeftHeel", "RightHeel"):
            assert sites[semantic].source == "verified_geometry_bounds"
            assert sites[semantic].confidence < 1.0
            assert np.linalg.norm(sites[semantic].local_position) > 1e-9
    finally:
        adapter.close()


@pytest.mark.parametrize("model_id", VERIFIED_UNITREE_MAPS)
def test_unitree_semantic_paths_use_full_compiled_chains(model_id: str):
    adapter = _adapter_for(model_id)
    try:
        sites = build_semantic_sites(
            adapter,
            load_semantic_map(MAP_ROOT / f"{model_id}.json"),
            require_distal_site_offsets=True,
        )
        for pair, (expected_path, expected_coordinates) in EXPECTED_PATHS[model_id].items():
            reference, target = pair
            assert adapter.body_path(sites[reference].body_name, sites[target].body_name) == expected_path
            active = [adapter.coordinate(index).label for index in adapter.active_velocity_coordinates(sites[reference], sites[target])]
            assert active == expected_coordinates
    finally:
        adapter.close()


@pytest.mark.parametrize("model_id", VERIFIED_UNITREE_MAPS)
def test_unitree_distal_offsets_are_sole_and_terminal_geometry_not_body_origins(model_id: str):
    adapter = _adapter_for(model_id)
    try:
        sites = build_semantic_sites(
            adapter,
            load_semantic_map(MAP_ROOT / f"{model_id}.json", include_auxiliary=True),
            require_distal_site_offsets=True,
        )

        assert sites["LeftToe"].local_position[0] > sites["LeftFoot"].local_position[0] > sites["LeftHeel"].local_position[0]
        assert sites["RightToe"].local_position[0] > sites["RightFoot"].local_position[0] > sites["RightHeel"].local_position[0]
        assert sites["LeftFoot"].local_position[2] < 0.0
        assert sites["RightFoot"].local_position[2] < 0.0
        assert sites["LeftHand"].local_position[0] > 0.0
        assert sites["RightHand"].local_position[0] > 0.0
    finally:
        adapter.close()


def test_unitree_h1_urdf_semantic_map_uses_verified_snapshot_variant_evidence():
    expectation = json.loads((EXPECTATION_ROOT / "unitree_h1_urdf.json").read_text())
    assert expectation["verification_status"] == "blocked"
    assert expectation["blocker"]["code"] == "compiled_geometry_unavailable"

    payload = json.loads((MAP_ROOT / "unitree_h1_urdf.json").read_text())
    assert payload["verification_status"] == "verified"
    assert payload["model_source"]["path"] == "assets/robot_zoo/snapshots/unitree_h1_urdf/model.urdf"
    assert validate_verified_semantic_map_payload(payload) == []
    assert payload["semantics"]["Hips"]["evidence"][-1] == "variant_match:unitree_h1_mjcf"
    for semantic in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToe", "RightToe", "LeftHeel", "RightHeel"):
        assert payload["semantics"][semantic]["source"] == "verified_variant_geometry"
        assert any(
            evidence.startswith("variant_geometry:unitree_h1_mjcf:")
            for evidence in payload["semantics"][semantic]["evidence"]
        )

    entry = _manifest_entries()["unitree_h1_urdf"]
    resolved = resolve_robot_source(entry, allow_fetch=False)
    assert resolved.available
    assert Path(payload["model_source"]["path"]).exists()
