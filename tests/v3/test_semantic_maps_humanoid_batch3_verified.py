from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.robot_zoo import load_robot_zoo_manifest, resolve_robot_source, sha256_file
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites, load_semantic_map
from soma_retargeter.robotics.v3.semantic_validation import (
    classify_semantic_coverage,
    validate_no_fake_semantic_evidence,
    validate_nonzero_distal_sites,
    validate_verified_semantic_map_payload,
)
from soma_retargeter.robotics.v3.site_geometry import body_geometry_bounds


BATCH3_FULL_MAP_IDS = (
    "adam_lite_mjcf",
    "apptronik_apollo_mjcf",
    "fourier_n1_urdf",
    "fourier_n1_mjcf",
)

BERKELEY_IDS = (
    "berkeley_humanoid_urdf",
    "berkeley_humanoid_mjcf_direct",
)


def _map_path(model_id: str) -> Path:
    return Path("assets/robot_zoo/semantic_maps") / f"{model_id}.json"


def _expectation_path(model_id: str) -> Path:
    return Path("assets/robot_zoo/semantic_expectations") / f"{model_id}.json"


def _source_adapter(model_id: str) -> tuple[MuJoCoRuntimeModelAdapter, dict]:
    manifest = load_robot_zoo_manifest()
    entry = manifest.model_by_id[model_id]
    resolved = resolve_robot_source(entry, allow_fetch=False)
    assert resolved.available, resolved.to_json(manifest_path=manifest.path, manifest_sha256=manifest.sha256)
    adapter = MuJoCoRuntimeModelAdapter(resolved.path, model_format=entry.model_format)
    return adapter, {"entry": entry, "resolved": resolved}


def _active_labels(adapter: MuJoCoRuntimeModelAdapter, sites: dict, reference: str, target: str) -> set[str]:
    return {adapter.coordinate(index).label for index in adapter.active_velocity_coordinates(sites[reference], sites[target])}


def _assert_contains(labels: set[str], expected: set[str]) -> None:
    missing = expected - labels
    assert not missing, f"missing labels {sorted(missing)} from {sorted(labels)}"


def test_batch3_full_maps_are_verified_payloads_bound_to_local_compiled_models():
    for model_id in BATCH3_FULL_MAP_IDS:
        adapter, source = _source_adapter(model_id)
        payload = json.loads(_map_path(model_id).read_text())

        assert validate_verified_semantic_map_payload(payload) == []
        assert payload["model_id"] == model_id
        assert payload["model_fingerprint"] == adapter.fingerprint
        assert payload["model_source"]["sha256"] == sha256_file(source["resolved"].path)


def test_batch3_full_maps_build_nonzero_distal_sites_without_inferred_evidence():
    for model_id in BATCH3_FULL_MAP_IDS:
        adapter, _ = _source_adapter(model_id)
        semantic_map = load_semantic_map(_map_path(model_id), include_auxiliary=True)

        sites = build_semantic_sites(adapter, semantic_map, require_distal_site_offsets=True)

        assert classify_semantic_coverage(sites, expected_capability="positive").status == "full_humanoid_ready"
        assert validate_no_fake_semantic_evidence(sites) == []
        assert validate_nonzero_distal_sites(sites) == []
        for semantic in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToe", "RightToe", "LeftHeel", "RightHeel"):
            assert np.linalg.norm(sites[semantic].local_position) > 1e-9
            assert sites[semantic].source == "verified_geometry_bounds"
            assert sites[semantic].confidence < 1.0


def test_adam_lite_verified_map_uses_full_torso_arm_and_leg_chains():
    adapter, _ = _source_adapter("adam_lite_mjcf")
    sites = build_semantic_sites(adapter, load_semantic_map(_map_path("adam_lite_mjcf")))

    _assert_contains(_active_labels(adapter, sites, "Hips", "Chest"), {"waistRoll", "waistPitch", "waistYaw"})
    _assert_contains(
        _active_labels(adapter, sites, "Chest", "LeftHand"),
        {"shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left", "elbow_Left", "wristYaw_Left"},
    )
    _assert_contains(
        _active_labels(adapter, sites, "Chest", "RightHand"),
        {"shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right", "elbow_Right", "wristYaw_Right"},
    )
    _assert_contains(
        _active_labels(adapter, sites, "Hips", "LeftFoot"),
        {"hipPitch_Left", "hipRoll_Left", "hipYaw_Left", "kneePitch_Left", "anklePitch_Left", "ankleRoll_Left"},
    )


def test_apollo_verified_map_uses_full_torso_arm_and_leg_chains():
    adapter, _ = _source_adapter("apptronik_apollo_mjcf")
    sites = build_semantic_sites(adapter, load_semantic_map(_map_path("apptronik_apollo_mjcf")))

    _assert_contains(_active_labels(adapter, sites, "Hips", "Chest"), {"torso_yaw", "torso_roll", "torso_pitch"})
    _assert_contains(
        _active_labels(adapter, sites, "Chest", "LeftHand"),
        {"l_shoulder_aa", "l_shoulder_ie", "l_shoulder_fe", "l_elbow_fe", "l_wrist_roll", "l_wrist_yaw", "l_wrist_pitch"},
    )
    _assert_contains(
        _active_labels(adapter, sites, "Hips", "LeftFoot"),
        {"l_hip_ie", "l_hip_aa", "l_hip_fe", "l_knee_fe", "l_ankle_ie", "l_ankle_pd"},
    )


def test_fourier_n1_maps_reflect_compiled_urdf_and_mjcf_topology():
    cases = {
        "fourier_n1_urdf": {
            "hips": "world",
            "chest": "waist_yaw_link",
            "left_hand": "left_hand_yaw_link",
            "torso_labels": {"waist_yaw_joint"},
            "arm_labels": {"left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_yaw_joint"},
            "leg_labels": {"left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint"},
        },
        "fourier_n1_mjcf": {
            "hips": "base_link",
            "chest": "torso_link",
            "left_hand": "left_end_effector_link",
            "torso_labels": {"waist_yaw_joint"},
            "arm_labels": {"left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_yaw_joint"},
            "leg_labels": {"left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_pitch_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint"},
        },
    }
    for model_id, expected in cases.items():
        adapter, _ = _source_adapter(model_id)
        sites = build_semantic_sites(adapter, load_semantic_map(_map_path(model_id)))

        assert sites["Hips"].body_name == expected["hips"]
        assert sites["Chest"].body_name == expected["chest"]
        assert sites["LeftHand"].body_name == expected["left_hand"]
        _assert_contains(_active_labels(adapter, sites, "Hips", "Chest"), expected["torso_labels"])
        _assert_contains(_active_labels(adapter, sites, "Chest", "LeftHand"), expected["arm_labels"])
        _assert_contains(_active_labels(adapter, sites, "Hips", "LeftFoot"), expected["leg_labels"])


def test_berkeley_batch3_is_documented_as_partial_without_fabricated_hands():
    forbidden_upper_body_tokens = ("shoulder", "arm", "elbow", "wrist", "hand", "palm", "effector")
    foot_bodies = {
        "berkeley_humanoid_urdf": ("ll_faa", "lr_faa"),
        "berkeley_humanoid_mjcf_direct": ("ll_faa", "lr_faa"),
    }

    for model_id in BERKELEY_IDS:
        expectation = json.loads(_expectation_path(model_id).read_text())
        adapter, _ = _source_adapter(model_id)
        lowered = [name.lower() for name in adapter.body_names]

        assert not _map_path(model_id).exists()
        assert expectation["expected_capability"] == "partial_humanoid"
        assert {"LeftHand", "RightHand"} <= set(expectation["missing_required_semantics"])
        assert not any(token in body for token in forbidden_upper_body_tokens for body in lowered)
        for body_name in foot_bodies[model_id]:
            bounds = body_geometry_bounds(adapter, body_name)
            assert np.linalg.norm(bounds.span) > 1e-9

    lower_body_sites = {name: object() for name in ("Hips", "Chest", "LeftFoot", "RightFoot")}
    classification = classify_semantic_coverage(lower_body_sites, expected_capability="positive")
    assert classification.status == "partial_humanoid"
    assert classification.missing_required == ("LeftHand", "RightHand")
