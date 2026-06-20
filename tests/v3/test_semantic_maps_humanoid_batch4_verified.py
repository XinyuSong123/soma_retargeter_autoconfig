from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites, load_semantic_map
from soma_retargeter.robotics.v3.semantic_validation import validate_verified_semantic_map_payload
from soma_retargeter.robotics.v3.site_geometry import body_geometry_bounds


BATCH4_IDS = (
    "jvrc_urdf",
    "jvrc_mjcf",
    "elf2_urdf",
    "elf2_mjcf",
    "ergocub_urdf",
    "pal_talos_mjcf_direct",
)
REQUIRED_SEMANTICS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
REQUIRED_DISTAL = (
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
    "LeftToe",
    "RightToe",
    "LeftHeel",
    "RightHeel",
)


def _cache_root() -> Path:
    return Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")).expanduser()


def _map_path(model_id: str) -> Path:
    return Path("assets/robot_zoo/semantic_maps") / f"{model_id}.json"


def _expectation_path(model_id: str) -> Path:
    return Path("assets/robot_zoo/semantic_expectations") / f"{model_id}.json"


def _payload(model_id: str) -> dict:
    return json.loads(_map_path(model_id).read_text())


def _entry(model_id: str, semantic: str) -> dict:
    return _payload(model_id)["semantics"][semantic]


def _distal_norm(entry: dict) -> float:
    return float(np.linalg.norm(np.asarray(entry["local_position"], dtype=float)))


def test_batch4_verified_maps_match_expectations_and_payload_schema():
    for model_id in BATCH4_IDS:
        payload = _payload(model_id)
        expectation = json.loads(_expectation_path(model_id).read_text())

        assert payload["model_id"] == model_id
        assert expectation["model_id"] == model_id
        assert payload["verification_status"] == "verified"
        assert validate_verified_semantic_map_payload(payload) == []
        assert tuple(expectation["required_semantics"]) == REQUIRED_SEMANTICS
        assert tuple(expectation["required_distal_sites"]) == REQUIRED_DISTAL

        semantics = payload["semantics"]
        for semantic in REQUIRED_SEMANTICS:
            assert semantic in semantics
        for semantic in REQUIRED_DISTAL:
            entry = semantics[semantic]
            assert _distal_norm(entry) > expectation["site_gates"]["distal_local_position_norm_min_m"]
            assert entry["confidence"] <= expectation["site_gates"]["verified_confidence_max"]
            assert entry["evidence"]
            assert any(item.startswith(("topology:", "geometry:", "site:")) for item in entry["evidence"])


def test_batch4_mjcf_maps_resolve_on_actual_cached_mujoco_models():
    paths = {
        "jvrc_mjcf": _cache_root() / "jvrc_mj_description/xml/jvrc1.xml",
        "elf2_mjcf": _cache_root() / "bxi_robot_models/elf2_dof25/xml/scene.xml",
        "pal_talos_mjcf_direct": _cache_root() / "mujoco_menagerie/pal_talos/talos.xml",
    }
    for model_id, model_path in paths.items():
        assert model_path.exists()
        adapter = MuJoCoRuntimeModelAdapter(model_path)
        sites = build_semantic_sites(
            adapter,
            load_semantic_map(_map_path(model_id), include_auxiliary=True),
            require_distal_site_offsets=True,
        )

        for semantic in REQUIRED_DISTAL:
            assert np.linalg.norm(sites[semantic].local_position) > 1e-9
            adapter.resolve_body_name(sites[semantic].body_name)


def test_talos_feet_are_distal_compiled_sole_sites_not_proximal_legs():
    model_path = _cache_root() / "mujoco_menagerie/pal_talos/talos.xml"
    assert model_path.exists()
    adapter = MuJoCoRuntimeModelAdapter(model_path)
    sites = build_semantic_sites(
        adapter,
        load_semantic_map(_map_path("pal_talos_mjcf_direct"), include_auxiliary=True),
        require_distal_site_offsets=True,
    )

    assert sites["LeftFoot"].body_name == "leg_left_6_link"
    assert sites["RightFoot"].body_name == "leg_right_6_link"
    assert sites["LeftFoot"].body_name != "leg_left_1_link"
    assert sites["RightFoot"].body_name != "leg_right_1_link"

    left_path = adapter.body_path("base_link", sites["LeftFoot"].body_name)
    right_path = adapter.body_path("base_link", sites["RightFoot"].body_name)
    assert left_path == [
        "base_link",
        "leg_left_1_link",
        "leg_left_2_link",
        "leg_left_3_link",
        "leg_left_4_link",
        "leg_left_5_link",
        "leg_left_6_link",
    ]
    assert right_path[-1] == "leg_right_6_link"

    left_bounds = body_geometry_bounds(adapter, "leg_left_6_link")
    right_bounds = body_geometry_bounds(adapter, "leg_right_6_link")
    assert sites["LeftFoot"].local_position[2] <= left_bounds.minimum[2] + 0.01
    assert sites["RightFoot"].local_position[2] <= right_bounds.minimum[2] + 0.01
    assert _entry("pal_talos_mjcf_direct", "LeftFoot")["body"] == "leg_left_6_link"
    assert _entry("pal_talos_mjcf_direct", "RightFoot")["body"] == "leg_right_6_link"


def test_elf2_maps_record_reduced_terminal_arm_hand_sites_without_fabricated_wrists():
    for model_id in ("elf2_urdf", "elf2_mjcf"):
        left = _entry(model_id, "LeftHand")
        right = _entry(model_id, "RightHand")

        assert left["body"] == "l_elb_z_link"
        assert right["body"] == "r_elb_z_link"
        assert any("capability:reduced_arm_no_wrist_body" == item for item in left["evidence"])
        assert any("capability:reduced_arm_no_wrist_body" == item for item in right["evidence"])
        assert "wrist" not in left["body"].lower()
        assert "palm" not in left["body"].lower()


def test_jvrc_urdf_map_documents_runtime_loader_blocker_and_cross_checked_bounds():
    payload = _payload("jvrc_urdf")

    assert payload["verification_backend"] == "source_xml_with_jvrc_mjcf_crosscheck"
    assert "pycollada" in payload["blocked_runtime_loader"]
    for semantic in ("LeftHand", "RightHand", "LeftFoot", "RightFoot"):
        entry = payload["semantics"][semantic]
        assert _distal_norm(entry) > 1e-9
        assert any("jvrc_mjcf_compiled_bounds" in item for item in entry["evidence"])
        assert any("urdf_mesh_reference" in item for item in entry["evidence"])


def test_ergocub_map_uses_palm_and_ankle_sole_geometry_with_toe_heel_ordering():
    payload = _payload("ergocub_urdf")
    semantics = payload["semantics"]

    assert semantics["Hips"]["body"] == "root_link"
    assert semantics["Chest"]["body"] == "chest"
    assert semantics["LeftHand"]["body"] == "l_hand_palm"
    assert semantics["RightHand"]["body"] == "r_hand_palm"
    assert semantics["LeftFoot"]["body"] == "l_ankle_2"
    assert semantics["RightFoot"]["body"] == "r_ankle_2"

    assert semantics["LeftHeel"]["local_position"][0] < semantics["LeftFoot"]["local_position"][0]
    assert semantics["LeftToe"]["local_position"][0] > semantics["LeftFoot"]["local_position"][0]
    assert semantics["RightHeel"]["local_position"][0] < semantics["RightFoot"]["local_position"][0]
    assert semantics["RightToe"]["local_position"][0] > semantics["RightFoot"]["local_position"][0]
    assert any("urdf_fixed_frame:l_ankle_2>l_foot_rear>l_sole" in item for item in semantics["LeftFoot"]["evidence"])
