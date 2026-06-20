from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.model_adapter import SemanticSite
from soma_retargeter.robotics.v3.robot_zoo import load_robot_zoo_manifest
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites, load_semantic_map
from soma_retargeter.robotics.v3.semantic_validation import (
    _resolve_local_manifest_source,
    compute_verified_semantic_map_coverage,
    validate_no_fake_semantic_evidence,
    validate_nonzero_distal_sites,
    validate_verified_semantic_map_payload,
)
from soma_retargeter.robotics.v3.site_geometry import body_geometry_bounds


MAP_DIR = Path("assets/robot_zoo/semantic_maps")
ASSIGNED_IDS = (
    "robotis_op3_mjcf",
    "booster_t1_urdf",
    "booster_t1_mjcf",
    "toddlerbot_urdf",
    "toddlerbot_2xc_mjcf",
    "toddlerbot_2xm_mjcf",
)
EXPECTED_TORSO_COORDS = {
    "robotis_op3_mjcf": (),
    "booster_t1_urdf": ("Waist",),
    "booster_t1_mjcf": ("Waist",),
    "toddlerbot_urdf": ("waist_yaw", "waist_roll"),
    "toddlerbot_2xc_mjcf": ("waist_yaw", "waist_roll"),
    "toddlerbot_2xm_mjcf": ("waist_yaw", "waist_roll"),
}
TODDLER_MJCF_MODEL_SITES = {
    "LeftHand": "left_hand_center",
    "RightHand": "right_hand_center",
    "LeftFoot": "left_foot_center",
    "RightFoot": "right_foot_center",
}


@lru_cache(maxsize=None)
def _manifest_entry(model_id: str):
    manifest = load_robot_zoo_manifest()
    for entry in manifest.entries:
        if entry.id == model_id:
            return entry
    raise KeyError(model_id)


@lru_cache(maxsize=None)
def _adapter(model_id: str) -> MuJoCoRuntimeModelAdapter:
    entry = _manifest_entry(model_id)
    path, status, _ = _resolve_local_manifest_source(entry, robot_descriptions_cache=None, robot_zoo_cache=None)
    assert status == "available"
    assert path is not None
    return MuJoCoRuntimeModelAdapter(path, entry.model_format)


def _payload(model_id: str) -> dict:
    return json.loads((MAP_DIR / f"{model_id}.json").read_text())


def _sites(model_id: str) -> dict[str, SemanticSite]:
    semantic_map = load_semantic_map(MAP_DIR / f"{model_id}.json", include_auxiliary=True)
    return build_semantic_sites(_adapter(model_id), semantic_map, require_distal_site_offsets=True)


def _coord_labels(adapter: MuJoCoRuntimeModelAdapter, reference: SemanticSite, target: SemanticSite) -> tuple[str, ...]:
    indices = adapter.active_velocity_coordinates(reference, target)
    return tuple(adapter.coordinate(index).label for index in indices)


def test_compact_batch_maps_are_verified_and_bound_to_cached_model_fingerprints():
    for model_id in ASSIGNED_IDS:
        adapter = _adapter(model_id)
        payload = _payload(model_id)

        assert validate_verified_semantic_map_payload(payload) == []
        assert payload["model_id"] == model_id
        assert payload["model_fingerprint"] == adapter.fingerprint

        report = compute_verified_semantic_map_coverage()
        entry_by_id = {entry.model_id: entry for entry in report.entries}
        assert entry_by_id[model_id].source_available
        assert entry_by_id[model_id].semantic_map_status == "verified"


def test_compact_batch_maps_build_existing_sites_with_nonzero_distal_offsets():
    for model_id in ASSIGNED_IDS:
        sites = _sites(model_id)
        adapter = _adapter(model_id)

        assert validate_no_fake_semantic_evidence(sites) == []
        assert validate_nonzero_distal_sites(sites) == []

        for semantic_name, site in sites.items():
            assert adapter.resolve_body_name(site.body_name) == site.body_name
            assert site.source in {"verified_semantic_map", "verified_model_site", "verified_geometry_bounds"}
            assert site.confidence <= 0.99
            assert np.isfinite(site.local_position).all(), semantic_name
            assert np.isfinite(site.local_rotation_xyzw).all(), semantic_name


def test_torso_topology_preserves_op3_fixed_case_and_booster_toddler_waist_splits():
    for model_id, expected_coords in EXPECTED_TORSO_COORDS.items():
        sites = _sites(model_id)
        labels = _coord_labels(_adapter(model_id), sites["Hips"], sites["Chest"])

        assert labels == expected_coords

    for model_id in ("booster_t1_urdf", "booster_t1_mjcf"):
        sites = _sites(model_id)

        assert sites["Hips"].body_name == "Waist"
        assert sites["Chest"].body_name in {"world", "Trunk"}
        assert sites["Hips"].body_name != sites["Chest"].body_name
        assert _coord_labels(_adapter(model_id), sites["Hips"], sites["Chest"]) == ("Waist",)


def test_hand_sites_match_verified_geometry_or_compiled_model_sites():
    for model_id in ASSIGNED_IDS:
        adapter = _adapter(model_id)
        sites = _sites(model_id)

        if model_id in {"robotis_op3_mjcf", "booster_t1_urdf", "booster_t1_mjcf"}:
            for semantic_name, bound_attr in (("LeftHand", "maximum"), ("RightHand", "minimum")):
                site = sites[semantic_name]
                bounds = body_geometry_bounds(adapter, site.body_name)
                expected_y = getattr(bounds, bound_attr)[1]
                assert np.isclose(site.local_position[1], expected_y, atol=1e-9)
                assert any("distal_" in evidence for evidence in _payload(model_id)["semantics"][semantic_name]["evidence"])
            continue

        if model_id == "toddlerbot_urdf":
            for semantic_name in ("LeftHand", "RightHand"):
                site = sites[semantic_name]
                bounds = body_geometry_bounds(adapter, site.body_name)
                assert np.allclose(site.local_position[:2], bounds.center[:2], atol=1e-8)
                assert np.isclose(site.local_position[2], bounds.minimum[2], atol=1e-8)
            continue

        for semantic_name, model_site in TODDLER_MJCF_MODEL_SITES.items():
            if not semantic_name.endswith("Hand"):
                continue
            body, pos, quat = adapter.model_site_frame(model_site)
            site = sites[semantic_name]
            assert site.body_name == body
            assert np.allclose(site.local_position, pos, atol=1e-9)
            assert np.allclose(site.local_rotation_xyzw, quat, atol=1e-9)


def test_foot_sole_toe_and_heel_offsets_match_compiled_foot_bounds():
    for model_id in ASSIGNED_IDS:
        adapter = _adapter(model_id)
        sites = _sites(model_id)

        for side in ("Left", "Right"):
            foot = sites[f"{side}Foot"]
            toe = sites[f"{side}Toe"]
            heel = sites[f"{side}Heel"]
            bounds = body_geometry_bounds(adapter, foot.body_name)

            assert toe.body_name == foot.body_name
            assert heel.body_name == foot.body_name
            assert toe.local_position[0] > heel.local_position[0]
            assert np.isclose(toe.local_position[0], bounds.maximum[0], atol=1e-8)
            assert np.isclose(heel.local_position[0], bounds.minimum[0], atol=1e-8)
            assert np.isclose(toe.local_position[2], bounds.minimum[2], atol=1e-8)
            assert np.isclose(heel.local_position[2], bounds.minimum[2], atol=1e-8)

            if foot.source == "verified_geometry_bounds":
                assert np.isclose(foot.local_position[2], bounds.minimum[2], atol=1e-8)
            else:
                model_site = TODDLER_MJCF_MODEL_SITES[f"{side}Foot"]
                body, pos, quat = adapter.model_site_frame(model_site)
                assert foot.body_name == body
                assert np.allclose(foot.local_position, pos, atol=1e-9)
                assert np.allclose(foot.local_rotation_xyzw, quat, atol=1e-9)
                assert foot.local_position[2] <= bounds.minimum[2] + 0.002
