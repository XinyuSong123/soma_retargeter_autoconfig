from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import (
    build_semantic_sites,
    default_rpo_semantic_map,
    infer_semantic_map_from_body_names,
    load_semantic_map,
)
from soma_retargeter.robotics.v3.semantic_validation import (
    classify_semantic_coverage,
    validate_no_fake_semantic_evidence,
    validate_nonzero_distal_sites,
    validate_verified_semantic_map_payload,
)


class _NoWorldNoTopologyAdapter:
    body_names = ["root", "left_hand", "left_hand_terminal_with_long_name"]

    def resolve_body_name(self, body_name: str) -> str:
        if body_name not in self.body_names:
            raise KeyError(body_name)
        return body_name


def test_body_name_inference_does_not_use_string_length_as_depth():
    semantic_map = infer_semantic_map_from_body_names(_NoWorldNoTopologyAdapter())

    assert semantic_map["LeftHand"]["body"] == "left_hand"
    assert semantic_map["LeftHand"]["source"] == "inferred_body_name"
    assert semantic_map["LeftHand"]["confidence"] < 1.0
    assert "topology_unavailable" in semantic_map["LeftHand"]["evidence"]


def test_inferred_sites_are_not_marked_as_explicit_or_confidence_one():
    adapter = _NoWorldNoTopologyAdapter()
    semantic_map = infer_semantic_map_from_body_names(adapter)

    sites = build_semantic_sites(adapter, semantic_map)

    assert sites["LeftHand"].source == "inferred_body_name"
    assert sites["LeftHand"].confidence < 1.0
    assert validate_no_fake_semantic_evidence(sites) == []


def test_configured_string_sites_are_not_forced_to_confidence_one():
    adapter = _NoWorldNoTopologyAdapter()

    sites = build_semantic_sites(adapter, {"LeftHand": "left_hand"})

    assert sites["LeftHand"].source == "configured_body"
    assert sites["LeftHand"].confidence == 0.95
    assert validate_no_fake_semantic_evidence(sites) == []


def test_distal_zero_origin_gate_exposes_body_only_false_positive():
    adapter = _NoWorldNoTopologyAdapter()

    with pytest.raises(ValueError, match="body origin"):
        build_semantic_sites(
            adapter,
            {"LeftHand": "left_hand"},
            require_distal_site_offsets=True,
        )


def test_default_rpo_semantic_map_has_verified_nonzero_core_distal_offsets():
    adapter = MuJoCoRuntimeModelAdapter("assets/robots/atom01/mjcf/atom01.xml")

    sites = build_semantic_sites(adapter, default_rpo_semantic_map(), require_distal_site_offsets=True)

    for name in ("LeftHand", "RightHand", "LeftFoot", "RightFoot"):
        assert np.linalg.norm(sites[name].local_position) > 1e-9
        assert sites[name].source == "verified_geometry_bounds"
        assert sites[name].confidence < 1.0
    assert validate_nonzero_distal_sites(sites) == []


def test_verified_rpo_asset_has_toe_and_heel_auxiliary_anchors():
    adapter = MuJoCoRuntimeModelAdapter("assets/robots/atom01/mjcf/atom01.xml")
    semantic_map = load_semantic_map("assets/robot_zoo/semantic_maps/roboparty_rpo_local.json", include_auxiliary=True)

    sites = build_semantic_sites(adapter, semantic_map, require_distal_site_offsets=True)

    for name in ("LeftToe", "RightToe", "LeftHeel", "RightHeel"):
        assert np.linalg.norm(sites[name].local_position) > 1e-9
        assert sites[name].source == "verified_geometry_bounds"
    assert sites["LeftToe"].local_position[0] > sites["LeftHeel"].local_position[0]
    assert sites["RightToe"].local_position[0] > sites["RightHeel"].local_position[0]


def test_verified_rpo_map_payload_carries_evidence_and_fingerprint():
    payload = json.loads(Path("assets/robot_zoo/semantic_maps/roboparty_rpo_local.json").read_text())

    assert validate_verified_semantic_map_payload(payload) == []


def test_semantic_coverage_classifies_partial_and_negative_controls_without_fabricated_hands():
    lower_body_sites = {
        name: object()
        for name in ("Hips", "Chest", "LeftFoot", "RightFoot")
    }
    partial = classify_semantic_coverage(lower_body_sites, expected_capability="positive")
    negative = classify_semantic_coverage(lower_body_sites, expected_capability="negative_control")

    assert partial.status == "partial_humanoid"
    assert partial.missing_required == ("LeftHand", "RightHand")
    assert negative.status == "negative_control_passed"


def test_negative_control_full_humanoid_semantics_is_flagged():
    fabricated = {name: object() for name in ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")}

    classification = classify_semantic_coverage(fabricated, expected_capability="negative_control")

    assert classification.status == "semantic_false_positive"
    assert classification.issues[0].code == "negative_control_full_humanoid_semantics"
