from __future__ import annotations

import json

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import (
    build_semantic_sites,
    default_rpo_semantic_map,
    infer_semantic_map_from_body_names,
)
from soma_retargeter.robotics.v3.semantic_validation import (
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


def test_inferred_sites_are_not_marked_as_explicit_or_confidence_one():
    adapter = _NoWorldNoTopologyAdapter()
    semantic_map = infer_semantic_map_from_body_names(adapter)

    sites = build_semantic_sites(adapter, semantic_map)

    assert sites["LeftHand"].source == "inferred_body_name"
    assert sites["LeftHand"].confidence < 1.0
    assert validate_no_fake_semantic_evidence(sites) == []


def test_distal_zero_origin_gate_exposes_body_only_false_positive():
    adapter = _NoWorldNoTopologyAdapter()

    with pytest.raises(ValueError, match="body origin"):
        build_semantic_sites(
            adapter,
            {"LeftHand": "left_hand"},
            require_distal_site_offsets=True,
        )


def test_default_rpo_semantic_map_has_verified_nonzero_distal_offsets():
    adapter = MuJoCoRuntimeModelAdapter("assets/robots/atom01/mjcf/atom01.xml")

    sites = build_semantic_sites(adapter, default_rpo_semantic_map(), require_distal_site_offsets=True)

    for name in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToe", "RightToe", "LeftHeel", "RightHeel"):
        assert np.linalg.norm(sites[name].local_position) > 1e-9
        assert sites[name].source == "verified_geometry_bounds"
        assert sites[name].confidence < 1.0
    assert validate_nonzero_distal_sites(sites) == []


def test_verified_rpo_map_payload_carries_evidence():
    payload = json.loads(open("assets/robot_zoo/semantic_maps/roboparty_rpo_local.json").read())

    assert validate_verified_semantic_map_payload(payload) == []
