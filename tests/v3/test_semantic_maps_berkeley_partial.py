from __future__ import annotations

import json
from pathlib import Path

import mujoco

from soma_retargeter.robotics.v3.robot_zoo import load_robot_zoo_manifest, resolve_robot_source, sha256_file
from soma_retargeter.robotics.v3.semantic_validation import (
    classify_semantic_coverage,
    compute_verified_semantic_map_coverage,
)


EXPECTATION_ROOT = Path("assets/robot_zoo/semantic_expectations")
MAP_ROOT = Path("assets/robot_zoo/semantic_maps")
BERKELEY_IDS = ("berkeley_humanoid_urdf", "berkeley_humanoid_mjcf_direct")
FORBIDDEN_UPPER_BODY_TOKENS = ("shoulder", "arm", "elbow", "wrist", "hand", "palm", "effector")


def _expectation(model_id: str) -> dict:
    return json.loads((EXPECTATION_ROOT / f"{model_id}.json").read_text())


def _resolved_source(model_id: str):
    manifest = load_robot_zoo_manifest()
    entry = manifest.model_by_id[model_id]
    resolved = resolve_robot_source(entry, allow_fetch=False)
    assert resolved.available, resolved.to_json(manifest_path=manifest.path, manifest_sha256=manifest.sha256)
    return entry, resolved


def _compiled_names(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    model = mujoco.MjModel.from_xml_path(str(path))
    bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody))
    sites = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index) for index in range(model.nsite))
    return bodies, sites


def test_berkeley_expectations_are_structural_partial_blockers_not_verified_maps():
    for model_id in BERKELEY_IDS:
        payload = _expectation(model_id)

        assert not (MAP_ROOT / f"{model_id}.json").exists()
        assert payload["verification_status"] == "partial_blocked"
        assert payload["coverage_status"] == "structurally_incomplete"
        assert payload["blocker"]["code"] == "structurally_incomplete"
        assert {"LeftHand", "RightHand"} <= set(payload["missing_required_semantics"])
        assert payload["blocker"]["evidence"]


def test_berkeley_cached_models_have_no_upper_limb_or_hand_bodies():
    for model_id in BERKELEY_IDS:
        expectation = _expectation(model_id)
        _, resolved = _resolved_source(model_id)
        bodies, sites = _compiled_names(resolved.path)
        lowered_bodies = [body.lower() for body in bodies if body]
        lowered_sites = [site.lower() for site in sites if site]

        assert sha256_file(resolved.path) == expectation["blocker"]["local_file_sha256"]
        assert bodies == tuple(expectation["blocker"]["observed_compiled_bodies"])
        assert sites == tuple(expectation["blocker"]["observed_compiled_sites"])
        assert not any(token in name for token in FORBIDDEN_UPPER_BODY_TOKENS for name in lowered_bodies)
        assert not any(token in name for token in FORBIDDEN_UPPER_BODY_TOKENS for name in lowered_sites)


def test_berkeley_partial_blockers_are_reported_separately_from_missing_maps():
    report = compute_verified_semantic_map_coverage()
    entries = {entry.model_id: entry for entry in report.entries}

    for model_id in BERKELEY_IDS:
        entry = entries[model_id]
        assert entry.semantic_map_status == "missing"
        assert entry.coverage_status == "structurally_incomplete"
        assert entry.has_explicit_partial_blocker
        assert not entry.has_verified_map
        assert entry.blocker_code == "structurally_incomplete"
        assert model_id not in report.available_missing_verified_map_ids
        assert model_id in report.available_structurally_incomplete_ids


def test_berkeley_supported_semantics_do_not_resolve_to_full_humanoid_ready():
    for model_id in BERKELEY_IDS:
        supported = _expectation(model_id)["supported_semantics"]
        classification = classify_semantic_coverage({name: object() for name in supported}, expected_capability="positive")

        assert classification.status != "full_humanoid_ready"
        assert {"LeftHand", "RightHand"} <= set(classification.missing_required)
