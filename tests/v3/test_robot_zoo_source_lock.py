from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.tools.fetch_robot_zoo_sources import (
    DEFAULT_MANIFEST,
    DEFAULT_OVERLAY,
    DEFAULT_SOURCE_LOCK,
    checkout_group,
    load_sources,
    merge_manifest,
)


RECOVERED_SOURCE_IDS = {
    "cassie_urdf",
    "draco3_urdf",
    "sigmaban_urdf",
    "simple_humanoid_urdf",
    "atlas_drc_urdf",
    "atlas_v4_urdf",
    "romeo_urdf",
    "mujoco_humanoid_mjcf",
    "upkie_urdf",
    "bolt_urdf",
    "rhea_urdf",
    "fourier_gr1_urdf",
    "jaxon_urdf",
    "talos_urdf",
    "valkyrie_urdf",
    "robonaut2_urdf",
}

REAL_G1_23DOF_IDS = {
    "unitree_g1_23dof_urdf",
    "unitree_g1_23dof_mjcf",
}


def test_source_lock_covers_all_previously_unavailable_sources_and_real_g1_23dof():
    sources = load_sources(DEFAULT_SOURCE_LOCK)
    assert RECOVERED_SOURCE_IDS <= set(sources)
    assert REAL_G1_23DOF_IDS <= set(sources)
    assert len(sources) == 18


def test_source_lock_uses_fixed_refs_and_verified_git_blobs():
    sources = load_sources(DEFAULT_SOURCE_LOCK)
    for model_id, source in sources.items():
        assert source["ref"] not in {"main", "master", "HEAD"}, model_id
        assert len(source["github_blob_sha"]) == 40
        assert source["repository"].startswith("https://github.com/")
        assert source["model_path"]


def test_fetch_only_sources_remain_optional_and_outside_repository():
    payload = json.loads(DEFAULT_SOURCE_LOCK.read_text())
    fetch_only = {
        model_id
        for model_id, source in payload["sources"].items()
        if source["redistribution"] == "fetch_only"
    }
    assert fetch_only == {
        "fourier_gr1_urdf",
        "jaxon_urdf",
        "talos_urdf",
        "valkyrie_urdf",
        "robonaut2_urdf",
    }
    assert all(payload["sources"][model_id]["required"] is False for model_id in fetch_only)


def test_checkout_dry_run_does_not_touch_network_or_disk(tmp_path: Path):
    sources = load_sources(DEFAULT_SOURCE_LOCK)
    source = sources["unitree_g1_23dof_urdf"]
    checkout, head, reason = checkout_group(
        [("unitree_g1_23dof_urdf", source)],
        tmp_path,
        fetch=False,
        dry_run=True,
    )
    assert checkout is not None
    assert checkout.is_relative_to(tmp_path)
    assert head is None
    assert reason == ""
    assert not checkout.exists()


def test_overlay_adds_real_g1_23dof_models_to_resolved_manifest():
    merged = merge_manifest(DEFAULT_MANIFEST, DEFAULT_OVERLAY, {})
    by_id = {entry["id"]: entry for entry in merged["models"]}
    assert REAL_G1_23DOF_IDS <= set(by_id)
    assert by_id["unitree_g1_23dof_urdf"]["required"] is True
    assert by_id["unitree_g1_23dof_mjcf"]["required"] is True
    assert "synthetic" not in by_id["unitree_g1_23dof_urdf"]["notes"].lower()
