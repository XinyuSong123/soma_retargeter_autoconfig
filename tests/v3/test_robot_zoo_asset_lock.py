from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
LOCK = Path("assets/robot_zoo/source_lock.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_asset_manifest_and_lock_are_pinned_and_consistent():
    manifest = _load(MANIFEST)
    lock = _load(LOCK)

    assert manifest["schema_version"] == 2
    assert lock["schema_version"] == 2
    assert len(manifest["models"]) == lock["counts"]["model_count"] == 48
    assert sum(bool(entry["required"]) for entry in manifest["models"]) == lock["counts"]["required_count"] == 43

    rd_manifest = manifest["aggregators"]["robot_descriptions"]
    rd_lock = lock["providers"]["robot_descriptions"]
    assert rd_manifest["version"] == rd_lock["version"] == "2.0.0"
    assert rd_manifest["commit"] == rd_lock["commit"] == "2431c405312001e10cb0a8d5315dbb2e90f0c732"
    assert rd_manifest["repository_index_blob_sha"] == rd_lock["repository_index_blob_sha"]
    assert rd_manifest["description_index_blob_sha"] == rd_lock["description_index_blob_sha"]
    assert lock["dependencies"]["pycollada"] == "==0.9.3"


def test_real_g1_23dof_sources_are_explicit_and_pinned():
    manifest = _load(MANIFEST)
    models = {entry["id"]: entry for entry in manifest["models"]}
    expected = {
        "unitree_g1_23dof_urdf": "robots/g1_description/g1_23dof.urdf",
        "unitree_g1_23dof_mjcf": "robots/g1_description/g1_23dof.xml",
    }
    for model_id, repository_path in expected.items():
        entry = models[model_id]
        assert entry["required"] is True
        assert entry["source_family"] == "pinned_git"
        assert entry["repository_url"] == "https://github.com/unitreerobotics/unitree_ros.git"
        assert entry["repository_ref"] == "267182b8521c8d6a631bab1fe63836873237a525"
        assert entry["repository_path"] == repository_path


def test_asset_configuration_disallows_unlisted_assets():
    lock = _load(LOCK)
    assert lock["policy"]["allow_private_or_unlisted_assets"] is False


def test_every_manifest_source_family_is_supported_by_bootstrap():
    manifest = _load(MANIFEST)
    assert {entry["source_family"] for entry in manifest["models"]} <= {
        "local",
        "robot_descriptions",
        "mujoco_menagerie",
        "pinned_git",
    }
