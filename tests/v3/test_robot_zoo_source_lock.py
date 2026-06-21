from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path("assets/robot_zoo/robot_zoo_manifest.json")
SOURCE_LOCK = Path("assets/robot_zoo/source_lock.json")

PREVIOUSLY_UNAVAILABLE = {
    "atlas_drc_urdf",
    "atlas_v4_urdf",
    "bolt_urdf",
    "cassie_urdf",
    "draco3_urdf",
    "fourier_gr1_urdf",
    "jaxon_urdf",
    "mujoco_humanoid_mjcf",
    "rhea_urdf",
    "robonaut2_urdf",
    "romeo_urdf",
    "sigmaban_urdf",
    "simple_humanoid_urdf",
    "talos_urdf",
    "upkie_urdf",
    "valkyrie_urdf",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_source_lock_covers_every_previously_unavailable_entry():
    lock = _load(SOURCE_LOCK)
    locked_models = {
        model_id
        for repository in lock["repositories"].values()
        for model_id in repository["models"]
    }
    assert PREVIOUSLY_UNAVAILABLE <= locked_models
    assert PREVIOUSLY_UNAVAILABLE == set(lock["model_policy"])


def test_source_lock_matches_manifest_required_and_redistribution_policy():
    manifest = _load(MANIFEST)
    lock = _load(SOURCE_LOCK)
    manifest_by_id = {entry["id"]: entry for entry in manifest["models"]}
    for model_id, policy in lock["model_policy"].items():
        entry = manifest_by_id[model_id]
        assert policy["required"] is bool(entry["required"])
        assert policy["redistribution"] == entry["redistribution"]


def test_source_lock_uses_fixed_public_github_sources_and_safe_paths():
    lock = _load(SOURCE_LOCK)
    for name, repository in lock["repositories"].items():
        assert repository["url"].startswith("https://github.com/")
        assert repository["url"].endswith(".git")
        assert repository["ref"]
        cache_path = Path(repository["cache_path"])
        assert not cache_path.is_absolute()
        assert ".." not in cache_path.parts
        for model_id, relative_path in repository["models"].items():
            path = Path(relative_path)
            assert not path.is_absolute(), (name, model_id)
            assert ".." not in path.parts, (name, model_id)


def test_source_lock_contains_no_company_private_asset_identifiers():
    text = SOURCE_LOCK.read_text().lower()
    assert "cxxx_190" not in text
    assert "company" not in text
