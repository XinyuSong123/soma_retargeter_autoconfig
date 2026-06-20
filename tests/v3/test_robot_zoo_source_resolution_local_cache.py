from __future__ import annotations

import json
import sys
from pathlib import Path

from soma_retargeter.robotics.v3.robot_zoo import RobotZooEntry, display_path, resolve_robot_source


def _robot_descriptions_entry(model_id: str, description_name: str, model_format: str = "urdf") -> RobotZooEntry:
    return RobotZooEntry(
        raw={
            "id": model_id,
            "description_name": description_name,
            "format": model_format,
            "robot_class": "humanoid",
            "expected_capability": "positive",
            "license": "test",
            "redistribution": "kinematic_snapshot",
            "required": True,
            "source_family": "robot_descriptions",
            "notes": "",
        }
    )


def test_robot_descriptions_no_fetch_resolves_declared_local_cache_without_import(monkeypatch, tmp_path: Path):
    package_root = tmp_path / "site" / "robot_descriptions"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text('raise RuntimeError("package import should not run")\n')
    (package_root / "_repositories.py").write_text(
        """
from dataclasses import dataclass

@dataclass
class Repository:
    url: str
    commit: str
    cache_path: str

REPOSITORIES = {
    "public_robot_repo": Repository(
        url="https://example.invalid/public_robot_repo.git",
        commit="fixed",
        cache_path="public_robot",
    )
}
"""
    )
    (package_root / "public_bot_description.py").write_text(
        """
raise RuntimeError("robot description import should not run")
from os import path as _path
from ._cache import clone_to_cache as _clone_to_cache

REPOSITORY_PATH: str = _clone_to_cache("public_robot_repo")
PACKAGE_PATH: str = _path.join(REPOSITORY_PATH, "robot_description")
URDF_PATH: str = _path.join(PACKAGE_PATH, "public_bot.urdf")
"""
    )
    source = tmp_path / "cache" / "public_robot" / "robot_description" / "public_bot.urdf"
    source.parent.mkdir(parents=True)
    source.write_text("<robot name='public_bot'/>\n")
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_PACKAGE_ROOT", str(package_root))
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "cache"))
    sys.modules.pop("robot_descriptions.public_bot_description", None)

    resolved = resolve_robot_source(_robot_descriptions_entry("public_bot", "public_bot_description"))

    assert resolved.available
    assert resolved.path == source
    assert resolved.resolver == "robot_descriptions_local_cache"
    assert resolved.path_attr == "URDF_PATH"
    assert "robot_descriptions.public_bot_description" not in sys.modules


def test_robot_descriptions_no_fetch_preserves_unavailable_for_missing_declared_file(monkeypatch, tmp_path: Path):
    package_root = tmp_path / "site" / "robot_descriptions"
    package_root.mkdir(parents=True)
    (package_root / "_repositories.py").write_text(
        'REPOSITORIES = {"public_robot_repo": Repository(url="x", commit="fixed", cache_path="public_robot")}\n'
    )
    (package_root / "missing_bot_description.py").write_text(
        """
from os import path as _path
from ._cache import clone_to_cache as _clone_to_cache

REPOSITORY_PATH: str = _clone_to_cache("public_robot_repo")
PACKAGE_PATH: str = _path.join(REPOSITORY_PATH)
URDF_PATH: str = _path.join(PACKAGE_PATH, "missing.urdf")
"""
    )
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_PACKAGE_ROOT", str(package_root))
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "cache"))

    resolved = resolve_robot_source(_robot_descriptions_entry("missing_bot", "missing_bot_description"))

    assert not resolved.available
    assert resolved.status == "source_unavailable"
    assert resolved.resolver == "robot_descriptions_local_cache"
    assert "declared local cache files are unavailable" in resolved.reason


def test_direct_menagerie_no_fetch_resolves_robot_descriptions_cache(monkeypatch, tmp_path: Path):
    menagerie_dir = tmp_path / "cache" / "mujoco_menagerie" / "pal_talos"
    menagerie_dir.mkdir(parents=True)
    source = menagerie_dir / "talos.xml"
    source.write_text("<mujoco model='talos'/>\n")
    (menagerie_dir / "scene_position.xml").write_text("<mujoco/>\n")
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "cache"))
    entry = RobotZooEntry(
        raw={
            "id": "pal_talos_mjcf_direct",
            "description_name": None,
            "format": "mjcf",
            "robot_class": "humanoid",
            "expected_capability": "positive",
            "license": "BSD-3-Clause",
            "redistribution": "kinematic_snapshot",
            "required": True,
            "source_family": "mujoco_menagerie",
            "notes": "Menagerie directory: pal_talos",
        }
    )

    resolved = resolve_robot_source(entry)

    assert resolved.available
    assert resolved.path == source
    assert resolved.resolver == "mujoco_menagerie_cache"


def test_cache_paths_are_displayed_without_absolute_user_paths(monkeypatch, tmp_path: Path):
    cache = tmp_path / "robot_descriptions"
    source = cache / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
    source.parent.mkdir(parents=True)
    source.write_text("<mujoco/>\n")
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(cache))

    payload = json.dumps({"path": display_path(source)})

    assert str(tmp_path) not in payload
    assert "${ROBOT_DESCRIPTIONS_CACHE}/mujoco_menagerie/unitree_g1/g1.xml" in payload
