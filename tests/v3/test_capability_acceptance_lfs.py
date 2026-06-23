from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_retargeting_v3_capability import audit_lfs_policy


def test_lfs_policy_rejects_pointer_only_validation(tmp_path: Path) -> None:
    repo = tmp_path
    pointer = repo / "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "size 123",
            ]
        )
        + "\n"
    )
    lock = repo / "assets/robot_zoo/robot_zoo_lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"entries": {}}) + "\n")

    failures = audit_lfs_policy(
        repo,
        lock_path=lock,
        run_fsck=False,
        lfs_paths=["assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml"],
    )

    assert any("pointer-only" in failure for failure in failures)


def test_lfs_policy_rejects_mesh_and_fetch_only_snapshot_leakage(tmp_path: Path) -> None:
    repo = tmp_path
    mesh = repo / "assets/robot_zoo/snapshots/jaxon_urdf/meshes/link.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid leaked\nendsolid leaked\n")
    lock = repo / "assets/robot_zoo/robot_zoo_lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps(
            {
                "entries": {
                    "jaxon_urdf": {
                        "snapshot_status": "fetch_only",
                        "redistribution": "fetch_only",
                    }
                }
            }
        )
        + "\n"
    )

    failures = audit_lfs_policy(
        repo,
        lock_path=lock,
        run_fsck=False,
        lfs_paths=["assets/robot_zoo/snapshots/jaxon_urdf/model.urdf"],
    )

    assert any("mesh/texture leakage" in failure for failure in failures)
    assert any("fetch-only model has vendored snapshot directory" in failure for failure in failures)
    assert any("fetch-only model has LFS-tracked vendored payload" in failure for failure in failures)
