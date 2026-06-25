from __future__ import annotations

from pathlib import Path

from scripts.write_capability_provenance import (
    detect_lfs_pointer_files,
    validate_lfs_state,
)


def _valid_lfs_state() -> dict:
    return {
        "schema_version": 1,
        "git_lfs_version": "git-lfs/3.7.1",
        "fsck_returncode": 0,
        "pointer_files_detected": [],
        "missing_lfs_objects": [],
        "materialized_snapshot_count": 38,
        "tracked_paths": [
            "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml",
            "artifacts/retargeting_v3_step2_capability/test_results/junit.xml",
        ],
    }


def test_detect_lfs_pointer_files_reports_pointer_only_payload(tmp_path: Path) -> None:
    pointer = tmp_path / "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml"
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

    assert detect_lfs_pointer_files(tmp_path, [pointer]) == [
        "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml"
    ]


def test_lfs_state_rejects_pointer_only_lfs() -> None:
    state = _valid_lfs_state()
    state["pointer_files_detected"] = [
        "assets/robot_zoo/snapshots/unitree_g1_mjcf/model.xml",
    ]

    failures = validate_lfs_state(state)

    assert any("pointer_files_detected" in failure for failure in failures)


def test_lfs_state_rejects_failed_fsck_and_missing_objects() -> None:
    state = _valid_lfs_state()
    state["fsck_returncode"] = 2
    state["missing_lfs_objects"] = [
        "assets/robot_zoo/snapshots/jvrc_mjcf/model.xml",
    ]

    failures = validate_lfs_state(state)

    assert any("fsck_returncode" in failure for failure in failures)
    assert any("missing_lfs_objects" in failure for failure in failures)


def test_lfs_state_accepts_materialized_lfs_payloads() -> None:
    assert validate_lfs_state(_valid_lfs_state()) == []
