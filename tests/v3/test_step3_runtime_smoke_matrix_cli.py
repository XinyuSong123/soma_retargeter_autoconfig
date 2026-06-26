from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma_retargeter.tools import run_v3_runtime_shadow_smoke as smoke


def _write_profile(root: Path, model_id: str, *, status: str = "passed") -> Path:
    path = root / "per_robot" / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "model": {
                    "id": model_id,
                    "local_file_sha256": f"{model_id}-runtime-sha",
                    "source_resolution": {
                        "local_file_sha256": f"{model_id}-runtime-sha",
                        "path": f"assets/robot_zoo/snapshots/{model_id}/model.xml",
                    },
                },
                "semantic_sites": {
                    "Hips": {"body": "pelvis"},
                    "Chest": {"body": "chest"},
                    "LeftHand": {"body": "left_hand"},
                    "RightHand": {"body": "right_hand"},
                    "LeftFoot": {"body": "left_foot"},
                    "RightFoot": {"body": "right_foot"},
                },
                "rest_calibration": {"status": "passed"},
                "canonical_targets": {"status": "passed"},
                "capability_summary": {"status": "passed"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _write_bvh(path: Path, *, frames: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "HIERARCHY",
                "ROOT Hips",
                "{",
                "  OFFSET 0 0 0",
                "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
                "}",
                "MOTION",
                f"Frames: {frames}",
                "Frame Time: 0.0333333",
                *("0 0 0 0 0 0" for _ in range(frames)),
            ]
        )
        + "\n"
    )
    return path


def test_cli_writes_required_runtime_shadow_artifacts(tmp_path: Path) -> None:
    profile_root = tmp_path / "step2"
    artifact_root = tmp_path / "artifacts"
    _write_profile(profile_root, "roboparty_rpo_local")
    _write_profile(profile_root, "unitree_g1_mjcf")
    clips = [
        _write_bvh(tmp_path / "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh", frames=15),
        _write_bvh(tmp_path / "assets/motions/bvh/wave_R_001__A428.bvh", frames=14),
    ]

    rc = smoke.main(
        [
            "--artifact-root",
            str(artifact_root),
            "--profile-artifact-root",
            str(profile_root),
            "--robots",
            "roboparty_rpo",
            "unitree_g1",
            "--clips",
            *(str(clip) for clip in clips),
            "--modes",
            "disabled",
            "shadow",
            "override_experimental",
            "--max-frames",
            "8",
        ]
    )

    assert rc == 0
    for relative in smoke.REQUIRED_ARTIFACT_FILES:
        assert (artifact_root / relative).exists(), relative
    matrix = json.loads((artifact_root / "smoke_matrix.json").read_text())
    assert matrix["max_frames"] == 8
    assert len(matrix["rows"]) == 2 * 2 * 3
    assert {row["robot_type"] for row in matrix["rows"]} == {"roboparty_rpo", "unitree_g1"}
    assert {row["mode"] for row in matrix["rows"]} == {
        "disabled",
        "shadow",
        "override_experimental",
    }
    assert all(row["frame_count"] == 8 for row in matrix["rows"])
    assert matrix["status"] == "blocked"


def test_diagnostics_are_deterministic_and_do_not_leak_absolute_paths(tmp_path: Path) -> None:
    profile_root = tmp_path / "step2"
    _write_profile(profile_root, "roboparty_rpo_local")
    clip = _write_bvh(tmp_path / "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh", frames=6)
    first = tmp_path / "first"
    second = tmp_path / "second"

    common_args = [
        "--profile-artifact-root",
        str(profile_root),
        "--robots",
        "roboparty_rpo",
        "--clips",
        str(clip),
        "--modes",
        "disabled",
        "shadow",
        "--max-frames",
        "4",
    ]
    assert smoke.main(["--artifact-root", str(first), *common_args]) == 0
    assert smoke.main(["--artifact-root", str(second), *common_args]) == 0

    first_rerun = json.loads((first / "deterministic_rerun.json").read_text())
    second_rerun = json.loads((second / "deterministic_rerun.json").read_text())
    assert first_rerun["diagnostics_hash"] == second_rerun["diagnostics_hash"]
    assert first_rerun["deterministic"] is True
    artifact_text = "\n".join(
        path.read_text()
        for path in sorted(first.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt", ".xml"}
    )
    assert str(tmp_path) not in artifact_text
    assert "/mnt/" not in artifact_text
    assert "/home/" not in artifact_text


def test_fail_on_blocked_returns_nonzero(tmp_path: Path) -> None:
    profile_root = tmp_path / "step2"
    _write_profile(profile_root, "roboparty_rpo_local")
    clip = _write_bvh(tmp_path / "assets/motions/bvh/Neutral_walk_forward_002__A057.bvh", frames=6)

    rc = smoke.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--profile-artifact-root",
            str(profile_root),
            "--robots",
            "roboparty_rpo",
            "--clips",
            str(clip),
            "--modes",
            "shadow",
            "--max-frames",
            "4",
            "--fail-on-blocked",
        ]
    )

    assert rc == 1


def test_shadow_output_drift_is_a_hard_failure() -> None:
    rows = [
        {
            "robot_type": "roboparty_rpo",
            "clip_name": "clip.bvh",
            "mode": "shadow",
            "pipeline_summary": {
                "output_equal_to_disabled_baseline": False,
                "output_diff_max": 0.001,
            },
        }
    ]

    with pytest.raises(smoke.ShadowOutputChangedError, match="roboparty_rpo"):
        smoke.assert_shadow_outputs_match_disabled(rows, tolerance=0.0)
