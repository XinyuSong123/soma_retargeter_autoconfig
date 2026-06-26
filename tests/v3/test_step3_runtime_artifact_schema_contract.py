from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.tools import run_v3_runtime_shadow_smoke as smoke


def _write_profile(root: Path, model_id: str, *, status: str = "passed", sha: str | None = None) -> None:
    sha = sha or f"{model_id}-sha"
    path = root / "per_robot" / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "model": {
                    "id": model_id,
                    "local_file_sha256": sha,
                    "source_resolution": {
                        "local_file_sha256": sha,
                        "path": f"assets/robot_zoo/snapshots/{model_id}/model.xml",
                    },
                },
                "semantic_sites": {
                    name: {"body": name.lower()}
                    for name in smoke.DEFAULT_SEMANTIC_NAMES
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


def _write_bvh(path: Path, frames: int = 10) -> Path:
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


def _generate(tmp_path: Path) -> Path:
    profile_root = tmp_path / "profiles"
    artifact_root = tmp_path / "artifacts"
    _write_profile(profile_root, "roboparty_rpo_local", sha="rpo-sha")
    _write_profile(profile_root, "unitree_g1_mjcf", sha="g1-profile-sha")
    clip = _write_bvh(tmp_path / "assets/motions/bvh/wave_R_001__A428.bvh", frames=9)
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
            str(clip),
            "--modes",
            "disabled",
            "shadow",
            "override_experimental",
            "--max-frames",
            "5",
        ]
    )
    assert rc == 0
    return artifact_root


def test_profile_resolution_schema_contains_required_fields(tmp_path: Path) -> None:
    artifact_root = _generate(tmp_path)
    profile_resolution = json.loads((artifact_root / "profile_resolution.json").read_text())

    for robot_type in ("roboparty_rpo", "unitree_g1"):
        resolution = profile_resolution["robots"][robot_type]
        for field in smoke.PROFILE_RESOLUTION_REQUIRED_FIELDS:
            assert field in resolution, field
        assert resolution["robot_type"] == robot_type
        assert resolution["profile_artifact_path"].startswith(
            "artifacts/retargeting_v3_step2_capability/per_robot/"
        )
        assert not Path(resolution["profile_artifact_path"]).is_absolute()


def test_per_clip_diagnostics_schema_contains_required_fields(tmp_path: Path) -> None:
    artifact_root = _generate(tmp_path)
    target_deltas = next((artifact_root / "per_clip").glob("roboparty_rpo/*/target_deltas.json"))
    pipeline_summary = target_deltas.with_name("pipeline_summary.json")
    target_payload = json.loads(target_deltas.read_text())
    pipeline_payload = json.loads(pipeline_summary.read_text())

    for mode, payload in target_payload["modes"].items():
        assert payload["mode"] == mode
        for field in smoke.TARGET_DELTAS_REQUIRED_FIELDS:
            assert field in payload, field
        assert payload["frame_count"] == 5
        assert set(payload["semantic_names"]) == set(smoke.DEFAULT_SEMANTIC_NAMES)
        assert set(payload["per_semantic"]) == set(smoke.DEFAULT_SEMANTIC_NAMES)
        assert payload["capability_policy"]["exact"] in {"blocked", "passed", "not_applicable"}

    for mode, payload in pipeline_payload["modes"].items():
        assert payload["mode"] == mode
        for field in smoke.PIPELINE_SUMMARY_REQUIRED_FIELDS:
            assert field in payload, field
        assert payload["output_frame_count"] == 5


def test_acceptance_ledger_and_test_result_schema_are_written(tmp_path: Path) -> None:
    artifact_root = _generate(tmp_path)
    ledger = json.loads((artifact_root / "acceptance_ledger.json").read_text())
    pytest_summary = json.loads((artifact_root / "test_results/pytest_summary.json").read_text())
    junit = (artifact_root / "test_results/junit.xml").read_text()
    pytest_text = (artifact_root / "test_results/pytest.txt").read_text()

    assert ledger["schema_version"] == 1
    assert ledger["status"] == "blocked"
    assert {gate["name"] for gate in ledger["gates"]} >= {
        "required_matrix_materialized",
        "shadow_output_equal_to_disabled",
        "diagnostics_deterministic",
        "no_local_absolute_paths",
    }
    assert pytest_summary["status"] == "not_run_by_smoke_runner"
    assert "not run by smoke runner" in pytest_text
    assert "<testsuite" in junit


def test_matrix_builder_rejects_unknown_mode() -> None:
    try:
        smoke.build_matrix(
            robots=["roboparty_rpo"],
            clips=["assets/motions/bvh/wave_R_001__A428.bvh"],
            modes=["shadow", "quality_eval"],
            max_frames=3,
        )
    except ValueError as exc:
        assert "quality_eval" in str(exc)
    else:
        raise AssertionError("unknown mode was accepted")
