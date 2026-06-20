from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from soma_retargeter.robotics.v3.validation import (
    DEFAULT_LOW_DISCREPANCY_COUNT,
    MANIFEST_MODEL_ID_BY_REPORT_ID,
    REQUIRED_ARTIFACT_IDS,
    write_validation_artifacts,
)


@pytest.fixture(scope="module")
def validation_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("retargeting_v3_artifacts")
    failures_dir = out / "failures"
    failures_dir.mkdir(parents=True)
    (failures_dir / "stale_failure.json").write_text('{"failures": ["old"]}\n')

    summary = write_validation_artifacts(out, low_discrepancy_count=1)
    assert summary["manifest"]["model_count"] >= len(REQUIRED_ARTIFACT_IDS)
    assert len(summary["reports"]) == summary["manifest"]["model_count"]
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _artifact_path(root: Path, value: str) -> Path:
    return root / value.replace("${RETARGETING_V3_ARTIFACTS}/", "")


def test_default_low_discrepancy_count_matches_goal():
    assert DEFAULT_LOW_DISCREPANCY_COUNT == 32


def test_summary_green_run_leaves_failures_empty(validation_artifacts: Path):
    summary = _load(validation_artifacts / "summary.json")
    assert summary["status_counts"]["passed"] >= 1
    assert summary["source_unavailable_count"] >= 1
    assert summary["failure_artifacts_count"] == 0
    assert list((validation_artifacts / "failures").glob("*.json")) == []


def test_per_robot_reproduction_commands_reference_saved_semantic_maps(validation_artifacts: Path):
    commands = (validation_artifacts / "commands.txt").read_text().splitlines()
    compile_commands = [line for line in commands if "compile_kinematic_profile_v3" in line]
    summary = _load(validation_artifacts / "summary.json")
    assert len(compile_commands) == summary["manifest"]["model_count"]

    for command in compile_commands:
        argv = shlex.split(command)
        assert "--robot-id" in argv
        assert "--output" in argv
        report_path = _artifact_path(validation_artifacts, argv[argv.index("--output") + 1])
        if report_path.exists():
            report = _load(report_path)
            assert report["reproduction_command"] == command
            artifact = report["model"]["semantic_map_artifact"]
            if isinstance(artifact, str):
                semantic_map = _artifact_path(validation_artifacts, artifact)
                if not semantic_map.exists():
                    semantic_map = validation_artifacts / "semantic_maps" / f"{report['model']['id']}.json"
                assert semantic_map.exists()
                payload = _load(semantic_map)
                assert payload["semantics"]
                assert payload["source"] in {"verified_semantic_map", "inferred_from_newton_body_names"}


def test_reports_include_model_and_loader_provenance(validation_artifacts: Path):
    for report_id in REQUIRED_ARTIFACT_IDS:
        report = _load(validation_artifacts / "per_robot" / f"{report_id}.json")
        model = report["model"]
        runtime = report["runtime_adapter"]

        if report["status"] == "passed":
            assert isinstance(model["local_file_sha256"], str)
            assert len(model["local_file_sha256"]) == 64
        else:
            assert model["local_file_sha256"]["status"] == "unavailable"
        assert "source_resolution" in model
        assert "manifest" in model
        expected_manifest_id = MANIFEST_MODEL_ID_BY_REPORT_ID.get(report_id)
        assert expected_manifest_id is not None
        assert model["manifest"]["status"] == "available"
        assert model["manifest"]["manifest_model_id"] == expected_manifest_id
        assert model["manifest"]["entry"]["id"] == expected_manifest_id
        assert model["manifest"]["manifest_path"] == "assets/robot_zoo/robot_zoo_manifest.json"
        assert len(model["manifest"]["manifest_sha256"]) == 64
        assert runtime["package_versions"]["newton"] != {"status": "unavailable"}
        assert runtime["loader_provenance"]["loader"]
        assert runtime["loader_provenance"]["compiled_model_manifest"]["status"] == "unavailable"


def test_complete_model_cross_checks_are_materialized(validation_artifacts: Path):
    checks = _load(validation_artifacts / "validation_checks.json")

    g1 = checks["g1_mjcf_urdf_equivalence"]
    assert g1["status"] == "passed"
    assert g1["strict_equivalent"]
    assert g1["differences"] == {}
    assert g1["coordinate_comparison"]["passed"]
