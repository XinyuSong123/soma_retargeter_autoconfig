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
    assert summary["compiled_count"] == len(REQUIRED_ARTIFACT_IDS)
    assert summary["missing_count"] == 0
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_default_low_discrepancy_count_matches_goal():
    assert DEFAULT_LOW_DISCREPANCY_COUNT == 32


def test_summary_green_run_leaves_failures_empty(validation_artifacts: Path):
    summary = _load(validation_artifacts / "summary.json")
    assert all(report["failures"] == [] for report in summary["reports"].values())
    assert summary["failure_artifacts_count"] == 0
    assert list((validation_artifacts / "failures").glob("*.json")) == []


def test_per_robot_reproduction_commands_reference_saved_semantic_maps(validation_artifacts: Path):
    commands = (validation_artifacts / "commands.txt").read_text().splitlines()
    compile_commands = [line for line in commands if "compile_kinematic_profile_v3" in line]
    assert len(compile_commands) == len(REQUIRED_ARTIFACT_IDS)

    for command in compile_commands:
        argv = shlex.split(command)
        assert "--semantic-map" in argv
        semantic_map = validation_artifacts / Path(argv[argv.index("--semantic-map") + 1]).relative_to(validation_artifacts)
        assert semantic_map.exists()
        payload = _load(semantic_map)
        assert payload["semantics"]
        assert payload["source"] in {"default_rpo_semantic_map", "inferred_from_newton_body_names"}

        report_path = validation_artifacts / Path(argv[argv.index("--output") + 1]).relative_to(validation_artifacts)
        report = _load(report_path)
        assert report["reproduction_command"] == command
        assert report["model"]["semantic_map_artifact"] == str(semantic_map)


def test_reports_include_model_and_loader_provenance(validation_artifacts: Path):
    for report_id in REQUIRED_ARTIFACT_IDS:
        report = _load(validation_artifacts / "per_robot" / f"{report_id}.json")
        model = report["model"]
        runtime = report["runtime_adapter"]

        assert isinstance(model["local_file_sha256"], str)
        assert len(model["local_file_sha256"]) == 64
        assert model["source"]["type"] in {"local_workspace_file", "robot_descriptions_module"}
        assert "manifest" in model
        expected_manifest_id = MANIFEST_MODEL_ID_BY_REPORT_ID.get(report_id)
        if expected_manifest_id is None:
            assert report_id == "unitree_g1_23dof"
            assert model["manifest"]["status"] == "unavailable"
        else:
            assert model["manifest"]["status"] == "available"
            assert model["manifest"]["manifest_model_id"] == expected_manifest_id
            assert model["manifest"]["entry"]["id"] == expected_manifest_id
            assert model["manifest"]["manifest_path"] == "assets/robot_zoo/robot_zoo_manifest.json"
            assert len(model["manifest"]["manifest_sha256"]) == 64
        assert runtime["package_versions"]["newton"] != {"status": "unavailable"}
        assert runtime["package_versions"]["robot-descriptions"] != {"status": "unavailable"}
        assert runtime["loader_provenance"]["loader"]
        assert runtime["loader_provenance"]["compiled_model_manifest"]["status"] == "unavailable"


def test_complete_model_cross_checks_are_materialized(validation_artifacts: Path):
    checks = _load(validation_artifacts / "validation_checks.json")

    g1 = checks["g1_mjcf_urdf_equivalence"]
    assert g1["status"] == "documented_limitation"
    assert g1["differences"]
    assert g1["tolerance"]["chain_coordinate_labels"] == "exact ordered match required for strict topology equivalence"

    rpo = checks["roboparty_rpo_distal_hand_endpoint"]
    assert rpo["status"] == "passed_with_documented_scope"
    assert rpo["checks"]["left_hand"]["full_arm_chain_coordinate_count"] >= 5
    assert rpo["checks"]["right_hand"]["full_arm_chain_includes_shoulder_and_elbow"]
    assert "not a separate palm/fingertip mesh anchor" in rpo["scope_note"]

    op3 = checks["robotis_op3_chest_demand_leakage"]
    assert op3["status"] == "passed"
    assert op3["torso_coordinate_labels"] == []
    assert op3["leg_coordinate_labels_in_torso_task"] == []
    assert op3["projection_status"] == "rank_zero"
