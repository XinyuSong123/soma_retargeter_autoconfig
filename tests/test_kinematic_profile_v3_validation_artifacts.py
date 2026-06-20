from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from soma_retargeter.robotics.v3.validation import (
    DEFAULT_LOW_DISCREPANCY_COUNT,
    MANIFEST_MODEL_ID_BY_REPORT_ID,
    REQUIRED_ARTIFACT_IDS,
    _sanitize_xml_artifact,
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


def test_generated_cross_format_xml_paths_are_sanitized(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache_root = tmp_path / "newton-cache"
    mesh_path = cache_root / "unitree_g1" / "meshes" / "pelvis.STL"
    mesh_path.parent.mkdir(parents=True)
    xml_path = tmp_path / "canonical.xml"
    xml_path.write_text(f'<mujoco><asset><mesh file="{mesh_path}"/></asset></mujoco>\n')
    monkeypatch.setenv("NEWTON_CACHE", str(cache_root))

    _sanitize_xml_artifact(xml_path)

    text = xml_path.read_text()
    assert str(tmp_path) not in text
    assert '${NEWTON_CACHE}/unitree_g1/meshes/pelvis.STL' in text


def test_summary_failure_artifacts_match_true_failure_statuses(validation_artifacts: Path):
    summary = _load(validation_artifacts / "summary.json")
    failure_statuses = {"model_load_failed", "semantic_failed", "algorithm_failed"}
    expected_failure_counts = {
        status: summary["status_counts"].get(status, 0)
        for status in sorted(failure_statuses)
        if summary["status_counts"].get(status, 0)
    }
    failure_reports = sorted((validation_artifacts / "failures").glob("*.json"))

    assert summary["status_counts"]["passed"] >= 1
    assert summary["source_unavailable_count"] >= 1
    assert summary["license_blocked_count"] == summary["status_counts"].get("license_blocked", 0)
    assert summary["model_load_failed_count"] == summary["status_counts"].get("model_load_failed", 0)
    assert summary["semantic_failed_count"] == summary["status_counts"].get("semantic_failed", 0)
    assert summary["algorithm_failed_count"] == summary["status_counts"].get("algorithm_failed", 0)
    assert summary["failure_artifact_status_counts"] == expected_failure_counts
    assert summary["failure_artifacts_count"] == sum(expected_failure_counts.values())
    assert len(failure_reports) == summary["failure_artifacts_count"]
    assert "stale_failure" not in {path.stem for path in failure_reports}
    for path in failure_reports:
        assert _load(path)["status"] in failure_statuses


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
                assert payload["source"] == "verified_semantic_map"


def test_required_external_reproducibility_protocol_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_name": "test-zoo",
                "models": [
                    {
                        "id": "missing_positive",
                        "description_name": None,
                        "format": "mjcf",
                        "robot_class": "humanoid",
                        "expected_capability": "positive",
                        "redistribution": "kinematic_snapshot",
                        "required": True,
                        "source_family": "local",
                        "source_path": str(tmp_path / "missing.xml"),
                    }
                ],
            }
        )
        + "\n"
    )
    monkeypatch.setattr("soma_retargeter.robotics.v3.validation._validation_checks", lambda output_dir: {})

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    commands = (tmp_path / "artifacts" / "commands.txt").read_text()
    protocol = summary["required_reproducibility_artifacts"]

    assert protocol["producer"] == "external_test_protocol"
    assert protocol["required_files"] == [
        "acceptance_ledger.json",
        "test_results/pytest.txt",
        "test_results/junit.xml",
        "test_results/coverage.json",
    ]
    assert "${RETARGETING_V3_ARTIFACTS}/acceptance_ledger.json" in commands
    assert "${RETARGETING_V3_ARTIFACTS}/test_results/pytest.txt" in commands
    assert "${RETARGETING_V3_ARTIFACTS}/test_results/junit.xml" in commands
    assert "${RETARGETING_V3_ARTIFACTS}/test_results/coverage.json" in commands


def test_reports_include_model_and_loader_provenance(validation_artifacts: Path):
    for report_id in REQUIRED_ARTIFACT_IDS:
        report = _load(validation_artifacts / "per_robot" / f"{report_id}.json")
        model = report["model"]
        runtime = report["runtime_adapter"]

        source_available = model["source_resolution"]["status"] == "available"
        if source_available:
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
    assert g1["status"] in {"passed", "incomplete", "source_unavailable"}
    if g1["status"] == "passed":
        assert g1["gate_a_status"] == "complete_passed"
        assert g1["gate_a_evidence_complete"]
        assert set(g1["evidence_statuses"].values()) == {"passed"}
    elif g1["status"] == "incomplete":
        assert not g1["gate_a_evidence_complete"]
        assert g1["strict_equivalent"]
        assert g1["gate_a_status"] == "incomplete"
        assert g1["differences"] == {}
        assert g1["coordinate_comparison"]["passed"]
        assert g1["evidence_statuses"]["semantic_fk"] == "unavailable"
        assert g1["evidence_statuses"]["canonical_projection"] == "unavailable"
    else:
        assert not g1["gate_a_evidence_complete"]
        assert g1["gate_a_status"] == "blocked"
        assert g1["differences"] == {"source": "unavailable"}
