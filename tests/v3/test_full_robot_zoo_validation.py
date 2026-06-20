from __future__ import annotations

import json
from pathlib import Path

import soma_retargeter.robotics.v3.validation as validation_module
from soma_retargeter.robotics.v3.robot_zoo import allowed_status_values
from soma_retargeter.robotics.v3.validation import write_validation_artifacts


def _write_manifest(path: Path, models: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "catalog_name": "test-zoo",
        "aggregators": {},
        "policies": {},
        "models": models,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _entry(model_id: str, **overrides) -> dict:
    entry = {
        "id": model_id,
        "description_name": None,
        "format": "mjcf",
        "robot_class": "humanoid",
        "expected_capability": "positive",
        "license": "test",
        "redistribution": "kinematic_snapshot",
        "required": True,
        "source_family": "local",
        "notes": "",
    }
    entry.update(overrides)
    return entry


def test_manifest_entries_all_materialize_structured_source_unavailable(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("missing_positive", source_path=str(tmp_path / "missing.xml")),
            _entry(
                "missing_negative",
                source_path=str(tmp_path / "missing-negative.xml"),
                robot_class="single_arm",
                expected_capability="negative_control",
            ),
        ],
    )

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    assert summary["manifest"]["model_count"] == 2
    assert set(summary["reports"]) == {"missing_positive", "missing_negative"}
    assert summary["status_counts"] == {"source_unavailable": 2}
    assert "compiled_count" not in summary
    assert "compiled" not in summary["manifest"]["allowed_statuses"]
    for model_id in summary["reports"]:
        report = json.loads((tmp_path / "artifacts" / "per_robot" / f"{model_id}.json").read_text())
        assert report["status"] == "source_unavailable"
        assert report["status"] in allowed_status_values()
        assert report["model"]["manifest"]["manifest_model_id"] == model_id
        assert report["reproduction_command"].startswith("python -m soma_retargeter.tools.compile_kinematic_profile_v3")


def test_status_enum_matches_goal_contract():
    assert set(allowed_status_values()) == {
        "passed",
        "partial_passed",
        "negative_control_passed",
        "algorithm_failed",
        "semantic_failed",
        "model_load_failed",
        "source_unavailable",
        "license_blocked",
    }
    assert "compiled" not in allowed_status_values()


def test_reproduction_commands_do_not_contain_local_absolute_paths(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("missing_positive", source_path=str(tmp_path / "missing.xml"))],
    )

    write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    commands = (tmp_path / "artifacts" / "commands.txt").read_text()
    assert str(tmp_path) not in commands
    assert "/mnt/ssd1/" not in commands
    assert "${RETARGETING_V3_ARTIFACTS}/per_robot/missing_positive.json" in commands
    assert "--robot-id missing_positive" in commands


def test_artifacts_do_not_leak_absolute_manifest_paths(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("missing_positive", source_path=str(tmp_path / "missing.xml"))],
    )

    write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    artifact_text = "\n".join(
        path.read_text()
        for path in sorted((tmp_path / "artifacts").rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt"}
    )
    assert str(tmp_path) not in artifact_text
    assert "/mnt/ssd1/" not in artifact_text
    assert "${LOCAL_SOURCE_PATH}/missing.xml" in artifact_text


def test_resolved_source_loader_exception_is_model_load_failed(monkeypatch, tmp_path: Path):
    source = tmp_path / "broken.xml"
    source.write_text("<not-a-model/>\n")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("broken_positive", source_path=str(source))],
    )

    def fail_semantic_map(*args, **kwargs):
        raise RuntimeError("loader failed")

    monkeypatch.setattr(validation_module, "_semantic_map_for_entry", fail_semantic_map)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "broken_positive.json").read_text())

    assert summary["status_counts"] == {"model_load_failed": 1}
    assert summary["model_load_failed_count"] == 1
    assert report["status"] == "model_load_failed"
    assert report["failures"] == ["RuntimeError: loader failed"]


def test_verified_semantic_map_path_is_loaded_without_inference(monkeypatch, tmp_path: Path):
    source = tmp_path / "robot.xml"
    source.write_text("<mujoco/>\n")
    semantic_map = tmp_path / "verified_semantics.json"
    semantic_map.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "semantics": {
                    "Hips": {"body": "pelvis", "source": "verified_map", "confidence": 0.99},
                    "Chest": {"body": "chest", "source": "verified_map", "confidence": 0.99},
                },
            }
        )
        + "\n"
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [_entry("verified_positive", source_path=str(source), semantic_map_path=str(semantic_map))],
    )
    captured = {}

    class FakeProfile:
        failures = []

        def to_json(self):
            return {
                "schema_version": 3,
                "model": {
                    "id": "verified_positive",
                    "path": str(source),
                    "format": "mjcf",
                    "backend": "newton",
                },
                "runtime_adapter": {},
                "failures": [],
                "warnings": [],
                "capability_status": "full_humanoid_ready",
            }

    def fake_compile(model, semantic_map_payload, **kwargs):
        captured["model"] = model
        captured["semantic_map"] = semantic_map_payload
        captured["kwargs"] = kwargs
        return FakeProfile()

    monkeypatch.setattr(validation_module, "compile_kinematic_profile_v3", fake_compile)

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)
    report = json.loads((tmp_path / "artifacts" / "per_robot" / "verified_positive.json").read_text())
    semantic_artifact = json.loads((tmp_path / "artifacts" / "semantic_maps" / "verified_positive.json").read_text())

    assert summary["status_counts"] == {"passed": 1}
    assert captured["semantic_map"]["Hips"]["body"] == "pelvis"
    assert report["semantic_map_resolution"]["source"] == "verified_semantic_map"
    assert semantic_artifact["resolution"]["source"] == "verified_semantic_map"
    assert str(tmp_path) not in json.dumps(report)


def test_cross_format_and_deterministic_scaffolds_are_written(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _entry("toy_urdf", format="urdf", source_path=str(tmp_path / "toy.urdf")),
            _entry("toy_mjcf", format="mjcf", source_path=str(tmp_path / "toy.xml")),
        ],
    )

    summary = write_validation_artifacts(tmp_path / "artifacts", manifest_path=manifest, low_discrepancy_count=1)

    cross_format = json.loads((tmp_path / "artifacts" / "cross_format.json").read_text())
    deterministic = json.loads((tmp_path / "artifacts" / "deterministic_rerun.json").read_text())
    assert cross_format["pairs"]["toy"]["status"] == "not_run"
    assert cross_format["gates"]["same_source_strict"]["status"] == "not_run"
    assert deterministic["status"] == "not_run"
    assert set(deterministic["models"]) == {"toy_urdf", "toy_mjcf"}
    assert summary["cross_format"] == cross_format
