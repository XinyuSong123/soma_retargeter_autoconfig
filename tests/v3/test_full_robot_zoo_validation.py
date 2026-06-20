from __future__ import annotations

import json
from pathlib import Path

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
