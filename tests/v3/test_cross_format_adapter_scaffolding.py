from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import soma_retargeter.robotics.v3.validation as validation_module
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.model_conversion import (
    REQUIRED_CANONICAL_PROJECTION_TASKS,
    compare_runtime_models,
    convert_urdf_to_canonical_mjcf,
)
from soma_retargeter.robotics.v3.target_builder import CANONICAL_MOTION_NAMES


def _write_urdf(path: Path) -> Path:
    path.write_text(
        """
<robot name="same_source">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="tip">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/>
    <child link="tip"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""".strip()
        + "\n"
    )
    return path


def _verified_semantic_map() -> dict:
    return {
        "Hips": {
            "body": "world",
            "source": "verified_semantic_map",
            "confidence": 0.99,
            "evidence": ["test_topology:world"],
        },
        "Chest": {
            "body": "tip",
            "source": "verified_semantic_map",
            "confidence": 0.99,
            "evidence": ["test_topology:tip"],
        },
    }


def _canonical_projection_report(
    *,
    target_source: str = "canonical_semantic_targets",
    desired_source: str = "canonical_targets.transforms",
) -> dict:
    tasks = {
        task: {
            "status": "projected",
            "converged": True,
            "residual": 0.0,
            "normalized_residual": 0.0,
            "normalization_scale": 1.0,
            "iterations": 1,
            "active_coordinates": [0],
            "desired_source": desired_source,
            "reference": "Hips",
            "target": "Chest",
        }
        for task in REQUIRED_CANONICAL_PROJECTION_TASKS
    }
    return {
        "motion_order": list(CANONICAL_MOTION_NAMES),
        "target_source": target_source,
        "failures": [],
        "unreachable_demands": [],
        "motions": {motion: {"tasks": deepcopy(tasks)} for motion in CANONICAL_MOTION_NAMES},
    }


def _neutral_test_fixture_projection() -> dict:
    return {
        "motion_order": ["neutral"],
        "target_source": "test_fixture",
        "failures": [],
        "unreachable_demands": [],
        "motions": {
            "neutral": {
                "tasks": {
                    "torso": {
                        "status": "projected",
                        "converged": True,
                        "residual": 0.0,
                        "normalized_residual": 0.0,
                        "normalization_scale": 1.0,
                        "iterations": 0,
                        "active_coordinates": [0],
                        "desired_source": "test_fixture",
                        "reference": "Hips",
                        "target": "Chest",
                    }
                }
            }
        },
    }


def test_same_source_urdf_to_canonical_mjcf_strict_equivalence(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"

    conversion = convert_urdf_to_canonical_mjcf(urdf, mjcf)
    assert mjcf.exists()
    assert conversion["settings"]["canonical_format"] == "mjcf"
    assert conversion["source_sha256"] != conversion["output_sha256"]
    assert conversion["runtime_signature"]["nv"] == 1

    left = MuJoCoRuntimeModelAdapter(urdf, model_format="urdf")
    right = MuJoCoRuntimeModelAdapter(mjcf, model_format="xml")
    report = compare_runtime_models(left, right, semantic_map=_verified_semantic_map())

    assert report["comparison_mode"] == "same_source_strict"
    assert report["schema_version"] == 2
    assert report["strict_equivalent"]
    assert report["gate_a_status"] == "incomplete"
    assert not report["gate_a_evidence_complete"]
    assert report["gate_a_required_sections"] == [
        "semantic_fk",
        "active_chains",
        "rank_summary",
        "canonical_projection",
    ]
    assert report["failures"] == []
    assert report["semantic_fk"]["status"] == "passed"
    assert report["semantic_fk"]["per_site"]["Chest"]["position_error"] == 0.0
    assert report["active_chains"]["status"] == "passed"
    assert report["active_chains"]["per_task"]["torso"]["left"]["coordinate_labels"] == ["hinge"]
    assert report["rank_summary"]["status"] == "passed"
    assert report["rank_summary"]["per_task"]["torso"]["left"]["rotation_rank"] == 1
    assert report["canonical_projection"]["status"] == "unavailable"
    assert report["canonical_projection"]["reason"] == "canonical_projection_reports_not_provided"
    assert len(report["left_fingerprint"]) == 64
    assert len(report["right_fingerprint"]) == 64


def test_same_source_gate_a_schema_records_unavailable_reasons_without_semantics(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"
    convert_urdf_to_canonical_mjcf(urdf, mjcf)

    left = MuJoCoRuntimeModelAdapter(urdf, model_format="urdf")
    right = MuJoCoRuntimeModelAdapter(mjcf, model_format="xml")
    report = compare_runtime_models(left, right)

    assert report["strict_equivalent"]
    assert report["gate_a_status"] == "incomplete"
    assert report["semantic_fk"]["status"] == "unavailable"
    assert report["semantic_fk"]["reason"] == "semantic_map_not_provided"
    assert report["active_chains"]["status"] == "unavailable"
    assert report["active_chains"]["reason"] == "semantic_sites_unavailable"
    assert report["rank_summary"]["status"] == "unavailable"
    assert report["rank_summary"]["reason"] == "semantic_sites_unavailable"
    assert report["canonical_projection"]["status"] == "unavailable"
    assert report["canonical_projection"]["reason"] == "canonical_projection_reports_not_provided"


def test_same_source_gate_a_rejects_neutral_only_test_fixture_projection_reports(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"
    convert_urdf_to_canonical_mjcf(urdf, mjcf)

    projection = _neutral_test_fixture_projection()
    left = MuJoCoRuntimeModelAdapter(urdf, model_format="urdf")
    right = MuJoCoRuntimeModelAdapter(mjcf, model_format="xml")
    report = compare_runtime_models(
        left,
        right,
        semantic_map=_verified_semantic_map(),
        canonical_projection_reports=(projection, deepcopy(projection)),
    )

    assert report["failures"] == []
    assert report["strict_equivalent"]
    assert report["gate_a_status"] == "incomplete"
    assert not report["gate_a_evidence_complete"]
    assert report["canonical_projection"]["status"] == "unavailable"
    assert report["canonical_projection"]["reason"] == "canonical_projection_reports_neutral_only"


def test_same_source_gate_a_can_complete_when_canonical_projection_reports_are_supplied(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"
    convert_urdf_to_canonical_mjcf(urdf, mjcf)

    projection = _canonical_projection_report()
    left = MuJoCoRuntimeModelAdapter(urdf, model_format="urdf")
    right = MuJoCoRuntimeModelAdapter(mjcf, model_format="xml")
    report = compare_runtime_models(
        left,
        right,
        semantic_map=_verified_semantic_map(),
        canonical_projection_reports=(projection, deepcopy(projection)),
    )

    assert report["failures"] == []
    assert report["strict_equivalent"]
    assert report["gate_a_status"] == "complete_passed"
    assert report["gate_a_evidence_complete"]
    assert report["canonical_projection"]["status"] == "passed"
    assert report["canonical_projection"]["failures"] == []


def test_same_source_gate_a_requires_verified_semantic_map(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"
    convert_urdf_to_canonical_mjcf(urdf, mjcf)

    projection = _canonical_projection_report()
    left = MuJoCoRuntimeModelAdapter(urdf, model_format="urdf")
    right = MuJoCoRuntimeModelAdapter(mjcf, model_format="xml")
    report = compare_runtime_models(
        left,
        right,
        semantic_map={"Hips": "world", "Chest": "tip"},
        canonical_projection_reports=(projection, deepcopy(projection)),
    )

    assert report["strict_equivalent"]
    assert report["gate_a_status"] == "incomplete"
    assert report["semantic_fk"]["status"] == "unavailable"
    assert report["semantic_fk"]["reason"] == "verified_semantic_map_not_provided"
    assert report["semantic_map_evidence"]["unverified_semantics"] == ["Chest", "Hips"]


def test_validation_same_source_check_keeps_incomplete_gate_a_incomplete(monkeypatch, tmp_path: Path):
    source = _write_urdf(tmp_path / "same_source.urdf")
    equivalence = {
        "schema_version": 2,
        "strict_equivalent": True,
        "failures": [],
        "gate_a_status": "incomplete",
        "gate_a_evidence_complete": False,
        "gate_a_required_sections": [
            "semantic_fk",
            "active_chains",
            "rank_summary",
            "canonical_projection",
        ],
        "left_signature": {"nq": 1, "nv": 1},
        "right_signature": {"nq": 1, "nv": 1},
        "coordinate_comparison": {"passed": True, "failures": []},
        "semantic_fk": {
            "status": "unavailable",
            "passed": False,
            "reason": "semantic_map_not_provided",
            "failures": [],
        },
        "active_chains": {
            "status": "unavailable",
            "passed": False,
            "reason": "semantic_sites_unavailable",
            "failures": [],
        },
        "rank_summary": {
            "status": "unavailable",
            "passed": False,
            "reason": "semantic_sites_unavailable",
            "failures": [],
        },
        "canonical_projection": {
            "status": "unavailable",
            "passed": False,
            "reason": "canonical_projection_reports_not_provided",
            "failures": [],
        },
        "tolerances": {
            "position_atol": 1e-7,
            "rotation_atol": 1e-7,
            "rank_singular_value_atol": 1e-6,
            "projection_atol": 1e-9,
        },
    }

    class FakeAdapter:
        def __init__(self, path, *, model_format):
            self.path = path
            self.model_format = model_format

        def close(self):
            pass

    def fake_convert(source_path, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<mujoco/>\n")
        return {"source_sha256": "0" * 64, "output_sha256": "1" * 64}

    monkeypatch.setattr(validation_module, "_find_g1_same_source_urdf", lambda: source)
    monkeypatch.setattr(validation_module, "convert_urdf_to_canonical_mjcf", fake_convert)
    monkeypatch.setattr(validation_module, "MuJoCoRuntimeModelAdapter", FakeAdapter)
    monkeypatch.setattr(validation_module, "compare_runtime_models", lambda left, right: equivalence)

    check = validation_module._g1_same_source_strict_check(tmp_path / "artifacts")

    assert check["status"] == "incomplete"
    assert check["strict_equivalent"]
    assert check["gate_a_status"] == "incomplete"
    assert not check["gate_a_evidence_complete"]
    assert check["evidence_statuses"] == {
        "semantic_fk": "unavailable",
        "active_chains": "unavailable",
        "rank_summary": "unavailable",
        "canonical_projection": "unavailable",
    }
    assert check["evidence_incomplete_reasons"]["semantic_fk"] == "semantic_map_not_provided"
    assert check["evidence_incomplete_reasons"]["canonical_projection"] == (
        "canonical_projection_reports_not_provided"
    )


def test_validation_same_source_check_passes_only_complete_gate_a(monkeypatch, tmp_path: Path):
    source = _write_urdf(tmp_path / "same_source.urdf")
    passed_section = {"status": "passed", "passed": True, "failures": []}
    equivalence = {
        "schema_version": 2,
        "strict_equivalent": True,
        "failures": [],
        "gate_a_status": "complete_passed",
        "gate_a_evidence_complete": True,
        "gate_a_required_sections": [
            "semantic_fk",
            "active_chains",
            "rank_summary",
            "canonical_projection",
        ],
        "left_signature": {"nq": 1, "nv": 1},
        "right_signature": {"nq": 1, "nv": 1},
        "coordinate_comparison": {"passed": True, "failures": []},
        "semantic_fk": passed_section,
        "active_chains": passed_section,
        "rank_summary": passed_section,
        "canonical_projection": passed_section,
        "tolerances": {
            "position_atol": 1e-7,
            "rotation_atol": 1e-7,
            "rank_singular_value_atol": 1e-6,
            "projection_atol": 1e-9,
        },
    }

    class FakeAdapter:
        def __init__(self, path, *, model_format):
            self.path = path
            self.model_format = model_format

        def close(self):
            pass

    def fake_convert(source_path, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<mujoco/>\n")
        return {"source_sha256": "0" * 64, "output_sha256": "1" * 64}

    monkeypatch.setattr(validation_module, "_find_g1_same_source_urdf", lambda: source)
    monkeypatch.setattr(validation_module, "convert_urdf_to_canonical_mjcf", fake_convert)
    monkeypatch.setattr(validation_module, "MuJoCoRuntimeModelAdapter", FakeAdapter)
    monkeypatch.setattr(validation_module, "compare_runtime_models", lambda left, right: equivalence)

    check = validation_module._g1_same_source_strict_check(tmp_path / "artifacts")

    assert check["status"] == "passed"
    assert check["gate_a_status"] == "complete_passed"
    assert check["gate_a_evidence_complete"]
    assert check["evidence_incomplete_reasons"] == {}


def test_validation_same_source_check_uses_manifest_semantics_and_projection_reports(monkeypatch, tmp_path: Path):
    source = _write_urdf(tmp_path / "same_source.urdf")
    semantic_map_path = tmp_path / "unitree_g1_urdf_semantics.json"
    semantic_map_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "verification_status": "verified",
                "semantics": _verified_semantic_map(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "unitree_g1_urdf",
                        "description_name": None,
                        "format": "urdf",
                        "robot_class": "humanoid",
                        "expected_capability": "positive",
                        "license": "test",
                        "redistribution": "kinematic_snapshot",
                        "required": True,
                        "source_family": "local",
                        "source_path": str(source),
                        "semantic_map_path": str(semantic_map_path),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    source_projection = _canonical_projection_report()
    generated_projection = deepcopy(source_projection)
    captured = {}

    class FakeAdapter:
        def __init__(self, path, *, model_format):
            self.path = path
            self.model_format = model_format

        def close(self):
            pass

    def fake_convert(source_path, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("<mujoco/>\n")
        return {"source_sha256": "0" * 64, "output_sha256": "1" * 64}

    def fake_compare(left, right, *, semantic_map=None, canonical_projection_reports=None):
        captured["left_path"] = left.path
        captured["right_path"] = right.path
        captured["semantic_map"] = semantic_map
        captured["canonical_projection_reports"] = canonical_projection_reports
        passed_section = {"status": "passed", "passed": True, "failures": []}
        return {
            "schema_version": 2,
            "strict_equivalent": True,
            "failures": [],
            "gate_a_status": "complete_passed",
            "gate_a_evidence_complete": True,
            "gate_a_required_sections": [
                "semantic_fk",
                "active_chains",
                "rank_summary",
                "canonical_projection",
            ],
            "left_signature": {"nq": 1, "nv": 1},
            "right_signature": {"nq": 1, "nv": 1},
            "coordinate_comparison": {"passed": True, "failures": []},
            "semantic_fk": passed_section,
            "active_chains": passed_section,
            "rank_summary": passed_section,
            "canonical_projection": passed_section,
            "tolerances": {
                "position_atol": 1e-7,
                "rotation_atol": 1e-7,
                "rank_singular_value_atol": 1e-6,
                "projection_atol": 1e-9,
            },
        }

    monkeypatch.setattr(validation_module, "convert_urdf_to_canonical_mjcf", fake_convert)
    monkeypatch.setattr(validation_module, "MuJoCoRuntimeModelAdapter", FakeAdapter)
    monkeypatch.setattr(validation_module, "compare_runtime_models", fake_compare)
    def fake_compile_projection(model_path, *, semantic_map, model_format, model_id):
        assert semantic_map["Chest"]["source"] == "verified_semantic_map"
        if model_format == "urdf":
            assert model_path == source
            assert model_id == "unitree_g1_same_source_source_urdf"
            return source_projection
        assert model_format == "mjcf"
        assert model_id == "unitree_g1_same_source_canonical_mjcf"
        return generated_projection

    monkeypatch.setattr(validation_module, "_compile_same_source_canonical_projection", fake_compile_projection)

    check = validation_module._g1_same_source_strict_check(
        tmp_path / "artifacts",
        manifest_path=manifest,
        full_reports={"unitree_g1_urdf": {"canonical_projection_reports": source_projection}},
    )

    assert check["status"] == "passed"
    assert captured["left_path"] == source
    assert captured["semantic_map"]["Chest"]["source"] == "verified_semantic_map"
    assert captured["canonical_projection_reports"] == (source_projection, generated_projection)
    assert check["source_resolution"]["resolver"] == "manifest_source_path"
    assert check["semantic_map_resolution"]["path"] == "${LOCAL_SOURCE_PATH}/unitree_g1_urdf_semantics.json"
    assert check["projection_report_resolution"]["status"] == "available"


def test_convert_robot_model_v3_cli_writes_report(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"
    report_path = tmp_path / "report.json"

    from soma_retargeter.tools.convert_robot_model_v3 import main
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "convert_robot_model_v3",
            "--input",
            str(urdf),
            "--output",
            str(mjcf),
            "--report",
            str(report_path),
            "--compare",
        ]
        main()
    finally:
        sys.argv = old_argv

    report = json.loads(report_path.read_text())
    assert report["output"] == str(mjcf)
    assert report["strict_equivalence"]["strict_equivalent"]
