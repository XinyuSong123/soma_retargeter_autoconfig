from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.model_conversion import compare_runtime_models, convert_urdf_to_canonical_mjcf


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
    report = compare_runtime_models(left, right, semantic_map={"Hips": "world", "Chest": "tip"})

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


def test_same_source_gate_a_can_complete_when_projection_reports_are_supplied(tmp_path: Path):
    urdf = _write_urdf(tmp_path / "same_source.urdf")
    mjcf = tmp_path / "same_source.canonical.xml"
    convert_urdf_to_canonical_mjcf(urdf, mjcf)

    projection = {
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
    left = MuJoCoRuntimeModelAdapter(urdf, model_format="urdf")
    right = MuJoCoRuntimeModelAdapter(mjcf, model_format="xml")
    report = compare_runtime_models(
        left,
        right,
        semantic_map={"Hips": "world", "Chest": "tip"},
        canonical_projection_reports=(projection, deepcopy(projection)),
    )

    assert report["failures"] == []
    assert report["strict_equivalent"]
    assert report["gate_a_status"] == "complete_passed"
    assert report["gate_a_evidence_complete"]
    assert report["canonical_projection"]["status"] == "passed"
    assert report["canonical_projection"]["failures"] == []


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
