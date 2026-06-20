from __future__ import annotations

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
    assert report["strict_equivalent"]
    assert report["failures"] == []
    assert report["semantic_fk"]["per_site"]["Chest"]["position_error"] == 0.0
    assert len(report["left_fingerprint"]) == 64
    assert len(report["right_fingerprint"]) == 64


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
