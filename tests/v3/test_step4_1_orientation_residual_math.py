from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from soma_retargeter.tools.step4_1_orientation_residual import canonicalize_quat_xyzw, rotation_log_residual
from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_quaternion_log_map_math_is_shortest_arc() -> None:
    runtime = np.eye(3)
    target = Rotation.from_euler("z", 45, degrees=True).as_matrix()

    residual = rotation_log_residual(runtime, target)

    assert residual["finite"] is True
    assert residual["q_runtime_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert abs(residual["angle_radians"] - np.deg2rad(45)) < 1e-12
    assert canonicalize_quat_xyzw([0.0, 0.0, 0.0, -2.0]) == [-0.0, -0.0, -0.0, 1.0]


def test_step4_1_audit_rejects_nonfinite_log_map_count(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    audit = read_json(artifact_dir / "orientation_residual_math_audit.json")
    audit["finite_log_map_count"] = audit["row_count"] - 1
    write_json(artifact_dir / "orientation_residual_math_audit.json", audit)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["orientation_residual_math"] >= 1

