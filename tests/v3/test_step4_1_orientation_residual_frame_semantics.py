from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_frame_semantics_records_required_conventions(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    matrix = read_json(artifact_dir / "orientation_frame_semantics_matrix.json")
    row = matrix["rows"][0]

    assert matrix["row_count"] >= 128
    assert row["target_frame"]
    assert row["runtime_frame"]
    assert row["source_frame"]
    assert row["quaternion_order"] == "xyzw"
    assert row["sign_canonicalized"] is True


def test_step4_1_audit_rejects_missing_quaternion_canonicalization(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    matrix = read_json(artifact_dir / "orientation_frame_semantics_matrix.json")
    matrix["rows"][0]["sign_canonicalized"] = False
    write_json(artifact_dir / "orientation_frame_semantics_matrix.json", matrix)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["orientation_frame_semantics"] >= 1

