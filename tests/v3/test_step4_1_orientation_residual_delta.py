from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_delta_records_orientation_breakthrough(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    delta = read_json(artifact_dir / "orientation_delta_vs_step4_0.json")
    quality_delta = read_json(artifact_dir / "quality_delta_vs_step4_0.json")

    assert delta["accepted_breakthrough"] is True
    assert delta["p95_rotation_residual_p95_delta"] <= -0.25
    assert quality_delta["primary_quality_breakthrough"] is True


def test_step4_1_audit_rejects_delta_regression(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    qdelta = read_json(artifact_dir / "quality_delta_vs_step4_0.json")
    qdelta["regressions"] = [{"field": "solver_backed_count"}]
    write_json(artifact_dir / "quality_delta_vs_step4_0.json", qdelta)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["quality_delta_vs_step4_0"] >= 1

