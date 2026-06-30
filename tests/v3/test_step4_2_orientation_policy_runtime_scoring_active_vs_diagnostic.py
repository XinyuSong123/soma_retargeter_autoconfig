from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json, write_passing_fixture


def test_step4_2_records_active_diagnostic_and_production_policy_roles(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    matrix = read_json(artifact_dir / "active_vs_diagnostic_policy_matrix.json")

    assert matrix["row_count"] == 32
    assert all(row["active_for_step4_2_scoring"] is True for row in matrix["rows"])
    assert all(row["production_default_changed"] is False for row in matrix["rows"])
    assert run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root).blocking_count == 0


def test_step4_2_rejects_default_enabled_runtime_override(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_override_default_enabled"] = True
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["release_candidate_status"] >= 1
