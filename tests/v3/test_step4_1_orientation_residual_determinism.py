from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_1_orientation_residual_breakthrough import run_audit
from tests.v3.step4_1_orientation_residual_fixture import read_json, write_json, write_passing_fixture


def test_step4_1_determinism_covers_all_44_rows(tmp_path: Path) -> None:
    artifact_dir, _, _ = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")

    assert deterministic["deterministic_compared_count"] == 44
    assert deterministic["deterministic_matched_count"] == 44


def test_step4_1_audit_rejects_determinism_mismatch(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")
    deterministic["deterministic_matched_count"] = 43
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["deterministic_rerun"] >= 1

