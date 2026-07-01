from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_3_normalized_residual_gate_reconciliation_fixture import write_passing_fixture


def test_step4_3_determinism_requires_44_of_44(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    deterministic = read_json(artifact_dir / "deterministic_rerun.json")
    deterministic["matched_count"] = 43
    deterministic["deterministic_matched_count"] = 43
    write_json(artifact_dir / "deterministic_rerun.json", deterministic)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["deterministic_rerun"] >= 1
