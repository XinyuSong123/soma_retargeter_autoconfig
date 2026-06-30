from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_2_orientation_policy_runtime_scoring import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import write_passing_fixture


def test_step4_2_rejects_robot_specific_threshold_shortcut(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    target = source_root / "soma_retargeter/tools/step4_2_orientation_policy_runtime_scoring.py"
    target.parent.mkdir(parents=True)
    target.write_text("per_robot_threshold = {'full_00': 0.1}\n", encoding="utf-8")

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["no_robot_specific_tuning"] >= 1
