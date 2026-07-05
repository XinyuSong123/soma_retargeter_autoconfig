from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.audit_retargeting_v3_step4_5_release_quality_v2_generalization import run_audit


ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_5_release_quality_v2_generalization")
BASELINE_STEP4_4_DIR = Path("artifacts/retargeting_v3_step4_4_release_quality_v2_validation")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_artifacts(tmp_path: Path) -> Path:
    target = tmp_path / "step4_5"
    shutil.copytree(ARTIFACT_DIR, target)
    return target


def test_step4_5_audit_accepts_committed_blocked_generalization_artifacts() -> None:
    result = run_audit(ARTIFACT_DIR, BASELINE_STEP4_4_DIR, Path("."))

    assert result.blocking_count == 0
    assert result.status == "BLOCKED_CLIP_GENERALIZATION"
    assert result.clip_manifest["clip_count"] == 11
    assert result.strict_policy_counts == {
        "release_quality_candidate_blocked_count": 32,
        "rows_below_candidate_pass_gate": 0,
        "rows_below_candidate_warn_gate": 0,
    }


def test_step4_5_audit_rejects_counted_task_class_weighting(tmp_path: Path) -> None:
    artifact_dir = copy_artifacts(tmp_path)
    contract = read_json(artifact_dir / "task_level_residual_contract.json")
    contract["task_class_weighting_counted"] = True
    write_json(artifact_dir / "task_level_residual_contract.json", contract)

    result = run_audit(artifact_dir, BASELINE_STEP4_4_DIR, Path("."))

    assert result.blocking_count >= 1
    assert result.gate_counts["task_contract"] >= 1

