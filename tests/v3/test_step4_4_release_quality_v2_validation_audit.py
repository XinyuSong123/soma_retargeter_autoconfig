from __future__ import annotations

from pathlib import Path

from scripts.audit_retargeting_v3_step4_4_release_quality_v2_validation import run_audit
from tests.v3.step4_2_orientation_policy_runtime_scoring_fixture import read_json, write_json
from tests.v3.step4_4_release_quality_v2_validation_fixture import write_passing_fixture


def test_step4_4_audit_rejects_missing_candidate_warned_deep_audit(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    (artifact_dir / "candidate_warned_deep_audit.json").unlink()

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["missing_required_artifacts"] >= 1


def test_step4_4_audit_rejects_candidate_counted_as_legacy_pass(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    summary = read_json(artifact_dir / "quality_summary.json")
    summary["runtime_quality_passed_count"] = 1
    summary["runtime_quality_warned_count"] = 31
    write_json(artifact_dir / "quality_summary.json", summary)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["status_semantics"] >= 1


def test_step4_4_audit_rejects_threshold_lowering(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    gate = read_json(artifact_dir / "gate_reconciliation_v3_report.json")
    gate["candidate_gate_inputs"]["candidate_warn"] = 0.65
    gate["candidate_gate_inputs"]["candidate_thresholds_lowered"] = True
    write_json(artifact_dir / "gate_reconciliation_v3_report.json", gate)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["threshold_integrity"] >= 1


def test_step4_4_audit_rejects_missing_global_method_trial(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    stress = read_json(artifact_dir / "release_quality_v2_stress_test.json")
    stress["global_methods_tried"] = [
        row for row in stress["global_methods_tried"] if row["method"] != "global_task_class_residual_weighting"
    ]
    write_json(artifact_dir / "release_quality_v2_stress_test.json", stress)

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["stress_test"] >= 1


def test_step4_4_audit_rejects_specific_model_shortcut(tmp_path: Path) -> None:
    artifact_dir, baseline_dir, source_root = write_passing_fixture(tmp_path)
    target = source_root / "soma_retargeter/tools/step4_4_release_quality_v2_validation.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("accepted_models = {'full_00'}\n", encoding="utf-8")

    result = run_audit(artifact_dir=artifact_dir, baseline_artifact_dir=baseline_dir, source_root=source_root)

    assert result.status.startswith("BLOCKED")
    assert result.gate_counts["no_robot_specific_tuning"] >= 1
