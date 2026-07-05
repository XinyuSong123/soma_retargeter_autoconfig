#!/usr/bin/env python3
"""Audit Step 4.5 release-quality-v2 expanded-suite generalization artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_5_release_quality_v2_generalization")
DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_4_release_quality_v2_validation")
DEFAULT_SOURCE_ROOT = Path(".")
EXPECTED_FULL_MODELS = 32
MIN_EXPANDED_CLIP_COUNT = 5
VALID_STATUSES = {
    "PASS_RC",
    "BLOCKED_CLIP_GENERALIZATION",
    "BLOCKED_TASK_CONTRACT",
    "BLOCKED_HIPS_ROTATION_RESIDUAL",
    "BLOCKED_CI_OR_PROVENANCE",
}
REQUIRED_ARTIFACT_FILES = (
    "quality_summary.json",
    "acceptance_ledger.json",
    "expanded_clip_manifest.json",
    "release_quality_v2_expanded_validation_matrix.json",
    "candidate_warned_expanded_stability.json",
    "task_level_residual_contract.json",
    "hips_root_rotation_residual_decomposition.json",
    "solver_convergence_weak_global_diagnostics.json",
    "release_quality_v2_generalization_readiness.json",
    "quality_delta_vs_step4_4.json",
    "full_pipeline_matrix.json",
    "clip_matrix.json",
    "gate_reconciliation_v2_report.json",
    "pipeline_config.json",
    "solver_config.json",
    "deterministic_rerun.json",
    "commands.txt",
)


@dataclass(frozen=True)
class Finding:
    gate: str
    severity: str
    subject: str
    message: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class AuditResult:
    status: str
    quality_status: str
    artifact_dir: str
    baseline_step4_4_artifact_dir: str
    source_root: str
    blocking_count: int
    finding_count: int
    gate_counts: dict[str, int]
    findings: list[Finding]
    clip_manifest: dict[str, Any]
    candidate_stability: dict[str, Any]
    strict_policy_counts: dict[str, Any]
    task_contract: dict[str, Any]
    hips_root_rotation: dict[str, Any]
    solver_convergence_weak: dict[str, Any]
    final_head_ci: dict[str, Any]
    lfs: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    baseline_step4_4_artifact_dir: Path = DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    require_final_head_ci: bool = False,
) -> AuditResult:
    artifact_dir = Path(artifact_dir).resolve()
    baseline_step4_4_artifact_dir = Path(baseline_step4_4_artifact_dir).resolve()
    source_root = Path(source_root).resolve()

    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    manifest = _read_json(artifact_dir / "expanded_clip_manifest.json")
    validation = _read_json(artifact_dir / "release_quality_v2_expanded_validation_matrix.json")
    stability = _read_json(artifact_dir / "candidate_warned_expanded_stability.json")
    task_contract = _read_json(artifact_dir / "task_level_residual_contract.json")
    hips = _read_json(artifact_dir / "hips_root_rotation_residual_decomposition.json")
    solver_weak = _read_json(artifact_dir / "solver_convergence_weak_global_diagnostics.json")
    readiness = _read_json(artifact_dir / "release_quality_v2_generalization_readiness.json")
    delta = _read_json(artifact_dir / "quality_delta_vs_step4_4.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    clip_matrix = _read_json(artifact_dir / "clip_matrix.json")
    gate_v2 = _read_json(artifact_dir / "gate_reconciliation_v2_report.json")
    pipeline_config = _read_json(artifact_dir / "pipeline_config.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    baseline_summary = _read_json(baseline_step4_4_artifact_dir / "quality_summary.json")
    baseline_gate = _read_json(baseline_step4_4_artifact_dir / "gate_reconciliation_v3_report.json")

    findings: list[Finding] = []
    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_baseline_step4_4(baseline_summary, baseline_gate))
    findings.extend(_audit_expanded_clip_manifest(manifest, clip_matrix, pipeline_config))
    findings.extend(_audit_expanded_validation(validation, summary, gate_v2, full_pipeline, manifest))
    findings.extend(_audit_candidate_stability(stability, summary, baseline_gate))
    findings.extend(_audit_task_contract(task_contract))
    findings.extend(_audit_hips_root_rotation(hips, validation))
    findings.extend(_audit_solver_convergence_weak(solver_weak, manifest))
    findings.extend(_audit_readiness(readiness, validation, stability, task_contract, hips))
    findings.extend(_audit_quality_delta(delta, baseline_summary, validation, summary))
    findings.extend(_audit_no_leakage_or_weakening(summary, ledger, validation, task_contract))
    findings.extend(_audit_deterministic(deterministic))
    findings.extend(_audit_lfs(summary, ledger))
    if require_final_head_ci:
        findings.extend(_audit_final_head_ci(_final_head_ci(summary, ledger), source_root))

    gate_counts = dict(Counter(finding.gate for finding in findings))
    blocking_count = sum(1 for finding in findings if finding.severity == "error")
    quality_status = str(summary.get("release_candidate_status") or ledger.get("release_candidate_status") or "")
    if blocking_count:
        status = "BLOCKED_CI_OR_PROVENANCE" if require_final_head_ci and gate_counts.get("final_head_ci") else "BLOCKED_TASK_CONTRACT"
    elif quality_status in VALID_STATUSES:
        status = quality_status
    else:
        status = "BLOCKED_TASK_CONTRACT"
    return AuditResult(
        status=status,
        quality_status=quality_status,
        artifact_dir=_display_path(artifact_dir),
        baseline_step4_4_artifact_dir=_display_path(baseline_step4_4_artifact_dir),
        source_root=_display_path(source_root),
        blocking_count=blocking_count,
        finding_count=len(findings),
        gate_counts=gate_counts,
        findings=findings,
        clip_manifest={
            "clip_count": manifest.get("clip_count"),
            "manifest_hash": manifest.get("manifest_hash"),
            "clip_ids": manifest.get("clip_ids", []),
        },
        candidate_stability={
            "original_rows_remain_stable_on_expanded_suite": stability.get("original_rows_remain_stable_on_expanded_suite"),
            "original_stable_under_expanded_guard_count": stability.get("original_stable_under_expanded_guard_count"),
            "newly_stable_under_expanded_guard_count": stability.get("newly_stable_under_expanded_guard_count"),
        },
        strict_policy_counts=validation.get("strict_policy_counts", {}),
        task_contract={
            "contract_status": task_contract.get("contract_status"),
            "task_class_weighting_counted": task_contract.get("task_class_weighting_counted"),
        },
        hips_root_rotation={
            "primary_blocker": hips.get("primary_blocker"),
            "hips_dominant_model_count": hips.get("hips_dominant_model_count"),
            "hips_rotation_dominant_model_count": hips.get("hips_rotation_dominant_model_count"),
        },
        solver_convergence_weak={
            "weak_row_count": solver_weak.get("weak_row_count"),
            "global_solver_diagnostics_only": solver_weak.get("global_solver_diagnostics_only"),
        },
        final_head_ci=_final_head_ci(summary, ledger),
        lfs=_lfs(summary, ledger),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--baseline-step4-4-artifact-dir", default=str(DEFAULT_BASELINE_STEP4_4_ARTIFACT_DIR))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-json")
    parser.add_argument("--write-report", dest="output_json")
    parser.add_argument("--require-final-head-ci", action="store_true")
    args = parser.parse_args(argv)
    result = run_audit(
        artifact_dir=Path(args.artifact_dir),
        baseline_step4_4_artifact_dir=Path(args.baseline_step4_4_artifact_dir),
        source_root=Path(args.source_root),
        require_final_head_ci=args.require_final_head_ci,
    )
    payload = result.to_json()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if result.blocking_count else 0


def _audit_required_artifacts(artifact_dir: Path) -> list[Finding]:
    findings = []
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(_finding("required_artifacts", relative, "required Step 4.5 artifact is missing", {}))
    return findings


def _audit_baseline_step4_4(summary: dict[str, Any], gate: dict[str, Any]) -> list[Finding]:
    findings = []
    expected = {
        "release_candidate_status": "PASS_RC",
        "runtime_quality_passed_count": 0,
        "runtime_quality_warned_count": 32,
        "runtime_quality_failed_count": 0,
        "release_quality_candidate_warned_count": 9,
        "release_quality_candidate_blocked_count": 23,
        "rows_below_candidate_warn_gate": 9,
        "rows_below_candidate_pass_gate": 0,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            findings.append(_finding("baseline_step4_4", key, "baseline Step 4.4 truth mismatch", {"actual": summary.get(key), "expected": expected_value}))
    if len(gate.get("rows_below_candidate_warn_gate", [])) != 9:
        findings.append(_finding("baseline_step4_4", "gate_reconciliation_v3_report.json", "Step 4.4 gate report must expose nine candidate-warned rows", {"actual": gate.get("rows_below_candidate_warn_gate")}))
    return findings


def _audit_expanded_clip_manifest(manifest: dict[str, Any], clip_matrix: dict[str, Any], pipeline_config: dict[str, Any]) -> list[Finding]:
    findings = []
    clip_count = _as_int(manifest.get("clip_count"))
    if clip_count < MIN_EXPANDED_CLIP_COUNT:
        findings.append(_finding("expanded_clip_manifest", "clip_count", "expanded clip manifest is not larger than Step 4.4", {"clip_count": clip_count}))
    if manifest.get("frozen_before_step4_5_metrics_finalization") is not True:
        findings.append(_finding("expanded_clip_manifest", "frozen_before_step4_5_metrics_finalization", "expanded clip manifest must be frozen before Step 4.5 finalization", {}))
    for field in ("no_clip_removed_after_metric_review", "hard_clip_removal_used", "robot_specific_clip_selection_used"):
        expected = False if field.endswith("_used") else True
        if manifest.get(field) is not expected:
            findings.append(_finding("expanded_clip_manifest", field, "manifest clip selection safety field mismatch", {"actual": manifest.get(field), "expected": expected}))
    matrix_clip_ids = sorted({str(row.get("clip_id")) for row in _full_clip_rows(clip_matrix) if row.get("clip_id")})
    manifest_clip_ids = sorted(str(value) for value in manifest.get("clip_ids", []))
    if matrix_clip_ids != manifest_clip_ids:
        findings.append(_finding("expanded_clip_manifest", "clip_ids", "manifest clip ids must match full clip matrix", {"manifest": manifest_clip_ids, "matrix": matrix_clip_ids}))
    configured = pipeline_config.get("config", {}) if isinstance(pipeline_config.get("config"), dict) else {}
    if _as_int(configured.get("solver_smoke_clip_limit")) != clip_count:
        findings.append(_finding("expanded_clip_manifest", "solver_smoke_clip_limit", "solver smoke limit must cover every frozen clip", {"actual": configured.get("solver_smoke_clip_limit"), "clip_count": clip_count}))
    return findings


def _audit_expanded_validation(
    validation: dict[str, Any],
    summary: dict[str, Any],
    gate_v2: dict[str, Any],
    full_pipeline: dict[str, Any],
    manifest: dict[str, Any],
) -> list[Finding]:
    findings = []
    model_count = _as_int(validation.get("model_count"))
    clip_count = _as_int(validation.get("clip_count"))
    if model_count != EXPECTED_FULL_MODELS:
        findings.append(_finding("expanded_validation", "model_count", "expanded validation must cover 32 full humanoid rows", {"actual": model_count}))
    if clip_count != _as_int(manifest.get("clip_count")):
        findings.append(_finding("expanded_validation", "clip_count", "validation clip count must match manifest", {"validation": clip_count, "manifest": manifest.get("clip_count")}))
    rows = validation.get("rows", [])
    if len(rows) != model_count * clip_count:
        findings.append(_finding("expanded_validation", "row_count", "validation must contain one row per full model and expanded clip", {"actual": len(rows), "expected": model_count * clip_count}))
    full_rows = _full_rows(full_pipeline)
    if len(full_rows) != EXPECTED_FULL_MODELS:
        findings.append(_finding("expanded_validation", "full_pipeline_matrix.json", "full pipeline matrix must preserve the 32 full-humanoid model set", {"actual": len(full_rows)}))
    inputs = gate_v2.get("candidate_gate_inputs", {})
    if inputs.get("candidate_warn") != 0.6 or inputs.get("candidate_pass") != 0.45:
        findings.append(_finding("threshold_integrity", "gate_reconciliation_v2_report.json", "candidate thresholds must remain unchanged", {"inputs": inputs}))
    strict = validation.get("strict_policy_counts", {})
    if strict.get("rows_below_candidate_warn_gate") != summary.get("rows_below_candidate_warn_gate"):
        findings.append(_finding("expanded_validation", "strict_policy_counts", "summary warn count must match strict policy", {"summary": summary.get("rows_below_candidate_warn_gate"), "strict": strict.get("rows_below_candidate_warn_gate")}))
    if strict.get("release_quality_candidate_blocked_count") != summary.get("release_quality_candidate_blocked_count"):
        findings.append(_finding("expanded_validation", "strict_policy_counts", "summary blocked count must match strict policy", {"summary": summary.get("release_quality_candidate_blocked_count"), "strict": strict.get("release_quality_candidate_blocked_count")}))
    for field in ("mean_aggregation_diagnostic_only", "promotion_requires_worst_clip_and_p95_evidence", "legacy_gates_unchanged", "candidate_release_gates_not_counted_as_legacy_runtime_passed"):
        if validation.get(field) is not True:
            findings.append(_finding("expanded_validation", field, "expanded validation safety field must be true", {"actual": validation.get(field)}))
    for field in ("thresholds_lowered", "robot_specific_tuning_used", "hard_clip_removal_used", "raw_residual_hiding_used"):
        if validation.get(field) is not False:
            findings.append(_finding("expanded_validation", field, "expanded validation prohibition field must be false", {"actual": validation.get(field)}))
    return findings


def _audit_candidate_stability(stability: dict[str, Any], summary: dict[str, Any], baseline_gate: dict[str, Any]) -> list[Finding]:
    findings = []
    if _as_int(stability.get("original_step4_3_candidate_warned_count")) != 6:
        findings.append(_finding("candidate_stability", "original_step4_3_candidate_warned_count", "must audit the six original Step 4.3 candidate-warned rows", {"actual": stability.get("original_step4_3_candidate_warned_count")}))
    if _as_int(stability.get("newly_step4_4_candidate_warned_count")) != 3:
        findings.append(_finding("candidate_stability", "newly_step4_4_candidate_warned_count", "must audit the three newly Step 4.4 candidate-warned rows", {"actual": stability.get("newly_step4_4_candidate_warned_count")}))
    if len(stability.get("rows", [])) != 9:
        findings.append(_finding("candidate_stability", "rows", "candidate stability audit must cover all nine Step 4.4 candidate-warned rows", {"actual": len(stability.get("rows", []))}))
    baseline_warned = set(str(value) for value in baseline_gate.get("rows_below_candidate_warn_gate", []))
    audited = set(str(row.get("model_id")) for row in stability.get("rows", []))
    if audited != baseline_warned:
        findings.append(_finding("candidate_stability", "rows", "candidate stability rows must match Step 4.4 candidate-warned rows", {"audited": sorted(audited), "baseline": sorted(baseline_warned)}))
    if summary.get("release_candidate_status") == "PASS_RC" and stability.get("candidate_warned_generalization_passed") is not True:
        findings.append(_finding("candidate_stability", "PASS_RC", "PASS_RC requires original rows and at least one newly warned row to remain stable", {}))
    if summary.get("release_candidate_status") == "BLOCKED_CLIP_GENERALIZATION" and stability.get("candidate_warned_generalization_passed") is True:
        findings.append(_finding("candidate_stability", "BLOCKED_CLIP_GENERALIZATION", "clip generalization cannot be blocked when stability passed", {}))
    return findings


def _audit_task_contract(contract: dict[str, Any]) -> list[Finding]:
    findings = []
    if contract.get("contract_status") != "defined_audited_diagnostic_only":
        findings.append(_finding("task_contract", "contract_status", "task-level residual contract must be explicitly defined and diagnostic-only", {"actual": contract.get("contract_status")}))
    if contract.get("task_class_weighting_counted") is not False:
        findings.append(_finding("task_contract", "task_class_weighting_counted", "task-class weighting must not be counted in Step 4.5", {"actual": contract.get("task_class_weighting_counted")}))
    tasks = contract.get("task_classes", [])
    if {row.get("task") for row in tasks} != {"torso", "left_hand", "right_hand", "left_foot", "right_foot"}:
        findings.append(_finding("task_contract", "task_classes", "task contract must cover every runtime task class", {"tasks": tasks}))
    requirements = contract.get("audit_requirements", {})
    true_fields = ("raw_residual_retained", "component_decomposition_required", "worst_clip_guard_required", "p95_clip_guard_required", "no_robot_specific_weights", "no_clip_specific_weights", "no_semantic_downweighting_to_hide_hips_root_rotation", "legacy_gates_unchanged")
    for field in true_fields:
        if requirements.get(field) is not True:
            findings.append(_finding("task_contract", field, "task contract audit requirement must be true", {"actual": requirements.get(field)}))
    for field in ("robot_specific_tuning_used", "thresholds_lowered", "raw_residual_hiding_used"):
        if contract.get(field) is not False:
            findings.append(_finding("task_contract", field, "task contract prohibition field must be false", {"actual": contract.get(field)}))
    return findings


def _audit_hips_root_rotation(hips: dict[str, Any], validation: dict[str, Any]) -> list[Finding]:
    findings = []
    if hips.get("primary_blocker") != "BLOCKED_HIPS_ROTATION_RESIDUAL":
        findings.append(_finding("hips_root_rotation", "primary_blocker", "Hips/root rotation must be the localized residual blocker", {"actual": hips.get("primary_blocker")}))
    if _as_int(hips.get("model_count")) != EXPECTED_FULL_MODELS:
        findings.append(_finding("hips_root_rotation", "model_count", "Hips/root decomposition must cover all full humanoid models", {"actual": hips.get("model_count")}))
    if _as_int(hips.get("hips_dominant_model_count")) < 24:
        findings.append(_finding("hips_root_rotation", "hips_dominant_model_count", "Hips/root residual should dominate most blocked models", {"actual": hips.get("hips_dominant_model_count")}))
    if _as_int(hips.get("hips_rotation_dominant_model_count")) < 24:
        findings.append(_finding("hips_root_rotation", "hips_rotation_dominant_model_count", "Hips/root rotation should dominate most blocked models", {"actual": hips.get("hips_rotation_dominant_model_count")}))
    if hips.get("raw_residual_retained") is not True or hips.get("task_class_weighting_counted") is not False:
        findings.append(_finding("hips_root_rotation", "residual_transparency", "Hips/root residual decomposition must retain raw residuals and avoid counted weights", {"raw": hips.get("raw_residual_retained"), "weighting": hips.get("task_class_weighting_counted")}))
    if len(hips.get("rows", [])) != _as_int(validation.get("model_count")):
        findings.append(_finding("hips_root_rotation", "rows", "Hips/root rows must align with expanded validation models", {"hips": len(hips.get("rows", [])), "validation": validation.get("model_count")}))
    return findings


def _audit_solver_convergence_weak(solver_weak: dict[str, Any], manifest: dict[str, Any]) -> list[Finding]:
    findings = []
    if solver_weak.get("global_solver_diagnostics_only") is not True:
        findings.append(_finding("solver_convergence_weak", "global_solver_diagnostics_only", "solver weak investigation must be global-only diagnostics", {"actual": solver_weak.get("global_solver_diagnostics_only")}))
    if solver_weak.get("robot_specific_tuning_used") is not False:
        findings.append(_finding("solver_convergence_weak", "robot_specific_tuning_used", "solver weak investigation must not use robot-specific tuning", {"actual": solver_weak.get("robot_specific_tuning_used")}))
    rows = solver_weak.get("rows", [])
    if _as_int(solver_weak.get("weak_row_count")) != len(rows):
        findings.append(_finding("solver_convergence_weak", "weak_row_count", "weak row count must match rows", {"count": solver_weak.get("weak_row_count"), "rows": len(rows)}))
    for row in rows:
        if row.get("clip_count") != manifest.get("clip_count"):
            findings.append(_finding("solver_convergence_weak", str(row.get("model_id")), "solver weak row must include every frozen clip", {"clip_count": row.get("clip_count"), "expected": manifest.get("clip_count")}))
        if row.get("global_solver_config_only") is not True or row.get("robot_specific_solver_tuning_used") is not False:
            findings.append(_finding("solver_convergence_weak", str(row.get("model_id")), "solver weak row violates global-only diagnostic contract", row))
    return findings


def _audit_readiness(
    readiness: dict[str, Any],
    validation: dict[str, Any],
    stability: dict[str, Any],
    task_contract: dict[str, Any],
    hips: dict[str, Any],
) -> list[Finding]:
    findings = []
    if readiness.get("decision") != "keep_diagnostic_only" or readiness.get("promote_to_release_candidate_gate") is not False:
        findings.append(_finding("readiness", "decision", "Step 4.5 must keep mean aggregation diagnostic-only unless strict evidence supports promotion", {"decision": readiness.get("decision"), "promote": readiness.get("promote_to_release_candidate_gate")}))
    if readiness.get("mean_aggregation_diagnostic_only") is not True:
        findings.append(_finding("readiness", "mean_aggregation_diagnostic_only", "mean aggregation must stay diagnostic-only", {"actual": readiness.get("mean_aggregation_diagnostic_only")}))
    if readiness.get("candidate_warned_generalization_passed") != stability.get("candidate_warned_generalization_passed"):
        findings.append(_finding("readiness", "candidate_warned_generalization_passed", "readiness must reflect stability audit", {"readiness": readiness.get("candidate_warned_generalization_passed"), "stability": stability.get("candidate_warned_generalization_passed")}))
    if readiness.get("task_level_residual_contract_status") != task_contract.get("contract_status"):
        findings.append(_finding("readiness", "task_level_residual_contract_status", "readiness must reference task contract status", {}))
    if readiness.get("hips_root_rotation_blocker_status") != hips.get("primary_blocker"):
        findings.append(_finding("readiness", "hips_root_rotation_blocker_status", "readiness must reference Hips/root rotation blocker", {}))
    if readiness.get("strict_worst_clip_p95_policy_supported") != (validation.get("strict_policy_counts", {}).get("rows_below_candidate_warn_gate", 0) > 0):
        findings.append(_finding("readiness", "strict_worst_clip_p95_policy_supported", "readiness strict support field mismatch", {}))
    return findings


def _audit_quality_delta(delta: dict[str, Any], baseline_summary: dict[str, Any], validation: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    baseline = delta.get("baseline_counts", {})
    expanded = delta.get("expanded_strict_policy_counts", {})
    if baseline.get("rows_below_candidate_warn_gate") != baseline_summary.get("rows_below_candidate_warn_gate"):
        findings.append(_finding("quality_delta", "baseline_counts", "delta baseline must match Step 4.4 summary", {"delta": baseline, "summary": baseline_summary.get("rows_below_candidate_warn_gate")}))
    if expanded != validation.get("strict_policy_counts", {}) | {"runtime_quality_passed_count": expanded.get("runtime_quality_passed_count"), "runtime_quality_failed_count": expanded.get("runtime_quality_failed_count")}:
        expected = dict(validation.get("strict_policy_counts", {}))
        expected["runtime_quality_passed_count"] = expanded.get("runtime_quality_passed_count")
        expected["runtime_quality_failed_count"] = expanded.get("runtime_quality_failed_count")
        if expanded != expected:
            findings.append(_finding("quality_delta", "expanded_strict_policy_counts", "delta expanded counts must match validation strict policy", {"delta": expanded, "expected": expected}))
    if delta.get("release_candidate_status") != summary.get("release_candidate_status"):
        findings.append(_finding("quality_delta", "release_candidate_status", "delta status must match summary", {"delta": delta.get("release_candidate_status"), "summary": summary.get("release_candidate_status")}))
    for field in ("legacy_gates_unchanged",):
        if delta.get(field) is not True:
            findings.append(_finding("quality_delta", field, "quality delta safety field must be true", {"actual": delta.get(field)}))
    for field in ("candidate_thresholds_lowered", "raw_residual_hiding_used", "robot_specific_tuning_used", "hard_clip_removal_used"):
        if delta.get(field) is not False:
            findings.append(_finding("quality_delta", field, "quality delta prohibition field must be false", {"actual": delta.get(field)}))
    return findings


def _audit_no_leakage_or_weakening(summary: dict[str, Any], ledger: dict[str, Any], validation: dict[str, Any], task_contract: dict[str, Any]) -> list[Finding]:
    findings = []
    if summary.get("runtime_quality_passed_count") != 0 or ledger.get("runtime_quality_passed_count") != 0:
        findings.append(_finding("no_leakage", "runtime_quality_passed_count", "candidate status must not leak into legacy runtime passed count", {"summary": summary.get("runtime_quality_passed_count"), "ledger": ledger.get("runtime_quality_passed_count")}))
    if summary.get("runtime_quality_failed_count") != 0 or ledger.get("runtime_quality_failed_count") != 0:
        findings.append(_finding("legacy_invariants", "runtime_quality_failed_count", "Step 4.5 must preserve zero legacy failures", {"summary": summary.get("runtime_quality_failed_count"), "ledger": ledger.get("runtime_quality_failed_count")}))
    for field in ("legacy_gates_unchanged", "candidate_release_gates_not_counted_as_legacy_runtime_passed", "mean_aggregation_diagnostic_only"):
        if summary.get(field) is not True:
            findings.append(_finding("no_leakage", field, "summary safety field must be true", {"actual": summary.get(field)}))
    for field in ("candidate_thresholds_lowered", "robot_specific_tuning_used", "hard_clip_removal_used", "raw_residual_hiding_used", "task_class_weighting_counted"):
        if summary.get(field) is not False:
            findings.append(_finding("prohibitions", field, "summary prohibition field must be false", {"actual": summary.get(field)}))
    if validation.get("candidate_warn_gate") != 0.6 or validation.get("candidate_pass_gate") != 0.45:
        findings.append(_finding("threshold_integrity", "release_quality_v2_expanded_validation_matrix.json", "candidate thresholds changed", {"warn": validation.get("candidate_warn_gate"), "pass": validation.get("candidate_pass_gate")}))
    if task_contract.get("task_class_weighting_counted") is not False:
        findings.append(_finding("prohibitions", "task_class_weighting_counted", "task-class weighting must not be counted", {"actual": task_contract.get("task_class_weighting_counted")}))
    return findings


def _audit_deterministic(deterministic: dict[str, Any]) -> list[Finding]:
    findings = []
    if deterministic.get("status") != "passed" or deterministic.get("deterministic") is not True:
        findings.append(_finding("determinism", "deterministic_rerun.json", "deterministic rerun must pass", {"status": deterministic.get("status"), "deterministic": deterministic.get("deterministic")}))
    if _as_int(deterministic.get("deterministic_compared_count")) < 44 or deterministic.get("deterministic_compared_count") != deterministic.get("deterministic_matched_count"):
        findings.append(_finding("determinism", "deterministic_rerun.json", "deterministic compared and matched counts are invalid", {"compared": deterministic.get("deterministic_compared_count"), "matched": deterministic.get("deterministic_matched_count")}))
    return findings


def _audit_lfs(summary: dict[str, Any], ledger: dict[str, Any]) -> list[Finding]:
    lfs = _lfs(summary, ledger)
    if lfs.get("git_lfs_fsck") != "OK":
        return [_finding("lfs", "git_lfs_fsck", "Git LFS fsck evidence must be OK", lfs)]
    return []


def _audit_final_head_ci(final_head_ci: dict[str, Any], source_root: Path) -> list[Finding]:
    findings = []
    head = _git_stdout(source_root, "rev-parse", "HEAD")
    if not final_head_ci:
        findings.append(_finding("final_head_ci", "final_head_ci", "final-head CI evidence is missing", {}))
        return findings
    if final_head_ci.get("conclusion") != "success":
        findings.append(_finding("final_head_ci", "conclusion", "final-head CI conclusion must be success", final_head_ci))
    if final_head_ci.get("head_sha") != head:
        findings.append(_finding("final_head_ci", "head_sha", "final-head CI SHA must match current HEAD", {"ci": final_head_ci.get("head_sha"), "head": head}))
    jobs = final_head_ci.get("job_conclusions", {})
    if not jobs or any(value != "success" for value in jobs.values()):
        findings.append(_finding("final_head_ci", "job_conclusions", "all final-head CI jobs must succeed", {"jobs": jobs}))
    return findings


def _final_head_ci(summary: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    value = ledger.get("final_head_ci") or summary.get("final_head_ci") or {}
    return value if isinstance(value, dict) else {}


def _lfs(summary: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    value = ledger.get("lfs") or summary.get("lfs") or {}
    return value if isinstance(value, dict) else {}


def _full_rows(payload: Any) -> list[dict[str, Any]]:
    return [row for row in _matrix_rows(payload) if row.get("category") == "full_humanoid_profile"]


def _full_clip_rows(payload: Any) -> list[dict[str, Any]]:
    return [row for row in _matrix_rows(payload) if row.get("category") == "full_humanoid_profile"]


def _matrix_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("rows", "matrix", "model_rows", "exports"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finding(gate: str, subject: str, message: str, evidence: dict[str, Any]) -> Finding:
    return Finding(gate=gate, severity="error", subject=subject, message=message, evidence=evidence)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _git_stdout(source_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
