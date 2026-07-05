#!/usr/bin/env python3
"""Audit Step 4.4 release-quality v2 validation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_4_release_quality_v2_validation")
DEFAULT_BASELINE_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation")
DEFAULT_SOURCE_ROOT = Path(".")
EXPECTED_MATRIX_ROWS = 44
EXPECTED_CATEGORY_COUNTS = {
    "full_humanoid_profile": 32,
    "partial_humanoid_profile": 3,
    "negative_control": 9,
}
REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "commands.txt",
    "quality_summary.json",
    "acceptance_ledger.json",
    "full_pipeline_matrix.json",
    "clip_matrix.json",
    "solver_smoke_matrix.json",
    "generic_smoke_matrix.json",
    "deterministic_rerun.json",
    "quality_delta_vs_step4_3.json",
    "release_quality_v2_validation_matrix.json",
    "candidate_warned_deep_audit.json",
    "release_quality_v2_blocker_taxonomy.json",
    "release_quality_v2_stress_test.json",
    "release_quality_v2_promotion_readiness.json",
    "gate_reconciliation_v3_report.json",
    "normalization_integrity_v2_report.json",
    "trajectory_export_manifest.json",
    "temporal_continuity_matrix.json",
    "support_contact_diagnostics.json",
    "collision_proxy_diagnostics.json",
    "pipeline_config.json",
    "solver_config.json",
    "pipeline_controls_reference.json",
    "red_team_report.json",
    "test_results/pytest.txt",
    "test_results/pytest_summary.json",
    "test_results/junit.xml",
)
PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES = (
    "pipeline_controls_reference.json",
    "pipeline_backed_matrix.json",
)
VALID_RELEASE_STATUSES = {
    "PASS_RC",
    "BLOCKED_RELEASE_QUALITY_V2_VALIDATION",
    "BLOCKED_CANDIDATE_GATE_ROBUSTNESS",
    "BLOCKED_CLIP_GENERALIZATION",
    "BLOCKED_PIPELINE_REGRESSION",
    "BLOCKED_CI_OR_PROVENANCE",
}
ACCEPTABLE_BLOCKED_STATUSES = {
    "BLOCKED_RELEASE_QUALITY_V2_VALIDATION",
    "BLOCKED_CANDIDATE_GATE_ROBUSTNESS",
    "BLOCKED_CLIP_GENERALIZATION",
}
REQUIRED_BLOCKER_CATEGORIES = {
    "orientation_integrated_fixed_global_scale_residual_p95_above_candidate_warn_gate",
    "clip_instability",
    "task_class_dominance",
    "temporal_jump",
    "support_contact_issue",
    "collision_proxy_issue",
    "solver_convergence_weak",
    "normalization_ambiguity",
}
REQUIRED_GLOBAL_METHODS = {
    "global_clip_aggregation_policy",
    "global_task_class_residual_weighting",
    "global_worst_clip_guard",
    "global_solver_retry_line_search_refinement",
    "global_temporal_consistency_residual_penalty",
}
SCAN_PATHS = (
    "soma_retargeter/runtime/v3",
    "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
    "soma_retargeter/tools/step4_4_release_quality_v2_validation.py",
    "scripts/audit_retargeting_v3_step4_4_release_quality_v2_validation.py",
)
CLOSED_ARTIFACT_DIRS = (
    "artifacts/retargeting_v3_step3_1_1_solver_backed_smoke",
    "artifacts/retargeting_v3_step3_2_solver_backed_acceptance",
    "artifacts/retargeting_v3_step3_3_global_solver_quality",
    "artifacts/retargeting_v3_step3_4_global_residual_quality",
    "artifacts/retargeting_v3_step4_full_pipeline_acceptance",
    "artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough",
    "artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring",
    "artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation",
)
CONCRETE_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", flags=re.IGNORECASE)
CONCRETE_RUN_ID_RE = re.compile(r"\b[1-9][0-9]{5,}\b")
SUSPICIOUS_SPECIFIC_RE = re.compile(
    r"per_robot_threshold|by_robot_weight|model_id_threshold|accepted_models|allowed_models|"
    r"force_pass|force_promote|whitelist|blacklist|denylist",
    flags=re.IGNORECASE,
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
    baseline_artifact_dir: str
    source_root: str
    matrix_row_count: int
    blocking_count: int
    finding_count: int
    gate_counts: dict[str, int]
    findings: list[Finding]
    invariants: dict[str, Any]
    candidate_counts: dict[str, Any]
    legacy_counts: dict[str, Any]
    normalization_integrity: dict[str, Any]
    candidate_stability: dict[str, Any]
    blocker_taxonomy_summary: dict[str, Any]
    final_head_ci: dict[str, Any]
    lfs: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    baseline_artifact_dir: Path = DEFAULT_BASELINE_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    require_final_head_ci: bool = False,
) -> AuditResult:
    artifact_dir = Path(artifact_dir).resolve()
    baseline_artifact_dir = Path(baseline_artifact_dir).resolve()
    source_root = Path(source_root).resolve()

    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    quality_delta = _read_json(artifact_dir / "quality_delta_vs_step4_3.json")
    validation_matrix = _read_json(artifact_dir / "release_quality_v2_validation_matrix.json")
    candidate_audit = _read_json(artifact_dir / "candidate_warned_deep_audit.json")
    blocker_taxonomy = _read_json(artifact_dir / "release_quality_v2_blocker_taxonomy.json")
    stress_test = _read_json(artifact_dir / "release_quality_v2_stress_test.json")
    promotion = _read_json(artifact_dir / "release_quality_v2_promotion_readiness.json")
    gate_report = _read_json(artifact_dir / "gate_reconciliation_v3_report.json")
    normalization = _read_json(artifact_dir / "normalization_integrity_v2_report.json")
    trajectory = _read_json(artifact_dir / "trajectory_export_manifest.json")
    temporal = _read_json(artifact_dir / "temporal_continuity_matrix.json")
    support = _read_json(artifact_dir / "support_contact_diagnostics.json")
    collision = _read_json(artifact_dir / "collision_proxy_diagnostics.json")
    pipeline_config = _read_json(artifact_dir / "pipeline_config.json")
    solver_config = _read_json(artifact_dir / "solver_config.json")
    red_team = _read_json(artifact_dir / "red_team_report.json")
    baseline_summary = _read_json(baseline_artifact_dir / "quality_summary.json")
    baseline_gate = _read_json(baseline_artifact_dir / "gate_reconciliation_v2_report.json")

    full_rows = _matrix_rows(full_pipeline)
    solver_rows = _matrix_rows(solver_smoke)
    generic_rows = _matrix_rows(generic_smoke)
    final_head_ci = _final_head_ci_record(ledger, summary)
    lfs = _lfs_record(ledger, summary)

    findings: list[Finding] = []
    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_baseline_step4_3(baseline_artifact_dir, baseline_summary, baseline_gate))
    findings.extend(_audit_matrix_shape_and_partition(full_rows, summary))
    findings.extend(_audit_solver_and_runtime_counts(full_rows, solver_rows, generic_rows, summary))
    findings.extend(_audit_runtime_quality_label_honesty(full_rows, summary, gate_report))
    findings.extend(_audit_validation_matrix(validation_matrix, full_rows))
    findings.extend(_audit_candidate_warned_deep_audit(candidate_audit, baseline_summary, baseline_gate))
    findings.extend(_audit_blocker_taxonomy(blocker_taxonomy, baseline_summary, baseline_gate))
    findings.extend(_audit_stress_test(stress_test, gate_report))
    findings.extend(_audit_promotion_readiness(promotion))
    findings.extend(_audit_gate_reconciliation_v3(gate_report, summary, quality_delta, baseline_summary))
    findings.extend(_audit_normalization_integrity(normalization))
    findings.extend(_audit_quality_delta(quality_delta, baseline_summary, summary, gate_report))
    findings.extend(_audit_deterministic(deterministic, summary))
    findings.extend(_audit_exports_and_diagnostics(trajectory, temporal, support, collision, summary))
    findings.extend(_audit_configs(pipeline_config, solver_config))
    findings.extend(_audit_pipeline_controls(artifact_dir))
    findings.extend(_audit_release_candidate_status(summary, ledger, gate_report, quality_delta, candidate_audit, promotion))
    findings.extend(_audit_no_readiness_claims(summary, ledger, promotion, red_team))
    findings.extend(_audit_no_specific_tuning(source_root))
    findings.extend(_audit_closed_artifact_trees(source_root))
    if require_final_head_ci:
        findings.extend(_audit_final_head_ci(final_head_ci, source_root))
        findings.extend(_audit_lfs_evidence(lfs))

    gate_counts = dict(Counter(finding.gate for finding in findings))
    blocking_count = sum(1 for finding in findings if finding.severity == "error")
    release_status = str(summary.get("release_candidate_status") or ledger.get("release_candidate_status") or "")
    if blocking_count:
        status = "BLOCKED_CI_OR_PROVENANCE" if require_final_head_ci and gate_counts.get("final_head_ci") else "BLOCKED_PIPELINE_REGRESSION"
    elif release_status in VALID_RELEASE_STATUSES:
        status = release_status
    else:
        status = "BLOCKED_PIPELINE_REGRESSION"
    return AuditResult(
        status=status,
        quality_status=release_status,
        artifact_dir=_display_path(artifact_dir),
        baseline_artifact_dir=_display_path(baseline_artifact_dir),
        source_root=_display_path(source_root),
        matrix_row_count=len(full_rows),
        blocking_count=blocking_count,
        finding_count=len(findings),
        gate_counts=gate_counts,
        findings=findings,
        invariants=_invariants(summary, full_rows, deterministic, trajectory, temporal, support, collision),
        candidate_counts=_candidate_counts(summary, gate_report, quality_delta),
        legacy_counts=_legacy_counts(summary),
        normalization_integrity=_normalization_summary(normalization),
        candidate_stability=_candidate_stability(candidate_audit, stress_test),
        blocker_taxonomy_summary=_blocker_taxonomy_summary(blocker_taxonomy),
        final_head_ci=final_head_ci,
        lfs=lfs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--baseline-step4-3-artifact-dir", "--baseline-artifact-dir", dest="baseline_artifact_dir", default=str(DEFAULT_BASELINE_ARTIFACT_DIR))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-json")
    parser.add_argument("--write-report", dest="output_json")
    parser.add_argument("--require-final-head-ci", action="store_true")
    args = parser.parse_args(argv)
    result = run_audit(
        artifact_dir=Path(args.artifact_dir),
        baseline_artifact_dir=Path(args.baseline_artifact_dir),
        source_root=Path(args.source_root),
        require_final_head_ci=args.require_final_head_ci,
    )
    payload = result.to_json()
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.blocking_count == 0 else 1


def _audit_required_artifacts(artifact_dir: Path) -> list[Finding]:
    findings = []
    if not artifact_dir.exists():
        return [_finding("missing_required_artifacts", _display_path(artifact_dir), "Step 4.4 artifact directory is missing", {})]
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(_finding("missing_required_artifacts", relative, "required Step 4.4 artifact is missing", {}))
    if not any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        findings.append(_finding("missing_required_artifacts", "pipeline controls", "pipeline control evidence is missing", {}))
    return findings


def _audit_baseline_step4_3(
    baseline_artifact_dir: Path,
    baseline_summary: dict[str, Any],
    baseline_gate: dict[str, Any],
) -> list[Finding]:
    findings = []
    if not baseline_artifact_dir.exists():
        findings.append(_finding("baseline_step4_3", _display_path(baseline_artifact_dir), "Step 4.3 baseline artifact dir is missing", {}))
    expected = {
        "release_candidate_status": "PASS_RC",
        "runtime_quality_passed_count": 0,
        "runtime_quality_warned_count": 32,
        "runtime_quality_failed_count": 0,
        "high_residual_warning_count": 32,
        "rows_below_candidate_warn_gate": 6,
        "rows_below_candidate_pass_gate": 0,
        "solver_backed_count": 32,
        "residual_only_count": 0,
    }
    for field, expected_value in expected.items():
        actual = baseline_summary.get(field)
        if isinstance(expected_value, int):
            if _as_int(actual) != expected_value:
                findings.append(_finding("baseline_step4_3", field, "baseline Step 4.3 summary mismatch", {"actual": actual, "expected": expected_value}))
        elif actual != expected_value:
            findings.append(_finding("baseline_step4_3", field, "baseline Step 4.3 summary mismatch", {"actual": actual, "expected": expected_value}))
    if len(baseline_gate.get("rows_below_candidate_warn_gate", [])) != 6:
        findings.append(_finding("baseline_step4_3", "gate_reconciliation_v2_report.json", "baseline Step 4.3 candidate warned row count must be 6", {"actual": len(baseline_gate.get("rows_below_candidate_warn_gate", []))}))
    return findings


def _audit_matrix_shape_and_partition(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if len(rows) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "full_pipeline_matrix.json", "full pipeline matrix must contain 44 rows", {"actual": len(rows)}))
    counts = dict(Counter(str(row.get("category")) for row in rows))
    if counts != EXPECTED_CATEGORY_COUNTS:
        findings.append(_finding("partition_32_3_9", "full_pipeline_matrix.json", "category partition must be 32/3/9", {"actual": counts}))
    for field, expected in (("in_scope_total", 44), ("full_humanoid_total", 32), ("partial_total", 3), ("negative_total", 9)):
        if _as_int(summary.get(field)) != expected:
            findings.append(_finding("partition_32_3_9", "quality_summary.json", f"{field} mismatch", {"actual": summary.get(field), "expected": expected}))
    return findings


def _audit_solver_and_runtime_counts(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    full_rows = [row for row in rows if row.get("category") == "full_humanoid_profile"]
    solver_backed_rows = sum(1 for row in full_rows if row.get("solver_backed") is True)
    if solver_backed_rows < 32 or _as_int(summary.get("solver_backed_count")) < 32:
        findings.append(_finding("solver_backed_counts", "solver_backed_count", "solver-backed full coverage regressed below 32", {"rows": solver_backed_rows, "summary": summary.get("solver_backed_count")}))
    if _as_int(summary.get("residual_only_count")) != 0:
        findings.append(_finding("solver_backed_counts", "residual_only_count", "residual_only_count must be 0", {"actual": summary.get("residual_only_count")}))
    if _as_int(summary.get("runtime_quality_failed_count")) != 0:
        findings.append(_finding("runtime_quality_status", "runtime_quality_failed_count", "runtime_quality_failed_count must be 0", {"actual": summary.get("runtime_quality_failed_count")}))
    solver_models = {_model_id(row) for row in solver_rows if row.get("category") == "full_humanoid_profile" and row.get("solver_backed") is True}
    generic_models = {_model_id(row) for row in generic_rows if row.get("category") == "full_humanoid_profile" and row.get("solver_backed") is True}
    if len(solver_models) < 32:
        findings.append(_finding("solver_backed_counts", "solver_smoke_matrix.json", "solver smoke must preserve 32 full humanoids", {"actual": len(solver_models)}))
    if len(generic_models) < 32:
        findings.append(_finding("solver_backed_counts", "generic_smoke_matrix.json", "generic smoke must preserve 32 full humanoids", {"actual": len(generic_models)}))
    return findings


def _audit_runtime_quality_label_honesty(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    gate_report: dict[str, Any],
) -> list[Finding]:
    findings = []
    if _as_int(summary.get("runtime_quality_passed_count")) != 0:
        findings.append(_finding("status_semantics", "quality_summary.json", "Step 4.4 must not count candidate rows as legacy runtime_quality_passed", {"actual": summary.get("runtime_quality_passed_count")}))
    if gate_report.get("candidate_release_gates_not_counted_as_legacy_runtime_passed") is not True:
        findings.append(_finding("status_semantics", "gate_reconciliation_v3_report.json", "candidate release gates must stay separate from legacy runtime pass counts", {}))
    for index, row in enumerate(rows):
        if row.get("category") != "full_humanoid_profile":
            continue
        if _row_status(row) == "runtime_quality_passed" and _as_float(row.get("normalized_task_residual_p95")) > 0.15:
            findings.append(_finding("status_semantics", f"full_pipeline_matrix.json[{index}]", "legacy pass row lacks unchanged legacy residual gate evidence", {"model_id": _model_id(row)}))
    return findings


def _audit_validation_matrix(validation: dict[str, Any], full_rows: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    rows = _matrix_rows(validation)
    full_model_count = sum(1 for row in full_rows if row.get("category") == "full_humanoid_profile")
    if validation.get("model_count") != full_model_count:
        findings.append(_finding("validation_matrix", "release_quality_v2_validation_matrix.json", "validation matrix model count mismatch", {"actual": validation.get("model_count"), "expected": full_model_count}))
    if len(rows) < full_model_count:
        findings.append(_finding("validation_matrix", "release_quality_v2_validation_matrix.json", "validation matrix must contain at least one row per full model", {"rows": len(rows), "models": full_model_count}))
    required = {
        "model_id",
        "clip_id",
        "candidate_metric_value",
        "candidate_status",
        "hard_safety_status",
        "legacy_status",
        "v2_status",
        "gate_blockers",
        "temporal_diagnostics",
        "support_contact_diagnostics",
        "collision_proxy_diagnostics",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            findings.append(_finding("validation_matrix", f"row[{index}]", "validation matrix row is missing required fields", {"missing": missing}))
        if not isinstance(row.get("hard_safety_status"), dict) or "passed" not in row.get("hard_safety_status", {}):
            findings.append(_finding("validation_matrix", f"row[{index}]", "hard safety status must be structured", {"row": row}))
        if row.get("candidate_status") == "release_quality_candidate_passed" and row.get("hard_safety_status", {}).get("passed") is not True:
            findings.append(_finding("validation_matrix", f"row[{index}]", "candidate pass requires hard safety evidence", {"row": row}))
        if row.get("candidate_status") == "release_quality_candidate_passed":
            for diagnostic_key in ("temporal_diagnostics", "support_contact_diagnostics", "collision_proxy_diagnostics"):
                if not isinstance(row.get(diagnostic_key), dict) or not row.get(diagnostic_key):
                    findings.append(_finding("validation_matrix", f"row[{index}]", "candidate pass requires temporal/contact/collision diagnostics", {"missing_or_empty": diagnostic_key}))
    return findings


def _audit_candidate_warned_deep_audit(
    candidate_audit: dict[str, Any],
    baseline_summary: dict[str, Any],
    baseline_gate: dict[str, Any],
) -> list[Finding]:
    findings = []
    rows = _matrix_rows(candidate_audit)
    baseline_warned = len(baseline_gate.get("rows_below_candidate_warn_gate", [])) or _as_int(baseline_summary.get("rows_below_candidate_warn_gate"))
    if len(rows) != baseline_warned:
        findings.append(_finding("candidate_warned_deep_audit", "candidate_warned_deep_audit.json", "deep audit must cover current baseline candidate-warned rows", {"rows": len(rows), "baseline_warned": baseline_warned}))
    if candidate_audit.get("all_stable_across_four_clips") is not True:
        findings.append(_finding("candidate_warned_deep_audit", "all_stable_across_four_clips", "candidate-warned rows must be stable across all four clips", {"value": candidate_audit.get("all_stable_across_four_clips")}))
    required = {
        "why_below_candidate_warn_gate",
        "stable_across_all_four_clips",
        "one_clip_dominates_result",
        "temporal_contact_collision_diagnostics_healthy",
        "what_separates_from_blocked_rows",
    }
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            findings.append(_finding("candidate_warned_deep_audit", str(row.get("model_id")), "candidate warned deep audit row is incomplete", {"missing": missing}))
    return findings


def _audit_blocker_taxonomy(
    taxonomy: dict[str, Any],
    baseline_summary: dict[str, Any],
    baseline_gate: dict[str, Any],
) -> list[Finding]:
    findings = []
    baseline_warned = len(baseline_gate.get("rows_below_candidate_warn_gate", [])) or _as_int(baseline_summary.get("rows_below_candidate_warn_gate"))
    expected_blocked = 32 - baseline_warned
    if _as_int(taxonomy.get("blocked_row_count")) != expected_blocked:
        findings.append(_finding("blocker_taxonomy", "release_quality_v2_blocker_taxonomy.json", "blocked taxonomy must cover Step 4.3 blocked rows", {"actual": taxonomy.get("blocked_row_count"), "expected": expected_blocked}))
    categories = set(taxonomy.get("required_categories", [])) | set((taxonomy.get("category_counts") or {}).keys())
    missing = sorted(REQUIRED_BLOCKER_CATEGORIES - categories)
    if missing:
        findings.append(_finding("blocker_taxonomy", "category_counts", "blocker taxonomy is missing required categories", {"missing": missing}))
    return findings


def _audit_stress_test(stress: dict[str, Any], gate_report: dict[str, Any]) -> list[Finding]:
    findings = []
    for field in (
        "per_clip_stability",
        "worst_clip_status",
        "mean_vs_p95_vs_max_sensitivity",
        "threshold_sensitivity_without_changing_selected_thresholds",
        "deterministic_rerun_stability",
        "raw_residual_monotonicity",
        "global_methods_tried",
        "selected_policy_counts",
    ):
        if field not in stress:
            findings.append(_finding("stress_test", "release_quality_v2_stress_test.json", "required stress-test field missing", {"field": field}))
    methods = {str(row.get("method")) for row in stress.get("global_methods_tried", []) if isinstance(row, dict)}
    missing_methods = sorted(REQUIRED_GLOBAL_METHODS - methods)
    if missing_methods:
        findings.append(_finding("stress_test", "global_methods_tried", "not all required global methods were tried or rejected explicitly", {"missing": missing_methods}))
    if stress.get("threshold_sensitivity_without_changing_selected_thresholds", {}).get("selected_thresholds_unchanged") is not True:
        findings.append(_finding("threshold_integrity", "release_quality_v2_stress_test.json", "selected thresholds must remain unchanged", {}))
    selected = stress.get("selected_policy_counts", {})
    if _as_int(selected.get("rows_below_candidate_warn_gate")) != len(gate_report.get("rows_below_candidate_warn_gate", [])):
        findings.append(_finding("stress_test", "selected_policy_counts", "selected policy count must match gate report", {"stress": selected, "gate": len(gate_report.get("rows_below_candidate_warn_gate", []))}))
    return findings


def _audit_promotion_readiness(promotion: dict[str, Any]) -> list[Finding]:
    findings = []
    if promotion.get("decision") not in {"keep_diagnostic_only", "promote_to_release_candidate_gate", "blocked_pending_more_validation"}:
        findings.append(_finding("promotion_readiness", "decision", "promotion readiness decision is invalid", {"decision": promotion.get("decision")}))
    for field, expected in (
        ("production_default_change_allowed", False),
        ("runtime_override_default_enabled", False),
        ("legacy_gates_unchanged", True),
    ):
        if promotion.get(field) is not expected:
            findings.append(_finding("promotion_readiness", field, "promotion readiness safety field mismatch", {"actual": promotion.get(field), "expected": expected}))
    for field in ("why", "risks", "required_future_evidence"):
        if not promotion.get(field):
            findings.append(_finding("promotion_readiness", field, "promotion readiness must include rationale, risks, and future evidence", {}))
    return findings


def _audit_gate_reconciliation_v3(
    gate_report: dict[str, Any],
    summary: dict[str, Any],
    quality_delta: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    if gate_report.get("legacy_gates_unchanged") is not True:
        findings.append(_finding("gate_reconciliation_v3", "legacy_gates_unchanged", "legacy gates must remain unchanged", {}))
    if gate_report.get("runtime_quality_passed_count_uses_legacy_gates_only") is not True:
        findings.append(_finding("status_semantics", "gate_reconciliation_v3_report.json", "legacy runtime pass semantics must stay separate", {}))
    inputs = gate_report.get("candidate_gate_inputs", {})
    if inputs.get("candidate_thresholds_lowered") is True:
        findings.append(_finding("threshold_integrity", "candidate_gate_inputs", "candidate thresholds were lowered", {"inputs": inputs}))
    if _as_float(inputs.get("candidate_warn")) != 0.60 or _as_float(inputs.get("candidate_pass")) != 0.45:
        findings.append(_finding("threshold_integrity", "candidate_gate_inputs", "Step 4.4 must keep selected candidate thresholds unchanged", {"inputs": inputs}))
    warn_count = len(gate_report.get("rows_below_candidate_warn_gate", []))
    pass_count = len(gate_report.get("rows_below_candidate_pass_gate", []))
    if _as_int(summary.get("rows_below_candidate_warn_gate")) != warn_count:
        findings.append(_finding("gate_reconciliation_v3", "quality_summary.json", "candidate warned count mismatch", {"summary": summary.get("rows_below_candidate_warn_gate"), "gate": warn_count}))
    if _as_int(summary.get("rows_below_candidate_pass_gate")) != pass_count:
        findings.append(_finding("gate_reconciliation_v3", "quality_summary.json", "candidate pass count mismatch", {"summary": summary.get("rows_below_candidate_pass_gate"), "gate": pass_count}))
    baseline_counts = gate_report.get("baseline_step4_3_counts", {})
    if _as_int(baseline_counts.get("rows_below_candidate_warn_gate")) != _as_int(baseline_summary.get("rows_below_candidate_warn_gate")):
        findings.append(_finding("gate_reconciliation_v3", "baseline_step4_3_counts", "baseline count mismatch", {"gate": baseline_counts, "baseline": baseline_summary.get("rows_below_candidate_warn_gate")}))
    if quality_delta.get("current_counts", {}).get("rows_below_candidate_warn_gate") != warn_count:
        findings.append(_finding("quality_delta_vs_step4_3", "quality_delta_vs_step4_3.json", "quality delta current warned count mismatch", {}))
    return findings


def _audit_normalization_integrity(normalization: dict[str, Any]) -> list[Finding]:
    findings = []
    for field in ("denominator_inflation_detected", "normalization_hides_raw_residual_regression", "candidate_thresholds_lowered", "robot_specific_tuning_used", "clip_removal_used"):
        if normalization.get(field) is True:
            findings.append(_finding("normalization_integrity", field, "normalization integrity or global-method prohibition failed", {"value": normalization.get(field)}))
    for field in ("raw_residual_always_retained", "legacy_gates_unchanged", "candidate_release_gates_not_counted_as_legacy_runtime_quality_passed"):
        if normalization.get(field) is not True:
            findings.append(_finding("normalization_integrity", field, "normalization integrity field must be true", {"value": normalization.get(field)}))
    if _as_int(normalization.get("raw_residual_regression_count")) != 0:
        findings.append(_finding("normalization_integrity", "raw_residual_regression_count", "raw residual regression count must be zero", {"actual": normalization.get("raw_residual_regression_count")}))
    if normalization.get("candidate_denominator_scope") in {"row_local", "clip_local", "model_local"}:
        findings.append(_finding("normalization_integrity", "candidate_denominator_scope", "candidate denominator must not be row/model/clip local", {"value": normalization.get("candidate_denominator_scope")}))
    return findings


def _audit_quality_delta(
    quality_delta: dict[str, Any],
    baseline_summary: dict[str, Any],
    summary: dict[str, Any],
    gate_report: dict[str, Any],
) -> list[Finding]:
    findings = []
    if quality_delta.get("legacy_gates_unchanged") is not True:
        findings.append(_finding("quality_delta_vs_step4_3", "legacy_gates_unchanged", "quality delta must preserve legacy gates", {}))
    if _as_int(quality_delta.get("raw_residual_regression_count")) != 0:
        findings.append(_finding("normalization_integrity", "quality_delta_vs_step4_3.json", "raw residual regression must remain zero", {}))
    baseline_counts = quality_delta.get("baseline_counts", {})
    current_counts = quality_delta.get("current_counts", {})
    if _as_int(baseline_counts.get("rows_below_candidate_warn_gate")) != _as_int(baseline_summary.get("rows_below_candidate_warn_gate")):
        findings.append(_finding("quality_delta_vs_step4_3", "baseline_counts", "baseline warned count mismatch", {"delta": baseline_counts, "baseline": baseline_summary.get("rows_below_candidate_warn_gate")}))
    if _as_int(current_counts.get("rows_below_candidate_warn_gate")) != _as_int(summary.get("rows_below_candidate_warn_gate")):
        findings.append(_finding("quality_delta_vs_step4_3", "current_counts", "current warned count mismatch", {"delta": current_counts, "summary": summary.get("rows_below_candidate_warn_gate")}))
    if _as_int(current_counts.get("release_quality_candidate_blocked_count")) != _as_int(gate_report.get("release_quality_candidate_blocked_count")):
        findings.append(_finding("quality_delta_vs_step4_3", "current_counts", "current blocked count mismatch", {"delta": current_counts, "gate": gate_report.get("release_quality_candidate_blocked_count")}))
    return findings


def _audit_deterministic(deterministic: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if deterministic.get("status") != "passed" or deterministic.get("deterministic") is not True:
        findings.append(_finding("deterministic_rerun", "deterministic_rerun.json", "deterministic rerun must pass", {"status": deterministic.get("status")}))
    for key in ("compared_count", "matched_count", "deterministic_compared_count", "deterministic_matched_count"):
        if _as_int(deterministic.get(key)) != 44:
            findings.append(_finding("deterministic_rerun", "deterministic_rerun.json", f"{key} must equal 44", {"actual": deterministic.get(key)}))
    for key in ("deterministic_compared_count", "deterministic_matched_count"):
        if _as_int(summary.get(key)) != 44:
            findings.append(_finding("deterministic_rerun", "quality_summary.json", f"{key} must equal 44", {"actual": summary.get(key)}))
    return findings


def _audit_exports_and_diagnostics(
    trajectory: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    exports = _matrix_rows(trajectory)
    if len(exports) < 128:
        findings.append(_finding("trajectory_exports", "trajectory_export_manifest.json", "expected at least 128 trajectory exports", {"rows": len(exports)}))
    for index, row in enumerate(exports):
        finite = row.get("finite_qpos", row.get("qpos_finite", row.get("finite")))
        if finite is not True or _as_int(row.get("nan_count")) != 0 or _as_int(row.get("inf_count")) != 0:
            findings.append(_finding("trajectory_exports", f"row[{index}]", "export must be finite with zero NaN/Inf", {"row": row}))
    temporal_rows = _matrix_rows(temporal)
    support_rows = _matrix_rows(support)
    collision_rows = _matrix_rows(collision)
    if len(temporal_rows) < 128:
        findings.append(_finding("temporal_continuity", "temporal_continuity_matrix.json", "temporal diagnostics are missing rows", {"rows": len(temporal_rows)}))
    if len(support_rows) < 128:
        findings.append(_finding("contact_collision_diagnostics", "support_contact_diagnostics.json", "support/contact diagnostics are missing rows", {"rows": len(support_rows)}))
    if len(collision_rows) < 128:
        findings.append(_finding("contact_collision_diagnostics", "collision_proxy_diagnostics.json", "collision diagnostics are missing rows", {"rows": len(collision_rows)}))
    finite_temporal = sum(1 for row in temporal_rows if row.get("finite", row.get("finite_velocity", True) and row.get("finite_acceleration", True)) is True)
    if finite_temporal != len(temporal_rows) or _as_int(summary.get("temporal_continuity_finite_count")) != finite_temporal:
        findings.append(_finding("temporal_continuity", "temporal_continuity_matrix.json", "temporal diagnostics must be finite", {"finite": finite_temporal, "rows": len(temporal_rows), "summary": summary.get("temporal_continuity_finite_count")}))
    for name, rows in (("support_contact_diagnostics.json", support_rows), ("collision_proxy_diagnostics.json", collision_rows)):
        for index, row in enumerate(rows):
            if row.get("finite", True) is not True:
                findings.append(_finding("contact_collision_diagnostics", f"{name}[{index}]", "diagnostic row must be finite", {"row": row}))
    return findings


def _audit_configs(pipeline_config: dict[str, Any], solver_config: dict[str, Any]) -> list[Finding]:
    findings = []
    for name, payload in (("pipeline_config.json", pipeline_config), ("solver_config.json", solver_config)):
        if payload.get("global_config") is not True or payload.get("robot_specific_tuning") is True:
            findings.append(_finding("global_config", name, "config must be global and non-specific", {"payload": payload}))
        if _contains_specific_tuning(payload):
            findings.append(_finding("no_robot_specific_tuning", name, "config contains robot-specific tuning or per-robot weights", {}))
    config = pipeline_config.get("config") if isinstance(pipeline_config.get("config"), dict) else {}
    policy = solver_config.get("release_quality_v2_validation_policy") if isinstance(solver_config.get("release_quality_v2_validation_policy"), dict) else {}
    if config.get("legacy_gates_unchanged") is not True or policy.get("legacy_runtime_quality_gates_changed") is True:
        findings.append(_finding("global_config", "config", "legacy gates must remain unchanged", {"config": config, "policy": policy}))
    if config.get("candidate_thresholds_lowered") is True or policy.get("candidate_thresholds_lowered") is True:
        findings.append(_finding("threshold_integrity", "config", "candidate thresholds must not be lowered", {"config": config, "policy": policy}))
    if config.get("production_default_changed") is True or policy.get("production_default_changed") is True:
        findings.append(_finding("global_config", "config", "production default must not change", {"config": config, "policy": policy}))
    if config.get("runtime_override_default_enabled") is True or policy.get("runtime_override_default_enabled") is True:
        findings.append(_finding("global_config", "config", "runtime override must not default-enable", {"config": config, "policy": policy}))
    return findings


def _audit_pipeline_controls(artifact_dir: Path) -> list[Finding]:
    if any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        return []
    return [_finding("pipeline_controls", "pipeline controls", "pipeline controls are missing", {})]


def _audit_release_candidate_status(
    summary: dict[str, Any],
    ledger: dict[str, Any],
    gate_report: dict[str, Any],
    quality_delta: dict[str, Any],
    candidate_audit: dict[str, Any],
    promotion: dict[str, Any],
) -> list[Finding]:
    findings = []
    status = str(summary.get("release_candidate_status") or "")
    if status not in VALID_RELEASE_STATUSES:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "invalid Step 4.4 release status", {"status": status}))
    if str(ledger.get("release_candidate_status") or "") != status:
        findings.append(_finding("release_candidate_status", "acceptance_ledger.json", "ledger status must match summary", {"ledger": ledger.get("release_candidate_status"), "summary": status}))
    improvement = _has_step4_4_improvement(gate_report, quality_delta)
    if status == "PASS_RC":
        requirements = {
            "release_candidate_status_has_real_basis": improvement or promotion.get("promote_to_release_candidate_gate") is True,
            "runtime_quality_failed_count_zero": _as_int(summary.get("runtime_quality_failed_count")) == 0,
            "legacy_gates_unchanged": summary.get("legacy_gates_unchanged") is True,
            "candidate_warned_rows_stable": candidate_audit.get("all_stable_across_four_clips") is True,
            "runtime_quality_passed_not_candidate_mixed": _as_int(summary.get("runtime_quality_passed_count")) == 0,
        }
        if not all(requirements.values()):
            findings.append(_finding("release_candidate_status", "quality_summary.json", "PASS_RC requirements are not met", requirements))
    elif status in ACCEPTABLE_BLOCKED_STATUSES:
        if not _blocked_diagnostics_complete(candidate_audit, gate_report, quality_delta, promotion):
            findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked status requires complete diagnostics", {"status": status}))
    elif status.startswith("BLOCKED") and improvement:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked status contradicts Step 4.4 improvement", {"status": status}))
    return findings


def _has_step4_4_improvement(gate_report: dict[str, Any], quality_delta: dict[str, Any]) -> bool:
    baseline = gate_report.get("baseline_step4_3_counts", {})
    return bool(
        len(gate_report.get("rows_below_candidate_pass_gate", [])) > _as_int(baseline.get("rows_below_candidate_pass_gate"))
        or len(gate_report.get("rows_below_candidate_warn_gate", [])) > _as_int(baseline.get("rows_below_candidate_warn_gate"))
        or _as_int(gate_report.get("release_quality_candidate_blocked_count")) < _as_int(baseline.get("release_quality_candidate_blocked_count"))
        or bool(quality_delta.get("improvements"))
    )


def _blocked_diagnostics_complete(
    candidate_audit: dict[str, Any],
    gate_report: dict[str, Any],
    quality_delta: dict[str, Any],
    promotion: dict[str, Any],
) -> bool:
    return bool(candidate_audit and gate_report and quality_delta and promotion and promotion.get("why") and promotion.get("risks"))


def _contains_specific_tuning(value: Any) -> bool:
    suspicious_keys = {
        "per_robot_solver_weights",
        "per_robot_weights",
        "per_robot_ik_weights",
        "per_robot_thresholds",
        "robot_specific_thresholds",
        "model_id_thresholds",
        "whitelist",
        "blacklist",
        "allowlist",
        "denylist",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in suspicious_keys or SUSPICIOUS_SPECIFIC_RE.search(key_text):
                return True
            if _contains_specific_tuning(item):
                return True
    elif isinstance(value, list):
        return any(_contains_specific_tuning(item) for item in value)
    elif isinstance(value, str):
        return bool(SUSPICIOUS_SPECIFIC_RE.search(value))
    return False


def _audit_no_readiness_claims(*payloads: dict[str, Any]) -> list[Finding]:
    findings = []
    for index, payload in enumerate(payloads):
        claims = _positive_readiness_claims(payload)
        if claims:
            findings.append(_finding("readiness_claims", f"payload[{index}]", "visual/deployment readiness must not be claimed in Step 4.4", {"claims": claims}))
    return findings


def _positive_readiness_claims(value: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_text = str(key).lower()
            if key_text in {
                "visual_ready",
                "visual_readiness",
                "visual_readiness_claimed",
                "deployment_ready",
                "deployment_readiness",
                "deployment_readiness_claimed",
            } and item is True:
                out.append(path)
            out.extend(_positive_readiness_claims(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.extend(_positive_readiness_claims(item, f"{prefix}[{index}]"))
    return out


def _audit_no_specific_tuning(source_root: Path) -> list[Finding]:
    findings = []
    for relative in SCAN_PATHS:
        path = source_root / relative
        files = [path] if path.is_file() else sorted(path.rglob("*.py")) if path.is_dir() else []
        for file_path in files:
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines, start=1):
                if (
                    "SUSPICIOUS_SPECIFIC_RE" in line
                    or "per_robot_threshold|by_robot_weight" in line
                    or "force_pass|force_promote" in line
                    or "accepted_models|allowed_models" in line
                    or (file_path.name == "audit_retargeting_v3_step4_4_release_quality_v2_validation.py" and line.strip().startswith('"'))
                ):
                    continue
                if SUSPICIOUS_SPECIFIC_RE.search(line):
                    findings.append(_finding("no_robot_specific_tuning", f"{_display_path(file_path)}:{index}", "suspicious specific tuning shortcut", {"line": line.strip()}))
    return findings


def _audit_closed_artifact_trees(source_root: Path) -> list[Finding]:
    if not (source_root / ".git").exists():
        return []
    findings = []
    for relative in CLOSED_ARTIFACT_DIRS:
        status = _git_stdout(source_root, "status", "--short", "--", relative)
        if status:
            findings.append(_finding("closed_artifact_trees", relative, "closed artifact tree has local modifications", {"git_status": status}))
    return findings


def _audit_final_head_ci(record: dict[str, Any], source_root: Path) -> list[Finding]:
    findings = []
    current_head = _git_stdout(source_root, "rev-parse", "HEAD")
    if not record:
        return [_finding("final_head_ci", "acceptance_ledger.json", "final HEAD CI evidence is missing", {})]
    if not _is_concrete_run_id(record.get("workflow_run_id")):
        findings.append(_finding("final_head_ci", "workflow_run_id", "workflow_run_id must be concrete", {"record": record}))
    if current_head and record.get("head_sha") != current_head:
        findings.append(_finding("final_head_ci", "head_sha", "CI head_sha must equal current HEAD", {"record": record.get("head_sha"), "current_head": current_head}))
    if record.get("conclusion") != "success":
        findings.append(_finding("final_head_ci", "conclusion", "CI conclusion must be success", {"conclusion": record.get("conclusion")}))
    return findings


def _audit_lfs_evidence(record: dict[str, Any]) -> list[Finding]:
    if not record:
        return [_finding("lfs", "acceptance_ledger.json", "Git LFS fsck evidence is missing", {})]
    if record.get("git_lfs_fsck") != "OK" or _as_int(record.get("exit_code")) != 0:
        return [_finding("lfs", "acceptance_ledger.json", "Git LFS fsck evidence must be OK", {"record": record})]
    return []


def _final_head_ci_record(acceptance_ledger: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    for payload in (acceptance_ledger, summary):
        for key in ("final_head_ci", "github_actions", "ci"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, dict):
                return value
    return {}


def _lfs_record(acceptance_ledger: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    for payload in (acceptance_ledger, summary):
        value = payload.get("lfs") if isinstance(payload, dict) else None
        if isinstance(value, dict):
            return value
    return {}


def _invariants(
    summary: dict[str, Any],
    full_rows: list[dict[str, Any]],
    deterministic: dict[str, Any],
    trajectory: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
) -> dict[str, Any]:
    partition = dict(Counter(str(row.get("category")) for row in full_rows))
    return {
        "row_count": len(full_rows),
        "partition": partition,
        "partition_32_3_9_preserved": partition == EXPECTED_CATEGORY_COUNTS,
        "solver_backed_count": _as_int(summary.get("solver_backed_count")),
        "residual_only_count": _as_int(summary.get("residual_only_count")),
        "runtime_quality_failed_count": _as_int(summary.get("runtime_quality_failed_count")),
        "legacy_gates_unchanged": summary.get("legacy_gates_unchanged") is True,
        "production_default_changed": summary.get("production_default_changed") is True,
        "runtime_override_default_enabled": summary.get("runtime_override_default_enabled") is True,
        "deterministic_compared_count": _as_int(deterministic.get("deterministic_compared_count", deterministic.get("compared_count"))),
        "deterministic_matched_count": _as_int(deterministic.get("deterministic_matched_count", deterministic.get("matched_count"))),
        "trajectory_exports_count": len(_matrix_rows(trajectory)),
        "temporal_diagnostic_count": len(_matrix_rows(temporal)),
        "support_contact_diagnostic_count": len(_matrix_rows(support)),
        "collision_proxy_diagnostic_count": len(_matrix_rows(collision)),
    }


def _candidate_counts(
    summary: dict[str, Any],
    gate_report: dict[str, Any],
    quality_delta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "release_quality_candidate_passed_count": _as_int(summary.get("release_quality_candidate_passed_count")),
        "release_quality_candidate_warned_count": _as_int(summary.get("release_quality_candidate_warned_count")),
        "release_quality_candidate_blocked_count": _as_int(summary.get("release_quality_candidate_blocked_count")),
        "rows_below_candidate_warn_gate": _as_int(summary.get("rows_below_candidate_warn_gate")),
        "rows_below_candidate_pass_gate": _as_int(summary.get("rows_below_candidate_pass_gate")),
        "candidate_warned_rows": gate_report.get("rows_below_candidate_warn_gate", []),
        "candidate_passed_rows": gate_report.get("rows_below_candidate_pass_gate", []),
        "deltas_vs_step4_3": quality_delta.get("count_deltas", {}),
    }


def _legacy_counts(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_quality_passed_count": _as_int(summary.get("runtime_quality_passed_count")),
        "runtime_quality_warned_count": _as_int(summary.get("runtime_quality_warned_count")),
        "runtime_quality_failed_count": _as_int(summary.get("runtime_quality_failed_count")),
        "high_residual_warning_count": _as_int(summary.get("high_residual_warning_count")),
    }


def _normalization_summary(normalization: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalization_policy_selected": normalization.get("normalization_policy_selected"),
        "candidate_denominator_scope": normalization.get("candidate_denominator_scope"),
        "fixed_global_scale": normalization.get("fixed_global_scale"),
        "legacy_gates_unchanged": normalization.get("legacy_gates_unchanged"),
        "raw_residual_regression_count": normalization.get("raw_residual_regression_count"),
        "denominator_inflation_detected": normalization.get("denominator_inflation_detected"),
        "normalization_hides_raw_residual_regression": normalization.get("normalization_hides_raw_residual_regression"),
        "denominator_robot_specific": normalization.get("denominator_robot_specific"),
        "candidate_release_gates_not_counted_as_legacy_runtime_quality_passed": normalization.get("candidate_release_gates_not_counted_as_legacy_runtime_quality_passed"),
    }


def _candidate_stability(candidate_audit: dict[str, Any], stress_test: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_step4_3_candidate_warned_count": candidate_audit.get("baseline_step4_3_candidate_warned_count"),
        "candidate_warned_rows": candidate_audit.get("candidate_warned_rows", []),
        "all_stable_across_four_clips": candidate_audit.get("all_stable_across_four_clips"),
        "diagnostics_healthy_count": candidate_audit.get("diagnostics_healthy_count"),
        "per_clip_stability": stress_test.get("per_clip_stability", {}),
        "worst_clip_status": stress_test.get("worst_clip_status", {}),
    }


def _blocker_taxonomy_summary(taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        "blocked_row_count": taxonomy.get("blocked_row_count"),
        "category_counts": taxonomy.get("category_counts", {}),
        "required_categories": taxonomy.get("required_categories", []),
    }


def _matrix_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("rows", "matrix", "exports", "model_rows"):
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


def _model_id(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or "")


def _row_status(row: dict[str, Any]) -> str:
    return str(row.get("runtime_quality_status") or row.get("quality_classification") or row.get("release_candidate_row_status") or "")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_concrete_run_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and bool(CONCRETE_RUN_ID_RE.fullmatch(str(value)))


def _git_stdout(source_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
