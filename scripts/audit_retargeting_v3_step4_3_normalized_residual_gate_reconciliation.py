#!/usr/bin/env python3
"""Audit Step 4.3 normalized residual gate reconciliation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_3_normalized_residual_gate_reconciliation")
DEFAULT_BASELINE_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_2_orientation_policy_runtime_scoring")
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
    "model_matrix.json",
    "full_pipeline_matrix.json",
    "clip_matrix.json",
    "solver_smoke_matrix.json",
    "generic_smoke_matrix.json",
    "deterministic_rerun.json",
    "quality_delta_vs_step4_2.json",
    "runtime_scoring_delta_vs_step4_2.json",
    "normalized_residual_scale_audit.json",
    "gate_semantics_audit.json",
    "normalization_candidate_matrix.json",
    "gate_candidate_matrix.json",
    "normalization_policy_selection.json",
    "gate_reconciliation_v2_report.json",
    "scoring_normalization_audit.json",
    "orientation_policy_runtime_impact_report.json",
    "orientation_integrated_residual_matrix.json",
    "active_vs_diagnostic_policy_matrix.json",
    "trajectory_export_manifest.json",
    "temporal_continuity_matrix.json",
    "support_contact_diagnostics.json",
    "collision_proxy_diagnostics.json",
    "pipeline_config.json",
    "solver_config.json",
    "red_team_report.json",
    "test_results/pytest.txt",
    "test_results/pytest_summary.json",
    "test_results/junit.xml",
)
PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES = (
    "pipeline_controls_reference.json",
    "pipeline_backed_matrix.json",
)
EXPECTED_CANDIDATES = {
    "candidate_0_current_legacy_normalized_gate",
    "candidate_1_fixed_global_body_scale_normalization",
    "candidate_2_semantic_task_scale_normalization",
    "candidate_3_orientation_integrated_scale_normalization",
    "candidate_4_raw_residual_percentile_gate_diagnostic",
    "candidate_5_two_metric_gate_raw_plus_normalized",
    "candidate_6_release_quality_gate_v2_candidate",
}
VALID_RELEASE_STATUSES = {
    "PASS_RC",
    "BLOCKED_GATE_RECONCILIATION",
    "BLOCKED_NORMALIZATION_INTEGRITY",
    "BLOCKED_SCORING_SCALE_AMBIGUITY",
    "BLOCKED_PIPELINE_REGRESSION",
    "BLOCKED_CI_OR_PROVENANCE",
}
ACCEPTABLE_BLOCKED_STATUSES = {
    "BLOCKED_GATE_RECONCILIATION",
    "BLOCKED_NORMALIZATION_INTEGRITY",
    "BLOCKED_SCORING_SCALE_AMBIGUITY",
}
REQUIRED_FINAL_HEAD_CI_JOBS = (
    "step4-3-static-and-unit",
    "step4-3-artifact-audit",
    "step4-3-gate-semantics-smoke",
    "step4-3-normalization-integrity-smoke",
    "step4-3-export-and-temporal-smoke",
    "step4-3-lfs-and-snapshot-smoke",
    "step4-3-pipeline-controls-reference",
    "step4-3-final-head-ci-evidence",
)
SCAN_PATHS = (
    "soma_retargeter/runtime/v3",
    "soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py",
    "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
    "soma_retargeter/tools/step4_3_normalized_residual_gate_reconciliation.py",
    "scripts/audit_retargeting_v3_step4_3_normalized_residual_gate_reconciliation.py",
)
CONCRETE_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", flags=re.IGNORECASE)
CONCRETE_RUN_ID_RE = re.compile(r"\b[1-9][0-9]{5,}\b")
SUSPICIOUS_ROBOT_SPECIFIC_RE = re.compile(
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
    artifact_dir: str
    baseline_artifact_dir: str
    source_root: str
    matrix_row_count: int
    blocking_count: int
    finding_count: int
    gate_counts: dict[str, int]
    findings: list[Finding]
    final_head_ci: dict[str, Any]

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
    environment = _read_json(artifact_dir / "environment.json")
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    runtime_delta = _read_json(artifact_dir / "runtime_scoring_delta_vs_step4_2.json")
    quality_delta = _read_json(artifact_dir / "quality_delta_vs_step4_2.json")
    scale_audit = _read_json(artifact_dir / "normalized_residual_scale_audit.json")
    gate_semantics = _read_json(artifact_dir / "gate_semantics_audit.json")
    normalization_candidates = _read_json(artifact_dir / "normalization_candidate_matrix.json")
    gate_candidates = _read_json(artifact_dir / "gate_candidate_matrix.json")
    policy_selection = _read_json(artifact_dir / "normalization_policy_selection.json")
    gate_report = _read_json(artifact_dir / "gate_reconciliation_v2_report.json")
    trajectory = _read_json(artifact_dir / "trajectory_export_manifest.json")
    temporal = _read_json(artifact_dir / "temporal_continuity_matrix.json")
    support = _read_json(artifact_dir / "support_contact_diagnostics.json")
    collision = _read_json(artifact_dir / "collision_proxy_diagnostics.json")
    solver_config = _read_json(artifact_dir / "solver_config.json")
    pipeline_config = _read_json(artifact_dir / "pipeline_config.json")
    baseline_summary = _read_json(baseline_artifact_dir / "quality_summary.json")

    rows = _matrix_rows(model_matrix)
    full_rows = _matrix_rows(full_pipeline)
    solver_rows = _matrix_rows(solver_smoke)
    generic_rows = _matrix_rows(generic_smoke)
    committed_final_head_ci = _final_head_ci_record(ledger, summary)
    live_final_head_ci = _live_final_head_ci_record(source_root) if require_final_head_ci else {}
    final_head_ci = live_final_head_ci or committed_final_head_ci

    findings: list[Finding] = []
    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_baseline_step4_2(baseline_artifact_dir, baseline_summary, summary, ledger, runtime_delta))
    findings.extend(_audit_clean_provenance(environment, summary, ledger, runtime_delta, source_root))
    findings.extend(_audit_matrix_shape_and_partition(rows, full_rows, summary))
    findings.extend(_audit_solver_and_status_counts(rows, solver_rows, summary))
    findings.extend(_audit_runtime_quality_label_honesty(rows, full_rows, solver_rows, generic_rows))
    findings.extend(_audit_negative_and_partial(rows, full_rows, summary))
    findings.extend(_audit_scale_audit(scale_audit, summary))
    findings.extend(_audit_gate_semantics(gate_semantics))
    findings.extend(_audit_candidate_matrices(normalization_candidates, gate_candidates, policy_selection))
    findings.extend(_audit_gate_reconciliation_v2(gate_report, summary, runtime_delta))
    findings.extend(_audit_runtime_delta(runtime_delta, quality_delta, baseline_summary, summary))
    findings.extend(_audit_release_candidate_status(summary, ledger, runtime_delta, gate_report, scale_audit))
    findings.extend(_audit_exports_and_temporal(trajectory, temporal, support, collision, summary))
    findings.extend(_audit_deterministic(deterministic, summary))
    findings.extend(_audit_configs(solver_config, pipeline_config))
    findings.extend(_audit_pipeline_controls(artifact_dir))
    findings.extend(_audit_no_robot_specific_tuning(source_root))
    if require_final_head_ci:
        findings.extend(_audit_final_head_ci(committed_final_head_ci, final_head_ci, source_root))

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
        artifact_dir=_display_path(artifact_dir),
        baseline_artifact_dir=_display_path(baseline_artifact_dir),
        source_root=_display_path(source_root),
        matrix_row_count=len(rows),
        blocking_count=blocking_count,
        finding_count=len(findings),
        gate_counts=gate_counts,
        findings=findings,
        final_head_ci=final_head_ci,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--baseline-step4-2-artifact-dir", "--baseline-artifact-dir", dest="baseline_artifact_dir", default=str(DEFAULT_BASELINE_ARTIFACT_DIR))
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
        return [_finding("missing_required_artifacts", _display_path(artifact_dir), "Step 4.3 artifact directory is missing", {})]
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(_finding("missing_required_artifacts", relative, "required Step 4.3 artifact is missing", {}))
    if not any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        findings.append(_finding("missing_required_artifacts", "pipeline controls", "pipeline control evidence is missing", {}))
    return findings


def _audit_baseline_step4_2(
    baseline_artifact_dir: Path,
    baseline_summary: dict[str, Any],
    summary: dict[str, Any],
    ledger: dict[str, Any],
    runtime_delta: dict[str, Any],
) -> list[Finding]:
    findings = []
    if not baseline_artifact_dir.exists():
        findings.append(_finding("baseline_step4_2", _display_path(baseline_artifact_dir), "Step 4.2 baseline artifact dir is missing", {}))
    expected = {
        "in_scope_total": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "solver_backed_count": 32,
        "residual_only_count": 0,
        "runtime_quality_failed_count": 0,
        "runtime_quality_passed_count": 0,
        "runtime_quality_warned_count": 32,
        "high_residual_warning_count": 32,
    }
    for field, expected_value in expected.items():
        if _as_int(baseline_summary.get(field)) != expected_value:
            findings.append(_finding("baseline_step4_2", field, "baseline Step 4.2 summary mismatch", {"actual": baseline_summary.get(field), "expected": expected_value}))
    if baseline_summary.get("release_candidate_status") != "PASS_RC":
        findings.append(_finding("baseline_step4_2", "release_candidate_status", "baseline Step 4.2 must be PASS_RC", {"actual": baseline_summary.get("release_candidate_status")}))
    base_head = summary.get("base_step4_2_final_head") or ledger.get("base_step4_2_final_head")
    if not _is_concrete_sha(base_head):
        findings.append(_finding("baseline_step4_2", "base_step4_2_final_head", "base Step 4.2 final HEAD must be concrete", {"value": base_head}))
    if runtime_delta.get("baseline_counts") and _as_int(runtime_delta["baseline_counts"].get("runtime_quality_warned_count")) != 32:
        findings.append(_finding("baseline_step4_2", "runtime_scoring_delta_vs_step4_2.json", "baseline counts must preserve Step 4.2 truth", {}))
    return findings


def _audit_clean_provenance(
    environment: dict[str, Any],
    summary: dict[str, Any],
    ledger: dict[str, Any],
    runtime_delta: dict[str, Any],
    source_root: Path,
) -> list[Finding]:
    findings = []
    for field in (
        "source_code_commit_remote_resolvable",
        "source_code_commit_is_artifact_commit_ancestor",
        "source_worktree_clean_before_run",
        "source_worktree_clean_after_run",
    ):
        if environment.get(field) is not True:
            findings.append(_finding("clean_provenance", "environment.json", "clean-provenance field must be true", {"field": field, "value": environment.get(field)}))
    if environment.get("git_status_short") not in {"", None}:
        findings.append(_finding("clean_provenance", "environment.json", "artifact records dirty source status", {"git_status_short": environment.get("git_status_short")}))
    if environment.get("core_diff_after_source_commit") != []:
        findings.append(_finding("clean_provenance", "environment.json", "core diff after source commit must be empty", {"value": environment.get("core_diff_after_source_commit")}))
    source_commit = str(environment.get("source_code_commit") or "")
    if not _is_concrete_sha(source_commit):
        findings.append(_finding("clean_provenance", "environment.json", "source_code_commit must be concrete SHA", {"source_code_commit": source_commit}))
    current_head = _git_stdout(source_root, "rev-parse", "HEAD")
    if current_head and source_commit and _git_returncode(source_root, "merge-base", "--is-ancestor", source_commit, current_head) != 0:
        findings.append(_finding("clean_provenance", "git", "source commit must be ancestor of current HEAD", {"source_commit": source_commit, "current_head": current_head}))
    for subject, payload in (("quality_summary.json", summary), ("acceptance_ledger.json", ledger), ("runtime_scoring_delta_vs_step4_2.json", runtime_delta)):
        commit = payload.get("source_code_commit") or payload.get("current_source_commit")
        if commit and source_commit and str(commit) != source_commit:
            findings.append(_finding("clean_provenance", subject, "source commit fields must agree", {"expected": source_commit, "actual": commit}))
    return findings


def _audit_matrix_shape_and_partition(rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if len(rows) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "model_matrix.json", "model matrix must contain 44 rows", {"actual": len(rows)}))
    if len(full_rows) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "full_pipeline_matrix.json", "full pipeline matrix must contain 44 rows", {"actual": len(full_rows)}))
    counts = dict(Counter(str(row.get("category")) for row in rows))
    if counts != EXPECTED_CATEGORY_COUNTS:
        findings.append(_finding("partition_32_3_9", "model_matrix.json", "category partition must be 32/3/9", {"actual": counts}))
    for field, expected in (("in_scope_total", 44), ("full_humanoid_total", 32), ("partial_total", 3), ("negative_total", 9)):
        if _as_int(summary.get(field)) != expected:
            findings.append(_finding("partition_32_3_9", "quality_summary.json", f"{field} mismatch", {"actual": summary.get(field), "expected": expected}))
    return findings


def _audit_solver_and_status_counts(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    model_full = [row for row in rows if row.get("category") == "full_humanoid_profile"]
    expected_counts = {
        "solver_backed_smoke_attempted_count": sum(1 for row in model_full if row.get("solver_backed_smoke_attempted") is True),
        "solver_backed_completed_count": sum(1 for row in model_full if row.get("solver_backed_smoke_completed") is True),
        "solver_backed_count": sum(1 for row in model_full if row.get("solver_backed") is True),
        "residual_only_count": sum(1 for row in model_full if row.get("residual_only") is True),
        "runtime_quality_failed_count": sum(1 for row in model_full if _row_status(row) == "runtime_quality_failed"),
    }
    for field in ("solver_backed_smoke_attempted_count", "solver_backed_completed_count", "solver_backed_count"):
        if expected_counts[field] < 32 or _as_int(summary.get(field)) != expected_counts[field]:
            findings.append(_finding("solver_backed_counts", field, "Step 4.3 must preserve solver-backed coverage", {"rows": expected_counts[field], "summary": summary.get(field)}))
    for field, expected in (("residual_only_count", 0), ("runtime_quality_failed_count", 0)):
        if expected_counts[field] != expected or _as_int(summary.get(field)) != expected:
            findings.append(_finding("solver_backed_counts", field, "Step 4.3 invariant mismatch", {"rows": expected_counts[field], "summary": summary.get(field), "expected": expected}))
    completed_models = {
        str(row.get("model_id"))
        for row in solver_rows
        if row.get("category") == "full_humanoid_profile"
        and row.get("solver_backed") is True
        and row.get("solver_backed_smoke_completed") is True
    }
    if len(completed_models) != 32:
        findings.append(_finding("solver_backed_counts", "solver_smoke_matrix.json", "solver smoke must cover all 32 full humanoids", {"completed": len(completed_models)}))
    return findings


def _audit_runtime_quality_label_honesty(
    rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings = []
    solver_models = {_model_id(row) for row in solver_rows if row.get("solver_backed") is True and row.get("solver_backed_smoke_completed") is True}
    generic_models = {_model_id(row) for row in generic_rows if row.get("solver_backed") is True and row.get("solver_backed_smoke_completed") is True}
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows)):
        for index, row in enumerate(collection):
            if row.get("category") != "full_humanoid_profile":
                continue
            status = _row_status(row)
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            if status not in {"runtime_quality_passed", "runtime_quality_warned", "runtime_quality_failed", "runtime_evaluation_completed"}:
                findings.append(_finding("runtime_quality_label_honesty", subject, "invalid full humanoid runtime quality status", {"status": status}))
            if status == "runtime_quality_passed":
                requirements = {
                    "solver_backed": row.get("solver_backed") is True,
                    "solver_backed_smoke_completed": row.get("solver_backed_smoke_completed") is True,
                    "residual_only_false": row.get("residual_only") is False,
                    "solver_matrix_evidence": _model_id(row) in solver_models,
                    "generic_matrix_evidence": _model_id(row) in generic_models,
                    "legacy_residual_pass_gate": _as_float(row.get("normalized_task_residual_p95")) <= 0.15,
                    "output_nan_zero": _as_int(row.get("output_nan_count")) == 0,
                    "output_inf_zero": _as_int(row.get("output_inf_count")) == 0,
                }
                if not all(requirements.values()):
                    findings.append(_finding("runtime_quality_label_honesty", subject, "runtime_quality_passed row lacks unchanged legacy gate evidence", requirements))
    return findings


def _audit_negative_and_partial(rows: list[dict[str, Any]], full_pipeline_rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    partial_count = 0
    negative_count = 0
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows)):
        for index, row in enumerate(collection):
            if row.get("category") not in {"partial_humanoid_profile", "negative_control"}:
                continue
            partial_count += 1 if collection_name == "model_matrix.json" and row.get("category") == "partial_humanoid_profile" else 0
            negative_count += 1 if collection_name == "model_matrix.json" and row.get("category") == "negative_control" else 0
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            if _row_status(row) == "runtime_quality_passed":
                findings.append(_finding("negative_and_partial_not_promoted", subject, "partial/negative rows must not be promoted", {}))
            for flag in ("solver_backed", "solver_backed_smoke_attempted", "solver_backed_smoke_completed"):
                if row.get(flag) is True:
                    findings.append(_finding("negative_and_partial_not_promoted", subject, f"partial/negative row must not set {flag}", {"value": row.get(flag)}))
    if _as_int(summary.get("partial_runtime_passed_count")) != partial_count:
        findings.append(_finding("negative_and_partial_not_promoted", "quality_summary.json", "partial rows must not count as full humanoid passes", {"summary": summary.get("partial_runtime_passed_count"), "rows": partial_count}))
    if _as_int(summary.get("negative_control_runtime_passed_count")) != negative_count:
        findings.append(_finding("negative_and_partial_not_promoted", "quality_summary.json", "negative controls must not be promoted", {"summary": summary.get("negative_control_runtime_passed_count"), "rows": negative_count}))
    return findings


def _audit_scale_audit(scale_audit: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    required = (
        "legacy_normalization_formula",
        "denominator_scope",
        "denominator_follows_current_row_max",
        "row_local_denominator_saturation_detected",
        "why_normalized_p95_remains_near_one",
        "raw_vs_legacy_normalized_spearman",
        "task_class_dominance",
        "rows_improve_raw_but_remain_legacy_normalized_blocked_count",
        "raw_residual_always_retained",
        "denominator_inflation_detected",
        "normalization_hides_raw_residual_regression",
    )
    for field in required:
        if field not in scale_audit:
            findings.append(_finding("normalized_residual_scale_audit", "normalized_residual_scale_audit.json", "required scale-audit field missing", {"field": field}))
    if scale_audit.get("denominator_scope") != "row_local_legacy_metric":
        findings.append(_finding("normalized_residual_scale_audit", "denominator_scope", "legacy denominator scope must be row-local", {"value": scale_audit.get("denominator_scope")}))
    if scale_audit.get("raw_residual_always_retained") is not True:
        findings.append(_finding("normalization_integrity", "normalized_residual_scale_audit.json", "raw residual must be retained", {}))
    if scale_audit.get("denominator_inflation_detected") is True or summary.get("denominator_inflation_detected") is True:
        findings.append(_finding("normalization_integrity", "normalized_residual_scale_audit.json", "denominator inflation detected", {}))
    if scale_audit.get("normalization_hides_raw_residual_regression") is True or summary.get("normalization_hides_raw_residual_regression") is True:
        findings.append(_finding("normalization_integrity", "normalized_residual_scale_audit.json", "normalization hides raw residual regression", {}))
    dominance = scale_audit.get("task_class_dominance")
    if not isinstance(dominance, dict) or not dominance.get("dominant_semantic_counts"):
        findings.append(_finding("normalized_residual_scale_audit", "task_class_dominance", "task class dominance must be audited", {"value": dominance}))
    return findings


def _audit_gate_semantics(gate_semantics: dict[str, Any]) -> list[Finding]:
    findings = []
    required = (
        "legacy_thresholds",
        "normalized_task_residual_p95_pass_semantics",
        "normalized_task_residual_p95_warn_semantics",
        "designed_for_metric",
        "active_parent_relative_residual_needs_new_metric_name",
        "new_metric_name",
        "gate_categories",
        "legacy_gates_unchanged",
        "candidate_gates_do_not_replace_legacy_runtime_quality_passed",
    )
    for field in required:
        if field not in gate_semantics:
            findings.append(_finding("gate_semantics", "gate_semantics_audit.json", "required gate-semantics field missing", {"field": field}))
    if gate_semantics.get("legacy_gates_unchanged") is not True:
        findings.append(_finding("gate_semantics", "gate_semantics_audit.json", "legacy gates must remain unchanged", {}))
    if gate_semantics.get("candidate_gates_do_not_replace_legacy_runtime_quality_passed") is not True:
        findings.append(_finding("gate_semantics", "gate_semantics_audit.json", "candidate gates must not replace legacy runtime pass semantics", {}))
    return findings


def _audit_candidate_matrices(
    normalization_candidates: dict[str, Any],
    gate_candidates: dict[str, Any],
    policy_selection: dict[str, Any],
) -> list[Finding]:
    findings = []
    norm_ids = {str(row.get("candidate_id")) for row in _matrix_rows(normalization_candidates)}
    gate_ids = {str(row.get("candidate_id")) for row in _matrix_rows(gate_candidates)}
    if not EXPECTED_CANDIDATES.issubset(norm_ids):
        findings.append(_finding("candidate_policies", "normalization_candidate_matrix.json", "normalization candidates are incomplete", {"missing": sorted(EXPECTED_CANDIDATES - norm_ids)}))
    if not EXPECTED_CANDIDATES.issubset(gate_ids):
        findings.append(_finding("candidate_policies", "gate_candidate_matrix.json", "gate candidates are incomplete", {"missing": sorted(EXPECTED_CANDIDATES - gate_ids)}))
    for name, payload in (("normalization_candidate_matrix.json", normalization_candidates), ("gate_candidate_matrix.json", gate_candidates)):
        if payload.get("robot_specific_tuning_used") is True or payload.get("uses_model_id_thresholds") is True or payload.get("uses_clip_id_thresholds") is True:
            findings.append(_finding("candidate_policies", name, "candidate selection must be global", {"payload": payload}))
        for row in _matrix_rows(payload):
            if row.get("robot_specific_tuning_used") is True or row.get("uses_model_id_threshold") is True or row.get("uses_clip_id_threshold") is True:
                findings.append(_finding("candidate_policies", f"{name}:{row.get('candidate_id')}", "candidate uses forbidden specific tuning", {"row": row}))
    if policy_selection.get("normalization_policy_selected") != "candidate_1_fixed_global_body_scale_normalization":
        findings.append(_finding("candidate_policies", "normalization_policy_selection.json", "unexpected normalization policy", {"value": policy_selection.get("normalization_policy_selected")}))
    if policy_selection.get("gate_policy_selected") != "candidate_6_release_quality_gate_v2_candidate":
        findings.append(_finding("candidate_policies", "normalization_policy_selection.json", "unexpected gate policy", {"value": policy_selection.get("gate_policy_selected")}))
    for field in ("legacy_gates_unchanged", "candidate_release_gates_defined", "candidate_release_gates_active", "runtime_quality_passed_count_uses_legacy_gates_only"):
        if policy_selection.get(field) is not True:
            findings.append(_finding("candidate_policies", "normalization_policy_selection.json", "policy selection safety field must be true", {"field": field, "value": policy_selection.get(field)}))
    return findings


def _audit_gate_reconciliation_v2(gate_report: dict[str, Any], summary: dict[str, Any], runtime_delta: dict[str, Any]) -> list[Finding]:
    findings = []
    required = (
        "active_scoring_metrics",
        "diagnostic_only_metrics",
        "legacy_gate_inputs",
        "candidate_gate_inputs",
        "hard_safety_gate_inputs",
        "release_quality_gate_inputs",
        "pass_gate_thresholds_unchanged_for_legacy",
        "candidate_release_gate_thresholds_if_any",
        "rows_below_legacy_warn_gate",
        "rows_below_candidate_warn_gate",
        "rows_below_candidate_pass_gate",
        "rows_newly_passing_under_candidate",
        "rows_still_warned_under_candidate",
        "why_no_pass_if_zero",
        "candidate_release_gates_not_counted_as_legacy_runtime_passed",
    )
    for field in required:
        if field not in gate_report:
            findings.append(_finding("gate_reconciliation_v2", "gate_reconciliation_v2_report.json", "required gate reconciliation field missing", {"field": field}))
    if gate_report.get("pass_gate_thresholds_unchanged_for_legacy") is not True or gate_report.get("legacy_gates_unchanged") is not True:
        findings.append(_finding("gate_reconciliation_v2", "gate_reconciliation_v2_report.json", "legacy gates must remain unchanged", {}))
    if gate_report.get("candidate_release_gates_not_counted_as_legacy_runtime_passed") is not True:
        findings.append(_finding("status_semantics", "gate_reconciliation_v2_report.json", "candidate release gates must stay separate from legacy runtime pass counts", {}))
    candidate_warn = len(gate_report.get("rows_below_candidate_warn_gate", []))
    if _as_int(summary.get("rows_below_candidate_warn_gate")) != candidate_warn:
        findings.append(_finding("gate_reconciliation_v2", "quality_summary.json", "candidate warn count mismatch", {"summary": summary.get("rows_below_candidate_warn_gate"), "gate_report": candidate_warn}))
    if _as_int(runtime_delta.get("rows_below_candidate_warn_gate")) != candidate_warn:
        findings.append(_finding("gate_reconciliation_v2", "runtime_scoring_delta_vs_step4_2.json", "candidate warn count mismatch", {"delta": runtime_delta.get("rows_below_candidate_warn_gate"), "gate_report": candidate_warn}))
    for row in _matrix_rows(gate_report):
        status = row.get("release_quality_v2_status")
        if status == "release_quality_candidate_passed" and row.get("hard_safety_passed") is not True:
            findings.append(_finding("status_semantics", f"gate_reconciliation_v2_report:{row.get('model_id')}", "candidate pass row lacks hard safety evidence", {"row": row}))
    return findings


def _audit_runtime_delta(
    runtime_delta: dict[str, Any],
    quality_delta: dict[str, Any],
    baseline_summary: dict[str, Any],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    if not runtime_delta:
        return [_finding("runtime_scoring_delta_vs_step4_2", "runtime_scoring_delta_vs_step4_2.json", "runtime scoring delta is missing", {})]
    for field in ("in_scope_total", "full_humanoid_total", "partial_total", "negative_total", "solver_backed_count", "residual_only_count", "runtime_quality_failed_count"):
        baseline_counts = runtime_delta.get("baseline_counts", {})
        current_counts = runtime_delta.get("current_counts", {})
        if field in baseline_counts and _as_int(baseline_counts.get(field)) != _as_int(baseline_summary.get(field)):
            findings.append(_finding("runtime_scoring_delta_vs_step4_2", field, "baseline count mismatch", {"delta": baseline_counts.get(field), "baseline": baseline_summary.get(field)}))
        if field in current_counts and _as_int(current_counts.get(field)) != _as_int(summary.get(field)):
            findings.append(_finding("runtime_scoring_delta_vs_step4_2", field, "current count mismatch", {"delta": current_counts.get(field), "summary": summary.get(field)}))
    required = (
        "runtime_quality_passed_count_delta",
        "runtime_quality_warned_count_delta",
        "high_residual_warning_count_delta",
        "p95_normalized_task_residual_p95_delta",
        "p95_orientation_integrated_residual_delta",
        "raw_residual_regression_count",
        "denominator_inflation_detected",
        "normalization_hides_raw_residual_regression",
        "rows_below_candidate_warn_gate",
        "rows_below_candidate_pass_gate",
        "gate_blocker_taxonomy",
    )
    for field in required:
        if field not in runtime_delta:
            findings.append(_finding("runtime_scoring_delta_vs_step4_2", "runtime_scoring_delta_vs_step4_2.json", "required delta field missing", {"field": field}))
    if runtime_delta.get("regressions"):
        findings.append(_finding("runtime_scoring_delta_vs_step4_2", "regressions", "runtime scoring delta records pipeline regressions", {"regressions": runtime_delta.get("regressions")}))
    if runtime_delta.get("denominator_inflation_detected") is True or runtime_delta.get("normalization_hides_raw_residual_regression") is True:
        findings.append(_finding("normalization_integrity", "runtime_scoring_delta_vs_step4_2.json", "normalization integrity failed", {"runtime_delta": runtime_delta}))
    if quality_delta.get("runtime_scoring_delta_vs_step4_2") in ({}, None):
        findings.append(_finding("runtime_scoring_delta_vs_step4_2", "quality_delta_vs_step4_2.json", "quality delta must reference runtime scoring delta", {}))
    return findings


def _audit_release_candidate_status(
    summary: dict[str, Any],
    ledger: dict[str, Any],
    runtime_delta: dict[str, Any],
    gate_report: dict[str, Any],
    scale_audit: dict[str, Any],
) -> list[Finding]:
    findings = []
    status = str(summary.get("release_candidate_status") or "")
    ledger_status = str(ledger.get("release_candidate_status") or "")
    if status not in VALID_RELEASE_STATUSES:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "invalid Step 4.3 release status", {"status": status}))
    if ledger_status != status:
        findings.append(_finding("release_candidate_status", "acceptance_ledger.json", "ledger status must match summary", {"ledger": ledger_status, "summary": status}))
    impact = _has_real_breakthrough(summary, runtime_delta, gate_report)
    if status == "PASS_RC":
        requirements = {
            "primary_quality_breakthrough": summary.get("primary_quality_breakthrough") is True and runtime_delta.get("primary_quality_breakthrough") is True,
            "real_gate_or_scoring_reconciliation_impact": impact,
            "legacy_gates_unchanged": summary.get("legacy_gates_unchanged") is True,
            "candidate_release_gates_defined": summary.get("candidate_release_gates_defined") is True,
            "runtime_quality_failed_count_zero": _as_int(summary.get("runtime_quality_failed_count")) == 0,
            "raw_residual_regression_zero": _as_int(runtime_delta.get("raw_residual_regression_count")) == 0,
            "no_denominator_inflation": scale_audit.get("denominator_inflation_detected") is False,
            "no_normalization_hidden_regression": scale_audit.get("normalization_hides_raw_residual_regression") is False,
        }
        if not all(requirements.values()):
            findings.append(_finding("release_candidate_status", "quality_summary.json", "PASS_RC requirements are not met", requirements))
    elif status in ACCEPTABLE_BLOCKED_STATUSES:
        if not _blocked_diagnostics_complete(scale_audit, gate_report, runtime_delta):
            findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked gate/normalization status requires complete diagnostics", {"status": status}))
    elif status.startswith("BLOCKED") and impact and not runtime_delta.get("regressions"):
        findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked status contradicts gate reconciliation impact", {"status": status}))
    return findings


def _audit_exports_and_temporal(
    trajectory: dict[str, Any],
    temporal: dict[str, Any],
    support: dict[str, Any],
    collision: dict[str, Any],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    exports = _matrix_rows(trajectory)
    if len(exports) < 128:
        findings.append(_finding("trajectory_exports", "trajectory_export_manifest.json", "expected 128 full pipeline exports", {"rows": len(exports)}))
    for index, row in enumerate(exports):
        finite = row.get("finite_qpos", row.get("qpos_finite", row.get("finite")))
        if finite is not True or _as_int(row.get("nan_count")) != 0 or _as_int(row.get("inf_count")) != 0:
            findings.append(_finding("trajectory_exports", f"row[{index}]", "export must be finite with zero NaN/Inf", {"row": row}))
    temporal_rows = _matrix_rows(temporal)
    finite_count = sum(1 for row in temporal_rows if row.get("finite", row.get("finite_velocity") is True and row.get("finite_acceleration") is True) is True)
    if finite_count != len(temporal_rows) or _as_int(summary.get("temporal_continuity_finite_count")) != finite_count:
        findings.append(_finding("temporal_continuity", "temporal_continuity_matrix.json", "temporal diagnostics must be finite", {"finite": finite_count, "rows": len(temporal_rows), "summary": summary.get("temporal_continuity_finite_count")}))
    for name, payload in (("support_contact_diagnostics.json", support), ("collision_proxy_diagnostics.json", collision)):
        rows = _matrix_rows(payload)
        if len(rows) < 128:
            findings.append(_finding("contact_collision_diagnostics", name, "diagnostic matrix must contain 128 rows", {"rows": len(rows)}))
        for index, row in enumerate(rows):
            if row.get("finite") is not True:
                findings.append(_finding("contact_collision_diagnostics", f"{name}[{index}]", "diagnostic row must be finite", {"row": row}))
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


def _audit_configs(solver_config: dict[str, Any], pipeline_config: dict[str, Any]) -> list[Finding]:
    findings = []
    for name, payload in (("solver_config.json", solver_config), ("pipeline_config.json", pipeline_config)):
        if payload.get("robot_specific_tuning") is True or payload.get("global_config") is False:
            findings.append(_finding("global_config", name, "config must be global and non robot-specific", {"payload": payload}))
    config = pipeline_config.get("config") if isinstance(pipeline_config.get("config"), dict) else {}
    policy = solver_config.get("normalized_residual_gate_reconciliation_policy") if isinstance(solver_config.get("normalized_residual_gate_reconciliation_policy"), dict) else {}
    if config.get("enable_normalized_residual_gate_reconciliation") is not True:
        findings.append(_finding("global_config", "pipeline_config.json", "Step 4.3 reconciliation flag must be recorded", {"config": config}))
    if policy.get("legacy_runtime_quality_gates_changed") is True or config.get("legacy_gates_unchanged") is not True:
        findings.append(_finding("global_config", "config", "legacy runtime quality gates must remain unchanged", {"policy": policy, "config": config}))
    if policy.get("production_default_changed") is True or config.get("production_default_changed") is True or config.get("orientation_policy_production_default_changed") is True:
        findings.append(_finding("global_config", "config", "production default must remain unchanged", {"policy": policy, "config": config}))
    if policy.get("runtime_override_default_enabled") is True or config.get("runtime_override_default_enabled") is True:
        findings.append(_finding("global_config", "config", "runtime override must not default-enable", {"policy": policy, "config": config}))
    return findings


def _audit_pipeline_controls(artifact_dir: Path) -> list[Finding]:
    if any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        return []
    return [_finding("pipeline_controls", "pipeline controls", "pipeline controls are missing", {})]


def _audit_no_robot_specific_tuning(source_root: Path) -> list[Finding]:
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
                    "SUSPICIOUS_ROBOT_SPECIFIC_RE" in line
                    or "per_robot_threshold|by_robot_weight" in line
                    or "force_pass|force_promote" in line
                    or "uses_model_id_threshold" in line
                    or "uses_clip_id_threshold" in line
                ):
                    continue
                if SUSPICIOUS_ROBOT_SPECIFIC_RE.search(line):
                    findings.append(_finding("no_robot_specific_tuning", f"{_display_path(file_path)}:{index}", "suspicious robot/model-specific tuning shortcut", {"line": line.strip()}))
    return findings


def _audit_final_head_ci(committed_record: dict[str, Any], record: dict[str, Any], source_root: Path) -> list[Finding]:
    findings = []
    current_head = _git_stdout(source_root, "rev-parse", "HEAD")
    if not record:
        return [_finding("final_head_ci", "github", "final HEAD CI evidence is missing", {})]
    if not _is_concrete_run_id(record.get("workflow_run_id")):
        findings.append(_finding("final_head_ci", "workflow_run_id", "workflow_run_id must be concrete", {"record": record}))
    if record.get("head_sha") != current_head:
        findings.append(_finding("final_head_ci", "head_sha", "CI head_sha must equal current HEAD", {"record": record.get("head_sha"), "current_head": current_head}))
    if record.get("conclusion") != "success":
        findings.append(_finding("final_head_ci", "conclusion", "CI conclusion must be success", {"conclusion": record.get("conclusion")}))
    job_conclusions = _job_conclusions(record.get("job_conclusions"))
    for job in REQUIRED_FINAL_HEAD_CI_JOBS:
        if job_conclusions.get(job) != "success":
            findings.append(_finding("final_head_ci", job, "required Step 4.3 CI job is not successful", {"job_conclusions": job_conclusions, "committed_record": committed_record}))
    return findings


def _has_real_breakthrough(summary: dict[str, Any], runtime_delta: dict[str, Any], gate_report: dict[str, Any]) -> bool:
    return bool(
        (_as_int(summary.get("runtime_quality_passed_count")) or 0) > 0
        or (_as_int(summary.get("high_residual_warning_count")) is not None and _as_int(summary.get("high_residual_warning_count")) < 32)
        or len(gate_report.get("rows_below_candidate_warn_gate", [])) > 0
        or len(gate_report.get("rows_below_candidate_pass_gate", [])) > 0
        or float(runtime_delta.get("p95_normalized_task_residual_p95_delta", 0.0) or 0.0) <= -0.10
    )


def _blocked_diagnostics_complete(scale_audit: dict[str, Any], gate_report: dict[str, Any], runtime_delta: dict[str, Any]) -> bool:
    return bool(scale_audit and gate_report and runtime_delta and gate_report.get("why_no_pass_if_zero") is not None)


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


def _row_status(row: dict[str, Any]) -> str:
    return str(row.get("runtime_quality_status") or row.get("quality_classification") or row.get("release_candidate_row_status") or "")


def _model_id(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or "")


def _final_head_ci_record(acceptance_ledger: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    for payload in (acceptance_ledger, summary):
        for key in ("final_head_ci", "github_actions", "ci"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, dict):
                return value
    return {}


def _live_final_head_ci_record(source_root: Path) -> dict[str, Any]:
    head = _git_stdout(source_root, "rev-parse", "HEAD")
    repo = _git_remote_repo_full_name(source_root)
    if not head or not repo:
        return {}
    url = f"https://api.github.com/repos/{repo}/commits/{head}/check-runs?per_page=100"
    payload = _github_json(url)
    candidates = _final_head_ci_candidates(payload.get("check_runs"), head)
    return candidates[0] if candidates else {}


def _final_head_ci_candidates(check_runs: Any, head_sha: str) -> list[dict[str, Any]]:
    if not isinstance(check_runs, list):
        return []
    by_run: dict[str, dict[str, Any]] = {}
    for check in check_runs:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        conclusion = str(check.get("conclusion") or "")
        workflow_run_id = _workflow_run_id_from_details_url(check.get("details_url"))
        if not workflow_run_id:
            continue
        record = by_run.setdefault(
            workflow_run_id,
            {"workflow_run_id": workflow_run_id, "head_sha": head_sha, "conclusion": "success", "job_conclusions": {}},
        )
        record["job_conclusions"][name] = conclusion
        if conclusion != "success":
            record["conclusion"] = conclusion or "unknown"
    return [record for record in by_run.values() if any(job in record["job_conclusions"] for job in REQUIRED_FINAL_HEAD_CI_JOBS)]


def _workflow_run_id_from_details_url(value: Any) -> str:
    if not value:
        return ""
    match = re.search(r"/runs/(\d+)", str(value))
    return match.group(1) if match else ""


def _github_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codex-step4-3-audit"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def _git_remote_repo_full_name(source_root: Path) -> str:
    remote = _git_stdout(source_root, "remote", "get-url", "origin")
    if remote.startswith("git@github.com:"):
        remote = remote.split(":", 1)[1]
    parsed = urllib.parse.urlparse(remote)
    path = parsed.path if parsed.path else remote
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def _job_conclusions(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finding(gate: str, subject: str, message: str, evidence: dict[str, Any]) -> Finding:
    return Finding(gate=gate, severity="error", subject=subject, message=message, evidence=evidence)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_concrete_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(CONCRETE_SHA_RE.fullmatch(value))


def _is_concrete_run_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and bool(CONCRETE_RUN_ID_RE.fullmatch(str(value)))


def _git_stdout(source_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_returncode(source_root: Path, *args: str) -> int:
    try:
        return subprocess.run(["git", *args], cwd=source_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode
    except Exception:
        return 1


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
