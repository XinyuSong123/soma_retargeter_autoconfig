#!/usr/bin/env python3
"""Audit Step 4.0 full-pipeline acceptance artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_full_pipeline_acceptance")
DEFAULT_BASELINE_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_4_global_residual_quality")
DEFAULT_SOURCE_ROOT = Path(".")
EXPECTED_BASE_STEP3_4_FINAL_HEAD = "77e7c02393a6678ccab40cdb847021d7d94392c9"
EXPECTED_MATRIX_ROWS = 44
EXPECTED_CATEGORY_COUNTS = {
    "full_humanoid_profile": 32,
    "partial_humanoid_profile": 3,
    "negative_control": 9,
}
REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "commands.txt",
    "model_matrix.json",
    "clip_matrix.json",
    "solver_smoke_matrix.json",
    "generic_smoke_matrix.json",
    "full_pipeline_matrix.json",
    "quality_summary.json",
    "acceptance_ledger.json",
    "deterministic_rerun.json",
    "quality_delta_vs_step3_4.json",
    "residual_taxonomy.json",
    "orientation_residual_taxonomy.json",
    "normalization_audit.json",
    "task_coverage_matrix.json",
    "anchor_reliability_matrix.json",
    "solver_config.json",
    "pipeline_config.json",
    "solver_diagnostics_matrix.json",
    "temporal_continuity_matrix.json",
    "support_contact_diagnostics.json",
    "collision_proxy_diagnostics.json",
    "trajectory_export_manifest.json",
    "red_team_report.json",
    "test_results/pytest.txt",
    "test_results/pytest_summary.json",
    "test_results/junit.xml",
)
PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES = (
    "pipeline_controls_reference.json",
    "pipeline_backed_matrix.json",
)
REQUIRED_FINAL_HEAD_CI_JOBS = (
    "step4-static-and-unit",
    "step4-artifact-audit",
    "step4-lfs-and-snapshot-smoke",
    "step4-export-and-trajectory-smoke",
    "step4-pipeline-controls-reference",
)
CORE_DIFF_PATHS = ("soma_retargeter", "tests", "scripts", ".github")
SCAN_PATHS = (
    "soma_retargeter/runtime/v3",
    "soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py",
    "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
    "scripts/audit_retargeting_v3_step4_full_pipeline_acceptance.py",
)
CONCRETE_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", flags=re.IGNORECASE)
CONCRETE_RUN_ID_RE = re.compile(r"\b[1-9][0-9]{5,}\b")
SUSPICIOUS_TABLE_RE = re.compile(
    r"(threshold|gate|limit|tolerance|weight|damping|lambda|trust|line_search|clip|residual|joint_limit)"
    r".*(by_model|by_robot|per_model|per_robot|model_id|robot_type|robot)|"
    r"(by_model|by_robot|per_model|per_robot|model_id|robot_type|robot)"
    r".*(threshold|gate|limit|tolerance|weight|damping|lambda|trust|line_search|clip|residual|joint_limit)",
    flags=re.IGNORECASE,
)
PROMOTION_SHORTCUT_RE = re.compile(
    r"white_?list|allow_?list|black_?list|block_?list|deny_?list|skip_?list|force_?pass|"
    r"force_?promote|accepted_models|allowed_models|excluded_models|known_good|known_bad",
    flags=re.IGNORECASE,
)
PASS_STATUSES = {"PASS_RC", "PASS_DIAGNOSTIC_ONLY"}
BLOCKED_STATUSES = {"BLOCKED_RESIDUAL_QUALITY", "BLOCKED_PIPELINE_REGRESSION", "BLOCKED_CI_OR_PROVENANCE"}
VALID_RELEASE_STATUSES = PASS_STATUSES | BLOCKED_STATUSES


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
    artifact_dir = artifact_dir.resolve()
    baseline_artifact_dir = baseline_artifact_dir.resolve()
    source_root = source_root.resolve()

    environment = _read_json(artifact_dir / "environment.json")
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    clip_matrix = _read_json(artifact_dir / "clip_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    delta = _read_json(artifact_dir / "quality_delta_vs_step3_4.json")
    taxonomy = _read_json(artifact_dir / "residual_taxonomy.json")
    orientation = _read_json(artifact_dir / "orientation_residual_taxonomy.json")
    normalization = _read_json(artifact_dir / "normalization_audit.json")
    task_coverage = _read_json(artifact_dir / "task_coverage_matrix.json")
    anchor_reliability = _read_json(artifact_dir / "anchor_reliability_matrix.json")
    solver_config = _read_json(artifact_dir / "solver_config.json")
    pipeline_config = _read_json(artifact_dir / "pipeline_config.json")
    solver_diagnostics = _read_json(artifact_dir / "solver_diagnostics_matrix.json")
    temporal = _read_json(artifact_dir / "temporal_continuity_matrix.json")
    support = _read_json(artifact_dir / "support_contact_diagnostics.json")
    collision = _read_json(artifact_dir / "collision_proxy_diagnostics.json")
    trajectory = _read_json(artifact_dir / "trajectory_export_manifest.json")
    baseline_summary = _read_json(baseline_artifact_dir / "quality_summary.json")

    rows = _matrix_rows(model_matrix)
    full_rows = _matrix_rows(full_pipeline)
    solver_rows = _matrix_rows(solver_smoke)
    generic_rows = _matrix_rows(generic_smoke)
    diagnostic_rows = _matrix_rows(solver_diagnostics)
    committed_final_head_ci = _final_head_ci_record(ledger, summary)
    live_final_head_ci = _live_final_head_ci_record(source_root) if require_final_head_ci else {}
    final_head_ci = live_final_head_ci or committed_final_head_ci

    findings: list[Finding] = []
    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_baseline_artifacts(baseline_artifact_dir, baseline_summary))
    findings.extend(_audit_clean_provenance(environment, summary, ledger, delta, source_root))
    findings.extend(_audit_matrix_shape(rows, full_rows, summary))
    findings.extend(_audit_partition(rows, summary))
    findings.extend(_audit_base_step3_4(summary, ledger, delta, solver_config, pipeline_config))
    findings.extend(_audit_solver_counts(rows, full_rows, solver_rows, diagnostic_rows, summary))
    findings.extend(_audit_label_honesty(rows, full_rows, solver_rows, generic_rows, deterministic))
    findings.extend(_audit_negative_and_partial(rows, full_rows))
    findings.extend(_audit_numeric_fields(rows, full_rows, diagnostic_rows))
    findings.extend(_audit_step4_evidence(taxonomy, orientation, task_coverage, anchor_reliability, rows))
    findings.extend(_audit_trajectory_exports(trajectory, full_rows, summary))
    findings.extend(_audit_temporal(temporal, summary))
    findings.extend(_audit_contact_collision(support, collision, summary))
    findings.extend(_audit_delta(delta, baseline_summary, summary))
    findings.extend(_audit_quality_breakthrough(delta, summary))
    findings.extend(_audit_normalization_integrity(normalization, delta, rows, full_rows, diagnostic_rows))
    findings.extend(_audit_configs(solver_config, pipeline_config, diagnostic_rows, full_rows))
    findings.extend(_audit_deterministic(deterministic, summary))
    findings.extend(_audit_release_candidate_status(summary, ledger, delta))
    findings.extend(_audit_pipeline_controls(artifact_dir))
    findings.extend(_audit_no_robot_specific_tuning(source_root))
    findings.extend(_audit_closed_artifacts_unchanged(source_root))
    if require_final_head_ci:
        findings.extend(_audit_final_head_ci(committed_final_head_ci, final_head_ci, source_root))

    gate_counts = dict(Counter(finding.gate for finding in findings))
    blocking_count = sum(1 for finding in findings if finding.severity == "error")
    release_status = str(summary.get("release_candidate_status") or ledger.get("release_candidate_status") or "BLOCKED_PIPELINE_REGRESSION")
    if blocking_count:
        status = "BLOCKED_CI_OR_PROVENANCE" if require_final_head_ci and gate_counts.get("final_head_ci") else "BLOCKED_PIPELINE_REGRESSION"
    elif release_status in PASS_STATUSES:
        status = release_status
    else:
        status = release_status if release_status in BLOCKED_STATUSES else "BLOCKED_RESIDUAL_QUALITY"
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
    parser.add_argument("--baseline-step3-4-artifact-dir", "--baseline-artifact-dir", dest="baseline_artifact_dir", default=str(DEFAULT_BASELINE_ARTIFACT_DIR))
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
        return [_finding("missing_required_artifacts", _display_path(artifact_dir), "Step 4 artifact directory is missing", {})]
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(_finding("missing_required_artifacts", relative, "required Step 4 artifact is missing", {}))
    if not any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        findings.append(_finding("missing_required_artifacts", "pipeline controls", "pipeline control evidence is missing", {}))
    return findings


def _audit_baseline_artifacts(baseline_artifact_dir: Path, baseline_summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if not baseline_artifact_dir.exists():
        findings.append(_finding("baseline_step3_4", _display_path(baseline_artifact_dir), "Step 3.4 baseline artifact dir is missing", {}))
    expected = {
        "in_scope_total": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "solver_backed_count": 32,
        "residual_only_count": 0,
        "runtime_quality_failed_count": 0,
        "high_residual_warning_count": 32,
    }
    for field, value in expected.items():
        if _as_int(baseline_summary.get(field)) != value:
            findings.append(_finding("baseline_step3_4", field, "baseline Step 3.4 summary mismatch", {"actual": baseline_summary.get(field), "expected": value}))
    return findings


def _audit_clean_provenance(
    environment: dict[str, Any],
    summary: dict[str, Any],
    ledger: dict[str, Any],
    delta: dict[str, Any],
    source_root: Path,
) -> list[Finding]:
    findings = []
    status_text = "\n".join(str(value) for value in (environment.get("git_status_short"), environment.get("source_git_status_short_before_run")) if value)
    for line in [line.strip() for line in status_text.splitlines() if line.strip()]:
        findings.append(_finding("clean_provenance", "environment.json", "artifact provenance records dirty source status", {"status_line": line}))
    for field in (
        "source_code_commit_remote_resolvable",
        "source_code_commit_is_artifact_commit_ancestor",
        "source_worktree_clean_before_run",
        "source_worktree_clean_after_run",
    ):
        if environment.get(field) is not True:
            findings.append(_finding("clean_provenance", "environment.json", "clean-provenance field must be true", {"field": field, "value": environment.get(field)}))
    if environment.get("core_diff_after_source_commit") != []:
        findings.append(_finding("clean_provenance", "environment.json", "core diff after source commit must be empty", {"value": environment.get("core_diff_after_source_commit")}))
    source_commit = str(environment.get("source_code_commit") or "")
    if not _is_concrete_sha(source_commit):
        findings.append(_finding("clean_provenance", "environment.json", "source_code_commit must be concrete SHA", {"source_code_commit": source_commit}))
    else:
        current_head = _git_stdout(source_root, "rev-parse", "HEAD")
        if current_head and _git_returncode(source_root, "merge-base", "--is-ancestor", source_commit, current_head) != 0:
            findings.append(_finding("clean_provenance", "git", "source commit must be an ancestor of current artifact HEAD", {"source_commit": source_commit, "current_head": current_head}))
        if current_head:
            core_diff = _git_stdout(source_root, "diff", "--name-only", f"{source_commit}..{current_head}", "--", *CORE_DIFF_PATHS)
            if core_diff.strip():
                findings.append(_finding("clean_provenance", "git", "core source changed after artifact source commit", {"changed_files": core_diff.splitlines()}))
    for subject, payload in (("quality_summary.json", summary), ("acceptance_ledger.json", ledger), ("quality_delta_vs_step3_4.json", delta)):
        commit = payload.get("source_code_commit") or payload.get("current_source_commit")
        if commit and source_commit and str(commit) != source_commit:
            findings.append(_finding("clean_provenance", subject, "source commit fields must agree", {"expected": source_commit, "actual": commit}))
    return findings


def _audit_matrix_shape(rows: list[dict[str, Any]], full_pipeline_rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if len(rows) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "model_matrix.json", "model matrix must contain exactly 44 rows", {"actual": len(rows)}))
    if len(full_pipeline_rows) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "full_pipeline_matrix.json", "full pipeline matrix must contain exactly 44 rows", {"actual": len(full_pipeline_rows)}))
    if len({_model_id(row) for row in rows if _model_id(row)}) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "model_matrix.json", "model matrix must contain 44 unique model ids", {}))
    for key in ("row_count", "in_scope_total", "matrix_row_count"):
        if key in summary and _as_int(summary.get(key)) != EXPECTED_MATRIX_ROWS:
            findings.append(_finding("matrix_shape", "quality_summary.json", f"{key} must equal 44", {"actual": summary.get(key)}))
    return findings


def _audit_partition(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    category_counts = dict(Counter(str(row.get("category")) for row in rows))
    if category_counts != EXPECTED_CATEGORY_COUNTS:
        findings.append(_finding("partition_32_3_9", "model_matrix.json", "category partition must be 32/3/9", {"actual": category_counts}))
    source_counts = _status_bucket_counts(rows)
    expected_status = {"passed": 32, "partial_passed": 3, "negative_control_passed": 9}
    if source_counts != expected_status:
        findings.append(_finding("partition_32_3_9", "model_matrix.json", "source status partition must remain 32/3/9", {"actual": source_counts}))
    for field, expected in (("full_humanoid_total", 32), ("partial_total", 3), ("negative_total", 9)):
        if _as_int(summary.get(field)) != expected:
            findings.append(_finding("partition_32_3_9", "quality_summary.json", f"{field} must match 32/3/9 partition", {"actual": summary.get(field), "expected": expected}))
    return findings


def _audit_base_step3_4(
    summary: dict[str, Any],
    ledger: dict[str, Any],
    delta: dict[str, Any],
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> list[Finding]:
    findings = []
    for subject, payload in (
        ("quality_summary.json", summary),
        ("acceptance_ledger.json", ledger),
        ("quality_delta_vs_step3_4.json", delta),
        ("solver_config.json", solver_config),
        ("pipeline_config.json", pipeline_config),
    ):
        value = payload.get("base_step3_4_final_head") or payload.get("baseline_final_head")
        if value != EXPECTED_BASE_STEP3_4_FINAL_HEAD:
            findings.append(_finding("base_step3_4_final_head", subject, "base Step 3.4 final HEAD mismatch", {"actual": value, "expected": EXPECTED_BASE_STEP3_4_FINAL_HEAD}))
    return findings


def _audit_solver_counts(
    rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    full_rows = [row for row in rows if row.get("category") == "full_humanoid_profile"]
    pipeline_full_rows = [row for row in full_pipeline_rows if row.get("category") == "full_humanoid_profile"]
    checks = {
        "solver_backed_smoke_attempted_count": sum(1 for row in full_rows if row.get("solver_backed_smoke_attempted") is True),
        "solver_backed_completed_count": sum(1 for row in full_rows if row.get("solver_backed_smoke_completed") is True),
        "solver_backed_count": sum(1 for row in full_rows if row.get("solver_backed") is True),
        "residual_only_count": sum(1 for row in full_rows if row.get("residual_only") is True),
        "runtime_quality_failed_count": sum(1 for row in full_rows if _row_final_status(row) == "runtime_quality_failed"),
    }
    expected_min = {
        "solver_backed_smoke_attempted_count": 32,
        "solver_backed_completed_count": 32,
        "solver_backed_count": 32,
    }
    for field, expected in expected_min.items():
        if checks[field] < expected or _as_int(summary.get(field)) is None or _as_int(summary.get(field)) < expected:
            findings.append(_finding("solver_backed_counts", field, "Step 4 must preserve solver-backed coverage", {"row_count": checks[field], "summary": summary.get(field), "expected_min": expected}))
    for field, expected in (("residual_only_count", 0), ("runtime_quality_failed_count", 0)):
        if checks[field] != expected or _as_int(summary.get(field)) != expected:
            findings.append(_finding("solver_backed_counts", field, "Step 4 core invariant mismatch", {"row_count": checks[field], "summary": summary.get(field), "expected": expected}))
    if len(pipeline_full_rows) != 32:
        findings.append(_finding("solver_backed_counts", "full_pipeline_matrix.json", "full pipeline matrix must contain 32 full-humanoid rows", {"actual": len(pipeline_full_rows)}))
    solver_full_model_ids = {_model_id(row) for row in solver_rows if row.get("category") == "full_humanoid_profile"}
    solver_completed_model_ids = {
        _model_id(row)
        for row in solver_rows
        if row.get("category") == "full_humanoid_profile"
        and row.get("solver_backed") is True
        and row.get("solver_backed_smoke_completed") is True
    }
    if len(solver_rows) < 32 or len(solver_full_model_ids) != 32:
        findings.append(
            _finding(
                "solver_backed_counts",
                "solver_smoke_matrix.json",
                "solver smoke matrix must contain at least one full-humanoid row for each model",
                {"row_count": len(solver_rows), "unique_full_humanoid_models": len(solver_full_model_ids)},
            )
        )
    if len(solver_completed_model_ids) != 32:
        findings.append(
            _finding(
                "solver_backed_counts",
                "solver_smoke_matrix.json",
                "solver smoke matrix must preserve completed solver-backed evidence for all 32 full-humanoid models",
                {"completed_full_humanoid_models": len(solver_completed_model_ids)},
            )
        )
    if len(diagnostic_rows) != 32:
        findings.append(_finding("solver_backed_counts", "solver_diagnostics_matrix.json", "solver diagnostics matrix must contain 32 full-humanoid rows", {"actual": len(diagnostic_rows)}))
    return findings


def _audit_label_honesty(
    rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    deterministic: dict[str, Any],
) -> list[Finding]:
    findings = []
    solver_models = {_model_id(row) for row in solver_rows if row.get("solver_backed") is True and row.get("solver_backed_smoke_completed") is True}
    generic_models = {_model_id(row) for row in generic_rows if row.get("solver_backed") is True and row.get("solver_backed_smoke_completed") is True}
    deterministic_ok = _as_int(deterministic.get("matched_count")) == 44 and _as_int(deterministic.get("compared_count")) == 44
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows)):
        for index, row in enumerate(collection):
            if row.get("category") != "full_humanoid_profile":
                continue
            status = _row_final_status(row)
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            if status not in {"runtime_quality_passed", "runtime_quality_warned", "runtime_quality_failed", "runtime_evaluation_completed"}:
                findings.append(_finding("runtime_quality_label_honesty", subject, "full humanoid status must be an exact runtime-quality enum", {"status": status}))
            high_residual = float(row.get("normalized_task_residual_p95", 1.0) or 1.0) > 0.15
            reasons = set(str(value) for value in row.get("runtime_quality_warning_reasons", row.get("warning_reasons", row.get("failure_or_warning_reasons", []))))
            if high_residual and status in {"runtime_quality_passed", "runtime_evaluation_completed"}:
                findings.append(_finding("runtime_quality_label_honesty", subject, "high residual row cannot be relabeled as passed/completed", {"status": status, "residual_p95": row.get("normalized_task_residual_p95")}))
            if high_residual and status == "runtime_quality_warned" and not ({"high_task_residual", "rotation_residual_dominates"} & reasons):
                findings.append(_finding("runtime_quality_label_honesty", subject, "high residual warning reason must remain explicit", {"reasons": sorted(reasons)}))
            if status != "runtime_quality_passed":
                continue
            model_id = _model_id(row)
            required = {
                "solver_backed_smoke_attempted": row.get("solver_backed_smoke_attempted") is True,
                "solver_backed_smoke_completed": row.get("solver_backed_smoke_completed") is True,
                "solver_backed": row.get("solver_backed") is True,
                "residual_only_false": row.get("residual_only") is False,
                "solver_matrix_evidence": model_id in solver_models,
                "generic_matrix_evidence": (model_id in generic_models) or not generic_rows,
                "deterministic_matched": deterministic_ok,
                "output_nan_count_zero": _as_int(row.get("output_nan_count")) == 0,
                "output_inf_count_zero": _as_int(row.get("output_inf_count")) == 0,
                "joint_limit_valid": _as_int(row.get("joint_limit_violation_count")) == 0 or float(row.get("max_joint_limit_violation", 1.0) or 1.0) <= 1e-5,
                "residual_pass_gate": float(row.get("normalized_task_residual_p95", 1.0) or 1.0) <= 0.15,
            }
            if not all(required.values()):
                findings.append(_finding("runtime_quality_label_honesty", subject, "runtime_quality_passed row does not satisfy all pass gates", required))
    return findings


def _audit_negative_and_partial(rows: list[dict[str, Any]], full_pipeline_rows: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows)):
        for index, row in enumerate(collection):
            if row.get("category") not in {"partial_humanoid_profile", "negative_control"}:
                continue
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            if _row_final_status(row) == "runtime_quality_passed":
                findings.append(_finding("negative_and_partial_not_promoted", subject, "partial/negative rows must not be runtime_quality_passed", {}))
            for flag in ("solver_backed", "solver_backed_smoke_attempted", "solver_backed_smoke_completed"):
                if row.get(flag) is True:
                    findings.append(_finding("negative_and_partial_not_promoted", subject, f"partial/negative row must not set {flag}", {"value": row.get(flag)}))
            if row.get("category") == "negative_control":
                for flag in ("promoted_to_runtime_quality", "quality_evaluated", "override_allowed", "humanoid_profile_generated"):
                    if row.get(flag) is True:
                        findings.append(_finding("negative_and_partial_not_promoted", subject, f"negative row has forbidden promotion flag {flag}", {"value": row.get(flag)}))
    return findings


def _audit_numeric_fields(
    rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings = []
    required = (
        "frame_count",
        "normalized_task_residual_mean",
        "normalized_task_residual_p95",
        "normalized_task_residual_max",
        "raw_task_residual_mean",
        "raw_task_residual_p95",
        "raw_task_residual_max",
        "joint_limit_violation_count",
        "max_joint_limit_violation",
        "output_nan_count",
        "output_inf_count",
        "task_anchor_count",
        "task_coverage_ratio",
        "anchor_reliability_score",
    )
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows), ("solver_diagnostics_matrix.json", diagnostic_rows)):
        for index, row in enumerate(collection):
            if row.get("category") != "full_humanoid_profile":
                continue
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            for field in required:
                if field not in row:
                    findings.append(_finding("numeric_metrics", subject, "required numeric field missing", {"field": field}))
                    continue
                number = _as_float(row.get(field))
                if number is None or number < 0:
                    findings.append(_finding("numeric_metrics", subject, "numeric field must be finite and non-negative", {"field": field, "value": row.get(field)}))
            for prefix in ("normalized_task_residual", "raw_task_residual", "task_residual", "target_translation_error", "target_rotation_error", "rotation_residual", "translation_residual"):
                findings.extend(_audit_ordered_triplet(subject, row, prefix))
            if row.get("solver_backed_smoke_completed") is True:
                for field in ("output_nan_count", "output_inf_count"):
                    if _as_int(row.get(field)) != 0:
                        findings.append(_finding("numeric_metrics", subject, "completed rows must have zero NaN/Inf counts", {"field": field, "value": row.get(field)}))
    return findings


def _audit_step4_evidence(
    taxonomy: dict[str, Any],
    orientation: dict[str, Any],
    task_coverage: dict[str, Any],
    anchor_reliability: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[Finding]:
    findings = []
    if len(_matrix_rows(taxonomy)) != 32:
        findings.append(_finding("step4_evidence", "residual_taxonomy.json", "residual taxonomy must contain 32 full-humanoid rows", {"actual": len(_matrix_rows(taxonomy))}))
    if len(_matrix_rows(orientation)) != 32:
        findings.append(_finding("step4_evidence", "orientation_residual_taxonomy.json", "orientation residual taxonomy must contain 32 full-humanoid rows", {"actual": len(_matrix_rows(orientation))}))
    if len(_matrix_rows(task_coverage)) != 32:
        findings.append(_finding("step4_evidence", "task_coverage_matrix.json", "task coverage matrix must contain 32 full-humanoid rows", {"actual": len(_matrix_rows(task_coverage))}))
    model_rows = anchor_reliability.get("model_rows") if isinstance(anchor_reliability.get("model_rows"), list) else []
    if len(model_rows) != 32:
        findings.append(_finding("step4_evidence", "anchor_reliability_matrix.json", "anchor reliability matrix must contain 32 model rows", {"actual": len(model_rows)}))
    if not taxonomy.get("aggregate_buckets"):
        findings.append(_finding("step4_evidence", "residual_taxonomy.json", "aggregate residual buckets are missing", {}))
    if not (orientation.get("aggregate_buckets") or orientation.get("summary")):
        findings.append(_finding("step4_evidence", "orientation_residual_taxonomy.json", "orientation summary is missing", {}))
    for row in rows:
        if row.get("category") != "full_humanoid_profile":
            continue
        if _as_float(row.get("task_coverage_ratio")) is None or _as_float(row.get("anchor_reliability_score")) is None:
            findings.append(_finding("step4_evidence", _model_id(row), "model row lacks task/anchor evidence", {}))
    return findings


def _audit_trajectory_exports(trajectory: dict[str, Any], full_pipeline_rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    rows = _matrix_rows(trajectory)
    full_models = {_model_id(row) for row in full_pipeline_rows if row.get("category") == "full_humanoid_profile"}
    if not rows:
        findings.append(_finding("trajectory_exports", "trajectory_export_manifest.json", "trajectory exports are missing", {}))
    if len(rows) < len(full_models):
        findings.append(_finding("trajectory_exports", "trajectory_export_manifest.json", "expected at least one export per full humanoid", {"exports": len(rows), "full_models": len(full_models)}))
    for index, row in enumerate(rows):
        subject = f"trajectory_export_manifest.json[{index}] {_model_id(row)}"
        finite_value = row.get("finite_qpos", row.get("qpos_finite", row.get("finite")))
        if finite_value is not True:
            findings.append(_finding("trajectory_exports", subject, "export must record finite qpos", {"finite": finite_value}))
        for field in ("nan_count", "inf_count"):
            if _as_int(row.get(field)) != 0:
                findings.append(_finding("trajectory_exports", subject, "export must have zero NaN/Inf counts", {"field": field, "value": row.get(field)}))
        if not row.get("export_hash"):
            findings.append(_finding("trajectory_exports", subject, "export_hash is required", {}))
    if _as_int(summary.get("trajectory_exports_count")) is not None and _as_int(summary.get("trajectory_exports_count")) != len(rows):
        findings.append(_finding("trajectory_exports", "quality_summary.json", "trajectory_exports_count must match manifest", {"summary": summary.get("trajectory_exports_count"), "manifest": len(rows)}))
    return findings


def _audit_temporal(temporal: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    rows = _matrix_rows(temporal)
    if not rows:
        return [_finding("temporal_continuity", "temporal_continuity_matrix.json", "temporal continuity matrix is missing", {})]
    finite_count = 0
    for index, row in enumerate(rows):
        subject = f"temporal_continuity_matrix.json[{index}] {_model_id(row)}"
        finite = row.get("finite")
        if finite is None:
            finite = row.get("finite_velocity") is True and row.get("finite_acceleration") is True
        if finite is not True:
            findings.append(_finding("temporal_continuity", subject, "temporal row must record finite velocities/accelerations", {"finite": finite}))
        else:
            finite_count += 1
        for field in ("nan_count", "inf_count"):
            if field in row and _as_int(row.get(field)) != 0:
                findings.append(_finding("temporal_continuity", subject, "temporal diagnostics must have zero NaN/Inf counts", {"field": field, "value": row.get(field)}))
        for field in ("temporal_jump_count", "joint_velocity_p95", "joint_acceleration_p95", "velocity_p95", "acceleration_p95"):
            if field in row:
                number = _as_float(row.get(field))
                if number is None or number < 0:
                    findings.append(_finding("temporal_continuity", subject, "temporal metric must be finite and non-negative", {"field": field, "value": row.get(field)}))
    expected = _as_int(summary.get("temporal_continuity_finite_count"))
    if expected is not None and expected != finite_count:
        findings.append(_finding("temporal_continuity", "quality_summary.json", "temporal_continuity_finite_count must match finite rows", {"summary": expected, "rows": finite_count}))
    return findings


def _audit_contact_collision(
    support: dict[str, Any],
    collision: dict[str, Any],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    support_rows = _matrix_rows(support)
    collision_rows = _matrix_rows(collision)
    if not support_rows:
        findings.append(_finding("contact_collision_diagnostics", "support_contact_diagnostics.json", "support/contact diagnostics are missing", {}))
    if not collision_rows:
        findings.append(_finding("contact_collision_diagnostics", "collision_proxy_diagnostics.json", "collision proxy diagnostics are missing", {}))
    for name, rows in (("support_contact_diagnostics.json", support_rows), ("collision_proxy_diagnostics.json", collision_rows)):
        for index, row in enumerate(rows):
            if row.get("finite") is not True:
                findings.append(_finding("contact_collision_diagnostics", f"{name}[{index}] {_model_id(row)}", "diagnostic row must be finite", {"finite": row.get("finite")}))
    if _as_int(summary.get("support_contact_diagnostic_count")) is not None and _as_int(summary.get("support_contact_diagnostic_count")) != len(support_rows):
        findings.append(_finding("contact_collision_diagnostics", "quality_summary.json", "support_contact_diagnostic_count mismatch", {"summary": summary.get("support_contact_diagnostic_count"), "rows": len(support_rows)}))
    if _as_int(summary.get("collision_proxy_diagnostic_count")) is not None and _as_int(summary.get("collision_proxy_diagnostic_count")) != len(collision_rows):
        findings.append(_finding("contact_collision_diagnostics", "quality_summary.json", "collision_proxy_diagnostic_count mismatch", {"summary": summary.get("collision_proxy_diagnostic_count"), "rows": len(collision_rows)}))
    return findings


def _audit_delta(delta: dict[str, Any], baseline_summary: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if not delta:
        return [_finding("quality_delta_vs_step3_4", "quality_delta_vs_step3_4.json", "delta artifact is missing or unreadable", {})]
    if delta.get("baseline_final_head") != EXPECTED_BASE_STEP3_4_FINAL_HEAD:
        findings.append(_finding("quality_delta_vs_step3_4", "baseline_final_head", "delta baseline final head mismatch", {"actual": delta.get("baseline_final_head")}))
    baseline_counts = delta.get("baseline_counts") if isinstance(delta.get("baseline_counts"), dict) else {}
    current_counts = delta.get("current_counts") if isinstance(delta.get("current_counts"), dict) else {}
    count_deltas = delta.get("count_deltas") if isinstance(delta.get("count_deltas"), dict) else {}
    for key in (
        "in_scope_total",
        "full_humanoid_total",
        "partial_total",
        "negative_total",
        "solver_backed_count",
        "residual_only_count",
        "runtime_quality_failed_count",
        "runtime_quality_passed_count",
        "high_residual_warning_count",
    ):
        if key in baseline_counts and _as_int(baseline_counts.get(key)) != _as_int(baseline_summary.get(key)):
            findings.append(_finding("quality_delta_vs_step3_4", key, "baseline delta count does not match Step 3.4 summary", {"delta": baseline_counts.get(key), "baseline": baseline_summary.get(key)}))
        if key in current_counts and _as_int(current_counts.get(key)) != _as_int(summary.get(key)):
            findings.append(_finding("quality_delta_vs_step3_4", key, "current delta count does not match Step 4 summary", {"delta": current_counts.get(key), "summary": summary.get(key)}))
        if key in current_counts and key in baseline_counts:
            expected_delta = (_as_int(current_counts.get(key)) or 0) - (_as_int(baseline_counts.get(key)) or 0)
            if _as_int(count_deltas.get(key)) != expected_delta:
                findings.append(_finding("quality_delta_vs_step3_4", key, "count delta arithmetic is inconsistent", {"actual": count_deltas.get(key), "expected": expected_delta}))
    if delta.get("regressions"):
        findings.append(_finding("quality_delta_vs_step3_4", "regressions", "Step 4 delta contains regressions", {"regressions": delta.get("regressions")}))
    return findings


def _audit_quality_breakthrough(delta: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    release_status = str(summary.get("release_candidate_status") or "")
    if release_status in BLOCKED_STATUSES:
        return []
    if _quality_breakthrough(delta, summary):
        return []
    return [
        _finding(
            "quality_breakthrough",
            "quality_delta_vs_step3_4.json",
            "Step 4 requires a runtime pass, lower high-residual count, or audited residual/orientation breakthrough",
            {
                "runtime_quality_passed_count": summary.get("runtime_quality_passed_count"),
                "high_residual_warning_count": summary.get("high_residual_warning_count"),
                "improvements": delta.get("improvements"),
            },
        )
    ]


def _audit_normalization_integrity(
    normalization: dict[str, Any],
    delta: dict[str, Any],
    rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings = []
    norm_delta = delta.get("normalization_deltas") if isinstance(delta.get("normalization_deltas"), dict) else {}
    if normalization.get("normalization_hides_raw_regression") is True or norm_delta.get("normalization_hides_raw_regression") is True:
        findings.append(_finding("normalization_integrity", "normalization_audit.json", "normalization improvement hides raw residual regression", {"normalization": normalization, "delta": norm_delta}))
    if normalization.get("denominator_inflation_detected") is True:
        findings.append(_finding("normalization_integrity", "normalization_audit.json", "denominator inflation detected", {"suspicious_rows": normalization.get("suspicious_rows")}))
    if _as_int(normalization.get("robot_specific_denominator_count")) not in {None, 0}:
        findings.append(_finding("normalization_integrity", "normalization_audit.json", "robot-specific normalization denominator detected", {"count": normalization.get("robot_specific_denominator_count")}))
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows), ("solver_diagnostics_matrix.json", diagnostic_rows), ("normalization_audit.json", _matrix_rows(normalization))):
        for index, row in enumerate(collection):
            if row.get("category") not in {None, "full_humanoid_profile"} and collection_name == "normalization_audit.json":
                continue
            if row.get("category") not in {"full_humanoid_profile", None} and collection_name != "normalization_audit.json":
                continue
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            if row.get("residual_denominator_robot_specific") is True:
                findings.append(_finding("normalization_integrity", subject, "residual denominator must not be robot-specific", {}))
            denominator = _as_float(row.get("residual_denominator"))
            raw_max = _as_float(row.get("raw_task_residual_max", row.get("task_residual_max", row.get("raw_task_residual_p95"))))
            normalized_max = _as_float(row.get("normalized_task_residual_max", row.get("normalized_task_residual_p95")))
            if denominator is not None and raw_max is not None and normalized_max is not None and denominator > 0:
                if normalized_max > 0 and raw_max / denominator > normalized_max + 1.0:
                    findings.append(_finding("normalization_integrity", subject, "normalization denominator appears inflated relative to raw residual", {"raw": raw_max, "denominator": denominator, "normalized": normalized_max}))
    return findings


def _audit_configs(
    solver_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    diagnostic_rows: list[dict[str, Any]],
    full_pipeline_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings = []
    for name, payload in (("solver_config.json", solver_config), ("pipeline_config.json", pipeline_config)):
        if payload.get("global_config") is False or payload.get("robot_specific_tuning") is True:
            findings.append(_finding("global_config", name, "Step 4 config must be global and non robot-specific", {"global_config": payload.get("global_config"), "robot_specific_tuning": payload.get("robot_specific_tuning")}))
    solver_hash = solver_config.get("solver_config_hash")
    pipeline_hash = pipeline_config.get("pipeline_config_hash")
    if not isinstance(solver_hash, str) or not solver_hash:
        findings.append(_finding("global_config", "solver_config.json", "solver_config_hash is missing", {}))
    if not isinstance(pipeline_hash, str) or not pipeline_hash:
        findings.append(_finding("global_config", "pipeline_config.json", "pipeline_config_hash is missing", {}))
    for row in diagnostic_rows:
        if solver_hash and row.get("solver_config_hash") not in {None, solver_hash}:
            findings.append(_finding("global_config", "solver_diagnostics_matrix.json", "diagnostic row solver_config_hash mismatch", {"model_id": row.get("model_id")}))
    for row in full_pipeline_rows:
        if pipeline_hash and row.get("pipeline_config_hash") not in {None, pipeline_hash}:
            findings.append(_finding("global_config", "full_pipeline_matrix.json", "pipeline_config_hash mismatch", {"model_id": row.get("model_id")}))
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


def _audit_release_candidate_status(summary: dict[str, Any], ledger: dict[str, Any], delta: dict[str, Any]) -> list[Finding]:
    findings = []
    summary_status = str(summary.get("release_candidate_status") or "")
    ledger_status = str(ledger.get("release_candidate_status") or ledger.get("verdict") or "")
    if summary_status not in VALID_RELEASE_STATUSES:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "release_candidate_status has invalid enum", {"status": summary_status}))
    if ledger_status != summary_status:
        findings.append(_finding("release_candidate_status", "acceptance_ledger.json", "ledger release status must match quality summary", {"ledger": ledger_status, "summary": summary_status}))
    breakthrough = _quality_breakthrough(delta, summary)
    regressions = bool(delta.get("regressions")) or _as_int(summary.get("runtime_quality_failed_count")) not in {0, None}
    if summary_status == "PASS_RC":
        required = {
            "runtime_quality_passed_count_positive": _as_int(summary.get("runtime_quality_passed_count")) is not None and _as_int(summary.get("runtime_quality_passed_count")) > 0,
            "runtime_quality_failed_count_zero": _as_int(summary.get("runtime_quality_failed_count")) == 0,
            "breakthrough": breakthrough,
            "no_regressions": not regressions,
        }
        if not all(required.values()):
            findings.append(_finding("release_candidate_status", "quality_summary.json", "PASS_RC status requirements are not met", required))
    if summary_status == "PASS_DIAGNOSTIC_ONLY" and (_as_int(summary.get("runtime_quality_passed_count")) or 0) > 0:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "PASS_DIAGNOSTIC_ONLY must not hide runtime pass rows", {"runtime_quality_passed_count": summary.get("runtime_quality_passed_count")}))
    if summary_status in PASS_STATUSES and not breakthrough:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "passing Step 4 status requires a primary quality breakthrough", {"status": summary_status}))
    if summary_status.startswith("BLOCKED") and breakthrough and not regressions:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked status contradicts available breakthrough evidence", {"status": summary_status}))
    return findings


def _audit_pipeline_controls(artifact_dir: Path) -> list[Finding]:
    if any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        return []
    return [_finding("pipeline_controls", "pipeline controls", "pipeline controls are missing", {})]


def _audit_no_robot_specific_tuning(source_root: Path) -> list[Finding]:
    findings = []
    for path in _scan_files(source_root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if _benign_scan_line(line):
                continue
            if SUSPICIOUS_TABLE_RE.search(line):
                findings.append(_finding("no_robot_specific_tuning", f"{_display_path(path)}:{index}", "suspicious per-model/per-robot tuning table", {"line": line.strip()}))
            if PROMOTION_SHORTCUT_RE.search(line) and re.search(r"model_id|robot_type|runtime_quality|pass|promote", line, re.IGNORECASE):
                findings.append(_finding("no_robot_specific_tuning", f"{_display_path(path)}:{index}", "suspicious allow/force promotion shortcut", {"line": line.strip()}))
    return findings


def _benign_scan_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith(("from ", "import "))
        or "robotics.v3" in stripped
        or "per-robot tuning" in stripped
        or "per_clip" in stripped
        or "SUSPICIOUS_TABLE_RE" in stripped
        or "PROMOTION_SHORTCUT_RE" in stripped
        or "accepted_models|allowed_models" in stripped
        or "quality_delta_key" in stripped
        or "residual_denominator_robot_specific" in stripped
        or "robot-specific" in stripped
        or "non robot-specific" in stripped
        or "model_id\" in row" in stripped
        or "model_id') in row" in stripped
        or "smoke_by_key" in stripped
        or "diagnostics_by_model" in stripped
        or "clips_by_model" in stripped
        or "exports_by_model" in stripped
        or "temporal_by_model" in stripped
        or "support_by_model" in stripped
        or "collision_by_model" in stripped
        or 'artifact_dir / "exports" / "per_model"' in stripped
    )


def _audit_closed_artifacts_unchanged(source_root: Path) -> list[Finding]:
    if _git_returncode(source_root, "rev-parse", "--is-inside-work-tree") != 0:
        return []
    changed = _git_stdout(
        source_root,
        "diff",
        "--name-only",
        f"{EXPECTED_BASE_STEP3_4_FINAL_HEAD}..HEAD",
        "--",
        "artifacts/retargeting_v3_step2",
        "artifacts/retargeting_v3_step2_numerical",
        "artifacts/retargeting_v3_step2_capability",
        "artifacts/retargeting_v3_step3_runtime_quality",
        "artifacts/retargeting_v3_step3_2_solver_backed_smoke",
        "artifacts/retargeting_v3_step3_3_global_solver_quality",
        "artifacts/retargeting_v3_step3_4_global_residual_quality",
    )
    if changed.strip():
        return [_finding("closed_artifacts_unchanged", "git", "closed Step 2/3 artifact trees changed after Step 3.4", {"changed_files": changed.splitlines()})]
    return []


def _audit_final_head_ci(committed_record: dict[str, Any], record: dict[str, Any], source_root: Path) -> list[Finding]:
    findings = []
    current_head = _git_stdout(source_root, "rev-parse", "HEAD")
    if not record:
        return [_finding("final_head_ci", "github", "final HEAD CI evidence is missing", {})]
    if not _is_concrete_run_id(record.get("workflow_run_id")):
        findings.append(_finding("final_head_ci", "workflow_run_id", "final-head CI workflow_run_id must be concrete", {"record": record}))
    if record.get("head_sha") != current_head:
        findings.append(_finding("final_head_ci", "head_sha", "final-head CI head_sha must equal current HEAD", {"record": record.get("head_sha"), "current_head": current_head}))
    if record.get("conclusion") != "success":
        findings.append(_finding("final_head_ci", "conclusion", "final-head CI conclusion must be success", {"conclusion": record.get("conclusion")}))
    job_conclusions = _job_conclusions(record.get("job_conclusions"))
    for job in REQUIRED_FINAL_HEAD_CI_JOBS:
        if job_conclusions.get(job) != "success":
            findings.append(_finding("final_head_ci", job, "required Step 4 CI job is not successful", {"job_conclusions": job_conclusions, "committed_record": committed_record}))
    return findings


def _quality_breakthrough(delta: dict[str, Any], summary: dict[str, Any]) -> bool:
    if _as_int(summary.get("runtime_quality_passed_count")) and _as_int(summary.get("runtime_quality_passed_count")) > 0:
        return True
    if _as_int(summary.get("high_residual_warning_count")) is not None and _as_int(summary.get("high_residual_warning_count")) < 32:
        return True
    if delta.get("primary_quality_breakthrough") is True:
        return True
    orientation = delta.get("orientation_residual_deltas") if isinstance(delta.get("orientation_residual_deltas"), dict) else {}
    if orientation.get("accepted_breakthrough") is True:
        return True
    return False


def _audit_ordered_triplet(subject: str, row: dict[str, Any], prefix: str) -> list[Finding]:
    findings = []
    fields = [f"{prefix}_{name}" for name in ("mean", "p95", "max")]
    if not all(field in row for field in fields):
        return findings
    values = [_as_float(row.get(field)) for field in fields]
    if any(value is None for value in values):
        return findings
    mean, p95, max_value = values
    if not (mean <= p95 + 1e-9 and p95 <= max_value + 1e-9):
        findings.append(_finding("numeric_metrics", subject, f"{prefix} mean/p95/max must be ordered", {"mean": mean, "p95": p95, "max": max_value}))
    return findings


def _scan_files(source_root: Path) -> list[Path]:
    out = []
    for relative in SCAN_PATHS:
        path = source_root / relative
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(child for child in path.rglob("*.py") if child.is_file()))
    return out


def _matrix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    for key in ("rows", "matrix", "exports", "model_rows"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _status_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(_status_bucket(row) for row in rows))


def _status_bucket(row: dict[str, Any]) -> str:
    category = row.get("category")
    source_status = str(row.get("source_status") or "")
    if category == "full_humanoid_profile":
        return "passed" if source_status in {"passed", "full_humanoid_profile"} else source_status
    if category == "partial_humanoid_profile":
        return "partial_passed" if source_status in {"partial_passed", "partial_humanoid_profile"} else source_status
    if category == "negative_control":
        return "negative_control_passed" if source_status in {"negative_control_passed", "negative_control"} else source_status
    return source_status or str(category)


def _row_final_status(row: dict[str, Any]) -> str:
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
    try:
        payload = _github_json(url)
    except Exception:
        return {}
    candidates = _final_head_ci_candidates(payload.get("check_runs"), head)
    if not candidates:
        return {}
    return candidates[0]


def _final_head_ci_candidates(check_runs: Any, head_sha: str) -> list[dict[str, Any]]:
    if not isinstance(check_runs, list):
        return []
    by_run: dict[str, dict[str, Any]] = {}
    for check in check_runs:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        conclusion = str(check.get("conclusion") or "")
        details = check.get("details_url")
        workflow_run_id = _workflow_run_id_from_details_url(details)
        if not workflow_run_id:
            continue
        record = by_run.setdefault(
            workflow_run_id,
            {
                "workflow_run_id": workflow_run_id,
                "head_sha": head_sha,
                "conclusion": "success",
                "job_conclusions": {},
            },
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
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codex-step4-audit"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def _git_remote_repo_full_name(source_root: Path) -> str:
    remote = _git_stdout(source_root, "remote", "get-url", "origin")
    if not remote:
        return ""
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


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
