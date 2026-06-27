#!/usr/bin/env python3
"""Audit Step 3.3 global solver-quality hardening artifacts."""

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


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_3_global_solver_quality")
DEFAULT_BASELINE_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_2_solver_backed_smoke")
DEFAULT_SOURCE_ROOT = Path(".")
EXPECTED_BASE_STEP3_2_FINAL_HEAD = "6ae0bfbc1153e3aba1291f38f0c82dfac6c2fa57"
EXPECTED_MATRIX_ROWS = 44
EXPECTED_CATEGORY_COUNTS = {
    "full_humanoid_profile": 32,
    "partial_humanoid_profile": 3,
    "negative_control": 9,
}
EXPECTED_STATUS_COUNTS = {
    "passed": 32,
    "partial_passed": 3,
    "negative_control_passed": 9,
}
REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "commands.txt",
    "model_matrix.json",
    "solver_smoke_matrix.json",
    "generic_smoke_matrix.json",
    "quality_summary.json",
    "acceptance_ledger.json",
    "deterministic_rerun.json",
    "quality_delta_vs_step3_2.json",
    "solver_config.json",
    "solver_diagnostics_matrix.json",
    "test_results/pytest.txt",
    "test_results/pytest_summary.json",
    "test_results/junit.xml",
)
PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES = (
    "pipeline_backed_matrix.json",
    "pipeline_controls_reference.json",
)
REQUIRED_FINAL_HEAD_CI_JOBS = (
    "step3-3-static-and-unit",
    "step3-3-artifact-audit",
)
CORE_DIFF_PATHS = ("soma_retargeter", "tests", "scripts", ".github")
SCAN_PATHS = (
    "soma_retargeter/runtime/v3",
    "soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py",
    "scripts/audit_retargeting_v3_step3_3_global_solver_quality.py",
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
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    delta = _read_json(artifact_dir / "quality_delta_vs_step3_2.json")
    solver_config = _read_json(artifact_dir / "solver_config.json")
    solver_diagnostics = _read_json(artifact_dir / "solver_diagnostics_matrix.json")
    baseline_summary = _read_json(baseline_artifact_dir / "quality_summary.json")

    rows = _matrix_rows(model_matrix)
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
    findings.extend(_audit_matrix_shape(rows, summary))
    findings.extend(_audit_partition(rows, summary))
    findings.extend(_audit_base_step3_2(summary, ledger, delta, solver_config))
    findings.extend(_audit_solver_counts(rows, solver_rows, diagnostic_rows, summary))
    findings.extend(_audit_label_honesty(rows, solver_rows, generic_rows, deterministic))
    findings.extend(_audit_negative_and_partial(rows))
    findings.extend(_audit_numeric_fields(rows, diagnostic_rows))
    findings.extend(_audit_delta(delta, baseline_summary, summary))
    findings.extend(_audit_solver_config(solver_config, diagnostic_rows))
    findings.extend(_audit_deterministic(deterministic, summary))
    findings.extend(_audit_pipeline_controls(artifact_dir))
    findings.extend(_audit_no_robot_specific_tuning(source_root))
    findings.extend(_audit_closed_artifacts_unchanged(source_root))
    if require_final_head_ci:
        findings.extend(_audit_final_head_ci(committed_final_head_ci, final_head_ci, source_root))

    gate_counts = dict(Counter(finding.gate for finding in findings))
    blocking_count = sum(1 for finding in findings if finding.severity == "error")
    return AuditResult(
        status="PASS" if blocking_count == 0 else "BLOCKED",
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
    parser.add_argument("--baseline-artifact-dir", default=str(DEFAULT_BASELINE_ARTIFACT_DIR))
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
    return 0 if result.status == "PASS" else 1


def _audit_required_artifacts(artifact_dir: Path) -> list[Finding]:
    findings = []
    if not artifact_dir.exists():
        return [_finding("missing_required_artifacts", _display_path(artifact_dir), "Step 3.3 artifact directory is missing", {})]
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(_finding("missing_required_artifacts", relative, "required Step 3.3 artifact is missing", {}))
    if not any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        findings.append(
            _finding(
                "missing_required_artifacts",
                "pipeline controls",
                "Step 3.3 artifacts must retain or reference pipeline control evidence",
                {"accepted_artifacts": list(PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES)},
            )
        )
    return findings


def _audit_baseline_artifacts(baseline_artifact_dir: Path, baseline_summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if not baseline_artifact_dir.exists():
        findings.append(_finding("baseline_step3_2", _display_path(baseline_artifact_dir), "Step 3.2 baseline artifact dir is missing", {}))
    if int(baseline_summary.get("runtime_quality_failed_count", -1) or -1) != 9:
        findings.append(
            _finding(
                "baseline_step3_2",
                "quality_summary.json",
                "baseline Step 3.2 failed count must be 9",
                {"actual": baseline_summary.get("runtime_quality_failed_count"), "expected": 9},
            )
        )
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
    for subject, payload in (("quality_summary.json", summary), ("acceptance_ledger.json", ledger), ("quality_delta_vs_step3_2.json", delta)):
        commit = payload.get("source_code_commit") or payload.get("current_source_commit")
        if commit and source_commit and str(commit) != source_commit:
            findings.append(_finding("clean_provenance", subject, "source commit fields must agree", {"expected": source_commit, "actual": commit}))
    return findings


def _audit_matrix_shape(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if len(rows) != EXPECTED_MATRIX_ROWS:
        findings.append(_finding("matrix_shape", "model_matrix.json", "model matrix must contain exactly 44 rows", {"actual": len(rows)}))
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
    status_counts = _status_bucket_counts(rows)
    if status_counts != EXPECTED_STATUS_COUNTS:
        findings.append(_finding("partition_32_3_9", "model_matrix.json", "source status partition must remain 32/3/9", {"actual": status_counts}))
    for field, expected in (("full_humanoid_total", 32), ("partial_total", 3), ("negative_total", 9)):
        if _as_int(summary.get(field)) != expected:
            findings.append(_finding("partition_32_3_9", "quality_summary.json", f"{field} must match 32/3/9 partition", {"actual": summary.get(field), "expected": expected}))
    return findings


def _audit_base_step3_2(summary: dict[str, Any], ledger: dict[str, Any], delta: dict[str, Any], solver_config: dict[str, Any]) -> list[Finding]:
    findings = []
    for subject, payload in (
        ("quality_summary.json", summary),
        ("acceptance_ledger.json", ledger),
        ("quality_delta_vs_step3_2.json", delta),
        ("solver_config.json", solver_config),
    ):
        value = payload.get("base_step3_2_final_head") or payload.get("baseline_final_head")
        if value != EXPECTED_BASE_STEP3_2_FINAL_HEAD:
            findings.append(_finding("base_step3_2_final_head", subject, "base Step 3.2 final HEAD mismatch", {"actual": value, "expected": EXPECTED_BASE_STEP3_2_FINAL_HEAD}))
    return findings


def _audit_solver_counts(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    full_rows = [row for row in rows if row.get("category") == "full_humanoid_profile"]
    checks = {
        "solver_backed_smoke_attempted_count": sum(1 for row in full_rows if row.get("solver_backed_smoke_attempted") is True),
        "solver_backed_completed_count": sum(1 for row in full_rows if row.get("solver_backed_smoke_completed") is True),
        "solver_backed_count": sum(1 for row in full_rows if row.get("solver_backed") is True),
        "residual_only_count": sum(1 for row in full_rows if row.get("residual_only") is True),
    }
    expected = {
        "solver_backed_smoke_attempted_count": 32,
        "solver_backed_completed_count": 32,
        "solver_backed_count": 32,
        "residual_only_count": 0,
    }
    for field, expected_value in expected.items():
        if checks[field] != expected_value or _as_int(summary.get(field)) != expected_value:
            findings.append(_finding("solver_backed_counts", field, "solver-backed invariant mismatch", {"row_count": checks[field], "summary": summary.get(field), "expected": expected_value}))
    if len(solver_rows) != 32:
        findings.append(_finding("solver_backed_counts", "solver_smoke_matrix.json", "solver smoke matrix must contain 32 full-humanoid rows", {"actual": len(solver_rows)}))
    if len(diagnostic_rows) != 32:
        findings.append(_finding("solver_backed_counts", "solver_diagnostics_matrix.json", "solver diagnostics matrix must contain 32 full-humanoid rows", {"actual": len(diagnostic_rows)}))
    if _as_int(summary.get("runtime_quality_failed_count")) is None or int(summary.get("runtime_quality_failed_count", 99)) >= 9:
        findings.append(_finding("quality_failure_reduction", "quality_summary.json", "runtime_quality_failed_count must be below the Step 3.2 baseline of 9", {"actual": summary.get("runtime_quality_failed_count"), "baseline": 9}))
    return findings


def _audit_label_honesty(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    deterministic: dict[str, Any],
) -> list[Finding]:
    findings = []
    solver_models = {_model_id(row) for row in solver_rows if row.get("solver_backed") is True and row.get("solver_backed_smoke_completed") is True}
    generic_models = {_model_id(row) for row in generic_rows if row.get("solver_backed") is True and row.get("solver_backed_smoke_completed") is True}
    deterministic_ok = _as_int(deterministic.get("matched_count")) == 44 and _as_int(deterministic.get("compared_count")) == 44
    for index, row in enumerate(rows):
        if row.get("category") != "full_humanoid_profile":
            continue
        status = _row_final_status(row)
        if status not in {"runtime_quality_passed", "runtime_quality_warned", "runtime_quality_failed", "runtime_evaluation_completed"}:
            findings.append(_finding("runtime_quality_label_honesty", _row_subject(index, row), "full humanoid status must be an exact runtime-quality enum", {"status": status}))
        if status != "runtime_quality_passed":
            continue
        model_id = _model_id(row)
        required = {
            "solver_backed_smoke_attempted": row.get("solver_backed_smoke_attempted") is True,
            "solver_backed_smoke_completed": row.get("solver_backed_smoke_completed") is True,
            "solver_backed": row.get("solver_backed") is True,
            "residual_only_false": row.get("residual_only") is False,
            "solver_matrix_evidence": model_id in solver_models,
            "generic_matrix_evidence": model_id in generic_models,
            "deterministic_matched": deterministic_ok,
            "output_nan_count_zero": _as_int(row.get("output_nan_count")) == 0,
            "output_inf_count_zero": _as_int(row.get("output_inf_count")) == 0,
            "joint_limit_valid": _as_int(row.get("joint_limit_violation_count")) == 0 or float(row.get("max_joint_limit_violation", 1.0) or 1.0) <= 1e-5,
            "residual_pass_gate": float(row.get("normalized_task_residual_p95", 1.0) or 1.0) <= 0.15,
        }
        if not all(required.values()):
            findings.append(_finding("runtime_quality_label_honesty", _row_subject(index, row), "runtime_quality_passed row does not satisfy all pass gates", required))
    return findings


def _audit_negative_and_partial(rows: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for index, row in enumerate(rows):
        if row.get("category") not in {"partial_humanoid_profile", "negative_control"}:
            continue
        subject = _row_subject(index, row)
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


def _audit_numeric_fields(rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    required = (
        "frame_count",
        "normalized_task_residual_mean",
        "normalized_task_residual_p95",
        "normalized_task_residual_max",
        "target_translation_error_mean",
        "target_translation_error_p95",
        "target_translation_error_max",
        "target_rotation_error_mean",
        "target_rotation_error_p95",
        "target_rotation_error_max",
        "joint_limit_violation_count",
        "max_joint_limit_violation",
        "output_nan_count",
        "output_inf_count",
        "runtime_seconds",
    )
    for collection_name, collection in (("model_matrix.json", rows), ("solver_diagnostics_matrix.json", diagnostic_rows)):
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
            for prefix in ("normalized_task_residual", "target_translation_error", "target_rotation_error"):
                findings.extend(_audit_ordered_triplet(subject, row, prefix))
            if row.get("solver_backed_smoke_completed") is True:
                for field in ("output_nan_count", "output_inf_count"):
                    if _as_int(row.get(field)) != 0:
                        findings.append(_finding("numeric_metrics", subject, "completed rows must have zero NaN/Inf counts", {"field": field, "value": row.get(field)}))
    return findings


def _audit_delta(delta: dict[str, Any], baseline_summary: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings = []
    if not delta:
        return [_finding("quality_delta_vs_step3_2", "quality_delta_vs_step3_2.json", "delta artifact is missing or unreadable", {})]
    baseline_counts = delta.get("baseline_counts") if isinstance(delta.get("baseline_counts"), dict) else {}
    current_counts = delta.get("current_counts") if isinstance(delta.get("current_counts"), dict) else {}
    count_deltas = delta.get("count_deltas") if isinstance(delta.get("count_deltas"), dict) else {}
    for key in ("in_scope_total", "full_humanoid_total", "partial_total", "negative_total", "solver_backed_count", "residual_only_count", "runtime_quality_failed_count"):
        if _as_int(baseline_counts.get(key)) != _as_int(baseline_summary.get(key)):
            findings.append(_finding("quality_delta_vs_step3_2", key, "baseline delta count does not match Step 3.2 summary", {"delta": baseline_counts.get(key), "baseline": baseline_summary.get(key)}))
        if _as_int(current_counts.get(key)) != _as_int(summary.get(key)):
            findings.append(_finding("quality_delta_vs_step3_2", key, "current delta count does not match Step 3.3 summary", {"delta": current_counts.get(key), "summary": summary.get(key)}))
        expected_delta = (_as_int(current_counts.get(key)) or 0) - (_as_int(baseline_counts.get(key)) or 0)
        if _as_int(count_deltas.get(key)) != expected_delta:
            findings.append(_finding("quality_delta_vs_step3_2", key, "count delta arithmetic is inconsistent", {"actual": count_deltas.get(key), "expected": expected_delta}))
    if _as_int(count_deltas.get("runtime_quality_failed_count")) is None or int(count_deltas.get("runtime_quality_failed_count", 0)) >= 0:
        findings.append(_finding("quality_delta_vs_step3_2", "runtime_quality_failed_count_delta", "failed-count delta must be negative", {"actual": count_deltas.get("runtime_quality_failed_count")}))
    if delta.get("verdict") != "PASS":
        findings.append(_finding("quality_delta_vs_step3_2", "verdict", "delta verdict must be PASS for Step 3.3 PASS", {"verdict": delta.get("verdict")}))
    return findings


def _audit_solver_config(solver_config: dict[str, Any], diagnostic_rows: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    config_hash = solver_config.get("solver_config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        findings.append(_finding("solver_config", "solver_config.json", "solver_config_hash is missing", {}))
    if solver_config.get("global_config") is not True or solver_config.get("robot_specific_tuning") is not False:
        findings.append(_finding("solver_config", "solver_config.json", "solver config must be global and non robot-specific", {"global_config": solver_config.get("global_config"), "robot_specific_tuning": solver_config.get("robot_specific_tuning")}))
    config = solver_config.get("config") if isinstance(solver_config.get("config"), dict) else {}
    if config.get("enable_global_quality_hardening") is not True or config.get("project_joint_limits") is not True:
        findings.append(_finding("solver_config", "solver_config.json", "global quality hardening and joint-limit projection must be enabled", {"config": config}))
    for row in diagnostic_rows:
        if row.get("solver_config_hash") != config_hash:
            findings.append(_finding("solver_config", "solver_diagnostics_matrix.json", "diagnostic row solver_config_hash mismatch", {"model_id": row.get("model_id")}))
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


def _audit_pipeline_controls(artifact_dir: Path) -> list[Finding]:
    if any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        return []
    return [_finding("pipeline_controls", "pipeline controls", "pipeline controls are missing", {})]


def _audit_no_robot_specific_tuning(source_root: Path) -> list[Finding]:
    findings = []
    for path in _scan_files(source_root):
        text = path.read_text(encoding="utf-8")
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
    )


def _audit_closed_artifacts_unchanged(source_root: Path) -> list[Finding]:
    findings = []
    changed = _git_stdout(
        source_root,
        "diff",
        "--name-only",
        f"{EXPECTED_BASE_STEP3_2_FINAL_HEAD}..HEAD",
        "--",
        "artifacts/retargeting_v3_step2",
        "artifacts/retargeting_v3_step2_numerical",
        "artifacts/retargeting_v3_step2_capability",
        "artifacts/retargeting_v3_step3_runtime_quality",
        "artifacts/retargeting_v3_step3_2_solver_backed_smoke",
    )
    if changed.strip():
        findings.append(_finding("closed_artifacts_unchanged", "git diff", "closed Step 2/3.1/3.2 artifacts must not change", {"changed_files": changed.splitlines()}))
    threshold_changes = [
        line
        for line in _git_stdout(source_root, "diff", "--name-only", f"{EXPECTED_BASE_STEP3_2_FINAL_HEAD}..HEAD").splitlines()
        if "threshold" in line.lower() and not line.startswith(str(DEFAULT_ARTIFACT_DIR))
    ]
    if threshold_changes:
        findings.append(_finding("closed_artifacts_unchanged", "threshold artifacts", "threshold artifacts outside Step 3.3 must not change", {"changed_files": threshold_changes}))
    return findings


def _audit_final_head_ci(committed_record: dict[str, Any], record: dict[str, Any], source_root: Path) -> list[Finding]:
    findings = []
    current_head = _git_stdout(source_root, "rev-parse", "HEAD")
    committed_head = committed_record.get("head_sha") or committed_record.get("final_head")
    if committed_record and current_head and _is_concrete_sha(committed_head) and str(committed_head).lower() != current_head.lower():
        findings.append(_finding("final_head_ci", "acceptance_ledger.json", "committed final-head CI evidence must match current HEAD", {"committed": committed_head, "current": current_head}))
    if not record:
        return [_finding("final_head_ci", "github_check_runs", "final HEAD CI evidence is missing", {})]
    if not _is_concrete_run_id(record.get("workflow_run_id") or record.get("run_id")):
        findings.append(_finding("final_head_ci", "workflow_run_id", "workflow run id must be concrete", {"value": record.get("workflow_run_id")}))
    head_sha = record.get("head_sha") or record.get("final_head")
    if not _is_concrete_sha(head_sha) or (current_head and str(head_sha).lower() != current_head.lower()):
        findings.append(_finding("final_head_ci", "head_sha", "CI evidence must be for current HEAD", {"head_sha": head_sha, "current": current_head}))
    if str(record.get("conclusion") or "").lower() != "success":
        findings.append(_finding("final_head_ci", "conclusion", "CI conclusion must be success", {"conclusion": record.get("conclusion")}))
    jobs = _job_conclusions(record.get("job_conclusions") or record.get("jobs"))
    for job in REQUIRED_FINAL_HEAD_CI_JOBS:
        if jobs.get(job) != "success":
            findings.append(_finding("final_head_ci", job, "required Step 3.3 CI job must be success", {"job_conclusions": jobs}))
    return findings


def _audit_ordered_triplet(subject: str, row: dict[str, Any], prefix: str) -> list[Finding]:
    mean = _as_float(row.get(f"{prefix}_mean"))
    p95 = _as_float(row.get(f"{prefix}_p95"))
    max_value = _as_float(row.get(f"{prefix}_max"))
    if mean is None or p95 is None or max_value is None or mean <= p95 <= max_value:
        return []
    return [_finding("numeric_metrics", subject, "mean/p95/max metrics must be ordered", {"prefix": prefix, "mean": mean, "p95": p95, "max": max_value})]


def _scan_files(source_root: Path) -> list[Path]:
    out = []
    for relative in SCAN_PATHS:
        path = source_root / relative
        if path.is_file() and path.suffix == ".py":
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(path.rglob("*.py")))
    return out


def _matrix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rows", "matrix", "models", "quality_matrix"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _status_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_status_bucket(row) for row in rows).items()))


def _status_bucket(row: dict[str, Any]) -> str:
    status = str(row.get("source_status") or row.get("status") or row.get("profile_status") or "").lower()
    if status in {"passed", "capability_limited_passed"}:
        return "passed"
    if status == "partial_passed":
        return "partial_passed"
    if status == "negative_control_passed":
        return "negative_control_passed"
    return status


def _row_final_status(row: dict[str, Any]) -> str:
    return str(row.get("runtime_quality_status") or row.get("final_step3_3_status") or row.get("quality_classification") or "").lower()


def _row_subject(index: int, row: dict[str, Any]) -> str:
    return f"model_matrix[{index}] {_model_id(row) or 'unknown_model'}"


def _model_id(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or row.get("profile_model_id") or row.get("robot_id") or "").strip()


def _final_head_ci_record(acceptance_ledger: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    for payload in (acceptance_ledger, summary):
        for key in ("final_head_ci", "ci"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _live_final_head_ci_record(source_root: Path) -> dict[str, Any]:
    head_sha = _git_stdout(source_root, "rev-parse", "HEAD")
    repo_full_name = _git_remote_repo_full_name(source_root)
    if not head_sha or not repo_full_name:
        return {}
    check_runs = _github_json(f"https://api.github.com/repos/{repo_full_name}/commits/{head_sha}/check-runs?per_page=100")
    candidates = _final_head_ci_candidates(check_runs.get("check_runs"), head_sha) if isinstance(check_runs, dict) else []
    for candidate in candidates:
        run_payload = _github_json(f"https://api.github.com/repos/{repo_full_name}/actions/runs/{candidate.get('workflow_run_id')}")
        if not isinstance(run_payload, dict):
            continue
        if str(run_payload.get("head_sha") or "").lower() != head_sha.lower():
            continue
        if str(run_payload.get("conclusion") or "").lower() != "success":
            continue
        candidate.update(
            {
                "workflow_name": run_payload.get("name"),
                "workflow_id": run_payload.get("workflow_id"),
                "run_number": run_payload.get("run_number"),
                "run_attempt": run_payload.get("run_attempt"),
                "head_branch": run_payload.get("head_branch"),
                "status": run_payload.get("status"),
                "conclusion": run_payload.get("conclusion"),
                "html_url": run_payload.get("html_url"),
                "repo": repo_full_name,
                "evidence_source": "github_check_runs_live",
            }
        )
        return candidate
    return {}


def _final_head_ci_candidates(check_runs: Any, head_sha: str) -> list[dict[str, Any]]:
    if not isinstance(check_runs, list):
        return []
    grouped: dict[str, dict[str, Any]] = {}
    relevant = set(REQUIRED_FINAL_HEAD_CI_JOBS) | {"step3-3-final-head-ci-evidence"}
    for check_run in check_runs:
        if not isinstance(check_run, dict):
            continue
        name = str(check_run.get("name") or "")
        if name not in relevant:
            continue
        run_id = _workflow_run_id_from_details_url(check_run.get("details_url"))
        if not run_id:
            continue
        group = grouped.setdefault(run_id, {"workflow_run_id": run_id, "head_sha": head_sha, "job_conclusions": {}, "completed_at": ""})
        group["job_conclusions"][name] = str(check_run.get("conclusion") or "").lower()
        completed_at = str(check_run.get("completed_at") or "")
        if completed_at > str(group.get("completed_at") or ""):
            group["completed_at"] = completed_at
    return sorted(
        [group for group in grouped.values() if all(group["job_conclusions"].get(job) == "success" for job in REQUIRED_FINAL_HEAD_CI_JOBS)],
        key=lambda item: str(item.get("completed_at") or ""),
        reverse=True,
    )


def _workflow_run_id_from_details_url(value: Any) -> str:
    match = re.search(r"/actions/runs/([1-9][0-9]{5,})(?:/|$)", str(value or ""))
    return match.group(1) if match else ""


def _github_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "soma-retargeter-step3-3-audit"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_remote_repo_full_name(source_root: Path) -> str:
    remote_url = _git_stdout(source_root, "remote", "get-url", "origin").removesuffix(".git")
    if remote_url.startswith("git@github.com:"):
        return remote_url.removeprefix("git@github.com:")
    parsed = urllib.parse.urlparse(remote_url)
    if parsed.netloc.lower() == "github.com":
        return parsed.path.strip("/")
    return ""


def _job_conclusions(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(child).lower() for key, child in value.items()}
    if isinstance(value, list):
        out = {}
        for item in value:
            if isinstance(item, dict) and (item.get("name") or item.get("job")):
                out[str(item.get("name") or item.get("job"))] = str(item.get("conclusion") or item.get("status") or "").lower()
        return out
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
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
    return isinstance(value, str) and CONCRETE_SHA_RE.fullmatch(value.strip()) is not None


def _is_concrete_run_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and CONCRETE_RUN_ID_RE.fullmatch(str(value).strip()) is not None


def _git_stdout(source_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=source_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_returncode(source_root: Path, *args: str) -> int:
    try:
        return subprocess.run(["git", *args], cwd=source_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=20).returncode
    except (OSError, subprocess.SubprocessError):
        return 1


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
