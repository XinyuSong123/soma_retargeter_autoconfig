#!/usr/bin/env python3
"""Audit Step 3.2 solver-backed generic runtime smoke artifacts.

The audit is read-only and Step 3.2-specific. It validates that the artifact
tree extends the closed Step 3.1.1 baseline with real solver-backed generic
smoke evidence while preserving the 44-row full-fleet accounting.
"""

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
from typing import Any, Iterable


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_2_solver_backed_smoke")
DEFAULT_SOURCE_ROOT = Path(".")
EXPECTED_BASE_STEP3_1_1_FINAL_HEAD = "26817de67bdda0cb315a1237b53c30e4d8199c78"
EXPECTED_MATRIX_ROWS = 44
EXPECTED_STATUS_COUNTS = {
    "passed": 32,
    "partial_passed": 3,
    "negative_control_passed": 9,
}
EXPECTED_CATEGORY_COUNTS = {
    "full_humanoid_profile": 32,
    "partial_humanoid_profile": 3,
    "negative_control": 9,
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
    "test_results/pytest.txt",
    "test_results/pytest_summary.json",
)
PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES = (
    "pipeline_backed_matrix.json",
    "pipeline_controls_reference.json",
)
ACCEPTANCE_GATES = (
    "missing_required_artifacts",
    "clean_provenance",
    "matrix_shape",
    "status_counts_32_3_9",
    "base_step3_1_1_final_head",
    "solver_backed_smoke_counts",
    "solver_evidence_present",
    "runtime_quality_label_honesty",
    "negative_and_partial_not_promoted",
    "quality_numeric_fields",
    "deterministic_rerun_matched",
    "final_head_ci",
)
PROVENANCE_REQUIRED_TRUE_FIELDS = (
    "source_code_commit_remote_resolvable",
    "source_code_commit_is_artifact_commit_ancestor",
    "source_worktree_clean_before_run",
    "source_worktree_clean_after_run",
)
REQUIRED_FINAL_HEAD_CI_JOBS = (
    "step3-2-static-and-unit",
    "step3-2-artifact-audit",
)
REQUIRED_FULL_ROW_NUMERIC_FIELDS = (
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
COUNT_NUMERIC_FIELDS = {
    "frame_count",
    "joint_limit_violation_count",
    "output_nan_count",
    "output_inf_count",
}
COMPLETED_ZERO_REQUIRED_FIELDS = {
    "output_nan_count",
    "output_inf_count",
}
RUNTIME_QUALITY_PASS_STATUSES = {
    "runtime_quality_passed",
    "quality_passed",
    "quality_accepted",
    "passed",
    "pass",
}
NEGATIVE_OR_PARTIAL_ALLOWED_FINAL_STATUSES = {
    "partial_runtime_passed",
    "partial_passed",
    "negative_control_runtime_passed",
    "negative_control_not_promoted",
    "negative_control_rejected",
    "runtime_quality_failed",
    "runtime_quality_warned",
    "runtime_evaluation_completed",
    "not_applicable",
}
CONCRETE_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b", flags=re.IGNORECASE)
CONCRETE_RUN_ID_RE = re.compile(r"\b[1-9][0-9]{5,}\b")


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
    source_root: str
    matrix_row_count: int
    status_counts: dict[str, int]
    finding_count: int
    blocking_count: int
    gate_counts: dict[str, int]
    findings: list[Finding]
    final_head_ci: dict[str, Any]

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    require_final_head_ci: bool = False,
) -> AuditResult:
    artifact_dir = artifact_dir.resolve()
    source_root = source_root.resolve()

    environment = _read_json(artifact_dir / "environment.json")
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")

    rows = _matrix_rows(model_matrix)
    solver_rows = _matrix_rows(solver_smoke)
    generic_rows = _matrix_rows(generic_smoke)
    committed_final_head_ci = _final_head_ci_record(ledger, summary)
    live_final_head_ci = _live_final_head_ci_record(source_root) if require_final_head_ci else {}
    final_head_ci = live_final_head_ci or committed_final_head_ci

    findings: list[Finding] = []
    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_clean_provenance(environment))
    findings.extend(_audit_matrix_shape(rows, summary))
    findings.extend(_audit_status_counts(rows, summary))
    findings.extend(_audit_base_final_head(summary, ledger))
    findings.extend(_audit_solver_counts(rows, solver_rows, generic_rows, summary))
    findings.extend(_audit_solver_evidence_present(rows, solver_rows, generic_rows))
    findings.extend(_audit_runtime_quality_label_honesty(rows, solver_rows, generic_rows))
    findings.extend(_audit_negative_and_partial_not_promoted(rows))
    findings.extend(_audit_quality_numeric_fields(rows))
    findings.extend(_audit_deterministic_rerun(deterministic, summary))
    if require_final_head_ci:
        findings.extend(
            _audit_final_head_ci(
                committed_final_head_ci,
                final_head_ci,
                source_root,
                require_final_head_ci=require_final_head_ci,
            )
        )

    gate_counts = {gate: 0 for gate in ACCEPTANCE_GATES}
    for finding in findings:
        gate_counts[finding.gate] = gate_counts.get(finding.gate, 0) + 1
    blocking_count = len([finding for finding in findings if finding.severity == "error"])
    return AuditResult(
        status="PASS" if blocking_count == 0 else "BLOCKED",
        artifact_dir=_display_path(artifact_dir),
        source_root=_display_path(source_root),
        matrix_row_count=len(rows),
        status_counts=_status_bucket_counts(rows),
        finding_count=len(findings),
        blocking_count=blocking_count,
        gate_counts=gate_counts,
        findings=findings,
        final_head_ci=final_head_ci,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-json")
    parser.add_argument("--write-report", dest="output_json")
    parser.add_argument(
        "--require-final-head-ci",
        action="store_true",
        help="Require committed final HEAD CI evidence. The default audit intentionally skips this gate.",
    )
    args = parser.parse_args(argv)

    result = run_audit(
        artifact_dir=Path(args.artifact_dir),
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
    findings: list[Finding] = []
    if not artifact_dir.exists():
        return [
            _finding(
                "missing_required_artifacts",
                _display_path(artifact_dir),
                "Step 3.2 solver-backed smoke artifact directory is missing",
                {"required_dir": _display_path(artifact_dir)},
            )
        ]
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(
                _finding(
                    "missing_required_artifacts",
                    relative,
                    "required Step 3.2 artifact file is missing",
                    {"required_artifact": relative},
                )
            )
    if not any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        findings.append(
            _finding(
                "missing_required_artifacts",
                "pipeline controls",
                "Step 3.2 artifacts must retain or reference pipeline control evidence",
                {"accepted_artifacts": list(PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES)},
            )
        )
    return findings


def _audit_clean_provenance(environment: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    git = environment.get("git") if isinstance(environment.get("git"), dict) else {}
    status_text = "\n".join(
        str(value)
        for value in (
            environment.get("git_status_short"),
            git.get("status_short") if isinstance(git, dict) else None,
        )
        if value
    )
    for line in _nonempty_lines(status_text):
        findings.append(
            _finding(
                "clean_provenance",
                "environment.json",
                "artifact provenance records a dirty worktree status line",
                {"status_line": line},
            )
        )
    for field in PROVENANCE_REQUIRED_TRUE_FIELDS:
        if environment.get(field) is not True:
            findings.append(
                _finding(
                    "clean_provenance",
                    "environment.json",
                    "clean-provenance field must be recorded as true",
                    {"field": field, "value": environment.get(field)},
                )
            )
    if environment.get("core_diff_after_source_commit") != []:
        findings.append(
            _finding(
                "clean_provenance",
                "environment.json",
                "core source diff after source commit must be recorded as empty",
                {"core_diff_after_source_commit": environment.get("core_diff_after_source_commit")},
            )
        )
    source_commit = environment.get("source_code_commit") or environment.get("source_commit") or git.get("head")
    if not _is_concrete_sha(source_commit):
        findings.append(
            _finding(
                "clean_provenance",
                "environment.json",
                "source_code_commit must be a concrete 40-character SHA",
                {"source_code_commit": source_commit},
            )
        )
    return findings


def _audit_matrix_shape(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if len(rows) != EXPECTED_MATRIX_ROWS:
        findings.append(
            _finding(
                "matrix_shape",
                "model_matrix.json",
                "Step 3.2 model matrix must contain exactly 44 rows",
                {"actual_rows": len(rows), "expected_rows": EXPECTED_MATRIX_ROWS},
            )
        )
    model_ids = [_model_id(row) for row in rows]
    unique_model_ids = {model_id for model_id in model_ids if model_id}
    duplicate_model_ids = sorted(_duplicates(model_id for model_id in model_ids if model_id))
    if len(unique_model_ids) != EXPECTED_MATRIX_ROWS:
        findings.append(
            _finding(
                "matrix_shape",
                "model_matrix.json",
                "Step 3.2 model matrix must contain 44 unique model_id values",
                {
                    "unique_model_ids": len(unique_model_ids),
                    "expected_unique_model_ids": EXPECTED_MATRIX_ROWS,
                    "duplicate_model_ids": duplicate_model_ids[:10],
                },
            )
        )
    for key in ("row_count", "in_scope_total", "matrix_row_count"):
        if key in summary and _as_int(summary.get(key)) != EXPECTED_MATRIX_ROWS:
            findings.append(
                _finding(
                    "matrix_shape",
                    "quality_summary.json",
                    f"summary {key} must equal 44",
                    {"field": key, "actual": summary.get(key), "expected": EXPECTED_MATRIX_ROWS},
                )
            )
    return findings


def _audit_status_counts(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    row_counts = _status_bucket_counts(rows)
    if row_counts != EXPECTED_STATUS_COUNTS:
        findings.append(
            _finding(
                "status_counts_32_3_9",
                "model_matrix.json",
                "Step 3.2 must preserve the 32/3/9 Step 2 terminal partition",
                {"actual_counts": row_counts, "expected_counts": EXPECTED_STATUS_COUNTS},
            )
        )
    summary_counts = _summary_counts(summary.get("status_counts") or summary.get("counts"))
    if summary_counts != EXPECTED_STATUS_COUNTS:
        findings.append(
            _finding(
                "status_counts_32_3_9",
                "quality_summary.json",
                "quality summary status_counts must match the 32/3/9 partition",
                {"actual_counts": summary_counts, "expected_counts": EXPECTED_STATUS_COUNTS},
            )
        )
    category_counts = Counter(_category(row) for row in rows if _category(row))
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        findings.append(
            _finding(
                "status_counts_32_3_9",
                "model_matrix.json",
                "Step 3.2 categories must remain 32 full humanoid, 3 partial, 9 negative",
                {"actual_counts": dict(category_counts), "expected_counts": EXPECTED_CATEGORY_COUNTS},
            )
        )
    for field, expected in (
        ("full_humanoid_total", 32),
        ("partial_total", 3),
        ("negative_total", 9),
    ):
        if _as_int(summary.get(field)) != expected:
            findings.append(
                _finding(
                    "status_counts_32_3_9",
                    "quality_summary.json",
                    f"summary {field} must match Step 3.2 category counts",
                    {"field": field, "actual": summary.get(field), "expected": expected},
                )
            )
    return findings


def _audit_base_final_head(summary: dict[str, Any], ledger: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    summary_head = summary.get("base_step3_1_1_final_head")
    if summary_head != EXPECTED_BASE_STEP3_1_1_FINAL_HEAD:
        findings.append(
            _finding(
                "base_step3_1_1_final_head",
                "quality_summary.json",
                "Step 3.2 must record the closed Step 3.1.1 final HEAD baseline",
                {
                    "actual": summary_head,
                    "expected": EXPECTED_BASE_STEP3_1_1_FINAL_HEAD,
                },
            )
        )
    ledger_head = ledger.get("base_step3_1_1_final_head")
    if ledger_head is not None and ledger_head != EXPECTED_BASE_STEP3_1_1_FINAL_HEAD:
        findings.append(
            _finding(
                "base_step3_1_1_final_head",
                "acceptance_ledger.json",
                "acceptance ledger base Step 3.1.1 final HEAD does not match the required baseline",
                {
                    "actual": ledger_head,
                    "expected": EXPECTED_BASE_STEP3_1_1_FINAL_HEAD,
                },
            )
        )
    return findings


def _audit_solver_counts(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    full_rows = [row for row in rows if _is_full_humanoid(row)]
    attempted_count = sum(1 for row in full_rows if _is_true(row.get("solver_backed_smoke_attempted")))
    completed_count = sum(1 for row in full_rows if _is_true(row.get("solver_backed_smoke_completed")))
    solver_backed_count = sum(1 for row in full_rows if _is_true(row.get("solver_backed")))
    solver_attempted_models = {
        _model_id(row)
        for row in solver_rows
        if _model_id(row) and _is_true(row.get("solver_backed_smoke_attempted") or row.get("attempted"))
    }
    solver_completed_models = {
        _model_id(row)
        for row in solver_rows
        if _model_id(row) and _is_true(row.get("solver_backed_smoke_completed") or row.get("completed"))
    }
    generic_solver_models = {_model_id(row) for row in generic_rows if _model_id(row) and _row_solver_backed(row)}

    if _as_int(summary.get("solver_backed_smoke_attempted_count")) != 32:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "quality_summary.json",
                "solver_backed_smoke_attempted_count must equal the 32 full-humanoid rows",
                {
                    "actual": summary.get("solver_backed_smoke_attempted_count"),
                    "expected": 32,
                },
            )
        )
    if attempted_count != 32:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "model_matrix.json",
                "all 32 full-humanoid rows must record solver_backed_smoke_attempted=true",
                {"actual_attempted_count": attempted_count, "expected": 32},
            )
        )
    if len(solver_attempted_models) != 32:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "solver_smoke_matrix.json",
                "solver smoke matrix must contain attempted evidence for all 32 full-humanoid rows",
                {"actual_attempted_models": len(solver_attempted_models), "expected": 32},
            )
        )
    summary_completed = _as_int(summary.get("solver_backed_completed_count"))
    if summary_completed is None or summary_completed <= 0:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "quality_summary.json",
                "solver_backed_completed_count must be greater than zero",
                {"actual": summary.get("solver_backed_completed_count")},
            )
        )
    if completed_count <= 0 or not solver_completed_models:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "model_matrix.json",
                "Step 3.2 must complete at least one solver-backed generic smoke",
                {
                    "completed_full_rows": completed_count,
                    "completed_solver_matrix_models": len(solver_completed_models),
                },
            )
        )
    summary_solver_backed = _as_int(summary.get("solver_backed_count"))
    if summary_solver_backed is None or summary_solver_backed <= 0:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "quality_summary.json",
                "solver_backed_count must be greater than zero",
                {"actual": summary.get("solver_backed_count")},
            )
        )
    if solver_backed_count <= 0 or not generic_solver_models:
        findings.append(
            _finding(
                "solver_backed_smoke_counts",
                "model_matrix.json",
                "Step 3.2 must record at least one solver_backed=true full-humanoid result",
                {
                    "solver_backed_full_rows": solver_backed_count,
                    "solver_backed_generic_models": len(generic_solver_models),
                },
            )
        )
    return findings


def _audit_solver_evidence_present(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    full_model_ids = {_model_id(row) for row in rows if _is_full_humanoid(row)}
    solver_model_ids = {_model_id(row) for row in solver_rows if _model_id(row)}
    generic_model_ids = {_model_id(row) for row in generic_rows if _model_id(row)}
    missing_solver = sorted(full_model_ids - solver_model_ids)
    missing_generic = sorted(full_model_ids - generic_model_ids)
    if missing_solver:
        findings.append(
            _finding(
                "solver_evidence_present",
                "solver_smoke_matrix.json",
                "solver smoke matrix is missing full-humanoid model evidence",
                {"missing_model_ids": missing_solver[:10], "missing_count": len(missing_solver)},
            )
        )
    if missing_generic:
        findings.append(
            _finding(
                "solver_evidence_present",
                "generic_smoke_matrix.json",
                "generic smoke matrix is missing full-humanoid model evidence",
                {"missing_model_ids": missing_generic[:10], "missing_count": len(missing_generic)},
            )
        )
    return findings


def _audit_runtime_quality_label_honesty(
    rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    solver_backed_models = {
        _model_id(row)
        for row in solver_rows + generic_rows
        if _model_id(row) and _row_solver_backed(row) and _row_completed(row)
    }
    residual_only_models = {
        _model_id(row)
        for row in rows + solver_rows + generic_rows
        if _model_id(row) and _is_residual_only(row)
    }
    for index, row in enumerate(rows):
        subject = _row_subject(index, row)
        model_id = _model_id(row)
        if _is_residual_only(row) and _is_runtime_quality_pass_row(row):
            findings.append(
                _finding(
                    "runtime_quality_label_honesty",
                    subject,
                    "residual-only rows must never be runtime_quality_passed",
                    {"solver_mode": _solver_mode(row), "runtime_quality_status": _row_final_status(row)},
                )
            )
        if not _is_runtime_quality_pass_row(row):
            continue
        required_flags = {
            "solver_backed_smoke_attempted": _is_true(row.get("solver_backed_smoke_attempted")),
            "solver_backed_smoke_completed": _is_true(row.get("solver_backed_smoke_completed")),
            "solver_backed": _is_true(row.get("solver_backed")),
            "completed_solver_evidence": model_id in solver_backed_models,
        }
        if not all(required_flags.values()):
            findings.append(
                _finding(
                    "runtime_quality_label_honesty",
                    subject,
                    "runtime_quality_passed rows require attempted, completed, solver_backed evidence",
                    required_flags,
                )
            )
        if model_id in residual_only_models and model_id not in solver_backed_models:
            findings.append(
                _finding(
                    "runtime_quality_label_honesty",
                    subject,
                    "runtime_quality_passed row is backed only by residual-only evidence",
                    {"model_id": model_id},
                )
            )
    return findings


def _audit_negative_and_partial_not_promoted(rows: list[dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(rows):
        bucket = _status_bucket(row)
        if bucket not in {"partial_passed", "negative_control_passed"}:
            continue
        subject = _row_subject(index, row)
        final_status = _row_final_status(row)
        if _is_runtime_quality_pass_row(row):
            findings.append(
                _finding(
                    "negative_and_partial_not_promoted",
                    subject,
                    "partial and negative-control rows must not be promoted to runtime_quality_passed",
                    {"source_status": bucket, "runtime_quality_status": final_status},
                )
            )
        if final_status and final_status not in NEGATIVE_OR_PARTIAL_ALLOWED_FINAL_STATUSES:
            findings.append(
                _finding(
                    "negative_and_partial_not_promoted",
                    subject,
                    "partial or negative-control row has an unexpected promoted final status",
                    {
                        "source_status": bucket,
                        "runtime_quality_status": final_status,
                        "allowed_statuses": sorted(NEGATIVE_OR_PARTIAL_ALLOWED_FINAL_STATUSES),
                    },
                )
            )
        for flag in ("solver_backed", "solver_backed_smoke_attempted", "solver_backed_smoke_completed"):
            if _is_true(row.get(flag)):
                findings.append(
                    _finding(
                        "negative_and_partial_not_promoted",
                        subject,
                        f"partial and negative-control rows must not set {flag}=true",
                        {"field": flag, "value": row.get(flag)},
                    )
                )
        if bucket == "negative_control_passed":
            expected_capability = str(row.get("expected_capability") or "").lower()
            if expected_capability != "negative_control":
                findings.append(
                    _finding(
                        "negative_and_partial_not_promoted",
                        subject,
                        "negative-control rows must retain expected_capability=negative_control",
                        {"expected_capability": row.get("expected_capability")},
                    )
                )
            for flag in ("promoted_to_runtime_quality", "quality_evaluated", "override_allowed", "humanoid_profile_generated"):
                if row.get(flag) is True:
                    findings.append(
                        _finding(
                            "negative_and_partial_not_promoted",
                            subject,
                            f"negative-control row has forbidden promotion flag {flag}=true",
                            {"field": flag, "value": row.get(flag)},
                        )
                    )
    return findings


def _audit_quality_numeric_fields(rows: list[dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(rows):
        subject = _row_subject(index, row)
        fields = REQUIRED_FULL_ROW_NUMERIC_FIELDS if _is_full_humanoid(row) else tuple(
            field for field in REQUIRED_FULL_ROW_NUMERIC_FIELDS if field in row
        )
        for field in fields:
            if field not in row:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "Step 3.2 row is missing a required numeric metric",
                        {"field": field},
                    )
                )
                continue
            number = _as_float(row.get(field))
            if number is None:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "Step 3.2 numeric metric must be finite",
                        {"field": field, "value": repr(row.get(field))},
                    )
                )
                continue
            if number < 0:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "Step 3.2 numeric metric must be non-negative",
                        {"field": field, "value": number},
                    )
                )
            if field in COUNT_NUMERIC_FIELDS and not number.is_integer():
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "Step 3.2 count metric must be integral",
                        {"field": field, "value": number},
                    )
                )
            if field == "frame_count" and number <= 0:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "Step 3.2 frame_count must be positive",
                        {"field": field, "value": number},
                    )
                )
            if _row_completed(row) and field in COMPLETED_ZERO_REQUIRED_FIELDS and number != 0:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "completed solver-backed rows must not report NaN/Inf output samples",
                        {"field": field, "value": number},
                    )
                )
        for prefix in ("normalized_task_residual", "target_translation_error", "target_rotation_error"):
            findings.extend(_audit_numeric_ordering(index, row, prefix))
    return findings


def _audit_deterministic_rerun(deterministic: dict[str, Any], summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    status = str(deterministic.get("status") or "").lower()
    if status != "passed":
        findings.append(
            _finding(
                "deterministic_rerun_matched",
                "deterministic_rerun.json",
                "deterministic_rerun.status must be passed",
                {"status": deterministic.get("status")},
            )
        )
    if deterministic.get("deterministic") is not True:
        findings.append(
            _finding(
                "deterministic_rerun_matched",
                "deterministic_rerun.json",
                "deterministic_rerun.deterministic must be true",
                {"deterministic": deterministic.get("deterministic")},
            )
        )
    compared = _deterministic_count(deterministic, "compared_count")
    matched = _deterministic_count(deterministic, "matched_count")
    for key, value in (("compared_count", compared), ("matched_count", matched)):
        if value != EXPECTED_MATRIX_ROWS:
            findings.append(
                _finding(
                    "deterministic_rerun_matched",
                    "deterministic_rerun.json",
                    f"deterministic_rerun {key} must cover all 44 rows",
                    {"field": key, "actual": value, "expected": EXPECTED_MATRIX_ROWS},
                )
            )
    if compared is not None and matched is not None and compared != matched:
        findings.append(
            _finding(
                "deterministic_rerun_matched",
                "deterministic_rerun.json",
                "deterministic_rerun matched_count must equal compared_count",
                {"compared_count": compared, "matched_count": matched},
            )
        )
    for field in ("deterministic_compared_count", "deterministic_matched_count"):
        if _as_int(summary.get(field)) != EXPECTED_MATRIX_ROWS:
            findings.append(
                _finding(
                    "deterministic_rerun_matched",
                    "quality_summary.json",
                    f"summary {field} must equal 44",
                    {"field": field, "actual": summary.get(field), "expected": EXPECTED_MATRIX_ROWS},
                )
            )
    return findings


def _audit_final_head_ci(
    committed_record: dict[str, Any],
    record: dict[str, Any],
    source_root: Path,
    *,
    require_final_head_ci: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    current_head = _git_current_head(source_root)
    committed_head = committed_record.get("head_sha") or committed_record.get("final_head")
    if committed_record and current_head and _is_concrete_sha(committed_head) and str(committed_head).lower() != current_head.lower():
        findings.append(
            _finding(
                "final_head_ci",
                "acceptance_ledger.json",
                "committed final HEAD CI evidence must match the current HEAD",
                {"committed_head_sha": committed_head, "current_head": current_head},
            )
        )
    if not record:
        if require_final_head_ci:
            findings.append(
                _finding(
                    "final_head_ci",
                    "acceptance_ledger.json",
                    "final HEAD CI evidence is missing",
                    {"required_field": "final_head_ci", "live_lookup": "github_check_runs_for_current_head"},
                )
            )
        return findings

    workflow_run_id = record.get("workflow_run_id") or record.get("run_id")
    head_sha = record.get("head_sha") or record.get("final_head")
    conclusion = str(record.get("conclusion") or "").lower()
    if not _is_concrete_run_id(workflow_run_id):
        findings.append(
            _finding(
                "final_head_ci",
                "acceptance_ledger.json",
                "workflow_run_id must be concrete final HEAD CI evidence",
                {"workflow_run_id": workflow_run_id},
            )
        )
    if not _is_concrete_sha(head_sha):
        findings.append(
            _finding(
                "final_head_ci",
                "acceptance_ledger.json",
                "head_sha must be a concrete 40-character SHA",
                {"head_sha": head_sha},
            )
        )
    elif current_head and str(head_sha).lower() != current_head.lower():
        findings.append(
            _finding(
                "final_head_ci",
                "acceptance_ledger.json",
                "final HEAD CI evidence must be for the current HEAD",
                {"head_sha": head_sha, "current_head": current_head},
            )
        )
    if conclusion != "success":
        findings.append(
            _finding(
                "final_head_ci",
                "acceptance_ledger.json",
                "final HEAD CI conclusion must be success",
                {"conclusion": record.get("conclusion")},
            )
        )
    job_conclusions = _job_conclusions(record.get("job_conclusions") or record.get("jobs"))
    if not job_conclusions:
        findings.append(
            _finding(
                "final_head_ci",
                "acceptance_ledger.json",
                "final HEAD CI job conclusions are missing",
                {"required_jobs": list(REQUIRED_FINAL_HEAD_CI_JOBS)},
            )
        )
    else:
        for job in REQUIRED_FINAL_HEAD_CI_JOBS:
            if job not in job_conclusions:
                findings.append(
                    _finding(
                        "final_head_ci",
                        "acceptance_ledger.json",
                        "final HEAD CI evidence is missing a required job conclusion",
                        {"required_job": job, "job_conclusions": job_conclusions},
                    )
                )
        for job, conclusion_value in sorted(job_conclusions.items()):
            if str(conclusion_value).lower() != "success":
                findings.append(
                    _finding(
                        "final_head_ci",
                        "acceptance_ledger.json",
                        "final HEAD CI job conclusion must be success",
                        {"job": job, "conclusion": conclusion_value},
                    )
                )
    return findings


def _audit_numeric_ordering(index: int, row: dict[str, Any], prefix: str) -> list[Finding]:
    mean = _as_float(row.get(f"{prefix}_mean"))
    p95 = _as_float(row.get(f"{prefix}_p95"))
    max_value = _as_float(row.get(f"{prefix}_max"))
    if mean is None or p95 is None or max_value is None:
        return []
    if mean <= p95 <= max_value:
        return []
    return [
        _finding(
            "quality_numeric_fields",
            _row_subject(index, row),
            "Step 3.2 mean/p95/max metrics must be ordered",
            {"prefix": prefix, "mean": mean, "p95": p95, "max": max_value},
        )
    ]


def _matrix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rows", "matrix", "models", "quality_matrix"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _status_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_status_bucket(row) for row in rows).items()))


def _summary_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return dict(sorted((str(key), _as_int(child) or 0) for key, child in value.items()))


def _status_bucket(row: dict[str, Any]) -> str:
    status = str(
        row.get("source_status")
        or row.get("status")
        or row.get("profile_status")
        or row.get("terminal_status")
        or ""
    ).lower()
    if status in {"passed", "capability_limited_passed", "quality_passed"}:
        return "passed"
    if status == "partial_passed":
        return "partial_passed"
    if status == "negative_control_passed":
        return "negative_control_passed"
    return status


def _category(row: dict[str, Any]) -> str:
    return str(row.get("category") or row.get("expected_category") or "").lower()


def _is_full_humanoid(row: dict[str, Any]) -> bool:
    return _category(row) == "full_humanoid_profile" or (
        _status_bucket(row) == "passed" and str(row.get("expected_capability") or "").lower() != "negative_control"
    )


def _row_final_status(row: dict[str, Any]) -> str:
    return str(
        row.get("runtime_quality_status")
        or row.get("final_step3_2_status")
        or row.get("final_step3_1_status")
        or row.get("quality_classification")
        or row.get("status")
        or ""
    ).lower()


def _is_runtime_quality_pass_row(row: dict[str, Any]) -> bool:
    return _row_final_status(row) in RUNTIME_QUALITY_PASS_STATUSES


def _row_completed(row: dict[str, Any]) -> bool:
    return _is_true(row.get("solver_backed_smoke_completed") or row.get("completed"))


def _row_solver_backed(row: dict[str, Any]) -> bool:
    if _is_true(row.get("solver_backed")):
        return True
    solver_type = str(row.get("solver_type") or row.get("solver_mode") or "").lower()
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    summary_solver_type = str(summary.get("solver_type") or summary.get("mode") or "").lower()
    return "solver" in solver_type or "solver" in summary_solver_type


def _is_residual_only(row: dict[str, Any]) -> bool:
    if _is_true(row.get("residual_only")):
        return True
    mode = _solver_mode(row)
    if "fk_residual" in mode or "residual_only" in mode:
        return True
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    residuals = summary.get("residuals") if isinstance(summary.get("residuals"), dict) else {}
    solver = str(residuals.get("solver") or "").lower()
    return solver in {"runtime_model_fk_residual_evaluation", "runtime_model_fk_residual_evaluation_only"}


def _solver_mode(row: dict[str, Any]) -> str:
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    return str(row.get("solver_mode") or row.get("mode") or summary.get("mode") or "").lower()


def _deterministic_count(payload: dict[str, Any], key: str) -> int | None:
    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    return _as_int(payload.get(key)) if key in payload else _as_int(totals.get(key))


def _final_head_ci_record(acceptance_ledger: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    for payload in (acceptance_ledger, summary):
        if not isinstance(payload, dict):
            continue
        for key in ("final_head_ci", "ci"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _live_final_head_ci_record(source_root: Path) -> dict[str, Any]:
    head_sha = _git_current_head(source_root)
    repo_full_name = _git_remote_repo_full_name(source_root)
    if not head_sha or not repo_full_name:
        return {}

    check_runs = _github_json(
        f"https://api.github.com/repos/{repo_full_name}/commits/{head_sha}/check-runs?per_page=100",
        accept="application/vnd.github+json",
    )
    if not isinstance(check_runs, dict):
        return {}
    candidates = _final_head_ci_candidates(check_runs.get("check_runs"), head_sha)
    for candidate in candidates:
        workflow_run_id = candidate.get("workflow_run_id")
        run_payload = _github_json(f"https://api.github.com/repos/{repo_full_name}/actions/runs/{workflow_run_id}")
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
                "created_at": run_payload.get("created_at"),
                "updated_at": run_payload.get("updated_at"),
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
    relevant_names = set(REQUIRED_FINAL_HEAD_CI_JOBS) | {"step3-2-final-head-ci-evidence"}
    for check_run in check_runs:
        if not isinstance(check_run, dict):
            continue
        name = str(check_run.get("name") or "")
        if name not in relevant_names:
            continue
        run_id = _workflow_run_id_from_details_url(check_run.get("details_url"))
        if not run_id:
            continue
        group = grouped.setdefault(
            run_id,
            {
                "workflow_run_id": run_id,
                "head_sha": head_sha,
                "job_conclusions": {},
                "check_run_ids": {},
                "check_run_urls": {},
                "completed_at": "",
            },
        )
        group["job_conclusions"][name] = str(check_run.get("conclusion") or "").lower()
        group["check_run_ids"][name] = check_run.get("id")
        group["check_run_urls"][name] = check_run.get("details_url")
        completed_at = str(check_run.get("completed_at") or "")
        if completed_at > str(group.get("completed_at") or ""):
            group["completed_at"] = completed_at

    candidates = []
    for group in grouped.values():
        job_conclusions = group.get("job_conclusions") if isinstance(group.get("job_conclusions"), dict) else {}
        if all(job_conclusions.get(job) == "success" for job in REQUIRED_FINAL_HEAD_CI_JOBS):
            candidates.append(group)
    return sorted(candidates, key=lambda item: str(item.get("completed_at") or ""), reverse=True)


def _workflow_run_id_from_details_url(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"/actions/runs/([1-9][0-9]{5,})(?:/|$)", text)
    return match.group(1) if match else ""


def _github_json(url: str, *, accept: str = "application/vnd.github+json") -> dict[str, Any]:
    headers = {
        "Accept": accept,
        "User-Agent": "soma-retargeter-step3-2-audit",
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_current_head(source_root: Path) -> str:
    return _git_stdout(source_root, "rev-parse", "HEAD")


def _git_remote_repo_full_name(source_root: Path) -> str:
    remote_url = _git_stdout(source_root, "remote", "get-url", "origin")
    if not remote_url:
        return ""
    remote_url = remote_url.removesuffix(".git")
    if remote_url.startswith("git@github.com:"):
        return remote_url.removeprefix("git@github.com:")
    parsed = urllib.parse.urlparse(remote_url)
    if parsed.netloc.lower() == "github.com":
        return parsed.path.strip("/")
    return ""


def _git_stdout(source_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=source_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _job_conclusions(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(child).lower() for key, child in value.items()}
    if isinstance(value, list):
        conclusions: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("job") or item.get("id")
                conclusion = item.get("conclusion") or item.get("status")
                if name:
                    conclusions[str(name)] = str(conclusion or "").lower()
            elif isinstance(item, str):
                conclusions[item] = item.lower()
        return conclusions
    return {}


def _model_id(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or row.get("profile_model_id") or row.get("robot_id") or "").strip()


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


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


def _row_subject(index: int, row: dict[str, Any]) -> str:
    return f"model_matrix[{index}] {_model_id(row) or 'unknown_model'}"


def _is_true(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes", "passed", "completed"}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _is_concrete_sha(value: Any) -> bool:
    return isinstance(value, str) and CONCRETE_SHA_RE.fullmatch(value.strip()) is not None


def _is_concrete_run_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and CONCRETE_RUN_ID_RE.fullmatch(str(value).strip()) is not None


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
