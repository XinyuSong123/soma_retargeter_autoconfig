#!/usr/bin/env python3
"""Agent F style audit for Step 3.1 full-fleet runtime-quality evidence.

The audit is read-only. It validates that the Step 3.1 artifact tree is a real
44-model full-fleet acceptance record, not a replay of the Step 3.0 RPO/G1
runtime-shadow smoke evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_runtime_quality")
DEFAULT_SOURCE_ROOT = Path(".")
AGENT_F_HANDOFF = Path("docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md")
EXPECTED_MATRIX_ROWS = 44
EXPECTED_STATUS_COUNTS = {
    "passed": 32,
    "partial_passed": 3,
    "negative_control_passed": 9,
}
MIN_NON_RPO_G1_ROWS = 40
REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "commands.txt",
    "model_matrix.json",
    "full_fleet_matrix.json",
    "quality_summary.json",
    "target_stream_matrix.json",
    "generic_smoke_matrix.json",
    "pipeline_backed_matrix.json",
    "pipeline_controls.json",
    "acceptance_ledger.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/pytest_summary.json",
)
ACCEPTANCE_GATES = (
    "missing_required_artifacts",
    "clean_provenance",
    "matrix_shape",
    "status_counts_32_3_9",
    "status_count_honesty",
    "non_rpo_g1_full_fleet",
    "negative_controls_not_promoted",
    "quality_numeric_fields",
    "runtime_quality_label_honesty",
    "absolute_path_leakage",
    "pipeline_controls_present",
    "final_head_ci",
    "full_repo_pytest_caveat",
    "stale_agent_f_verdict",
)
REQUIRED_QUALITY_NUMERIC_FIELDS = (
    "frame_count",
    "target_translation_error_mean",
    "target_translation_error_p95",
    "target_translation_error_max",
    "target_rotation_error_mean",
    "target_rotation_error_p95",
    "target_rotation_error_max",
    "output_nan_count",
    "output_inf_count",
    "joint_limit_violation_count",
    "max_joint_limit_violation",
    "runtime_seconds",
)
COUNT_NUMERIC_FIELDS = {
    "frame_count",
    "output_nan_count",
    "output_inf_count",
    "joint_limit_violation_count",
}
ZERO_REQUIRED_COUNT_FIELDS = {"output_nan_count", "output_inf_count"}
REQUIRED_PIPELINE_CONTROL_FLAGS = (
    "default_runtime_disabled_verified",
    "shadow_noop_verified",
    "override_explicit_only",
    "fingerprint_gate_enforced",
    "negative_controls_excluded",
    "artifact_paths_sanitized",
)
REQUIRED_ROW_CONTROL_FLAGS = (
    "legacy_default_unchanged",
    "shadow_noop_verified",
    "override_explicit_only",
    "fingerprint_gate_enforced",
)
FORBIDDEN_NEGATIVE_QUALITY_STATUSES = {
    "accepted",
    "pass",
    "passed",
    "promoted",
    "quality_accepted",
    "quality_passed",
    "runtime_quality_passed",
}
RUNTIME_QUALITY_PASS_STATUSES = {
    "accepted",
    "pass",
    "passed",
    "quality_accepted",
    "quality_passed",
    "runtime_quality_passed",
    "full_fleet_quality_pass",
}
RUNTIME_QUALITY_FAILURE_STATUSES = {
    "blocked",
    "failed",
    "quality_failed",
    "runtime_quality_failed",
}
RESIDUAL_ONLY_SMOKE_MODES = {
    "generic_fk_residual_smoke",
}
PROVENANCE_REQUIRED_TRUE_FIELDS = (
    "source_code_commit_remote_resolvable",
    "source_code_commit_is_artifact_commit_ancestor",
    "source_worktree_clean_before_run",
    "source_worktree_clean_after_run",
)
REQUIRED_FINAL_HEAD_CI_JOBS = (
    "full-fleet-static-and-unit",
    "full-fleet-artifact-audit",
    "lfs-and-snapshot-smoke",
    "pipeline-backed-regression",
    "quality-status-semantics",
)
ALLOWED_FULL_REPO_PYTEST_CLASSIFICATIONS = {
    "not_run_scoped_caveat",
    "full_repo_passed",
    "full_repo_failed",
}
LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w$])/(?:mnt|home|Users|tmp|var|private/var)/[^\s\"'<>),\]}`:]+"
)
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
    full_repo_pytest: dict[str, Any]
    final_head_ci: dict[str, Any]

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    def to_json(self) -> dict[str, Any]:
        return _sanitize_audit_payload(asdict(self))


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    require_final_head_ci: bool = False,
) -> AuditResult:
    artifact_dir = artifact_dir.resolve()
    source_root = source_root.resolve()
    findings: list[Finding] = []

    matrix_payload = _read_json(artifact_dir / "full_fleet_matrix.json")
    model_matrix_payload = _read_json(artifact_dir / "model_matrix.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    environment = _read_json(artifact_dir / "environment.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    controls = _read_json(artifact_dir / "pipeline_controls.json")
    acceptance_ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    rows = _matrix_rows(matrix_payload)
    generic_rows = _generic_smoke_rows(generic_smoke)
    committed_final_head_ci = _final_head_ci_record(acceptance_ledger, summary)
    live_final_head_ci = _live_final_head_ci_record(source_root) if require_final_head_ci else {}
    final_head_ci = live_final_head_ci or committed_final_head_ci

    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_clean_provenance(environment))
    findings.extend(_audit_matrix_shape(rows, summary))
    findings.extend(_audit_status_counts(rows, summary))
    findings.extend(_audit_status_count_honesty(rows, summary, model_matrix_payload, generic_rows))
    findings.extend(_audit_non_rpo_g1_full_fleet(rows, summary))
    findings.extend(_audit_negative_controls_not_promoted(rows))
    findings.extend(_audit_quality_numeric_fields(rows))
    findings.extend(_audit_runtime_quality_label_honesty(rows, generic_rows, summary))
    findings.extend(_audit_absolute_paths(artifact_dir, source_root))
    findings.extend(_audit_pipeline_controls(rows, controls))
    findings.extend(
        _audit_final_head_ci(
            committed_final_head_ci,
            final_head_ci,
            source_root,
            require_final_head_ci=require_final_head_ci,
        )
    )
    findings.extend(_audit_full_repo_pytest_caveat(acceptance_ledger, source_root))

    pre_verdict_status = "PASS" if not [finding for finding in findings if finding.severity == "error"] else "BLOCKED"
    findings.extend(_audit_agent_f_verdicts(acceptance_ledger, source_root, pre_verdict_status))

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
        full_repo_pytest=acceptance_ledger.get("full_repo_pytest", {}),
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
        help=(
            "Require committed/external final HEAD CI metadata. The default audit omits this "
            "self-referential gate so artifact-audit CI can run on the same commit it validates."
        ),
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
        findings.append(
            _finding(
                "missing_required_artifacts",
                _display_path(artifact_dir),
                "Step 3.1 runtime-quality artifact directory is missing",
                {"required_dir": _display_path(artifact_dir)},
            )
        )
        return findings
    for relative in REQUIRED_ARTIFACT_FILES:
        path = artifact_dir / relative
        if not path.exists():
            findings.append(
                _finding(
                    "missing_required_artifacts",
                    relative,
                    "required Step 3.1 artifact file is missing",
                    {"required_artifact": relative},
                )
            )
    return findings


def _audit_matrix_shape(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if len(rows) != EXPECTED_MATRIX_ROWS:
        findings.append(
            _finding(
                "matrix_shape",
                "full_fleet_matrix.json",
                "full-fleet quality matrix must contain exactly 44 rows",
                {"actual_rows": len(rows), "expected_rows": EXPECTED_MATRIX_ROWS},
            )
        )
    model_ids = [_model_id(row) for row in rows]
    duplicate_model_ids = sorted(_duplicates(model_id for model_id in model_ids if model_id))
    if len({model_id for model_id in model_ids if model_id}) != EXPECTED_MATRIX_ROWS:
        findings.append(
            _finding(
                "matrix_shape",
                "full_fleet_matrix.json",
                "full-fleet quality matrix must contain 44 unique model_id values",
                {
                    "unique_model_ids": len({model_id for model_id in model_ids if model_id}),
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
    if not rows:
        findings.append(
            _finding(
                "matrix_shape",
                "full_fleet_matrix.json",
                "full-fleet matrix is missing or not a JSON row list",
                {"accepted_keys": ["matrix", "rows", "models", "quality_matrix"]},
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
    core_diff = environment.get("core_diff_after_source_commit")
    if core_diff != []:
        findings.append(
            _finding(
                "clean_provenance",
                "environment.json",
                "core source diff after source commit must be recorded as empty",
                {"core_diff_after_source_commit": core_diff},
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
    artifact_commit = environment.get("artifact_commit")
    if artifact_commit is not None and not _is_concrete_sha(artifact_commit):
        findings.append(
            _finding(
                "clean_provenance",
                "environment.json",
                "artifact_commit must be a concrete 40-character SHA when recorded",
                {"artifact_commit": artifact_commit},
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
                "full_fleet_matrix.json",
                "matrix rows must preserve the 32/3/9 Step 2 terminal status partition",
                {"actual_counts": row_counts, "expected_counts": EXPECTED_STATUS_COUNTS},
            )
        )
    summary_counts = _summary_status_counts(summary)
    if summary_counts is None:
        findings.append(
            _finding(
                "status_counts_32_3_9",
                "quality_summary.json",
                "quality summary must record status counts",
                {"accepted_fields": ["status_counts", "counts"]},
            )
        )
    elif summary_counts != EXPECTED_STATUS_COUNTS:
        findings.append(
            _finding(
                "status_counts_32_3_9",
                "quality_summary.json",
                "quality summary counts must match the 32/3/9 full-fleet partition",
                {"actual_counts": summary_counts, "expected_counts": EXPECTED_STATUS_COUNTS},
            )
        )
    return findings


def _audit_status_count_honesty(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    model_matrix_payload: dict[str, Any],
    generic_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    final_counts = Counter(_row_final_status(row) for row in rows if _row_final_status(row))
    summary_final_counts = summary.get("final_status_counts")
    if not isinstance(summary_final_counts, dict):
        findings.append(
            _finding(
                "status_count_honesty",
                "quality_summary.json",
                "quality summary must record final_status_counts",
                {"actual_final_status_counts": dict(final_counts)},
            )
        )
    else:
        for status in sorted(set(final_counts) | set(str(key) for key in summary_final_counts)):
            actual = final_counts.get(status, 0)
            expected = _as_int(summary_final_counts.get(status)) or 0
            if actual != expected:
                findings.append(
                    _finding(
                        "status_count_honesty",
                        "quality_summary.json",
                        "summary final_status_counts must match full_fleet_matrix rows",
                        {"status": status, "actual": actual, "summary": summary_final_counts.get(status)},
                    )
                )
    model_rows = _matrix_rows(model_matrix_payload)
    if model_rows:
        model_counts = _status_bucket_counts(model_rows)
        row_counts = _status_bucket_counts(rows)
        if model_counts != row_counts:
            findings.append(
                _finding(
                    "status_count_honesty",
                    "model_matrix.json",
                    "model_matrix status partition must match full_fleet_matrix",
                    {"model_matrix_counts": model_counts, "full_fleet_counts": row_counts},
                )
            )
    if generic_rows:
        generic_success_count = _unique_smoke_model_count(generic_rows, _is_pass_status)
        generic_failed_count = _unique_smoke_model_count(generic_rows, _is_failure_status)
        for field, actual in (
            ("generic_smoke_success_count", generic_success_count),
            ("generic_smoke_failed_count", generic_failed_count),
        ):
            if field in summary and (_as_int(summary.get(field)) or 0) != actual:
                findings.append(
                    _finding(
                        "status_count_honesty",
                        "quality_summary.json",
                        f"summary {field} must match generic_smoke_matrix rows",
                        {"field": field, "actual": actual, "summary": summary.get(field)},
                    )
                )
    quality_failed_count = _as_int(summary.get("quality_failed_count"))
    if quality_failed_count is not None:
        inferred_problem_models = _quality_pass_problem_models(rows, generic_rows)
        if quality_failed_count < len(inferred_problem_models):
            findings.append(
                _finding(
                    "status_count_honesty",
                    "quality_summary.json",
                    "quality_failed_count understates rows that cannot honestly be runtime quality passes",
                    {
                        "quality_failed_count": quality_failed_count,
                        "minimum_inferred_quality_problem_count": len(inferred_problem_models),
                        "sample_models": sorted(inferred_problem_models)[:10],
                    },
                )
            )
    return findings


def _audit_non_rpo_g1_full_fleet(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    non_rpo_g1_rows = [row for row in rows if not _is_rpo_or_g1(row)]
    if len(non_rpo_g1_rows) < MIN_NON_RPO_G1_ROWS:
        findings.append(
            _finding(
                "non_rpo_g1_full_fleet",
                "full_fleet_matrix.json",
                "Step 3.1 evidence must be a full-fleet audit, not RPO/G1-only runtime smoke",
                {
                    "non_rpo_g1_rows": len(non_rpo_g1_rows),
                    "minimum_required": MIN_NON_RPO_G1_ROWS,
                    "total_rows": len(rows),
                },
            )
        )
    summary_non_rpo_g1 = _as_int(summary.get("non_rpo_g1_row_count"))
    if summary_non_rpo_g1 is not None and summary_non_rpo_g1 < MIN_NON_RPO_G1_ROWS:
        findings.append(
            _finding(
                "non_rpo_g1_full_fleet",
                "quality_summary.json",
                "summary reports too few non-RPO/G1 rows for full-fleet acceptance",
                {
                    "non_rpo_g1_row_count": summary_non_rpo_g1,
                    "minimum_required": MIN_NON_RPO_G1_ROWS,
                },
            )
        )
    return findings


def _audit_negative_controls_not_promoted(rows: list[dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(rows):
        if _status_bucket(row) != "negative_control_passed":
            continue
        subject = _row_subject(index, row)
        expected_capability = str(row.get("expected_capability") or "").lower()
        if expected_capability != "negative_control":
            findings.append(
                _finding(
                    "negative_controls_not_promoted",
                    subject,
                    "negative-control rows must explicitly retain expected_capability=negative_control",
                    {"expected_capability": row.get("expected_capability")},
                )
            )
        runtime_quality_status = str(
            row.get("runtime_quality_status")
            or row.get("quality_status")
            or row.get("quality_classification")
            or ""
        ).lower()
        if runtime_quality_status in FORBIDDEN_NEGATIVE_QUALITY_STATUSES:
            findings.append(
                _finding(
                    "negative_controls_not_promoted",
                    subject,
                    "negative-control row was promoted to a runtime quality pass status",
                    {"runtime_quality_status": runtime_quality_status},
                )
            )
        for key in ("promoted_to_runtime_quality", "quality_evaluated", "override_allowed", "humanoid_profile_generated"):
            if row.get(key) is True:
                findings.append(
                    _finding(
                        "negative_controls_not_promoted",
                        subject,
                        f"negative-control row has forbidden promotion flag {key}=true",
                        {"field": key, "value": row.get(key)},
                    )
                )
    return findings


def _audit_quality_numeric_fields(rows: list[dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(rows):
        if _status_bucket(row) == "negative_control_passed":
            required_fields = tuple(field for field in REQUIRED_QUALITY_NUMERIC_FIELDS if field in row)
        else:
            required_fields = REQUIRED_QUALITY_NUMERIC_FIELDS
        for field in required_fields:
            subject = _row_subject(index, row)
            if field not in row:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "runtime quality row is missing a required numeric field",
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
                        "runtime quality numeric field must be finite",
                        {"field": field, "value": repr(row.get(field))},
                    )
                )
                continue
            if number < 0:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "runtime quality numeric field must be non-negative",
                        {"field": field, "value": number},
                    )
                )
            if field in COUNT_NUMERIC_FIELDS and not float(number).is_integer():
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "runtime quality count field must be integral",
                        {"field": field, "value": number},
                    )
                )
            if field == "frame_count" and number <= 0:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "runtime quality frame_count must be positive",
                        {"field": field, "value": number},
                    )
                )
            if field in ZERO_REQUIRED_COUNT_FIELDS and number != 0:
                findings.append(
                    _finding(
                        "quality_numeric_fields",
                        subject,
                        "runtime quality output must not contain NaN/Inf samples",
                        {"field": field, "value": number},
                    )
                )
        findings.extend(_audit_numeric_ordering(index, row, "target_translation_error"))
        findings.extend(_audit_numeric_ordering(index, row, "target_rotation_error"))
    return findings


def _audit_runtime_quality_label_honesty(
    rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    residual_only_models: dict[str, list[dict[str, Any]]] = defaultdict(list)
    solver_backed_pass_models: set[str] = set()
    for index, smoke_row in enumerate(generic_rows):
        model_id = _model_id(smoke_row)
        category = str(smoke_row.get("category") or "").lower()
        mode = _smoke_mode(smoke_row)
        status = _smoke_status(smoke_row)
        if category == "full_humanoid_profile" and _is_residual_only_smoke(smoke_row):
            if model_id:
                residual_only_models[model_id].append(smoke_row)
            if _is_pass_status(status):
                findings.append(
                    _finding(
                        "runtime_quality_label_honesty",
                        f"generic_smoke_matrix[{index}] {model_id or 'unknown_model'}",
                        "residual-only FK smoke must not be labeled as a runtime quality pass",
                        {
                            "mode": mode,
                            "status": status,
                            "required_status_semantics": [
                                "runtime_evaluation_completed",
                                "runtime_quality_warned",
                                "runtime_quality_failed",
                            ],
                            "residual_metrics": _smoke_residual_metric_sample(smoke_row),
                        },
                    )
                )
        elif category == "full_humanoid_profile" and model_id and _is_pass_status(status):
            solver_backed_pass_models.add(model_id)
    for index, row in enumerate(rows):
        model_id = _model_id(row)
        if not _is_runtime_quality_pass_row(row):
            continue
        if _as_int(row.get("joint_limit_violation_count")) not in (None, 0):
            findings.append(
                _finding(
                    "runtime_quality_label_honesty",
                    _row_subject(index, row),
                    "runtime_quality_passed row reports joint-limit violations",
                    {
                        "joint_limit_violation_count": row.get("joint_limit_violation_count"),
                        "max_joint_limit_violation": row.get("max_joint_limit_violation"),
                    },
                )
            )
        max_violation = _as_float(row.get("max_joint_limit_violation"))
        if max_violation is not None and max_violation > 0:
            findings.append(
                _finding(
                    "runtime_quality_label_honesty",
                    _row_subject(index, row),
                    "runtime_quality_passed row reports nonzero max_joint_limit_violation",
                    {"max_joint_limit_violation": max_violation},
                )
            )
        if residual_only_models.get(model_id) and model_id not in solver_backed_pass_models:
            findings.append(
                _finding(
                    "runtime_quality_label_honesty",
                    _row_subject(index, row),
                    "runtime_quality_passed row is backed only by residual-only FK smoke evidence",
                    {
                        "model_id": model_id,
                        "final_status": _row_final_status(row),
                        "generic_smoke_modes": sorted({_smoke_mode(smoke) for smoke in residual_only_models[model_id]}),
                        "residual_metrics": _smoke_residual_metric_sample(residual_only_models[model_id][0]),
                    },
                )
            )
    if summary.get("quality_failed_count") == 0 and _quality_pass_problem_models(rows, generic_rows):
        findings.append(
            _finding(
                "runtime_quality_label_honesty",
                "quality_summary.json",
                "summary claims zero quality failures while pass-labeled rows have quality blockers",
                {"quality_failed_count": summary.get("quality_failed_count")},
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
            "runtime quality mean/p95/max fields must be ordered",
            {"prefix": prefix, "mean": mean, "p95": p95, "max": max_value},
        )
    ]


def _audit_absolute_paths(artifact_dir: Path, source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    candidate_files: list[Path] = []
    if artifact_dir.exists():
        candidate_files.extend(path for path in artifact_dir.rglob("*") if path.is_file())
    handoff = source_root / AGENT_F_HANDOFF
    if handoff.exists():
        candidate_files.append(handoff)
    for path in candidate_files:
        text = _read_text(path)
        for match in LOCAL_ABSOLUTE_PATH_RE.finditer(text):
            findings.append(
                _finding(
                    "absolute_path_leakage",
                    _display_path(path),
                    "local absolute path leaked into Step 3.1 evidence",
                    {"path": match.group(0)},
                )
            )
    return findings


def _audit_pipeline_controls(rows: list[dict[str, Any]], controls: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    control_flags = controls.get("controls") if isinstance(controls.get("controls"), dict) else controls
    if not control_flags:
        findings.append(
            _finding(
                "pipeline_controls_present",
                "pipeline_controls.json",
                "pipeline controls artifact is missing or empty",
                {"required_flags": list(REQUIRED_PIPELINE_CONTROL_FLAGS)},
            )
        )
    for flag in REQUIRED_PIPELINE_CONTROL_FLAGS:
        if not _is_pass(control_flags.get(flag) if isinstance(control_flags, dict) else None):
            findings.append(
                _finding(
                    "pipeline_controls_present",
                    "pipeline_controls.json",
                    "pipeline controls artifact is missing a required pass flag",
                    {"flag": flag, "value": control_flags.get(flag) if isinstance(control_flags, dict) else None},
                )
            )
    for index, row in enumerate(rows):
        subject = _row_subject(index, row)
        if not row.get("pipeline_control_id") and not row.get("control_artifact"):
            findings.append(
                _finding(
                    "pipeline_controls_present",
                    subject,
                    "matrix row is missing pipeline control linkage",
                    {"required_one_of": ["pipeline_control_id", "control_artifact"]},
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
    else:
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
    handoff_text = _read_text(source_root / AGENT_F_HANDOFF)
    handoff_has_static_record = _handoff_has_final_head_ci(handoff_text)
    handoff_has_live_policy = bool(record) and record.get("evidence_source") == "github_check_runs_live" and _handoff_has_live_final_head_ci_policy(handoff_text)
    if require_final_head_ci and not (handoff_has_static_record or handoff_has_live_policy):
        findings.append(
            _finding(
                "final_head_ci",
                str(AGENT_F_HANDOFF),
                "Agent F handoff must record static final HEAD CI evidence or the live GitHub check-runs closure policy",
                {},
            )
        )
    return findings


def _audit_full_repo_pytest_caveat(acceptance_ledger: dict[str, Any], source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    record = acceptance_ledger.get("full_repo_pytest")
    if not isinstance(record, dict):
        findings.append(
            _finding(
                "full_repo_pytest_caveat",
                "acceptance_ledger.json",
                "acceptance ledger must include full_repo_pytest caveat/classification",
                {"required_field": "full_repo_pytest"},
            )
        )
    else:
        status = str(record.get("status") or "").lower()
        classification = str(record.get("classification") or "").lower()
        caveat = str(record.get("caveat") or "")
        if classification not in ALLOWED_FULL_REPO_PYTEST_CLASSIFICATIONS:
            findings.append(
                _finding(
                    "full_repo_pytest_caveat",
                    "acceptance_ledger.json",
                    "full_repo_pytest classification is missing or invalid",
                    {
                        "classification": record.get("classification"),
                        "allowed": sorted(ALLOWED_FULL_REPO_PYTEST_CLASSIFICATIONS),
                    },
                )
            )
        if not caveat.strip():
            findings.append(
                _finding(
                    "full_repo_pytest_caveat",
                    "acceptance_ledger.json",
                    "full_repo_pytest caveat must be non-empty",
                    {},
                )
            )
        if status == "not_run":
            text = caveat.lower()
            if classification != "not_run_scoped_caveat" or "not run" not in text:
                findings.append(
                    _finding(
                        "full_repo_pytest_caveat",
                        "acceptance_ledger.json",
                        "not-run full repo pytest evidence must be classified as a scoped caveat",
                        {"status": record.get("status"), "classification": record.get("classification"), "caveat": caveat},
                    )
                )
        elif status in {"passed", "failed"}:
            if classification != f"full_repo_{status}":
                findings.append(
                    _finding(
                        "full_repo_pytest_caveat",
                        "acceptance_ledger.json",
                        "full repo pytest run status and classification disagree",
                        {"status": record.get("status"), "classification": record.get("classification")},
                    )
                )
        else:
            findings.append(
                _finding(
                    "full_repo_pytest_caveat",
                    "acceptance_ledger.json",
                    "full_repo_pytest status must be passed, failed, or not_run",
                    {"status": record.get("status")},
                )
            )
    handoff_text = _read_text(source_root / AGENT_F_HANDOFF).lower()
    if handoff_text:
        if "full repo pytest" not in handoff_text or "classification" not in handoff_text:
            findings.append(
                _finding(
                    "full_repo_pytest_caveat",
                    str(AGENT_F_HANDOFF),
                    "Agent F handoff must record full repo pytest caveat/classification",
                    {},
                )
            )
    return findings


def _audit_agent_f_verdicts(
    acceptance_ledger: dict[str, Any],
    source_root: Path,
    expected_status_before_verdict_check: str,
) -> list[Finding]:
    findings: list[Finding] = []
    declared = [
        ("acceptance_ledger.json", _normalize_verdict(acceptance_ledger.get("verdict") or acceptance_ledger.get("status"))),
        (str(AGENT_F_HANDOFF), _extract_verdict(_read_text(source_root / AGENT_F_HANDOFF))),
    ]
    for subject, verdict in declared:
        if verdict is None:
            findings.append(
                _finding(
                    "stale_agent_f_verdict",
                    subject,
                    "Agent F verdict evidence is missing PASS/BLOCKED",
                    {"expected_verdict": expected_status_before_verdict_check},
                )
            )
            continue
        if verdict != expected_status_before_verdict_check:
            findings.append(
                _finding(
                    "stale_agent_f_verdict",
                    subject,
                    "Agent F PASS/BLOCKED verdict does not match live audit result",
                    {"declared_verdict": verdict, "expected_verdict": expected_status_before_verdict_check},
                )
            )
    return findings


def _matrix_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("matrix", "rows", "models", "quality_matrix"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            return [row for row in value.values() if isinstance(row, dict)]
    return []


def _generic_smoke_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _matrix_rows(payload)


def _summary_status_counts(summary: dict[str, Any]) -> dict[str, int] | None:
    value = summary.get("status_counts") if isinstance(summary, dict) else None
    if not isinstance(value, dict):
        value = summary.get("counts") if isinstance(summary, dict) else None
    if not isinstance(value, dict):
        return None
    return _normalize_status_counts(value)


def _status_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in EXPECTED_STATUS_COUNTS}
    for row in rows:
        bucket = _status_bucket(row)
        if bucket in counts:
            counts[bucket] += 1
    return counts


def _normalize_status_counts(counts: dict[str, Any]) -> dict[str, int]:
    passed = (
        _as_int(counts.get("passed")) or 0
    ) + (_as_int(counts.get("capability_limited_passed")) or 0) + (_as_int(counts.get("quality_passed")) or 0)
    return {
        "passed": passed,
        "partial_passed": _as_int(counts.get("partial_passed")) or 0,
        "negative_control_passed": _as_int(counts.get("negative_control_passed")) or 0,
    }


def _status_bucket(row: dict[str, Any]) -> str:
    status = str(
        row.get("source_status")
        or row.get("profile_status")
        or row.get("terminal_status")
        or row.get("status")
        or ""
    ).lower()
    if status in {"passed", "capability_limited_passed", "quality_passed"}:
        return "passed"
    if status == "partial_passed":
        return "partial_passed"
    if status == "negative_control_passed":
        return "negative_control_passed"
    return status


def _row_final_status(row: dict[str, Any]) -> str:
    return str(
        row.get("final_step3_1_status")
        or row.get("runtime_quality_status")
        or row.get("quality_classification")
        or ""
    ).lower()


def _is_runtime_quality_pass_row(row: dict[str, Any]) -> bool:
    return _is_pass_status(_row_final_status(row))


def _is_pass_status(status: Any) -> bool:
    return str(status or "").lower() in RUNTIME_QUALITY_PASS_STATUSES


def _is_failure_status(status: Any) -> bool:
    return str(status or "").lower() in RUNTIME_QUALITY_FAILURE_STATUSES


def _unique_smoke_model_count(generic_rows: list[dict[str, Any]], predicate) -> int:
    return len(
        {
            _model_id(row)
            for row in generic_rows
            if _model_id(row) and predicate(_smoke_status(row))
        }
    )


def _smoke_status(row: dict[str, Any]) -> str:
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    return str(
        row.get("runtime_quality_status")
        or row.get("quality_classification")
        or summary.get("status")
        or row.get("status")
        or ""
    ).lower()


def _smoke_mode(row: dict[str, Any]) -> str:
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    return str(summary.get("mode") or row.get("mode") or "").lower()


def _is_residual_only_smoke(row: dict[str, Any]) -> bool:
    mode = _smoke_mode(row)
    if mode in RESIDUAL_ONLY_SMOKE_MODES or "fk_residual" in mode:
        return True
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    residuals = summary.get("residuals") if isinstance(summary.get("residuals"), dict) else {}
    solver = str(residuals.get("solver") or "").lower()
    return solver in {"runtime_model_fk_residual_evaluation", "runtime_model_fk_residual_evaluation_only"}


def _smoke_residual_metric_sample(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("smoke_summary") if isinstance(row.get("smoke_summary"), dict) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    keys = (
        "normalized_task_residual_mean",
        "normalized_task_residual_p95",
        "task_residual_mean",
        "task_residual_p95",
        "joint_limit_violation_count",
        "max_joint_limit_violation",
        "solver_iteration_mean",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _quality_pass_problem_models(rows: list[dict[str, Any]], generic_rows: list[dict[str, Any]]) -> set[str]:
    residual_only_models = {_model_id(row) for row in generic_rows if _is_residual_only_smoke(row)}
    solver_backed_models = {
        _model_id(row)
        for row in generic_rows
        if _model_id(row) and not _is_residual_only_smoke(row) and _is_pass_status(_smoke_status(row))
    }
    problem_models: set[str] = set()
    for row in rows:
        model_id = _model_id(row)
        if not model_id or not _is_runtime_quality_pass_row(row):
            continue
        joint_limit_count = _as_int(row.get("joint_limit_violation_count"))
        max_violation = _as_float(row.get("max_joint_limit_violation"))
        if joint_limit_count not in (None, 0) or (max_violation is not None and max_violation > 0):
            problem_models.add(model_id)
        if model_id in residual_only_models and model_id not in solver_backed_models:
            problem_models.add(model_id)
    return problem_models


def _model_id(row: dict[str, Any]) -> str:
    return str(row.get("model_id") or row.get("profile_model_id") or row.get("robot_id") or "").strip()


def _is_rpo_or_g1(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("model_id", "profile_model_id", "robot_id", "robot_type", "target_type")
    )
    return "roboparty_rpo" in text or "unitree_g1" in text


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


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
    relevant_names = set(REQUIRED_FINAL_HEAD_CI_JOBS) | {"final-head-ci-evidence"}
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
        "User-Agent": "soma-retargeter-step3-audit",
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


def _handoff_has_final_head_ci(text: str) -> bool:
    lower = text.lower()
    return (
        bool(re.search(r"workflow_run_id\s*[:=]\s*`?[1-9][0-9]{5,}`?", text, flags=re.IGNORECASE))
        and bool(re.search(r"(?:head_sha|head sha|final_head)\s*[:=]\s*`?[0-9a-f]{40}`?", text, flags=re.IGNORECASE))
        and ("conclusion=success" in lower or re.search(r"conclusion\s*[:=]\s*`?success`?", lower) is not None)
        and "job conclusion" in lower
        and "success" in lower
    )


def _handoff_has_live_final_head_ci_policy(text: str) -> bool:
    lower = text.lower()
    return (
        "live final-head ci" in lower
        and "github check-runs" in lower
        and "--require-final-head-ci" in lower
        and "current head" in lower
        and "job conclusions" in lower
    )


def _is_concrete_sha(value: Any) -> bool:
    return isinstance(value, str) and CONCRETE_SHA_RE.fullmatch(value.strip()) is not None


def _is_concrete_run_id(value: Any) -> bool:
    return isinstance(value, (str, int)) and CONCRETE_RUN_ID_RE.fullmatch(str(value).strip()) is not None


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _finding(gate: str, subject: str, message: str, evidence: dict[str, Any]) -> Finding:
    return Finding(gate=gate, severity="error", subject=subject, message=message, evidence=evidence)


def _row_subject(index: int, row: dict[str, Any]) -> str:
    return f"matrix[{index}] {_model_id(row) or 'unknown_model'}"


def _is_pass(value: Any) -> bool:
    return value is True or str(value).lower() in {"pass", "passed", "true", "ok"}


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


def _extract_verdict(text: str) -> str | None:
    match = re.search(r"\bverdict\s*[:=]\s*(PASS|BLOCKED)\b", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bStatus\s*[:=]\s*(PASS|BLOCKED)\b", text, flags=re.IGNORECASE)
    return _normalize_verdict(match.group(1)) if match else None


def _normalize_verdict(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in {"PASS", "BLOCKED"} else None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _sanitize_audit_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_audit_payload(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_audit_payload(child) for child in value]
    if isinstance(value, str):
        return LOCAL_ABSOLUTE_PATH_RE.sub(_sanitize_path_match, value)
    return value


def _sanitize_path_match(match: re.Match[str]) -> str:
    path = match.group(0)
    name = Path(path).name
    return f"${{LOCAL_SOURCE_PATH}}/{name}" if name else "${LOCAL_SOURCE_PATH}"


if __name__ == "__main__":
    raise SystemExit(main())
