#!/usr/bin/env python3
"""Agent F style audit for Step 3.1 full-fleet runtime-quality evidence.

The audit is read-only. It validates that the Step 3.1 artifact tree is a real
44-model full-fleet acceptance record, not a replay of the Step 3.0 RPO/G1
runtime-shadow smoke evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
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
    "full_fleet_matrix.json",
    "quality_summary.json",
    "pipeline_controls.json",
    "acceptance_ledger.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/pytest_summary.json",
)
ACCEPTANCE_GATES = (
    "missing_required_artifacts",
    "matrix_shape",
    "status_counts_32_3_9",
    "non_rpo_g1_full_fleet",
    "negative_controls_not_promoted",
    "quality_numeric_fields",
    "absolute_path_leakage",
    "pipeline_controls_present",
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
ALLOWED_FULL_REPO_PYTEST_CLASSIFICATIONS = {
    "not_run_scoped_caveat",
    "full_repo_passed",
    "full_repo_failed",
}
LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w$])/(?:mnt|home|Users|tmp|var|private/var)/[^\s\"'<>),\]}`:]+"
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
    source_root: str
    matrix_row_count: int
    status_counts: dict[str, int]
    finding_count: int
    blocking_count: int
    gate_counts: dict[str, int]
    findings: list[Finding]
    full_repo_pytest: dict[str, Any]

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    def to_json(self) -> dict[str, Any]:
        return _sanitize_audit_payload(asdict(self))


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> AuditResult:
    artifact_dir = artifact_dir.resolve()
    source_root = source_root.resolve()
    findings: list[Finding] = []

    matrix_payload = _read_json(artifact_dir / "full_fleet_matrix.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    controls = _read_json(artifact_dir / "pipeline_controls.json")
    acceptance_ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    rows = _matrix_rows(matrix_payload)

    findings.extend(_audit_required_artifacts(artifact_dir))
    findings.extend(_audit_matrix_shape(rows, summary))
    findings.extend(_audit_status_counts(rows, summary))
    findings.extend(_audit_non_rpo_g1_full_fleet(rows, summary))
    findings.extend(_audit_negative_controls_not_promoted(rows))
    findings.extend(_audit_quality_numeric_fields(rows))
    findings.extend(_audit_absolute_paths(artifact_dir, source_root))
    findings.extend(_audit_pipeline_controls(rows, controls))
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-json")
    parser.add_argument("--write-report", dest="output_json")
    args = parser.parse_args(argv)

    result = run_audit(artifact_dir=Path(args.artifact_dir), source_root=Path(args.source_root))
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
        modes = row.get("control_modes") or row.get("pipeline_control_modes") or []
        if not isinstance(modes, list) or not {"disabled", "shadow"}.issubset({str(mode) for mode in modes}):
            findings.append(
                _finding(
                    "pipeline_controls_present",
                    subject,
                    "matrix row must record disabled and shadow pipeline controls",
                    {"control_modes": modes},
                )
            )
        for flag in REQUIRED_ROW_CONTROL_FLAGS:
            if not _is_pass(row.get(flag)):
                findings.append(
                    _finding(
                        "pipeline_controls_present",
                        subject,
                        "matrix row is missing a required pipeline-control pass flag",
                        {"flag": flag, "value": row.get(flag)},
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
