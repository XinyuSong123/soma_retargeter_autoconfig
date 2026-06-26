#!/usr/bin/env python3
"""Independent red-team audit for Step 3 runtime-shadow acceptance evidence.

The audit is read-only. It checks the Step 3 artifact tree, CI workflow, and
Agent F verdict documents for false-positive patterns called out in goal.md.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step3_runtime_shadow")
DEFAULT_SOURCE_ROOT = Path(".")
STEP2_ARTIFACT_DIRS = (
    "artifacts/retargeting_v3_step2",
    "artifacts/retargeting_v3_step2_numerical",
    "artifacts/retargeting_v3_step2_capability",
)
WORKFLOW_PATH = Path(".github/workflows/retargeting_v3_step3_runtime_shadow.yml")
AGENT_F_HANDOFF = Path("docs/retargeting_v3/subagents/step3_agent_f_red_team.md")
ACCEPTANCE_DOC = Path("docs/retargeting_v3/STEP3_RUNTIME_SHADOW_ACCEPTANCE.md")
DEFAULT_OUTPUT_DIFF_TOLERANCE = 1e-12
REQUIRED_RUNTIME_ROBOTS = ("roboparty_rpo", "unitree_g1")
REQUIRED_RUNTIME_CLIPS = ("Neutral_walk_forward_002__A057", "wave_R_001__A428")
REQUIRED_CI_JOBS = (
    "runtime-shadow-unit-tests",
    "runtime-shadow-lfs-smoke",
    "runtime-shadow-artifact-audit",
)
REQUIRED_ARTIFACT_FILES = (
    "environment.json",
    "commands.txt",
    "profile_resolution.json",
    "shadow_summary.json",
    "override_smoke_summary.json",
    "acceptance_ledger.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/pytest_summary.json",
)
GOAL_FALSE_POSITIVE_GATES = (
    "default_output_changed",
    "shadow_mutates_ik_inputs",
    "override_without_explicit_config",
    "fingerprint_mismatch_silent",
    "partial_or_negative_profile_override",
    "lfs_pointer_profile_or_snapshot",
    "missing_diagnostics",
    "diagnostics_nan_inf",
    "local_absolute_path_leakage",
    "step2_artifacts_mutated",
    "missing_ci",
    "stale_agent_f_verdict",
)
GOAL_FALSE_POSITIVE_TRACEABILITY = {
    "fp01_default_output_changed": ("default_output_changed",),
    "fp02_shadow_mutates_ik_inputs": ("shadow_mutates_ik_inputs",),
    "fp03_override_enabled_without_explicit_config": ("override_without_explicit_config",),
    "fp04_fingerprint_mismatch_silently_accepted": ("fingerprint_mismatch_silent",),
    "fp05_partial_or_negative_profile_used_for_override": ("partial_or_negative_profile_override",),
    "fp06_lfs_pointer_profile_or_snapshot": ("lfs_pointer_profile_or_snapshot",),
    "fp07_missing_diagnostics": ("missing_diagnostics",),
    "fp08_diagnostics_nan_or_inf": ("diagnostics_nan_inf",),
    "fp09_local_absolute_path_leakage": ("local_absolute_path_leakage",),
    "fp10_step2_artifacts_mutated": ("step2_artifacts_mutated",),
    "fp11_no_ci": ("missing_ci",),
    "fp12_stale_agent_f_pass_fail_mismatch": ("stale_agent_f_verdict",),
}
LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w$])/(?:mnt|home|Users|tmp|var|private/var)/[^\s\"'<>),\]}`:]+"
)
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
OVERRIDE_FORBIDDEN_PROFILE_STATUSES = {
    "partial",
    "partial_passed",
    "negative",
    "negative_control",
    "negative_control_passed",
}
OVERRIDE_ALLOWED_PROFILE_STATUSES = {"passed"}
FAIL_CLOSED_STATUSES = {"fail_closed", "failed", "blocked", "shadow_fingerprint_skip"}


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
    finding_count: int
    blocking_count: int
    gate_counts: dict[str, int]
    findings: list[Finding]

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    def to_json(self) -> dict[str, Any]:
        return _sanitize_audit_payload(asdict(self))


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    *,
    skip_git_checks: bool = False,
) -> AuditResult:
    artifact_dir = artifact_dir.resolve()
    source_root = source_root.resolve()
    findings: list[Finding] = []

    environment = _read_json(artifact_dir / "environment.json")
    profile_resolution = _read_json(artifact_dir / "profile_resolution.json")
    shadow_summary = _read_json(artifact_dir / "shadow_summary.json")
    override_summary = _read_json(artifact_dir / "override_smoke_summary.json")
    acceptance_ledger = _read_json(artifact_dir / "acceptance_ledger.json")

    findings.extend(_audit_required_diagnostics(artifact_dir, shadow_summary))
    findings.extend(_audit_shadow_noop(shadow_summary, artifact_dir))
    findings.extend(_audit_override_summary(override_summary))
    findings.extend(_audit_profile_resolution(profile_resolution, source_root))
    findings.extend(_audit_nonfinite_diagnostics(artifact_dir))
    findings.extend(_audit_local_absolute_paths(artifact_dir, source_root))
    findings.extend(_audit_step2_artifact_mutation(environment, source_root, skip_git_checks=skip_git_checks))
    findings.extend(_audit_ci(source_root))

    pre_verdict_status = "PASS" if not [f for f in findings if f.severity == "error"] else "BLOCKED"
    findings.extend(_audit_agent_f_verdicts(source_root, acceptance_ledger, pre_verdict_status))

    gate_counts = {gate: 0 for gate in GOAL_FALSE_POSITIVE_GATES}
    for finding in findings:
        gate_counts[finding.gate] = gate_counts.get(finding.gate, 0) + 1
    blocking_count = len([finding for finding in findings if finding.severity == "error"])
    return AuditResult(
        status="PASS" if blocking_count == 0 else "BLOCKED",
        artifact_dir=_display_path(artifact_dir),
        source_root=_display_path(source_root),
        finding_count=len(findings),
        blocking_count=blocking_count,
        gate_counts=gate_counts,
        findings=findings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--skip-git-checks", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--write-report", dest="output_json")
    args = parser.parse_args(argv)

    result = run_audit(
        artifact_dir=Path(args.artifact_dir),
        source_root=Path(args.source_root),
        skip_git_checks=args.skip_git_checks,
    )
    payload = result.to_json()
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.status == "PASS" else 1


def _audit_required_diagnostics(artifact_dir: Path, shadow_summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if not artifact_dir.exists():
        return [
            _finding(
                "missing_diagnostics",
                str(artifact_dir),
                "Step 3 runtime-shadow artifact directory is missing",
                {"required_dir": _display_path(artifact_dir)},
            )
        ]
    for relative in REQUIRED_ARTIFACT_FILES:
        path = artifact_dir / relative
        if not path.exists():
            findings.append(
                _finding(
                    "missing_diagnostics",
                    relative,
                    "required Step 3 artifact is missing",
                    {"required_artifact": relative},
                )
            )
    per_clip = artifact_dir / "per_clip"
    if not per_clip.is_dir():
        findings.append(
            _finding(
                "missing_diagnostics",
                "per_clip",
                "required per-clip diagnostics directory is missing",
                {"required_dir": "per_clip"},
            )
        )
    for robot in REQUIRED_RUNTIME_ROBOTS:
        for clip in REQUIRED_RUNTIME_CLIPS:
            for name in ("target_deltas.json", "pipeline_summary.json"):
                relative = Path("per_clip") / robot / clip / name
                if not (artifact_dir / relative).exists():
                    findings.append(
                        _finding(
                            "missing_diagnostics",
                            str(relative),
                            "required per-clip runtime diagnostic is missing",
                            {"robot_type": robot, "clip_name": clip, "required_artifact": str(relative)},
                        )
                    )
    shadow_entries = _shadow_entries(shadow_summary)
    present = {(str(row.get("robot_type")), _clip_key(row.get("clip_name"))) for row in shadow_entries}
    for robot in REQUIRED_RUNTIME_ROBOTS:
        for clip in REQUIRED_RUNTIME_CLIPS:
            if (robot, clip) not in present:
                findings.append(
                    _finding(
                        "missing_diagnostics",
                        f"shadow_summary:{robot}:{clip}",
                        "shadow summary is missing a required robot/clip row",
                        {"robot_type": robot, "clip_name": clip},
                    )
                )
    return findings


def _audit_shadow_noop(shadow_summary: dict[str, Any], artifact_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(_shadow_entries(shadow_summary)):
        subject = _row_subject("shadow_summary", index, row)
        row_status = str(row.get("status") or row.get("resolution_status") or "").lower()
        row_blocked = row_status in FAIL_CLOSED_STATUSES or row_status == "blocked"
        if row_blocked:
            findings.append(
                _finding(
                    "missing_diagnostics",
                    subject,
                    "shadow row is blocked and does not prove runtime shadow behavior",
                    {"row": _row_identity(row), "reason": row.get("reason")},
                )
            )
            continue
        if row.get("ik_inputs_equal_to_disabled") is False or row.get("input_targets_equal_to_disabled") is False:
            findings.append(
                _finding(
                    "shadow_mutates_ik_inputs",
                    subject,
                    "shadow mode mutated IK inputs instead of remaining read-only",
                    {"row": _row_identity(row)},
                )
            )
        if not _is_true(row.get("output_equal_to_disabled_baseline")):
            findings.append(
                _finding(
                    "default_output_changed",
                    subject,
                    "shadow output is not equal to disabled baseline",
                    {"row": _row_identity(row), "output_diff_max": row.get("output_diff_max")},
                )
            )
        output_diff = _as_float(row.get("output_diff_max"))
        if output_diff is not None and output_diff > DEFAULT_OUTPUT_DIFF_TOLERANCE:
            findings.append(
                _finding(
                    "default_output_changed",
                    subject,
                    "shadow output diff exceeds default no-op tolerance",
                    {
                        "row": _row_identity(row),
                        "output_diff_max": output_diff,
                        "tolerance": DEFAULT_OUTPUT_DIFF_TOLERANCE,
                    },
                )
            )
        if row.get("diagnostics_written") is False:
            findings.append(
                _finding(
                    "missing_diagnostics",
                    subject,
                    "shadow row reports diagnostics were not written",
                    {"row": _row_identity(row)},
                )
            )
    for path in artifact_dir.glob("per_clip/*/*/pipeline_summary.json"):
        payload = _read_json(path)
        for summary in _mode_payloads(payload, "shadow"):
            subject = _display_path(path)
            row_status = str(summary.get("status") or "").lower()
            if row_status in FAIL_CLOSED_STATUSES or row_status == "blocked":
                findings.append(
                    _finding(
                        "missing_diagnostics",
                        subject,
                        "per-clip pipeline summary shadow mode is blocked",
                        {"reason": summary.get("reason")},
                    )
                )
                continue
            if not _is_true(summary.get("output_equal_to_disabled_baseline")):
                findings.append(
                    _finding(
                        "default_output_changed",
                        subject,
                        "per-clip pipeline summary reports shadow output changed",
                        {"output_diff_max": summary.get("output_diff_max")},
                    )
                )
            output_diff = _as_float(summary.get("output_diff_max"))
            if output_diff is not None and output_diff > DEFAULT_OUTPUT_DIFF_TOLERANCE:
                findings.append(
                    _finding(
                        "default_output_changed",
                        subject,
                        "per-clip pipeline summary output diff exceeds tolerance",
                        {"output_diff_max": output_diff, "tolerance": DEFAULT_OUTPUT_DIFF_TOLERANCE},
                    )
                )
            for key in ("nan_count", "inf_count"):
                count = _as_int(summary.get(key))
                if count and count > 0:
                    findings.append(
                        _finding(
                            "diagnostics_nan_inf",
                            subject,
                            f"pipeline summary reports non-finite output via {key}",
                            {key: count},
                        )
                    )
    return findings


def _audit_override_summary(override_summary: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    rpo_override_clips: set[str] = set()
    for index, row in enumerate(_override_entries(override_summary)):
        if str(row.get("mode")) != "override_experimental":
            continue
        subject = _row_subject("override_smoke_summary", index, row)
        resolution_status = str(row.get("resolution_status") or row.get("status") or "").lower()
        row_blocked = resolution_status in FAIL_CLOSED_STATUSES or resolution_status == "blocked"
        if row_blocked and str(row.get("robot_type")) == "roboparty_rpo":
            findings.append(
                _finding(
                    "missing_diagnostics",
                    subject,
                    "RPO override smoke row is blocked and does not prove finite override behavior",
                    {"row": _row_identity(row), "reason": row.get("reason")},
                )
            )
            rpo_override_clips.add(_clip_key(row.get("clip_name")))
            continue
        if row_blocked:
            continue
        if row.get("config_explicit") is not True and row.get("explicit_config") is not True:
            findings.append(
                _finding(
                    "override_without_explicit_config",
                    subject,
                    "override_experimental mode was enabled without explicit config evidence",
                    {"row": _row_identity(row)},
                )
            )
        profile_status = str(row.get("profile_status", "")).lower()
        forbidden_profile_status = profile_status in OVERRIDE_FORBIDDEN_PROFILE_STATUSES
        if forbidden_profile_status:
            findings.append(
                _finding(
                    "partial_or_negative_profile_override",
                    subject,
                    "partial or negative-control profile appears in override evidence",
                    {"profile_status": row.get("profile_status"), "row": _row_identity(row)},
                )
            )
        if (
            not forbidden_profile_status
            and resolution_status not in FAIL_CLOSED_STATUSES
            and profile_status
            and profile_status not in OVERRIDE_ALLOWED_PROFILE_STATUSES
        ):
            findings.append(
                _finding(
                    "partial_or_negative_profile_override",
                    subject,
                    "override used a profile status outside the allowed pass set",
                    {"profile_status": row.get("profile_status"), "row": _row_identity(row)},
                )
            )
        if str(row.get("robot_type")) == "roboparty_rpo" and resolution_status not in FAIL_CLOSED_STATUSES:
            if row.get("output_finite") is not True:
                findings.append(
                    _finding(
                        "diagnostics_nan_inf",
                        subject,
                        "RPO override smoke did not prove finite output",
                        {"row": _row_identity(row)},
                    )
                )
            if row.get("diagnostics_written") is not True:
                findings.append(
                    _finding(
                        "missing_diagnostics",
                        subject,
                        "RPO override smoke did not write diagnostics",
                        {"row": _row_identity(row)},
                    )
                )
            if row.get("experimental_label") is not True:
                findings.append(
                    _finding(
                        "override_without_explicit_config",
                        subject,
                        "override smoke is not clearly labeled experimental",
                        {"row": _row_identity(row)},
                    )
                )
            rpo_override_clips.add(_clip_key(row.get("clip_name")))
    for clip in REQUIRED_RUNTIME_CLIPS:
        if clip not in rpo_override_clips:
            findings.append(
                _finding(
                    "missing_diagnostics",
                    f"override_smoke_summary:roboparty_rpo:{clip}",
                    "required RPO override smoke row is missing",
                    {"robot_type": "roboparty_rpo", "clip_name": clip},
                )
            )
    return findings


def _audit_profile_resolution(profile_resolution: dict[str, Any], source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for index, row in enumerate(_profile_rows(profile_resolution)):
        subject = _row_subject("profile_resolution", index, row)
        fingerprint_match = row.get("fingerprint_match")
        source_hash_match = row.get("source_hash_match")
        mismatch = fingerprint_match is False or source_hash_match is False
        resolution_status = str(row.get("resolution_status", "")).lower()
        warnings = row.get("warnings") if isinstance(row.get("warnings"), list) else []
        errors = row.get("errors") if isinstance(row.get("errors"), list) else []
        reason_text = " ".join(str(value).lower() for value in [resolution_status, row.get("reason"), *warnings, *errors])
        if mismatch:
            silent = "mismatch" not in reason_text and not errors
            accepted = resolution_status in {"passed", "accepted", "override_enabled", "override_passed", "success"}
            strict_violation = row.get("strict_match_required") is True and resolution_status not in FAIL_CLOSED_STATUSES
            if silent or accepted or strict_violation:
                findings.append(
                    _finding(
                        "fingerprint_mismatch_silent",
                        subject,
                        "runtime/profile fingerprint mismatch was accepted or not explicitly recorded",
                        {
                            "row": _row_identity(row),
                            "fingerprint_match": fingerprint_match,
                            "source_hash_match": source_hash_match,
                            "resolution_status": row.get("resolution_status"),
                        },
                    )
                )
        for key in ("profile_artifact_path", "runtime_mjcf_path", "runtime_snapshot_path"):
            rel = row.get(key)
            if not rel:
                continue
            path = _resolve_source_path(source_root, str(rel))
            if _is_lfs_pointer_file(path):
                findings.append(
                    _finding(
                        "lfs_pointer_profile_or_snapshot",
                        subject,
                        f"{key} is a Git LFS pointer, not a materialized file",
                        {"field": key, "path": _display_path(path)},
                    )
                )
        if row.get("is_lfs_pointer") is True or row.get("profile_is_lfs_pointer") is True:
            findings.append(
                _finding(
                    "lfs_pointer_profile_or_snapshot",
                    subject,
                    "profile resolution metadata reports an LFS pointer",
                    {"row": _row_identity(row)},
                )
            )
    return findings


def _audit_nonfinite_diagnostics(artifact_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not artifact_dir.exists():
        return findings
    for path in artifact_dir.rglob("*.json"):
        payload = _read_json(path)
        for pointer, value in _iter_nonfinite(payload):
            findings.append(
                _finding(
                    "diagnostics_nan_inf",
                    _display_path(path),
                    "diagnostic JSON contains NaN or Inf",
                    {"json_pointer": pointer, "value": repr(value)},
                )
            )
    for path in artifact_dir.glob("per_clip/*/*/target_deltas.json"):
        payload = _read_json(path)
        for semantic, metrics in _per_semantic(payload).items():
            nan_count = _as_int(metrics.get("nan_count")) if isinstance(metrics, dict) else None
            finite_count = _as_int(metrics.get("finite_count")) if isinstance(metrics, dict) else None
            if nan_count and nan_count > 0:
                findings.append(
                    _finding(
                        "diagnostics_nan_inf",
                        _display_path(path),
                        "target delta diagnostics report NaN samples",
                        {"semantic": semantic, "nan_count": nan_count},
                    )
                )
            if finite_count is not None and finite_count < 0:
                findings.append(
                    _finding(
                        "diagnostics_nan_inf",
                        _display_path(path),
                        "target delta diagnostics report a negative finite_count",
                        {"semantic": semantic, "finite_count": finite_count},
                    )
                )
    return findings


def _audit_local_absolute_paths(artifact_dir: Path, source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    candidate_files: list[Path] = []
    if artifact_dir.exists():
        candidate_files.extend(path for path in artifact_dir.rglob("*") if path.is_file())
    for relative in (WORKFLOW_PATH, AGENT_F_HANDOFF, ACCEPTANCE_DOC):
        path = source_root / relative
        if path.exists():
            candidate_files.append(path)
    for path in candidate_files:
        text = _read_text(path)
        for match in LOCAL_ABSOLUTE_PATH_RE.finditer(text):
            findings.append(
                _finding(
                    "local_absolute_path_leakage",
                    _display_path(path),
                    "local absolute path leaked into Step 3 evidence",
                    {"path": match.group(0)},
                )
            )
    return findings


def _audit_step2_artifact_mutation(
    environment: dict[str, Any],
    source_root: Path,
    *,
    skip_git_checks: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    status_text = str(environment.get("step2_artifact_status_short") or environment.get("step2_git_status_short") or "")
    for line in _nonempty_lines(status_text):
        findings.append(
            _finding(
                "step2_artifacts_mutated",
                "environment.json",
                "environment reports Step 2 artifact mutations",
                {"status_line": line},
            )
        )
    if skip_git_checks:
        return findings
    git_dir = source_root / ".git"
    if not git_dir.exists():
        return findings
    result = subprocess.run(
        ["git", "status", "--short", *STEP2_ARTIFACT_DIRS],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        findings.append(
            _finding(
                "step2_artifacts_mutated",
                "git status",
                "could not verify Step 2 artifact cleanliness",
                {"returncode": result.returncode, "stderr": result.stderr.strip()},
            )
        )
        return findings
    for line in _nonempty_lines(result.stdout):
        findings.append(
            _finding(
                "step2_artifacts_mutated",
                "git status --short Step 2 artifacts",
                "Step 2 artifact path is modified in the worktree",
                {"status_line": line},
            )
        )
    return findings


def _audit_ci(source_root: Path) -> list[Finding]:
    workflow = source_root / WORKFLOW_PATH
    if not workflow.exists():
        return [
            _finding(
                "missing_ci",
                str(WORKFLOW_PATH),
                "Step 3 runtime-shadow CI workflow is missing",
                {"required_workflow": str(WORKFLOW_PATH)},
            )
        ]
    text = _read_text(workflow)
    findings: list[Finding] = []
    for job in REQUIRED_CI_JOBS:
        if not re.search(rf"^\s*{re.escape(job)}\s*:", text, flags=re.MULTILINE):
            findings.append(
                _finding(
                    "missing_ci",
                    str(WORKFLOW_PATH),
                    "Step 3 CI workflow is missing a required job",
                    {"required_job": job},
                )
            )
    return findings


def _audit_agent_f_verdicts(
    source_root: Path,
    acceptance_ledger: dict[str, Any],
    expected_status_before_verdict_check: str,
) -> list[Finding]:
    findings: list[Finding] = []
    declared: list[tuple[str, str | None]] = [
        ("acceptance_ledger.json", _normalize_verdict(acceptance_ledger.get("verdict") or acceptance_ledger.get("status"))),
    ]
    for relative in (AGENT_F_HANDOFF, ACCEPTANCE_DOC):
        path = source_root / relative
        declared.append((str(relative), _extract_verdict(_read_text(path))))
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


def _shadow_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _entry_list(payload, ("matrix", "shadow_matrix", "rows", "clips"))


def _override_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _entry_list(payload, ("smoke_matrix", "matrix", "rows", "clips"))


def _profile_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("profiles"), dict):
        return [row for row in payload["profiles"].values() if isinstance(row, dict)]
    if isinstance(payload.get("robots"), dict):
        return [row for row in payload["robots"].values() if isinstance(row, dict)]
    return _entry_list(payload, ("profiles", "rows", "resolutions", "robots"))


def _entry_list(payload: dict[str, Any], keys: Iterable[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            return [row for row in value.values() if isinstance(row, dict)]
    return []


def _per_semantic(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("per_semantic") if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else {}


def _mode_payloads(payload: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if payload.get("mode") == mode:
        return [payload]
    modes = payload.get("modes")
    if isinstance(modes, dict) and isinstance(modes.get(mode), dict):
        return [modes[mode]]
    return []


def _clip_key(value: Any) -> str:
    text = str(value or "")
    path = Path(text)
    if path.suffix:
        return path.stem
    return text


def _iter_nonfinite(value: Any, pointer: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, float) and not math.isfinite(value):
        yield pointer or "/", value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_nonfinite(child, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_nonfinite(child, f"{pointer}/{index}")


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


def _is_lfs_pointer_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            prefix = handle.read(128)
    except OSError:
        return False
    return prefix.decode("utf-8", errors="ignore").startswith(LFS_POINTER_PREFIX)


def _resolve_source_path(source_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return source_root / path


def _finding(gate: str, subject: str, message: str, evidence: dict[str, Any]) -> Finding:
    return Finding(gate=gate, severity="error", subject=subject, message=message, evidence=evidence)


def _row_subject(prefix: str, index: int, row: dict[str, Any]) -> str:
    robot = row.get("robot_type", "unknown_robot")
    clip = row.get("clip_name", row.get("mode", "unknown_clip"))
    return f"{prefix}[{index}] {robot} {clip}"


def _row_identity(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "robot_type",
        "clip_name",
        "mode",
        "profile_model_id",
        "profile_status",
        "resolution_status",
    )
    return {key: row.get(key) for key in keys if key in row}


def _is_true(value: Any) -> bool:
    return value is True or value == "true" or value == "passed"


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


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


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
