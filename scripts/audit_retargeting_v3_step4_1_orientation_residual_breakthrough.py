#!/usr/bin/env python3
"""Audit Step 4.1 orientation residual breakthrough artifacts."""

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


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_1_orientation_residual_breakthrough")
DEFAULT_BASELINE_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step4_full_pipeline_acceptance")
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
    "quality_delta_vs_step4_0.json",
    "orientation_delta_vs_step4_0.json",
    "release_candidate_impact_report.json",
    "orientation_frame_semantics_matrix.json",
    "orientation_residual_math_audit.json",
    "orientation_offset_candidate_matrix.json",
    "orientation_policy_selection.json",
    "orientation_clip_consistency_matrix.json",
    "normalization_audit.json",
    "pipeline_config.json",
    "solver_config.json",
    "trajectory_export_manifest.json",
    "temporal_continuity_matrix.json",
    "support_contact_diagnostics.json",
    "collision_proxy_diagnostics.json",
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
    "BLOCKED_ORIENTATION_SEMANTICS",
    "BLOCKED_TARGET_FRAME_AMBIGUITY",
    "BLOCKED_NORMALIZATION_INTEGRITY",
    "BLOCKED_PIPELINE_REGRESSION",
    "BLOCKED_CI_OR_PROVENANCE",
}
ACCEPTABLE_BLOCKED_QUALITY_STATUSES = {
    "BLOCKED_ORIENTATION_SEMANTICS",
    "BLOCKED_TARGET_FRAME_AMBIGUITY",
    "BLOCKED_NORMALIZATION_INTEGRITY",
}
REQUIRED_FINAL_HEAD_CI_JOBS = (
    "step4-1-static-and-unit",
    "step4-1-artifact-audit",
    "step4-1-orientation-math-smoke",
    "step4-1-export-and-temporal-smoke",
    "step4-1-lfs-and-snapshot-smoke",
    "step4-1-pipeline-controls-reference",
)
SCAN_PATHS = (
    "soma_retargeter/runtime/v3",
    "soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py",
    "soma_retargeter/tools/run_v3_full_pipeline_acceptance.py",
    "soma_retargeter/tools/step4_1_orientation_residual.py",
    "scripts/audit_retargeting_v3_step4_1_orientation_residual_breakthrough.py",
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

    environment = _read_json(artifact_dir / "environment.json")
    summary = _read_json(artifact_dir / "quality_summary.json")
    ledger = _read_json(artifact_dir / "acceptance_ledger.json")
    model_matrix = _read_json(artifact_dir / "model_matrix.json")
    full_pipeline = _read_json(artifact_dir / "full_pipeline_matrix.json")
    solver_smoke = _read_json(artifact_dir / "solver_smoke_matrix.json")
    generic_smoke = _read_json(artifact_dir / "generic_smoke_matrix.json")
    deterministic = _read_json(artifact_dir / "deterministic_rerun.json")
    quality_delta = _read_json(artifact_dir / "quality_delta_vs_step4_0.json")
    orientation_delta = _read_json(artifact_dir / "orientation_delta_vs_step4_0.json")
    impact = _read_json(artifact_dir / "release_candidate_impact_report.json")
    frame_semantics = _read_json(artifact_dir / "orientation_frame_semantics_matrix.json")
    math_audit = _read_json(artifact_dir / "orientation_residual_math_audit.json")
    offset_candidates = _read_json(artifact_dir / "orientation_offset_candidate_matrix.json")
    policy_selection = _read_json(artifact_dir / "orientation_policy_selection.json")
    clip_consistency = _read_json(artifact_dir / "orientation_clip_consistency_matrix.json")
    normalization = _read_json(artifact_dir / "normalization_audit.json")
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
    findings.extend(_audit_baseline_step4_0(baseline_artifact_dir, baseline_summary, summary, ledger, quality_delta))
    findings.extend(_audit_clean_provenance(environment, summary, ledger, quality_delta, source_root))
    findings.extend(_audit_matrix_shape_and_partition(rows, full_rows, summary))
    findings.extend(_audit_solver_and_status_counts(rows, full_rows, solver_rows, summary))
    findings.extend(_audit_label_honesty(rows, full_rows, solver_rows, generic_rows))
    findings.extend(_audit_negative_and_partial(rows, full_rows))
    findings.extend(_audit_orientation_artifacts(frame_semantics, math_audit, offset_candidates, policy_selection, clip_consistency))
    findings.extend(_audit_quality_delta(quality_delta, orientation_delta, baseline_summary, summary))
    findings.extend(_audit_release_candidate_status(summary, ledger, quality_delta, orientation_delta, impact))
    findings.extend(_audit_normalization(normalization, quality_delta))
    findings.extend(_audit_exports_and_temporal(trajectory, temporal, support, collision, summary))
    findings.extend(_audit_deterministic(deterministic, summary))
    findings.extend(_audit_configs(solver_config, pipeline_config, policy_selection))
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
    parser.add_argument("--baseline-step4-artifact-dir", "--baseline-artifact-dir", dest="baseline_artifact_dir", default=str(DEFAULT_BASELINE_ARTIFACT_DIR))
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
        return [_finding("missing_required_artifacts", _display_path(artifact_dir), "Step 4.1 artifact directory is missing", {})]
    for relative in REQUIRED_ARTIFACT_FILES:
        if not (artifact_dir / relative).exists():
            findings.append(_finding("missing_required_artifacts", relative, "required Step 4.1 artifact is missing", {}))
    if not any((artifact_dir / relative).exists() for relative in PIPELINE_CONTROL_ARTIFACT_ALTERNATIVES):
        findings.append(_finding("missing_required_artifacts", "pipeline controls", "pipeline control evidence is missing", {}))
    return findings


def _audit_baseline_step4_0(
    baseline_artifact_dir: Path,
    baseline_summary: dict[str, Any],
    summary: dict[str, Any],
    ledger: dict[str, Any],
    quality_delta: dict[str, Any],
) -> list[Finding]:
    findings = []
    if not baseline_artifact_dir.exists():
        findings.append(_finding("baseline_step4_0", _display_path(baseline_artifact_dir), "Step 4.0 baseline artifact dir is missing", {}))
    expected = {
        "in_scope_total": 44,
        "full_humanoid_total": 32,
        "partial_total": 3,
        "negative_total": 9,
        "solver_backed_count": 32,
        "residual_only_count": 0,
        "runtime_quality_failed_count": 0,
        "runtime_quality_passed_count": 0,
        "high_residual_warning_count": 32,
        "rotation_dominant_residual_count": 27,
    }
    for field, expected_value in expected.items():
        if _as_int(baseline_summary.get(field)) != expected_value:
            findings.append(
                _finding(
                    "baseline_step4_0",
                    field,
                    "baseline Step 4.0 summary mismatch",
                    {"actual": baseline_summary.get(field), "expected": expected_value},
                )
            )
    base_head = summary.get("base_step4_0_final_head") or ledger.get("base_step4_0_final_head")
    if not _is_concrete_sha(base_head):
        findings.append(_finding("baseline_step4_0", "base_step4_0_final_head", "base Step 4.0 final HEAD must be concrete", {"value": base_head}))
    if quality_delta.get("baseline_counts") and _as_int(quality_delta["baseline_counts"].get("runtime_quality_passed_count")) != 0:
        findings.append(_finding("baseline_step4_0", "quality_delta_vs_step4_0.json", "baseline counts must preserve blocked Step 4.0 truth", {}))
    return findings


def _audit_clean_provenance(
    environment: dict[str, Any],
    summary: dict[str, Any],
    ledger: dict[str, Any],
    quality_delta: dict[str, Any],
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
    for subject, payload in (("quality_summary.json", summary), ("acceptance_ledger.json", ledger), ("quality_delta_vs_step4_0.json", quality_delta)):
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
    full_rows: list[dict[str, Any]],
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
            findings.append(_finding("solver_backed_counts", field, "Step 4.1 must preserve solver-backed coverage", {"rows": expected_counts[field], "summary": summary.get(field)}))
    for field, expected in (("residual_only_count", 0), ("runtime_quality_failed_count", 0)):
        if expected_counts[field] != expected or _as_int(summary.get(field)) != expected:
            findings.append(_finding("solver_backed_counts", field, "Step 4.1 invariant mismatch", {"rows": expected_counts[field], "summary": summary.get(field), "expected": expected}))
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


def _audit_label_honesty(
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
            high_residual = float(row.get("normalized_task_residual_p95", 1.0) or 1.0) > 0.15
            if high_residual and status in {"runtime_quality_passed", "runtime_evaluation_completed"}:
                findings.append(_finding("runtime_quality_label_honesty", subject, "high residual row cannot be relabeled as passed", {"status": status}))
            if status == "runtime_quality_passed":
                requirements = {
                    "solver_backed": row.get("solver_backed") is True,
                    "solver_backed_smoke_completed": row.get("solver_backed_smoke_completed") is True,
                    "residual_only_false": row.get("residual_only") is False,
                    "solver_matrix_evidence": _model_id(row) in solver_models,
                    "generic_matrix_evidence": _model_id(row) in generic_models,
                    "residual_pass_gate": float(row.get("normalized_task_residual_p95", 1.0) or 1.0) <= 0.15,
                    "output_nan_zero": _as_int(row.get("output_nan_count")) == 0,
                    "output_inf_zero": _as_int(row.get("output_inf_count")) == 0,
                }
                if not all(requirements.values()):
                    findings.append(_finding("runtime_quality_label_honesty", subject, "runtime_quality_passed row lacks gate evidence", requirements))
    return findings


def _audit_negative_and_partial(rows: list[dict[str, Any]], full_pipeline_rows: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for collection_name, collection in (("model_matrix.json", rows), ("full_pipeline_matrix.json", full_pipeline_rows)):
        for index, row in enumerate(collection):
            if row.get("category") not in {"partial_humanoid_profile", "negative_control"}:
                continue
            subject = f"{collection_name}[{index}] {_model_id(row)}"
            if _row_status(row) == "runtime_quality_passed":
                findings.append(_finding("negative_and_partial_not_promoted", subject, "partial/negative rows must not be promoted", {}))
            for flag in ("solver_backed", "solver_backed_smoke_attempted", "solver_backed_smoke_completed"):
                if row.get(flag) is True:
                    findings.append(_finding("negative_and_partial_not_promoted", subject, f"partial/negative row must not set {flag}", {"value": row.get(flag)}))
    return findings


def _audit_orientation_artifacts(
    frame_semantics: dict[str, Any],
    math_audit: dict[str, Any],
    offset_candidates: dict[str, Any],
    policy_selection: dict[str, Any],
    clip_consistency: dict[str, Any],
) -> list[Finding]:
    findings = []
    frame_rows = _matrix_rows(frame_semantics)
    math_rows = _matrix_rows(math_audit)
    candidate_rows = _matrix_rows(offset_candidates)
    clip_rows = _matrix_rows(clip_consistency)
    if len(frame_rows) < 128:
        findings.append(_finding("orientation_frame_semantics", "orientation_frame_semantics_matrix.json", "frame semantics matrix is incomplete", {"rows": len(frame_rows)}))
    if len(math_rows) < 128 or _as_int(math_audit.get("finite_log_map_count")) != len(math_rows):
        findings.append(_finding("orientation_residual_math", "orientation_residual_math_audit.json", "log-map math audit must be finite and complete", {"rows": len(math_rows), "finite": math_audit.get("finite_log_map_count")}))
    if len(candidate_rows) < 5:
        findings.append(_finding("orientation_policy_selection", "orientation_offset_candidate_matrix.json", "candidate sweep must include global policies", {"rows": len(candidate_rows)}))
    if not clip_rows:
        findings.append(_finding("orientation_clip_consistency", "orientation_clip_consistency_matrix.json", "clip consistency matrix is missing", {}))
    for index, row in enumerate(frame_rows[: min(len(frame_rows), 20)]):
        for field in (
            "model_id",
            "clip_id",
            "semantic_anchor",
            "runtime_body_name",
            "target_frame",
            "runtime_frame",
            "source_frame",
            "quaternion_order",
            "sign_canonicalized",
            "rest_offset_policy",
            "parent_relative",
            "world_relative",
            "axis_convention_notes",
            "validity_status",
            "warning_reasons",
        ):
            if field not in row:
                findings.append(_finding("orientation_frame_semantics", f"row[{index}]", "required frame-semantics field missing", {"field": field}))
        if row.get("quaternion_order") != "xyzw" or row.get("sign_canonicalized") is not True:
            findings.append(_finding("orientation_frame_semantics", f"row[{index}]", "quaternion order/sign canonicalization must be recorded", {"row": row}))
    selected = policy_selection.get("global_selected_policy") if isinstance(policy_selection.get("global_selected_policy"), dict) else {}
    if not selected or selected.get("robot_specific_tuning_used") is not False:
        findings.append(_finding("orientation_policy_selection", "orientation_policy_selection.json", "selected policy must be global and non robot-specific", {"selected": selected}))
    if selected.get("production_default_changed") is True or selected.get("runtime_override_default_enabled") is True:
        findings.append(_finding("orientation_policy_selection", "orientation_policy_selection.json", "Step 4.1 must not default-enable runtime override", {"selected": selected}))
    if not policy_selection.get("candidate_policies"):
        findings.append(_finding("orientation_policy_selection", "orientation_policy_selection.json", "candidate policies are missing", {}))
    if policy_selection.get("raw_residual_regression_count") not in {0, None}:
        findings.append(_finding("orientation_policy_selection", "orientation_policy_selection.json", "selected policy cannot hide raw residual regression", {"count": policy_selection.get("raw_residual_regression_count")}))
    return findings


def _audit_quality_delta(
    quality_delta: dict[str, Any],
    orientation_delta: dict[str, Any],
    baseline_summary: dict[str, Any],
    summary: dict[str, Any],
) -> list[Finding]:
    findings = []
    if not quality_delta:
        return [_finding("quality_delta_vs_step4_0", "quality_delta_vs_step4_0.json", "quality delta is missing", {})]
    for field in ("in_scope_total", "full_humanoid_total", "partial_total", "negative_total", "solver_backed_count", "residual_only_count", "runtime_quality_failed_count"):
        baseline_counts = quality_delta.get("baseline_counts", {})
        current_counts = quality_delta.get("current_counts", {})
        if field in baseline_counts and _as_int(baseline_counts.get(field)) != _as_int(baseline_summary.get(field)):
            findings.append(_finding("quality_delta_vs_step4_0", field, "baseline count mismatch", {"delta": baseline_counts.get(field), "baseline": baseline_summary.get(field)}))
        if field in current_counts and _as_int(current_counts.get(field)) != _as_int(summary.get(field)):
            findings.append(_finding("quality_delta_vs_step4_0", field, "current count mismatch", {"delta": current_counts.get(field), "summary": summary.get(field)}))
    if quality_delta.get("regressions"):
        findings.append(_finding("quality_delta_vs_step4_0", "regressions", "quality delta records pipeline regressions", {"regressions": quality_delta.get("regressions")}))
    if not orientation_delta:
        findings.append(_finding("orientation_delta_vs_step4_0", "orientation_delta_vs_step4_0.json", "orientation delta is missing", {}))
    if orientation_delta.get("accepted_breakthrough") is True and float(orientation_delta.get("p95_rotation_residual_p95_delta", 0.0) or 0.0) > -0.25:
        findings.append(_finding("orientation_delta_vs_step4_0", "accepted_breakthrough", "accepted orientation breakthrough is too small", {"delta": orientation_delta.get("p95_rotation_residual_p95_delta")}))
    return findings


def _audit_release_candidate_status(
    summary: dict[str, Any],
    ledger: dict[str, Any],
    quality_delta: dict[str, Any],
    orientation_delta: dict[str, Any],
    impact: dict[str, Any],
) -> list[Finding]:
    findings = []
    status = str(summary.get("release_candidate_status") or "")
    ledger_status = str(ledger.get("release_candidate_status") or "")
    if status not in VALID_RELEASE_STATUSES:
        findings.append(_finding("release_candidate_status", "quality_summary.json", "invalid Step 4.1 release status", {"status": status}))
    if ledger_status != status:
        findings.append(_finding("release_candidate_status", "acceptance_ledger.json", "ledger status must match summary", {"ledger": ledger_status, "summary": status}))
    target = _strict_quality_target(summary, quality_delta, orientation_delta)
    if status == "PASS_RC":
        requirements = {
            "primary_quality_breakthrough": summary.get("primary_quality_breakthrough") is True and quality_delta.get("primary_quality_breakthrough") is True,
            "strict_quality_target": target,
            "runtime_quality_failed_count_zero": _as_int(summary.get("runtime_quality_failed_count")) == 0,
            "no_delta_regressions": not quality_delta.get("regressions"),
            "impact_report_matches": impact.get("release_candidate_status") == "PASS_RC",
        }
        if not all(requirements.values()):
            findings.append(_finding("release_candidate_status", "quality_summary.json", "PASS_RC requirements are not met", requirements))
    elif status in ACCEPTABLE_BLOCKED_QUALITY_STATUSES:
        if _orientation_diagnostics_complete(quality_delta, orientation_delta, impact) is not True:
            findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked orientation status requires complete diagnostics", {"status": status}))
    elif status.startswith("BLOCKED") and target and not quality_delta.get("regressions"):
        findings.append(_finding("release_candidate_status", "quality_summary.json", "blocked status contradicts available breakthrough evidence", {"status": status}))
    return findings


def _audit_normalization(normalization: dict[str, Any], quality_delta: dict[str, Any]) -> list[Finding]:
    findings = []
    norm_delta = quality_delta.get("normalization_deltas") if isinstance(quality_delta.get("normalization_deltas"), dict) else {}
    if normalization.get("normalization_hides_raw_residual_regression") is True or norm_delta.get("normalization_hides_raw_regression") is True:
        findings.append(_finding("normalization_integrity", "normalization_audit.json", "normalization hides raw residual regression", {"normalization": normalization, "delta": norm_delta}))
    if normalization.get("denominator_inflation_detected") is True:
        findings.append(_finding("normalization_integrity", "normalization_audit.json", "denominator inflation detected", {"suspicious_rows": normalization.get("suspicious_rows")}))
    if normalization.get("orientation_policy_changes_normalization") is True:
        findings.append(_finding("normalization_integrity", "normalization_audit.json", "orientation policy must not change normalization gates", {}))
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


def _audit_configs(solver_config: dict[str, Any], pipeline_config: dict[str, Any], policy_selection: dict[str, Any]) -> list[Finding]:
    findings = []
    for name, payload in (("solver_config.json", solver_config), ("pipeline_config.json", pipeline_config)):
        if payload.get("robot_specific_tuning") is True or payload.get("global_config") is False:
            findings.append(_finding("global_config", name, "config must be global and non robot-specific", {"payload": payload}))
    selected = policy_selection.get("global_selected_policy", {})
    if selected and selected.get("policy") not in json.dumps(solver_config):
        findings.append(_finding("global_config", "solver_config.json", "selected orientation policy must be recorded in solver config", {"selected": selected}))
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
                if "SUSPICIOUS_ROBOT_SPECIFIC_RE" in line:
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
            findings.append(_finding("final_head_ci", job, "required Step 4.1 CI job is not successful", {"job_conclusions": job_conclusions, "committed_record": committed_record}))
    return findings


def _strict_quality_target(summary: dict[str, Any], quality_delta: dict[str, Any], orientation_delta: dict[str, Any]) -> bool:
    count_deltas = quality_delta.get("count_deltas") if isinstance(quality_delta.get("count_deltas"), dict) else {}
    metric_deltas = quality_delta.get("metric_distribution_deltas") if isinstance(quality_delta.get("metric_distribution_deltas"), dict) else {}
    raw_delta = metric_deltas.get("raw_task_residual_p95", {}).get("delta", {}) if isinstance(metric_deltas.get("raw_task_residual_p95"), dict) else {}
    return bool(
        (_as_int(summary.get("runtime_quality_passed_count")) or 0) > 0
        or (_as_int(summary.get("high_residual_warning_count")) is not None and _as_int(summary.get("high_residual_warning_count")) < 32)
        or (_as_int(summary.get("rotation_dominant_residual_count")) is not None and _as_int(summary.get("rotation_dominant_residual_count")) < 27)
        or float(orientation_delta.get("p95_rotation_residual_p95_delta", 0.0) or 0.0) <= -0.25
        or (
            float(raw_delta.get("p95", 0.0) or 0.0) <= -0.10
            and quality_delta.get("normalization_deltas", {}).get("raw_residual_regression_count", 0) == 0
        )
        or count_deltas.get("runtime_quality_passed_count", 0) > 0
    )


def _orientation_diagnostics_complete(quality_delta: dict[str, Any], orientation_delta: dict[str, Any], impact: dict[str, Any]) -> bool:
    return bool(quality_delta and orientation_delta and impact and "orientation_policy_selection" in impact)


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
    return _final_head_ci_candidates(payload.get("check_runs"), head)[0] if _final_head_ci_candidates(payload.get("check_runs"), head) else {}


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
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codex-step4-1-audit"})
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
