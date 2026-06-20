#!/usr/bin/env python3
"""Independent Step-2 acceptance audit for retargeting v3 artifacts.

This script is intentionally read-only. It audits existing JSON artifacts and
source snippets for false-positive patterns called out in ``goal.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


DEFAULT_ARTIFACT_DIR = Path("artifacts/retargeting_v3_step2")
DEFAULT_SOURCE_ROOT = Path(".")
ZERO_TOL = 1e-12
ABSOLUTE_LOCAL_PREFIXES = ("/mnt/", "/home/", "/Users/")
FULL_HUMANOID_REPORTS = (
    "roboparty_rpo",
    "unitree_g1_mjcf",
    "unitree_g1_urdf",
    "unitree_g1_23dof",
    "unitree_h1",
    "robotis_op3",
    "booster_t1",
    "pal_talos",
)
ENDPOINT_SEMANTICS = ("LeftHand", "RightHand", "LeftFoot", "RightFoot")
CANONICAL_TARGET_KEYS = (
    "canonical_targets",
    "canonical_motions",
    "canonical_motion_suite",
    "canonical_source_motions",
)
CANONICAL_PROJECTION_KEYS = (
    "canonical_projection_reports",
    "canonical_motion_projection_reports",
    "per_motion_projection_reports",
)
REQUIRED_CANONICAL_MOTIONS = (
    "neutral",
    "root_translation",
    "global_root_yaw",
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "mixed_torso_rotation",
    "arms_forward",
    "elbow_bend",
    "overhead_reach",
    "squat",
    "single_step",
    "asymmetric_arm_reach",
    "crossed_body_reach",
    "extreme_but_valid_joint_limit_stress",
)
PROFILE_PASS_STATUSES = ("passed", "partial_passed")
REQUIRED_ARTIFACT_FILES = (
    "acceptance_ledger.json",
    "test_results/pytest.txt",
    "test_results/junit.xml",
    "test_results/coverage.json",
)
REQUIRED_SUBAGENT_HANDOFFS = (
    "step2_agent_a_handoff.md",
    "step2_agent_b_handoff.md",
    "step2_agent_c_handoff.md",
    "step2_agent_d_handoff.md",
    "step2_agent_e_handoff.md",
    "step2_agent_f_red_team.md",
)
GOAL_FALSE_POSITIVE_GATES = (
    "hardcoded_zero_calibration",
    "neutral_to_neutral_fake_projection",
    "canonical_targets_without_projection",
    "zero_offset_rpo_hand_sole",
    "body_name_depth_heuristic",
    "talos_proximal_foot_mapping",
    "booster_hips_chest_alias",
    "arbitrary_g1_equivalence",
    "dirty_artifact_metadata",
    "absolute_cache_paths",
    "inferred_semantics_confidence_one",
    "rank0_false_pass",
    "robot_name_special_cases",
    "legacy_offsets",
    "missing_formal_red_team_artifacts",
    "missing_required_artifacts",
    "cross_format_gates_not_run",
    "canonical_motion_suite_incomplete",
    "epsilon_stability_failed",
)

GOAL_FALSE_POSITIVE_TRACEABILITY = {
    "fp01_neutral_fk_projection": ("neutral_to_neutral_fake_projection",),
    "fp02_hardcoded_zero_rest_calibration": ("hardcoded_zero_calibration",),
    "fp03_rpo_endpoint_body_origin": ("zero_offset_rpo_hand_sole",),
    "fp04_canonical_targets_not_projected": ("canonical_targets_without_projection",),
    "fp05_body_name_length_depth": ("body_name_depth_heuristic",),
    "fp06_inferred_semantics_as_explicit_confidence_one": ("inferred_semantics_confidence_one",),
    "fp07_known_bad_talos_booster_semantics": (
        "talos_proximal_foot_mapping",
        "booster_hips_chest_alias",
    ),
    "fp08_g1_equivalence_documented_limitation": ("arbitrary_g1_equivalence",),
    "fp09_dirty_absolute_artifacts": ("dirty_artifact_metadata", "absolute_cache_paths"),
    "fp10_missing_formal_results_handoffs_red_team": ("missing_formal_red_team_artifacts",),
    "fp11_missing_junit_pytest_coverage_artifacts": ("missing_required_artifacts",),
    "fp12_cross_format_scaffold_not_run": ("cross_format_gates_not_run",),
    "fp13_incomplete_canonical_motion_suite": ("canonical_motion_suite_incomplete",),
    "fp14_unstable_jacobian_marked_pass": ("epsilon_stability_failed",),
}


@dataclass(frozen=True)
class Finding:
    gate: str
    severity: str
    subject: str
    message: str
    evidence: dict


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

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["findings"] = [asdict(finding) for finding in self.findings]
        return _sanitize_audit_payload(payload)


def run_audit(
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    source_root: Path = DEFAULT_SOURCE_ROOT,
) -> AuditResult:
    artifact_dir = artifact_dir.resolve()
    source_root = source_root.resolve()
    findings: list[Finding] = []

    summary = _read_json(artifact_dir / "summary.json")
    cross_format = _read_json(artifact_dir / "cross_format.json")
    validation_checks = _read_json(artifact_dir / "validation_checks.json")
    environment = _read_json(artifact_dir / "environment.json")
    commands_text = _read_text(artifact_dir / "commands.txt")
    per_robot = _load_per_robot(artifact_dir / "per_robot")
    semantic_maps = _load_per_robot(artifact_dir / "semantic_maps")

    findings.extend(_audit_required_artifact_files(artifact_dir))
    findings.extend(_audit_hardcoded_zero_calibration(source_root, per_robot))
    findings.extend(_audit_neutral_projection_only(per_robot))
    findings.extend(_audit_canonical_targets_projected(per_robot))
    findings.extend(_audit_canonical_motion_suite(per_robot))
    findings.extend(_audit_rpo_zero_endpoint_sites(per_robot.get("roboparty_rpo", {}), validation_checks))
    findings.extend(_audit_body_name_depth_heuristic(source_root))
    findings.extend(_audit_talos_foot_mapping(per_robot.get("pal_talos", {}), semantic_maps.get("pal_talos", {})))
    findings.extend(_audit_booster_hips_chest(per_robot.get("booster_t1", {}), semantic_maps.get("booster_t1", {})))
    findings.extend(_audit_g1_equivalence(summary, validation_checks))
    findings.extend(_audit_cross_format_gates(cross_format or summary.get("cross_format", {})))
    findings.extend(_audit_dirty_artifact_metadata(environment, source_root))
    findings.extend(_audit_absolute_paths(commands_text, per_robot))
    findings.extend(_audit_artifact_absolute_paths(artifact_dir))
    findings.extend(_audit_inferred_semantics_confidence(per_robot, semantic_maps))
    findings.extend(_audit_rank_zero_false_pass(per_robot))
    findings.extend(_audit_epsilon_stability(per_robot))
    findings.extend(_audit_robot_name_special_cases(source_root))
    findings.extend(_audit_legacy_offsets(source_root, per_robot))
    findings.extend(_audit_formal_red_team_artifacts(source_root))

    gate_counts = {gate: 0 for gate in GOAL_FALSE_POSITIVE_GATES}
    for finding in findings:
        gate_counts[finding.gate] = gate_counts.get(finding.gate, 0) + 1
    blocking = [finding for finding in findings if finding.severity == "error"]
    return AuditResult(
        status="PASS" if not blocking else "BLOCKED",
        artifact_dir=_display_audit_path(artifact_dir),
        source_root=_display_audit_path(source_root),
        finding_count=len(findings),
        blocking_count=len(blocking),
        gate_counts=gate_counts,
        findings=findings,
    )


def _audit_hardcoded_zero_calibration(source_root: Path, per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    rest_frames = source_root / "soma_retargeter/robotics/v3/rest_frames.py"
    text = _read_text(rest_frames)
    patterns = {
        "position": r"pos_errors\s*=\s*\{\s*name\s*:\s*0\.0\s+for\s+name\s+in\s+transforms\s*\}",
        "orientation": r"rot_errors\s*=\s*\{\s*name\s*:\s*0\.0\s+for\s+name\s+in\s+transforms\s*\}",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            findings.append(
                Finding(
                    "hardcoded_zero_calibration",
                    "error",
                    str(rest_frames.relative_to(source_root)),
                    f"rest calibration {label} errors are directly initialized to zero",
                    {"pattern": pattern},
                )
            )
    for robot_id, report in sorted(per_robot.items()):
        calibration = report.get("rest_calibration", {})
        pos = calibration.get("neutral_position_errors", {})
        rot = calibration.get("neutral_orientation_errors", {})
        all_zero = pos and rot and all(_is_zero(v) for v in pos.values()) and all(_is_zero(v) for v in rot.values())
        has_measurement_evidence = any(
            key in calibration
            for key in (
                "independent_measurements",
                "measured_neutral_errors",
                "calibration_residual_samples",
                "recomputed_neutral_errors",
            )
        )
        if all_zero and not has_measurement_evidence:
            findings.append(
                Finding(
                    "hardcoded_zero_calibration",
                    "error",
                    robot_id,
                    "artifact records exact zero calibration errors without independent measurement evidence",
                    {
                        "max_position_error": calibration.get("max_position_error"),
                        "max_orientation_error": calibration.get("max_orientation_error"),
                        "site_count": len(pos),
                    },
                )
            )
    return findings


def _audit_neutral_projection_only(per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    for robot_id, report in sorted(per_robot.items()):
        projections = report.get("projection_reports", {})
        if not projections:
            continue
        equal_zero = []
        for task, projection in sorted(projections.items()):
            desired = projection.get("desired")
            projected = projection.get("projected")
            residual = projection.get("residual")
            if _vectors_close(desired, projected) and _is_zero(residual):
                equal_zero.append(task)
        canonical_projection_keys = [
            key
            for key in (
                "canonical_projection_reports",
                "canonical_motion_projection_reports",
                "per_motion_projection_reports",
            )
            if key in report
        ]
        if equal_zero and len(equal_zero) == len(projections) and not canonical_projection_keys:
            findings.append(
                Finding(
                    "neutral_to_neutral_fake_projection",
                    "error",
                    robot_id,
                    "projection reports only prove desired==projected zero-residual neutral targets",
                    {"tasks": equal_zero, "canonical_projection_keys": canonical_projection_keys},
                )
            )
    return findings


def _audit_canonical_targets_projected(per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    for robot_id, report in sorted(per_robot.items()):
        target_keys = [key for key in CANONICAL_TARGET_KEYS if key in report]
        projection_keys = [key for key in CANONICAL_PROJECTION_KEYS if key in report]
        if target_keys and not projection_keys:
            findings.append(
                Finding(
                    "canonical_targets_without_projection",
                    "error",
                    robot_id,
                    "canonical targets or motions exist without per-motion torso/hand/foot projection reports",
                    {"canonical_target_keys": target_keys, "canonical_projection_keys": projection_keys},
                )
            )
    return findings


def _audit_canonical_motion_suite(per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    required = set(REQUIRED_CANONICAL_MOTIONS)
    for robot_id, report in sorted(per_robot.items()):
        if report.get("status") not in PROFILE_PASS_STATUSES:
            continue
        motion_order = _canonical_motion_order(report)
        if not motion_order:
            findings.append(
                Finding(
                    "canonical_motion_suite_incomplete",
                    "error",
                    robot_id,
                    "passed/partial profile has no canonical motion projection order",
                    {"required_motions": list(REQUIRED_CANONICAL_MOTIONS)},
                )
            )
            continue
        present = set(motion_order)
        missing = [motion for motion in REQUIRED_CANONICAL_MOTIONS if motion not in present]
        duplicate_count = len(motion_order) - len(present)
        if missing or len(present) < len(required) or duplicate_count:
            findings.append(
                Finding(
                    "canonical_motion_suite_incomplete",
                    "error",
                    robot_id,
                    "passed/partial profile does not include the full 15-motion canonical suite",
                    {
                        "motion_order": motion_order,
                        "motion_count": len(present),
                        "required_motion_count": len(required),
                        "missing": missing,
                        "duplicate_count": duplicate_count,
                    },
                )
            )
    return findings


def _canonical_motion_order(report: dict) -> list[str]:
    for key in CANONICAL_PROJECTION_KEYS:
        payload = report.get(key)
        if not isinstance(payload, dict):
            continue
        motion_order = payload.get("motion_order")
        if isinstance(motion_order, list):
            return [str(item) for item in motion_order]
        motions = payload.get("motions")
        if isinstance(motions, dict):
            return [str(item) for item in motions]
    return []


def _audit_rpo_zero_endpoint_sites(report: dict, validation_checks: dict) -> list[Finding]:
    findings: list[Finding] = []
    sites = report.get("semantic_sites", {})
    for semantic in ENDPOINT_SEMANTICS:
        site = sites.get(semantic, {})
        if _vector_is_zero(site.get("local_position")) and site.get("reason") in {"explicit_body", None}:
            findings.append(
                Finding(
                    "zero_offset_rpo_hand_sole",
                    "error",
                    f"roboparty_rpo:{semantic}",
                    "RPO endpoint semantic site uses the body origin instead of a distal hand or sole local site",
                    {
                        "body_name": site.get("body_name"),
                        "local_position": site.get("local_position"),
                        "reason": site.get("reason"),
                    },
                )
            )
    rpo_check = validation_checks.get("roboparty_rpo_distal_hand_endpoint", {})
    if rpo_check.get("status") == "passed_with_documented_scope":
        findings.append(
            Finding(
                "zero_offset_rpo_hand_sole",
                "error",
                "validation_checks.roboparty_rpo_distal_hand_endpoint",
                "validation check treats zero-offset RPO distal endpoints as a scoped pass",
                {"status": rpo_check.get("status"), "scope_note": rpo_check.get("scope_note")},
            )
        )
    return findings


def _audit_body_name_depth_heuristic(source_root: Path) -> list[Finding]:
    semantic_sites = source_root / "soma_retargeter/robotics/v3/semantic_sites.py"
    text = _read_text(semantic_sites)
    findings: list[Finding] = []
    risky_patterns = (r"len\s*\(\s*original\s*\)", r"len\s*\(\s*body_name\s*\)")
    if "body_path" in text:
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(re.search(pattern, line) for pattern in risky_patterns):
                findings.append(
                    Finding(
                        "body_name_depth_heuristic",
                        "error",
                        f"{semantic_sites.relative_to(source_root)}:{line_no}",
                        "semantic depth falls back to body-name string length when runtime topology lacks a world body",
                        {"line": line.strip()},
                    )
                )
    return findings


def _audit_talos_foot_mapping(report: dict, semantic_map: dict) -> list[Finding]:
    findings: list[Finding] = []
    for semantic, bad_body in (("LeftFoot", "leg_left_1_link"), ("RightFoot", "leg_right_1_link")):
        site = report.get("semantic_sites", {}).get(semantic, {})
        mapped = semantic_map.get("semantics", {}).get(semantic)
        if site.get("body_name") == bad_body or mapped == bad_body:
            findings.append(
                Finding(
                    "talos_proximal_foot_mapping",
                    "error",
                    f"pal_talos:{semantic}",
                    "TALOS foot maps to a proximal leg link instead of a distal foot or sole site",
                    {"semantic_map": mapped, "site_body": site.get("body_name")},
                )
            )
    return findings


def _audit_booster_hips_chest(report: dict, semantic_map: dict) -> list[Finding]:
    sites = report.get("semantic_sites", {})
    hips = sites.get("Hips", {}).get("body_name")
    chest = sites.get("Chest", {}).get("body_name")
    mapped = semantic_map.get("semantics", {})
    if hips and chest and hips == chest:
        return [
            Finding(
                "booster_hips_chest_alias",
                "error",
                "booster_t1:Hips/Chest",
                "Booster T1 Hips and Chest resolve to the same body, hiding waist/trunk capability",
                {
                    "site_body": hips,
                    "semantic_map_hips": mapped.get("Hips"),
                    "semantic_map_chest": mapped.get("Chest"),
                    "torso_chain": report.get("chains", {}).get("torso", {}),
                },
            )
        ]
    return []


def _audit_g1_equivalence(summary: dict, validation_checks: dict) -> list[Finding]:
    findings: list[Finding] = []
    check = validation_checks.get("g1_mjcf_urdf_equivalence", {})
    if check.get("status") != "passed":
        findings.append(
            Finding(
                "arbitrary_g1_equivalence",
                "error",
                "validation_checks.g1_mjcf_urdf_equivalence",
                "G1 URDF/MJCF equivalence is not a strict pass",
                {"status": check.get("status"), "differences": check.get("differences", {})},
            )
        )
    compiled_count = summary.get("compiled_count")
    failure_count = summary.get("failure_artifacts_count")
    if compiled_count == 9 and failure_count == 0 and check.get("status") != "passed":
        findings.append(
            Finding(
                "arbitrary_g1_equivalence",
                "error",
                "summary.json",
                "summary is green while G1 equivalence remains a documented limitation",
                {"compiled_count": compiled_count, "failure_artifacts_count": failure_count},
            )
        )
    return findings


def _audit_cross_format_gates(cross_format: dict) -> list[Finding]:
    findings: list[Finding] = []
    gates = cross_format.get("gates", {}) if isinstance(cross_format, dict) else {}
    required_gates = ("same_source_strict", "variant_compatibility")
    for gate_name in required_gates:
        gate = gates.get(gate_name, {})
        status = gate.get("status")
        if status != "passed":
            findings.append(
                Finding(
                    "cross_format_gates_not_run",
                    "error",
                    f"cross_format.gates.{gate_name}",
                    "required cross-format gate is not a completed pass",
                    {"status": status, "reason": gate.get("reason")},
                )
            )
    if not gates:
        findings.append(
            Finding(
                "cross_format_gates_not_run",
                "error",
                "cross_format.json",
                "cross-format artifact does not contain required gate results",
                {"required_gates": list(required_gates)},
            )
        )
    return findings


def _audit_dirty_artifact_metadata(environment: dict, source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    status = environment.get("git_status_short", "")
    if status.strip():
        findings.append(
            Finding(
                "dirty_artifact_metadata",
                "error",
                "environment.json",
                "artifacts were generated from a dirty worktree",
                {"git_status_short": status},
            )
        )
    current_head = _git(source_root, "rev-parse", "HEAD")
    artifact_head = environment.get("git_head")
    if current_head and artifact_head and current_head != artifact_head:
        source_drift = _source_changes_since_artifact_head(source_root, str(artifact_head), current_head)
        if source_drift:
            findings.append(
                Finding(
                    "dirty_artifact_metadata",
                    "error",
                    "environment.json",
                    "artifact git_head does not match the audited checkout HEAD",
                    {
                        "artifact_git_head": artifact_head,
                        "current_git_head": current_head,
                        "source_changes_since_artifact_head": source_drift[:50],
                    },
                )
            )
    return findings


def _source_changes_since_artifact_head(source_root: Path, artifact_head: str, current_head: str) -> list[str]:
    """Return non-artifact changes between the source commit and audited HEAD.

    Committed generated artifacts cannot record the commit hash that contains
    themselves. A clean artifact commit is acceptable only when every change
    after ``artifact_head`` is inside the generated artifact directory.
    """

    changed = _git(source_root, "diff", "--name-only", artifact_head, current_head)
    if not changed:
        return []
    artifact_prefix = "artifacts/retargeting_v3_step2/"
    return [
        path
        for path in changed.splitlines()
        if path and not path.startswith(artifact_prefix)
    ]


def _audit_required_artifact_files(artifact_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative in REQUIRED_ARTIFACT_FILES:
        path = artifact_dir / relative
        if not path.exists():
            findings.append(
                Finding(
                    "missing_required_artifacts",
                    "error",
                    relative,
                    "required reproducibility/test artifact is missing",
                    {"artifact_dir": _display_audit_path(artifact_dir)},
                )
            )
            continue
        try:
            empty = path.stat().st_size == 0
        except OSError:
            empty = True
        if empty:
            findings.append(
                Finding(
                    "missing_required_artifacts",
                    "error",
                    relative,
                    "required reproducibility/test artifact is empty",
                    {"artifact_dir": _display_audit_path(artifact_dir)},
                )
            )
    return findings


def _audit_absolute_paths(commands_text: str, per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    for idx, line in enumerate(commands_text.splitlines(), start=1):
        hits = _absolute_local_hits(line)
        if hits:
            findings.append(
                Finding(
                    "absolute_cache_paths",
                    "error",
                    f"commands.txt:{idx}",
                    "reproduction command contains local absolute cache path(s)",
                    {"paths": hits, "command": line},
                )
            )
    for robot_id, report in sorted(per_robot.items()):
        for field in ("reproduction_command",):
            hits = _absolute_local_hits(str(report.get(field, "")))
            if hits:
                findings.append(
                    Finding(
                        "absolute_cache_paths",
                        "error",
                        f"{robot_id}:{field}",
                        f"{field} contains local absolute cache path(s)",
                        {"paths": hits},
                    )
                )
        model_path = str(report.get("model", {}).get("path", ""))
        hits = _absolute_local_hits(model_path)
        if hits:
            findings.append(
                Finding(
                    "absolute_cache_paths",
                    "error",
                    f"{robot_id}:model.path",
                    "model.path contains a local absolute cache path",
                    {"paths": hits},
                )
            )
    return findings


def _audit_artifact_absolute_paths(artifact_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not artifact_dir.exists():
        return findings
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        text = _read_text(path)
        if not text:
            continue
        hits = _absolute_local_hits(text)
        if not hits:
            continue
        try:
            subject = str(path.relative_to(artifact_dir))
        except ValueError:
            subject = str(path)
        findings.append(
            Finding(
                "absolute_cache_paths",
                "error",
                subject,
                "artifact file contains local absolute path string(s)",
                {"paths": sorted(set(hits))[:20], "path_count": len(hits)},
            )
        )
    return findings


def _audit_inferred_semantics_confidence(per_robot: dict[str, dict], semantic_maps: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    for robot_id, semantic_map in sorted(semantic_maps.items()):
        if semantic_map.get("source") != "inferred_from_newton_body_names":
            continue
        report = per_robot.get(robot_id, {})
        for semantic, site in sorted(report.get("semantic_sites", {}).items()):
            if site.get("source") == "explicit_semantic_override" or float(site.get("confidence", 0.0)) >= 1.0:
                findings.append(
                    Finding(
                        "inferred_semantics_confidence_one",
                        "error",
                        f"{robot_id}:{semantic}",
                        "inferred body-name semantic is emitted as explicit override and/or confidence=1.0",
                        {
                            "semantic_map_source": semantic_map.get("source"),
                            "site_source": site.get("source"),
                            "confidence": site.get("confidence"),
                            "body_name": site.get("body_name"),
                        },
                    )
                )
    return findings


def _audit_rank_zero_false_pass(per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    for robot_id, report in sorted(per_robot.items()):
        for task, projection in sorted(report.get("projection_reports", {}).items()):
            if projection.get("status") == "rank_zero" and _is_zero(projection.get("residual")):
                if "unreachable_demand" not in projection and "demand_residual" not in projection:
                    findings.append(
                        Finding(
                            "rank0_false_pass",
                            "error",
                            f"{robot_id}:{task}",
                            "rank-zero projection reports zero residual without unreachable-demand evidence",
                            {
                                "status": projection.get("status"),
                                "residual": projection.get("residual"),
                                "desired": projection.get("desired"),
                                "projected": projection.get("projected"),
                            },
                        )
                    )
    return findings


def _audit_epsilon_stability(per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    for robot_id, report in sorted(per_robot.items()):
        if report.get("status") not in PROFILE_PASS_STATUSES:
            continue
        unstable_paths = [
            path
            for path, value in _walk_json(report.get("rank_stability", {}))
            if path.endswith("epsilon_stability_gate_passed") and value is False
        ]
        if unstable_paths:
            findings.append(
                Finding(
                    "epsilon_stability_failed",
                    "error",
                    robot_id,
                    "passed/partial profile contains epsilon_stability_gate_passed=false",
                    {"unstable_paths": unstable_paths[:50], "unstable_count": len(unstable_paths)},
                )
            )
    return findings


def _audit_robot_name_special_cases(source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    checked_files = [
        source_root / "soma_retargeter/robotics/v3/semantic_sites.py",
        source_root / "soma_retargeter/robotics/v3/profile.py",
        source_root / "soma_retargeter/robotics/v3/chain_projection.py",
        source_root / "soma_retargeter/robotics/v3/kinematic_paths.py",
        source_root / "soma_retargeter/robotics/v3/numerical_jacobian.py",
        source_root / "soma_retargeter/robotics/v3/reachability.py",
        source_root / "soma_retargeter/robotics/v3/rest_frames.py",
        source_root / "soma_retargeter/robotics/v3/target_builder.py",
    ]
    patterns = (
        "default_rpo_semantic_map",
        "roboparty_rpo",
        "unitree_g1",
        "booster_t1",
        "pal_talos",
        "robotis_op3",
        "berkeley_humanoid",
    )
    for path in checked_files:
        text = _read_text(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in patterns):
                findings.append(
                    Finding(
                        "robot_name_special_cases",
                        "error",
                        f"{path.relative_to(source_root)}:{line_no}",
                        "v3 core source contains robot-name-specific logic or default mapping",
                        {"line": line.strip()},
                    )
                )
    return findings


def _audit_legacy_offsets(source_root: Path, per_robot: dict[str, dict]) -> list[Finding]:
    findings: list[Finding] = []
    source_files = [
        source_root / "soma_retargeter/robotics/v3/source_rest.py",
        source_root / "soma_retargeter/robotics/v3/target_builder.py",
        source_root / "soma_retargeter/robotics/v3/rest_frames.py",
        source_root / "soma_retargeter/robotics/v3/profile.py",
    ]
    legacy_patterns = ("joint_offsets", "human_to_robot_scaler", "scaler_config", "optimized_offsets")
    for path in source_files:
        text = _read_text(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in legacy_patterns):
                findings.append(
                    Finding(
                        "legacy_offsets",
                        "error",
                        f"{path.relative_to(source_root)}:{line_no}",
                        "v3 source still references legacy scaler/offset fields",
                        {"line": line.strip()},
                    )
                )
    for robot_id, report in sorted(per_robot.items()):
        text = json.dumps(report, sort_keys=True)
        hits = [pattern for pattern in legacy_patterns if pattern in text]
        if hits:
            findings.append(
                Finding(
                    "legacy_offsets",
                    "error",
                    robot_id,
                    "artifact contains legacy scaler/offset field names",
                    {"patterns": hits},
                )
            )
    return findings


def _audit_formal_red_team_artifacts(source_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    required_files = (
        source_root / "docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md",
        source_root / "docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md",
    )
    for path in required_files:
        if not path.exists():
            findings.append(
                Finding(
                    "missing_formal_red_team_artifacts",
                    "error",
                    str(path.relative_to(source_root)),
                    "required formal Step-2 acceptance/report artifact is missing",
                    {},
                )
            )
    handoff_dir = source_root / "docs/retargeting_v3/subagents"
    for filename in REQUIRED_SUBAGENT_HANDOFFS:
        path = handoff_dir / filename
        if not path.exists():
            findings.append(
                Finding(
                    "missing_formal_red_team_artifacts",
                    "error",
                    str(path.relative_to(source_root)),
                    "required six-subagent handoff artifact is missing",
                    {},
                )
            )
    red_team = handoff_dir / "step2_agent_f_red_team.md"
    text = _read_text(red_team)
    if red_team.exists() and ("xhigh" not in text or ("PASS" not in text and "BLOCKED" not in text)):
        findings.append(
            Finding(
                "missing_formal_red_team_artifacts",
                "error",
                str(red_team.relative_to(source_root)),
                "Agent F handoff must record xhigh reasoning and an explicit PASS or BLOCKED result",
                {"contains_xhigh": "xhigh" in text, "contains_result": "PASS" in text or "BLOCKED" in text},
            )
        )
    return findings


def _load_per_robot(directory: Path) -> dict[str, dict]:
    if not directory.exists():
        return {}
    return {path.stem: _read_json(path) for path in sorted(directory.glob("*.json"))}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _is_zero(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and abs(number) <= ZERO_TOL


def _vector_is_zero(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(_is_zero(item) for item in value)


def _vectors_close(a: object, b: object) -> bool:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    try:
        return all(abs(float(x) - float(y)) <= ZERO_TOL for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


def _absolute_local_hits(text: str) -> list[str]:
    pattern = r"(?<![\w$])/(?:mnt|home|Users)/[^\s\"'<>),\]}`]+"
    return [match.group(0).rstrip(".,;:") for match in re.finditer(pattern, text)]


def _sanitize_audit_payload(value):
    if isinstance(value, dict):
        return {key: _sanitize_audit_payload(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_audit_payload(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_audit_payload(child) for child in value]
    if isinstance(value, str):
        return _sanitize_audit_string(value)
    return value


def _sanitize_audit_string(value: str) -> str:
    pattern = r"(?<![\w$])/(?:mnt|home|Users|tmp|var|private/var)/[^\s\"'<>),\]}`:]+"

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        path_text = token.rstrip(".,;")
        suffix = token[len(path_text) :]
        return f"{_display_audit_path(Path(path_text))}{suffix}"

    return re.sub(pattern, replace, value)


def _display_audit_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    for root in (Path.cwd(),):
        try:
            return str(resolved.relative_to(root.resolve()))
        except Exception:
            pass
    for variable, root in _audit_placeholder_roots():
        try:
            return f"${{{variable}}}/" + str(resolved.relative_to(root.resolve()))
        except Exception:
            pass
    return "${LOCAL_SOURCE_PATH}/" + resolved.name


def _audit_placeholder_roots() -> list[tuple[str, Path]]:
    descriptions_cache = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
    newton_cache = os.environ.get("NEWTON_CACHE")
    return [
        (
            "ROBOT_DESCRIPTIONS_CACHE",
            Path(descriptions_cache).expanduser() if descriptions_cache else Path.home() / ".cache" / "robot_descriptions",
        ),
        (
            "NEWTON_CACHE",
            Path(newton_cache).expanduser() if newton_cache else Path.home() / ".cache" / "newton",
        ),
    ]


def _walk_json(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        items: list[tuple[str, object]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_walk_json(child, child_prefix))
        return items
    if isinstance(value, list):
        items = []
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            items.extend(_walk_json(child, child_prefix))
        return items
    return [(prefix, value)]


def _git(source_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=source_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def write_junit(result: AuditResult, path: Path) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "retargeting_v3_step2_acceptance_audit",
            "tests": str(len(GOAL_FALSE_POSITIVE_GATES)),
            "failures": str(sum(1 for gate, count in result.gate_counts.items() if count)),
            "errors": "0",
            "skipped": "0",
        },
    )
    by_gate: dict[str, list[Finding]] = {gate: [] for gate in GOAL_FALSE_POSITIVE_GATES}
    for finding in result.findings:
        by_gate.setdefault(finding.gate, []).append(finding)
    for gate in GOAL_FALSE_POSITIVE_GATES:
        case = ET.SubElement(testsuite, "testcase", {"classname": "retargeting_v3_step2", "name": gate})
        gate_findings = by_gate.get(gate, [])
        if gate_findings:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "message": f"{len(gate_findings)} blocking finding(s)",
                    "type": "AcceptanceGateFailure",
                },
            )
            failure.text = "\n".join(f"{f.subject}: {f.message}" for f in gate_findings)
    tree = ET.ElementTree(testsuite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--junit-xml", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_audit(args.artifact_dir, args.source_root)
    payload = result.to_json()
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.junit_xml:
        write_junit(result, args.junit_xml)

    print(json.dumps({"status": result.status, "blocking_count": result.blocking_count, "gate_counts": result.gate_counts}, sort_keys=True))
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
