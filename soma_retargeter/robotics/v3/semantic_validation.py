"""Validation helpers for semantic-site evidence and morphology coverage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REQUIRED_HUMANOID_SEMANTICS = ("Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
LOWER_BODY_SEMANTICS = ("Hips", "Chest", "LeftFoot", "RightFoot")
DISTAL_SEMANTIC_SUFFIXES = ("Hand", "Foot", "Toe", "Heel")


@dataclass(frozen=True)
class SemanticValidationIssue:
    semantic_name: str
    code: str
    message: str

    def to_json(self) -> dict:
        return {
            "semantic_name": self.semantic_name,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class SemanticCoverageClassification:
    status: str
    missing_required: tuple[str, ...]
    available: tuple[str, ...]
    issues: tuple[SemanticValidationIssue, ...]

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "missing_required": list(self.missing_required),
            "available": list(self.available),
            "issues": [issue.to_json() for issue in self.issues],
        }


def is_inferred_source(source: str) -> bool:
    return source.startswith("inferred") or "inferred" in source


def validate_no_fake_semantic_evidence(sites: dict) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    for semantic_name, site in sites.items():
        source = str(getattr(site, "source", ""))
        confidence = float(getattr(site, "confidence", 0.0))
        if source == "explicit_semantic_override":
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "legacy_explicit_override_source",
                    "semantic evidence must distinguish configured, verified, and inferred provenance",
                )
            )
        if is_inferred_source(source) and confidence >= 1.0:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "inferred_confidence_one",
                    "inferred semantic evidence must not be assigned confidence=1.0",
                )
            )
        if confidence > 0.99:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "confidence_above_cap",
                    "semantic confidence must be capped below certainty",
                )
            )
    return issues


def validate_nonzero_distal_sites(
    sites: dict,
    *,
    distal_suffixes: tuple[str, ...] = DISTAL_SEMANTIC_SUFFIXES,
    atol: float = 1e-9,
) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    for semantic_name, site in sites.items():
        if not semantic_name.endswith(distal_suffixes):
            continue
        pos = np.asarray(getattr(site, "local_position", np.zeros(3)), dtype=float)
        if float(np.linalg.norm(pos)) <= atol:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "distal_site_at_body_origin",
                    "distal hand/sole/toe/heel site lacks nonzero local geometry evidence",
                )
            )
    return issues


def validate_verified_semantic_map_payload(payload: dict) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    semantics = payload.get("semantics", {})
    if payload.get("verification_status") != "verified":
        issues.append(
            SemanticValidationIssue(
                str(payload.get("model_id", "")),
                "map_not_verified",
                "semantic map payload must explicitly declare verification_status='verified'",
            )
        )
    if not payload.get("model_fingerprint"):
        issues.append(
            SemanticValidationIssue(
                str(payload.get("model_id", "")),
                "missing_model_fingerprint",
                "verified semantic map must bind to a model fingerprint",
            )
        )
    for semantic_name, entry in semantics.items():
        if isinstance(entry, str):
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "legacy_string_entry",
                    "verified maps must store evidence/provenance entries, not bare body strings",
                )
            )
            continue
        evidence = entry.get("evidence", [])
        if not evidence:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "missing_evidence",
                    "semantic map entry must include evidence",
                )
            )
        confidence = float(entry.get("confidence", 0.0))
        if confidence > 0.99:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "confidence_above_cap",
                    "verified map confidence must be capped below certainty",
                )
            )
    return issues


def classify_semantic_coverage(
    sites: dict,
    *,
    expected_capability: str = "positive",
) -> SemanticCoverageClassification:
    available = tuple(sorted(sites))
    missing = tuple(name for name in REQUIRED_HUMANOID_SEMANTICS if name not in sites)
    issues: list[SemanticValidationIssue] = []

    if expected_capability == "negative_control":
        if not missing:
            issues.append(
                SemanticValidationIssue(
                    "Robot",
                    "negative_control_full_humanoid_semantics",
                    "negative control unexpectedly resolved a complete humanoid semantic set",
                )
            )
            status = "semantic_false_positive"
        else:
            status = "negative_control_passed"
        return SemanticCoverageClassification(status, missing, available, tuple(issues))

    if not missing:
        return SemanticCoverageClassification("full_humanoid_ready", missing, available, tuple(issues))

    lower_body_ready = all(name in sites for name in LOWER_BODY_SEMANTICS)
    only_upper_missing = set(missing) <= {"LeftHand", "RightHand"}
    if lower_body_ready and only_upper_missing:
        return SemanticCoverageClassification("partial_humanoid", missing, available, tuple(issues))

    return SemanticCoverageClassification("semantic_incomplete", missing, available, tuple(issues))
