"""Validation helpers for semantic-site evidence and distal-site gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def is_inferred_source(source: str) -> bool:
    return source.startswith("inferred") or "inferred" in source


def validate_no_fake_semantic_evidence(sites: dict) -> list[SemanticValidationIssue]:
    issues: list[SemanticValidationIssue] = []
    for semantic_name, site in sites.items():
        source = str(getattr(site, "source", ""))
        confidence = float(getattr(site, "confidence", 0.0))
        if is_inferred_source(source) and confidence >= 1.0:
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "inferred_confidence_one",
                    "inferred semantic evidence must not be assigned confidence=1.0",
                )
            )
        if is_inferred_source(source) and source == "explicit_semantic_override":
            issues.append(
                SemanticValidationIssue(
                    semantic_name,
                    "inferred_marked_explicit",
                    "inferred semantics must not be marked as explicit overrides",
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
    return issues
