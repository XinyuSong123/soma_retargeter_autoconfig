from __future__ import annotations

import json
from pathlib import Path

from soma_retargeter.robotics.v3.robot_zoo import load_robot_zoo_manifest
from soma_retargeter.robotics.v3.semantic_validation import compute_verified_semantic_map_coverage


def _manifest_positive_humanoid_ids() -> tuple[str, ...]:
    manifest = load_robot_zoo_manifest()
    return tuple(
        entry.id
        for entry in manifest.entries
        if entry.expected_capability == "positive" and entry.robot_class == "humanoid"
    )


def test_manifest_positive_semantic_map_coverage_is_structured():
    report = compute_verified_semantic_map_coverage()
    manifest_ids = _manifest_positive_humanoid_ids()

    assert report.positive_humanoid_count == len(manifest_ids)
    assert tuple(entry.model_id for entry in report.entries) == manifest_ids
    assert report.positive_humanoid_count == report.available_count + report.unavailable_count
    assert report.verified_map_count == sum(entry.has_verified_map for entry in report.entries)
    assert report.partial_blocked_count == sum(entry.has_explicit_partial_blocker for entry in report.entries)
    assert all(entry.semantic_map_status in {"missing", "invalid", "verified"} for entry in report.entries)
    assert all(entry.coverage_status in {"verified_map", "structurally_incomplete", "partial_blocked", "blocked", "missing", "invalid"} for entry in report.entries)
    assert report.coverage_status_counts == {
        status: sum(entry.coverage_status == status for entry in report.entries)
        for status in report.coverage_status_counts
    }


def test_existing_manifest_semantic_maps_are_verified_payloads():
    report = compute_verified_semantic_map_coverage()

    invalid = [entry.to_json() for entry in report.entries if entry.semantic_map_status == "invalid"]

    assert invalid == []
    assert report.verified_map_count >= 1


def test_available_positive_humanoids_have_verified_semantic_maps():
    report = compute_verified_semantic_map_coverage()
    missing = report.available_missing_verified_map_ids

    partial_blocked = set(report.available_partial_blocked_ids)
    blocked = set(report.available_blocked_ids)
    assert not (partial_blocked & set(missing))
    assert not (blocked & set(missing))
    assert missing == (), (
        f"{len(missing)} locally available positive humanoids are missing verified semantic maps "
        "and have no explicit evidence-backed blocker: "
        f"{', '.join(missing)}\n"
        + json.dumps(report.to_json(), indent=2, sort_keys=True)
    )


def test_unavailable_positive_humanoids_are_reported_separately_from_map_gate():
    report = compute_verified_semantic_map_coverage()
    unavailable_missing = report.unavailable_missing_verified_map_ids
    available_missing = set(report.available_missing_verified_map_ids)

    assert unavailable_missing
    assert not (available_missing & set(unavailable_missing))


def test_available_positive_humanoid_without_map_or_blocker_remains_a_gate_gap(tmp_path: Path):
    source = tmp_path / "toy.xml"
    source.write_text("<mujoco><worldbody/></mujoco>\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "toy_positive_humanoid",
                        "description_name": None,
                        "format": "mjcf",
                        "robot_class": "humanoid",
                        "expected_capability": "positive",
                        "license": "test",
                        "redistribution": "local_existing",
                        "required": True,
                        "source_family": "local",
                        "source_path": str(source),
                        "notes": "",
                    }
                ],
            }
        )
    )

    report = compute_verified_semantic_map_coverage(
        manifest_path=manifest,
        semantic_maps_dir=tmp_path / "semantic_maps",
        semantic_expectations_dir=tmp_path / "semantic_expectations",
    )

    assert report.available_missing_verified_map_ids == ("toy_positive_humanoid",)
    assert report.entries[0].coverage_status == "missing"
    assert not report.entries[0].has_explicit_partial_blocker
