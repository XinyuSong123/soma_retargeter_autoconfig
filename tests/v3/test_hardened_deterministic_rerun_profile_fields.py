from __future__ import annotations

import copy

import soma_retargeter.robotics.v3.validation as validation_module


def _profile_report() -> dict:
    return {
        "status": "passed",
        "deterministic_hash": "stable-profile-hash",
        "rank_stability": {},
        "semantic_map_resolution": {"status": "available", "source": "test"},
        "semantic_sites": {},
        "task_certificate_summary": {"status": "available", "task_count": 1},
        "canonical_projection_reports": {
            "motion_order": ["neutral"],
            "target_source": "canonical_semantic_targets",
            "failures": [],
            "unreachable_demands": [],
            "motions": {
                "neutral": {
                    "tasks": {
                        "torso": {
                            "status": "converged",
                            "converged": True,
                            "residual": 0.0,
                            "normalized_residual": 0.0,
                            "normalization_scale": 1.0,
                            "iterations": 1,
                            "active_coordinates": [0, 1],
                            "desired_source": "canonical_targets.transforms",
                            "reference": "Hips",
                            "target": "Chest",
                            "chain_q": [0.0, 0.5],
                            "desired": [0.0, 0.0, 1.0],
                            "projected": [0.0, 0.0, 1.0],
                            "capability_certificate": {
                                "certificate_class": "exact_reachable",
                                "deterministic_digest": "digest-a",
                            },
                        }
                    }
                }
            },
        },
    }


def test_deterministic_profile_comparison_covers_status_q_task_vector_and_certificate_identity() -> None:
    first = _profile_report()
    second = copy.deepcopy(first)
    second["status"] = "capability_limited_passed"
    task = second["canonical_projection_reports"]["motions"]["neutral"]["tasks"]["torso"]
    task["chain_q"] = [0.0, 0.75]
    task["projected"] = [0.0, 0.1, 1.0]
    task["capability_certificate"] = {
        "certificate_class": "capability_limited_rank",
        "deterministic_digest": "digest-b",
    }

    comparison = validation_module._compare_deterministic_profile_reports(first, second)

    assert comparison["status"] == "mismatched"
    assert not comparison["comparisons"]["status"]["matched"]
    assert not comparison["comparisons"]["projection_q"]["matched"]
    assert not comparison["comparisons"]["projection_task_vectors"]["matched"]
    assert not comparison["comparisons"]["capability_certificate_identity"]["matched"]
    assert "motions.neutral.torso.chain_q[1]" in comparison["comparisons"]["projection_q"]["mismatch_paths"]
    assert (
        "motions.neutral.torso.projected[1]"
        in comparison["comparisons"]["projection_task_vectors"]["mismatch_paths"]
    )
    assert (
        "motions.neutral.torso.certificate_class"
        in comparison["comparisons"]["capability_certificate_identity"]["mismatch_paths"]
    )
    assert (
        "motions.neutral.torso.deterministic_digest"
        in comparison["comparisons"]["capability_certificate_identity"]["mismatch_paths"]
    )


def test_deterministic_comparison_fields_advertise_hardened_contract() -> None:
    assert validation_module._deterministic_comparison_fields() == [
        "status",
        "deterministic_hash",
        "rank_summary",
        "canonical_projection_residuals",
        "projection_q",
        "projection_task_vectors",
        "capability_certificate_identity",
        "semantic_site_evidence",
        "task_certificate_summary",
    ]
