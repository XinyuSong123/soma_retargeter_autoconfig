from __future__ import annotations

from copy import deepcopy

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import _deterministic_payload


def test_step3_4_deterministic_hash_ignores_runtime_seconds_but_includes_new_artifacts() -> None:
    first = _payload(runtime_seconds=0.1, taxonomy_bucket="rotation_residual_dominates")
    second = _payload(runtime_seconds=999.0, taxonomy_bucket="rotation_residual_dominates")

    assert _call(first)["diagnostics_hash"] == _call(second)["diagnostics_hash"]

    changed_taxonomy = _payload(runtime_seconds=0.1, taxonomy_bucket="translation_residual_dominates")
    assert _call(first)["diagnostics_hash"] != _call(changed_taxonomy)["diagnostics_hash"]


def _call(payload: dict) -> dict:
    return _deterministic_payload(**deepcopy(payload), enabled=True)


def _payload(*, runtime_seconds: float, taxonomy_bucket: str) -> dict:
    row = {
        "model_id": "full_00",
        "category": "full_humanoid_profile",
        "runtime_quality_status": "runtime_quality_warned",
        "solver_backed": True,
        "residual_only": False,
        "runtime_seconds": runtime_seconds,
    }
    quality_summary = {
        "schema_version": 1,
        "in_scope_total": 44,
        "deterministic_compared_count": 44,
        "deterministic_matched_count": 44,
    }
    return {
        "model_matrix": {"schema_version": 1, "in_scope_total": 44, "rows": [row]},
        "profile_matrix": {"schema_version": 1, "row_count": 44, "rows": []},
        "target_stream_matrix": {"schema_version": 1, "row_count": 0, "rows": []},
        "generic_smoke_matrix": {"schema_version": 1, "row_count": 1, "rows": [row]},
        "solver_smoke_matrix": {"schema_version": 1, "row_count": 1, "rows": [row]},
        "quality_summary": quality_summary,
        "solver_config": {"solver_config_hash": "hash"},
        "solver_diagnostics_matrix": {"solver_config_hash": "hash", "rows": [row]},
        "quality_delta_vs_step3_2": {},
        "quality_delta_vs_step3_3": {"count_deltas": {"runtime_quality_failed_count": 0}},
        "residual_taxonomy": {"rows": [{"model_id": "full_00", "blocker_buckets": [taxonomy_bucket]}]},
        "task_coverage_matrix": {"rows": [{"model_id": "full_00", "task_coverage_ratio": 1.0}]},
        "anchor_reliability_matrix": {"model_rows": [{"model_id": "full_00", "anchor_reliability_score": 1.0}]},
    }
