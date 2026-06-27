from __future__ import annotations

from copy import deepcopy
import inspect

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import _deterministic_payload


def test_step3_2_deterministic_hash_ignores_volatile_runtime_seconds() -> None:
    first = _minimal_deterministic_inputs(runtime_seconds=0.125)
    second = _minimal_deterministic_inputs(runtime_seconds=99.875)

    first_payload = _call_deterministic_payload(first)
    second_payload = _call_deterministic_payload(second)

    assert first_payload["diagnostics_hash"] == second_payload["diagnostics_hash"]
    assert first_payload["comparison"] in {
        "stable_json_excluding_runtime_seconds",
        "stable_json_runtime_seconds_tolerant",
    }


def _call_deterministic_payload(payload: dict) -> dict:
    signature = inspect.signature(_deterministic_payload)
    kwargs = dict(payload)
    if "solver_smoke_matrix" in signature.parameters:
        kwargs["solver_smoke_matrix"] = kwargs.pop("generic_smoke_matrix")
    return _deterministic_payload(**kwargs, enabled=True)


def _minimal_deterministic_inputs(*, runtime_seconds: float) -> dict:
    model_row = {
        "model_id": "example_full_humanoid",
        "category": "full_humanoid_profile",
        "runtime_quality_status": "runtime_quality_passed",
        "solver_backed": True,
        "residual_only": False,
        "runtime_seconds": runtime_seconds,
    }
    solver_row = {
        "model_id": "example_full_humanoid",
        "category": "full_humanoid_profile",
        "mode": "runtime_solver_smoke",
        "solver_backed_smoke_attempted": True,
        "solver_backed_smoke_completed": True,
        "solver_backed_smoke_metrics_finite": True,
        "solver_backed": True,
        "residual_only": False,
        "runtime_quality_status": "runtime_quality_passed",
        "metrics": {
            "runtime_seconds": runtime_seconds,
            "normalized_task_residual_p95": 0.01,
            "normalized_task_residual_max": 0.02,
            "solver_success_fraction": 1.0,
        },
    }
    quality_summary = {
        "schema_version": 1,
        "in_scope_total": 1,
        "step3_2_solver_backed_smoke_attempted_count": 1,
        "step3_2_solver_backed_smoke_completed_count": 1,
        "step3_2_solver_backed_smoke_passed_count": 1,
        "step3_2_solver_backed_smoke_failed_count": 0,
        "step3_2_solver_backed_smoke_finite_metrics_count": 1,
        "step3_2_solver_backed_runtime_quality_passed_count": 1,
        "step3_2_residual_only_runtime_quality_passed_count": 0,
        "runtime_seconds": runtime_seconds,
    }
    payload = {
        "model_matrix": {"schema_version": 1, "in_scope_total": 1, "rows": [model_row]},
        "profile_matrix": {"schema_version": 1, "row_count": 1, "rows": [{"model_id": "example_full_humanoid"}]},
        "target_stream_matrix": {"schema_version": 1, "row_count": 0, "rows": []},
        "generic_smoke_matrix": {"schema_version": 2, "row_count": 1, "rows": [solver_row]},
        "quality_summary": quality_summary,
    }
    return deepcopy(payload)
