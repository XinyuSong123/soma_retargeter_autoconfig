from __future__ import annotations

from soma_retargeter.tools.run_v3_full_fleet_runtime_quality import _task_coverage_matrix_payload


def test_step3_4_task_coverage_matrix_records_global_task_order() -> None:
    model_matrix = {
        "rows": [
            {
                "model_id": "full_00",
                "category": "full_humanoid_profile",
                "runtime_quality_status": "runtime_quality_warned",
                "task_anchor_count": 6,
                "task_anchor_semantic_counts": {"Chest": 3, "Hips": 3},
                "task_coverage_ratio": 1.0,
                "successful_task_coverage_ratio": 0.8,
                "runtime_quality_warning_reasons": ["high_task_residual"],
            }
        ]
    }
    solver_smoke = {
        "rows": [
            {
                "model_id": "full_00",
                "category": "full_humanoid_profile",
                "smoke_summary": {
                    "residuals": {
                        "task_coverage": {
                            "global_task_universe": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"],
                            "configured_task_order": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"],
                            "rows": [{"task": "torso", "attempted_frame_count": 1}],
                            "summary": {"available_task_count": 5, "attempted_task_count": 5, "successful_task_count": 4},
                        }
                    }
                },
            }
        ]
    }

    matrix = _task_coverage_matrix_payload(
        model_matrix,
        solver_smoke,
        {"rows": [{"model_id": "full_00", "category": "full_humanoid_profile"}]},
        {"solver_config_hash": "hash", "config": {"task_order": ["torso", "left_hand", "right_hand", "left_foot", "right_foot"]}},
    )

    assert matrix["row_count"] == 1
    assert matrix["rows"][0]["task_coverage_ratio"] == 1.0
    assert matrix["rows"][0]["configured_task_order"] == ["torso", "left_hand", "right_hand", "left_foot", "right_foot"]
