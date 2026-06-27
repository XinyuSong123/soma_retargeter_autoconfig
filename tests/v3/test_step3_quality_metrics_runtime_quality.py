from __future__ import annotations

import numpy as np

from soma_retargeter.runtime.v3.quality_metrics import smoke_output_metrics, target_stream_metrics


def test_step3_quality_metrics_report_finite_target_stream_stats() -> None:
    transforms = {
        "Hips": np.repeat(np.eye(4, dtype=float)[None, :, :], 4, axis=0),
        "Chest": np.repeat(np.eye(4, dtype=float)[None, :, :], 4, axis=0),
    }
    transforms["Chest"][:, 2, 3] = np.linspace(0.0, 0.3, 4)

    metrics = target_stream_metrics(transforms)

    assert metrics["finite_count"] == 8
    assert metrics["nan_count"] == 0
    assert metrics["se3_orthogonality_error_max"] == 0.0
    assert metrics["frame_to_frame_translation_velocity_max"] > 0.0


def test_step3_quality_metrics_report_output_and_joint_limit_stats() -> None:
    q = np.asarray([[0.0, 0.0], [0.1, 2.0], [0.2, 3.0]], dtype=float)
    coords = [
        {"index": 0, "limited": True, "lower": -1.0, "upper": 1.0},
        {"index": 1, "limited": True, "lower": -1.0, "upper": 1.0},
    ]

    metrics = smoke_output_metrics(q_sequence=q, coordinate_info=coords, residuals=np.asarray([0.0, 0.1, 0.2]), runtime_seconds=0.5)

    assert metrics["output_frame_count"] == 3
    assert metrics["joint_coord_count"] == 2
    assert metrics["nan_count"] == 0
    assert metrics["inf_count"] == 0
    assert metrics["joint_limit_violation_count"] == 2
    assert metrics["max_joint_limit_violation"] == 2.0


def test_step3_quality_metrics_use_qpos_address_for_joint_limits_when_available() -> None:
    q = np.asarray([[99.0, 0.0], [99.0, 0.5]], dtype=float)
    coords = [
        {"index": 0, "qpos_adr": 1, "limited": True, "lower": -1.0, "upper": 1.0},
    ]

    metrics = smoke_output_metrics(q_sequence=q, coordinate_info=coords, residuals=np.asarray([0.0]), runtime_seconds=0.1)

    assert metrics["joint_limit_violation_count"] == 0
    assert metrics["max_joint_limit_violation"] == 0.0
