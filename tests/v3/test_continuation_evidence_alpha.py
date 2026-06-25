from __future__ import annotations

import numpy as np

from soma_retargeter.robotics.v3.projection_certificate import ProjectionCertificateEvidence, build_projection_certificate


def test_continuation_gate_rejects_history_that_does_not_reach_alpha_one():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.zeros(3),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.zeros((3, 0)),
            seed_consensus={"checked": True, "passed": True, "start_count": 1},
            continuation_history=[
                {
                    "alpha_start": 0.0,
                    "alpha_end": 0.5,
                    "accepted": True,
                    "q_active": [],
                    "task_residual_norm": 0.0,
                }
            ],
        )
    ).to_json()

    assert cert["continuation"]["checked"] is True
    assert cert["continuation"]["final_alpha"] == 0.5
    assert cert["continuation"]["reached_alpha_one"] is False
    assert cert["gates"]["continuation"] is False
    assert cert["passed"] is False


def test_continuation_gate_accepts_strictly_increasing_alpha_one_history():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.zeros(3),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.zeros((3, 0)),
            seed_consensus={"checked": True, "passed": True, "start_count": 1},
            continuation_history=[
                {
                    "alpha_start": 0.0,
                    "alpha_end": 0.4,
                    "accepted": True,
                    "q_active": [],
                    "task_residual_norm": 0.0,
                },
                {
                    "alpha_start": 0.4,
                    "alpha_end": 1.0,
                    "accepted": True,
                    "q_active": [],
                    "task_residual_norm": 0.0,
                },
            ],
        )
    ).to_json()

    assert cert["continuation"]["accepted_alpha_strictly_increasing"] is True
    assert cert["continuation"]["reached_alpha_one"] is True
    assert cert["gates"]["continuation"] is True
