from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.capability_projection import project_endpoint_position_with_certificate
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.projection_certificate import (
    CERTIFICATE_CLASSES,
    ProjectionCertificateEvidence,
    build_projection_certificate,
)
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _single_slide_adapter(tmp_path: Path, *, upper: float):
    model = _write(
        tmp_path / f"single_slide_{upper}.xml",
        f"""
<mujoco model="single_slide">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="hand">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="reach_x" type="slide" axis="1 0 0" range="0 {upper}"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "chest",
            "Chest": "chest",
            "LeftHand": "hand",
            "RightHand": "hand",
            "LeftFoot": "chest",
            "RightFoot": "chest",
        },
    )
    return adapter, sites


def _project(adapter, sites, desired):
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    return project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.asarray(desired, dtype=float),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()["capability_certificate"]


def test_certificate_decisions_cover_exact_rank_limit_and_mixed_cases(tmp_path: Path):
    exact_adapter, exact_sites = _single_slide_adapter(tmp_path, upper=1.0)
    limited_adapter, limited_sites = _single_slide_adapter(tmp_path, upper=0.2)

    exact = _project(exact_adapter, exact_sites, [0.15, 0.0, 0.0])
    rank = _project(exact_adapter, exact_sites, [0.15, 0.2, 0.0])
    joint_limit = _project(limited_adapter, limited_sites, [0.5, 0.0, 0.0])
    mixed = _project(limited_adapter, limited_sites, [0.5, 0.2, 0.0])

    assert exact["certificate_class"] == "exact_reachable"
    assert rank["certificate_class"] == "capability_limited_rank"
    assert joint_limit["certificate_class"] == "capability_limited_joint_limits"
    assert mixed["certificate_class"] == "capability_limited_mixed"
    assert joint_limit["decomposition"]["active_limit_residual_norm"] > 0.299
    assert mixed["decomposition"]["active_limit_residual_norm"] > 0.299
    assert mixed["decomposition"]["rank_incompatible_residual_norm"] > 0.199
    assert joint_limit["gates"]["projected_gradient_kkt"] is True
    assert mixed["gates"]["projected_gradient_kkt"] is True


def test_pure_certificate_builder_classifies_failure_and_invalid_inputs():
    assert set(CERTIFICATE_CLASSES) == {
        "exact_reachable",
        "capability_limited_rank",
        "capability_limited_joint_limits",
        "capability_limited_mixed",
        "unsupported_rank_zero",
        "solver_failed",
        "numerical_invalid",
        "invalid_target_geometry",
    }

    invalid_geometry = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.zeros(2),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.eye(3),
        )
    )
    numerical_invalid = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.array([np.nan, 0.0, 0.0]),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.eye(3),
        )
    )
    solver_failed = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.array([1.0, 0.0, 0.0]),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.eye(3),
            converged=False,
            solver_status="failed",
            active_coordinates=[0],
        )
    )

    assert invalid_geometry.certificate_class == "invalid_target_geometry"
    assert numerical_invalid.certificate_class == "numerical_invalid"
    assert solver_failed.certificate_class == "solver_failed"
