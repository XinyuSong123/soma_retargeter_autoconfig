from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.canonical_projection import project_canonical_motion_sequence
from soma_retargeter.robotics.v3.capability_projection import project_endpoint_position_with_certificate
from soma_retargeter.robotics.v3.kinematic_paths import discover_paths
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.projection_certificate import ProjectionCertificateEvidence, build_projection_certificate
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites
from soma_retargeter.robotics.v3.spatial import transform
from soma_retargeter.robotics.v3.target_builder import SemanticTargets


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _single_slide_adapter(tmp_path: Path) -> tuple[MuJoCoRuntimeModelAdapter, dict]:
    model = _write(
        tmp_path / "single_slide_raw_evidence.xml",
        """
<mujoco model="single_slide_raw_evidence">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="hand">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="reach_x" type="slide" axis="1 0 0" range="0 1"/>
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


def test_capability_certificate_schema_v2_serializes_raw_recompute_inputs(tmp_path: Path):
    adapter, sites = _single_slide_adapter(tmp_path)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])

    payload = project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.25, 0.0, 0.0]),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()
    cert = payload["capability_certificate"]
    audit = cert["audit_evidence"]

    assert cert["schema_version"] == 2
    assert cert["passed"] is True
    assert cert["kkt_certificate"] == cert["kkt"] if "kkt_certificate" in cert else True
    assert set(audit) >= {
        "desired_vector",
        "projected_vector",
        "seed_vector",
        "normalized_residual_vector",
        "demand_vector",
        "relevant_task_jacobian",
        "active_coordinates",
        "q_active",
        "lower_bounds",
        "upper_bounds",
        "task_gradient",
        "prior_gradient",
        "seed_results",
        "continuation_history",
        "scalar_dtype",
        "normalization_scale",
    }
    np.testing.assert_allclose(audit["desired_vector"], payload["desired"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(audit["projected_vector"], payload["projected"], atol=0.0, rtol=0.0)
    assert audit["active_coordinates"] == active
    assert len(audit["q_active"]) == len(active)
    assert len(audit["lower_bounds"]) == len(active)
    assert len(audit["upper_bounds"]) == len(active)
    assert audit["scalar_dtype"] == "float64"


def test_canonical_projection_does_not_emit_synthetic_kkt_certificate(tmp_path: Path):
    adapter, sites = _single_slide_adapter(tmp_path)
    paths = discover_paths(adapter, sites)
    neutral = {
        "Hips": transform(),
        "Chest": transform(),
        "LeftHand": transform(),
        "RightHand": transform(),
        "LeftFoot": transform(),
        "RightFoot": transform(),
    }
    reach = {name: value.copy() for name, value in neutral.items()}
    reach["LeftHand"] = transform([0.2, 0.0, 0.0])

    report = project_canonical_motion_sequence(
        adapter,
        sites,
        paths,
        {
            "neutral": SemanticTargets(neutral, {}, mode="neutral"),
            "reach": SemanticTargets(reach, {}, mode="reach"),
        },
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()

    left_hand = report["motions"]["reach"]["tasks"]["left_hand"]
    assert "kkt_certificate" not in left_hand
    assert left_hand["capability_certificate"]["schema_version"] == 2


def test_rank_incompatible_zero_task_gradient_is_serialized_as_zero_not_epsilon():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.array([0.0, 1.0, 0.0]),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.array([[1.0], [0.0], [0.0]]),
            active_coordinates=[0],
            coordinate_values=np.array([0.0]),
            lower_bounds=np.array([-1.0]),
            upper_bounds=np.array([1.0]),
            seed_consensus={"checked": True, "passed": True, "start_count": 1},
            continuation_passed=True,
        )
    ).to_json()

    assert cert["certificate_class"] == "capability_limited_rank"
    assert cert["audit_evidence"]["task_gradient"] == [0.0]
    assert cert["kkt"]["task_gradient_inf_norm"] == 0.0
    assert cert["kkt"]["prior_cancellation_ratio"] == 0.0
