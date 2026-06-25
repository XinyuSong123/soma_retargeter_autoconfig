from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.projection_certificate import ProjectionCertificateEvidence, build_projection_certificate
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _two_link_revolute_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "two_link_seed_task_space.xml",
        """
<mujoco model="two_link_seed_task_space">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <body name="upper">
        <inertial pos="0.5 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-3.141592653589793 3.141592653589793"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01"/>
        <body name="forearm" pos="1 0 0">
          <inertial pos="0.5 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="elbow" type="hinge" axis="0 0 1" range="-3.141592653589793 3.141592653589793"/>
          <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )


def _endpoint_sites(adapter: MuJoCoRuntimeModelAdapter) -> dict:
    return build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": "base",
            "LeftHand": {"body": "forearm", "local_position": [1.0, 0.0, 0.0]},
            "RightHand": {"body": "forearm", "local_position": [1.0, 0.0, 0.0]},
            "LeftFoot": "base",
            "RightFoot": "base",
        },
    )


def test_seed_consensus_rejects_equal_residual_different_task_endpoint():
    cert = build_projection_certificate(
        ProjectionCertificateEvidence(
            task_block="translation",
            desired=np.zeros(3),
            projected=np.zeros(3),
            seed=np.zeros(3),
            jacobian=np.zeros((3, 0)),
            seed_consensus={
                "checked": True,
                "passed": True,
                "start_count": 2,
                "selected_seed_index": 0,
                "task_scale": 1.0,
                "tolerance": 1e-7,
                "global_seed_residual_tolerance": 1e-7,
                "seed_results": [
                    {
                        "seed_index": 0,
                        "accepted": True,
                        "normalized_residual": 0.0,
                        "final_task_vector": [0.0, 0.0, 0.0],
                        "certificate_class": "exact_reachable",
                    },
                    {
                        "seed_index": 1,
                        "accepted": True,
                        "normalized_residual": 0.0,
                        "final_task_vector": [1e-3, 0.0, 0.0],
                        "certificate_class": "exact_reachable",
                    },
                ],
            },
            continuation_passed=True,
        )
    ).to_json()

    assert cert["seed_consensus"]["competitive_seed_count"] == 2
    assert cert["seed_consensus"]["max_task_space_delta"] == 1e-3
    assert cert["seed_consensus"]["passed"] is False
    assert cert["gates"]["seed_consensus"] is False
    assert cert["passed"] is False


def test_solver_seed_results_include_task_space_endpoint_and_class(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_two_link_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])

    payload = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([1.0, 1.0, 0.0]),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()

    assert payload["seed_results"]
    for seed in payload["seed_results"]:
        assert "final_task_vector" in seed
        assert "normalized_residual" in seed
        assert "certificate_class" in seed
        assert "active_limits" in seed
        assert "final_q_active" in seed
        if seed["accepted"]:
            assert len(seed["final_task_vector"]) == 3
            assert len(seed["final_q_active"]) == len(active)
