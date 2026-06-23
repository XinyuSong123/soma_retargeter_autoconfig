from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position, project_torso_orientation
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites
from soma_retargeter.robotics.v3.spatial import so3_exp


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _two_link_revolute_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "two_link_seed_consensus.xml",
        """
<mujoco model="two_link_seed_consensus">
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


def _yaw_torso_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "orientation_seed_consensus.xml",
        """
<mujoco model="orientation_seed_consensus">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 1">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="torso_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
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


def _torso_sites(adapter: MuJoCoRuntimeModelAdapter) -> dict:
    return build_semantic_sites(
        adapter,
        {
            "Hips": "hips",
            "Chest": "chest",
            "LeftFoot": "hips",
            "RightFoot": "hips",
        },
    )


def test_two_local_minimum_chain_uses_deterministic_seed_consensus(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_two_link_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    desired = np.array([1.0, 1.0, 0.0])

    result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        desired,
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    )
    payload = result.to_json()

    assert result.status == "converged"
    assert result.residual < 1e-8
    np.testing.assert_allclose(result.projected, desired, atol=1e-8)
    assert payload["deterministic_start_count"] >= 7
    assert len(payload["seed_results"]) == payload["deterministic_start_count"]
    assert payload["selected_seed_index"] == min(
        seed["seed_index"] for seed in payload["seed_results"] if seed["task_residual_norm"] < 1e-8
    )
    assert payload["seed_results"] == sorted(payload["seed_results"], key=lambda seed: seed["seed_index"])


def test_projection_deterministic_rerun_reproduces_solution_and_history(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_two_link_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    desired = np.array([0.25, 1.55, 0.0])

    first = project_endpoint_position(adapter, adapter.neutral_q(), sites["Chest"], sites["LeftHand"], active, desired).to_json()
    second = project_endpoint_position(adapter, adapter.neutral_q(), sites["Chest"], sites["LeftHand"], active, desired).to_json()

    assert first["status"] == second["status"]
    np.testing.assert_allclose(first["chain_q"], second["chain_q"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first["projected"], second["projected"], atol=0.0, rtol=0.0)
    assert first["selected_seed_index"] == second["selected_seed_index"]
    assert first["continuation_history"] == second["continuation_history"]
    assert first["active_limit_kkt"] == second["active_limit_kkt"]


def test_nonidentity_orientation_target_uses_so3_log_jacobian_consistently(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_yaw_torso_model(tmp_path))
    sites = _torso_sites(adapter)
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    seed = adapter.set_velocity_coordinates(adapter.neutral_q(), active, np.array([0.15]))
    desired_angle = -0.45

    result = project_torso_orientation(adapter, seed, sites["Hips"], sites["Chest"], active, so3_exp([0.0, 0.0, desired_angle]))
    payload = result.to_json()

    assert result.status == "converged"
    assert result.residual < 1e-9
    np.testing.assert_allclose(result.chain_q[0], desired_angle, atol=1e-9)
    np.testing.assert_allclose(result.projected, [0.0, 0.0, desired_angle], atol=1e-9)
    assert payload["residual_parameterization"] == "so3_log"
    assert payload["so3_jacobian_convention"] == "left_perturbation_log_current_transpose_target"
    assert payload["active_limit_kkt"]["satisfied"]
