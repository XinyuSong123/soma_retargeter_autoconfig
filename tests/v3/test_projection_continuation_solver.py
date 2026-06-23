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


def _one_link_revolute_model(tmp_path: Path, *, limit: tuple[float, float] = (-3.141592653589793, 3.141592653589793)) -> Path:
    lower, upper = limit
    return _write(
        tmp_path / "one_link_revolute.xml",
        f"""
<mujoco model="one_link_revolute">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <body name="link">
        <inertial pos="0.5 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="yaw" type="hinge" axis="0 0 1" range="{lower} {upper}"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )


def _two_link_revolute_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "two_link_revolute.xml",
        """
<mujoco model="two_link_revolute">
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
        tmp_path / "yaw_torso.xml",
        """
<mujoco model="yaw_torso">
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


def _fixed_torso_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "fixed_torso.xml",
        """
<mujoco model="fixed_torso">
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 1"><geom type="sphere" size="0.01"/></body>
    </body>
  </worldbody>
</mujoco>
""",
    )


def _endpoint_sites(adapter: MuJoCoRuntimeModelAdapter, hand_body: str) -> dict:
    return build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": "base",
            "LeftHand": {"body": hand_body, "local_position": [1.0, 0.0, 0.0]},
            "RightHand": {"body": hand_body, "local_position": [1.0, 0.0, 0.0]},
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


def test_reachable_revolute_endpoint_uses_engine_jacobian_and_continuation(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_one_link_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter, "link")
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    desired_angle = 0.7
    desired = np.array([np.cos(desired_angle), np.sin(desired_angle), 0.0])

    result = project_endpoint_position(adapter, adapter.neutral_q(), sites["Chest"], sites["LeftHand"], active, desired)
    payload = result.to_json()

    assert result.status == "converged"
    assert result.residual < 1e-9
    np.testing.assert_allclose(result.projected, desired, atol=1e-9)
    np.testing.assert_allclose(result.chain_q[0], desired_angle, atol=1e-9)
    assert payload["residual_parameterization"] == "position"
    assert payload["residual_jacobian_source"] == "engine_relative_jacobian"
    assert payload["deterministic_start_count"] >= 3
    assert payload["continuation_history"]
    assert all(step["accepted"] for step in payload["continuation_history"])
    assert payload["active_limit_kkt"]["gradient_source"] == "task_only"
    assert payload["active_limit_kkt"]["satisfied"]


def test_unreachable_beyond_link_length_reports_task_only_stationarity(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_one_link_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter, "link")
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    desired = np.array([1.5, 0.2, 0.0])

    result = project_endpoint_position(adapter, adapter.neutral_q(), sites["Chest"], sites["LeftHand"], active, desired)
    payload = result.to_json()
    expected = desired / np.linalg.norm(desired)

    assert result.status == "converged/with_residual"
    np.testing.assert_allclose(result.projected, expected, atol=2e-7)
    assert abs(result.residual - (np.linalg.norm(desired) - 1.0)) < 2e-7
    assert payload["active_limit_kkt"]["active_lower"] == []
    assert payload["active_limit_kkt"]["active_upper"] == []
    assert payload["active_limit_kkt"]["stationarity_inf_norm"] <= payload["active_limit_kkt"]["stationarity_tolerance"]
    assert payload["task_gradient_inf_norm"] < 1e-7


def test_near_singular_arm_uses_deterministic_adaptive_subdivision(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_two_link_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter, "forearm")
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    q_target = np.array([2.4, -0.9])
    desired = np.array([np.cos(q_target[0]) + np.cos(np.sum(q_target)), np.sin(q_target[0]) + np.sin(np.sum(q_target)), 0.0])

    result = project_endpoint_position(adapter, adapter.neutral_q(), sites["Chest"], sites["LeftHand"], active, desired)
    payload = result.to_json()
    history = payload["continuation_history"]

    assert result.status == "converged"
    assert result.residual < 2e-8
    np.testing.assert_allclose(result.projected, desired, atol=2e-8)
    assert len(history) > 1
    assert history[0]["alpha_start"] == 0.0
    assert history[-1]["alpha_end"] == 1.0
    assert payload["deterministic_adaptive_subdivision"]["accepted_step_count"] == len(history)
    assert payload["deterministic_adaptive_subdivision"]["max_task_step_norm"] <= 0.2500000001


def test_yaw_only_torso_with_pitch_demand_uses_so3_log_task_gradient(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_yaw_torso_model(tmp_path))
    sites = _torso_sites(adapter)
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    desired = so3_exp(np.array([0.0, 0.35, 0.0]))

    result = project_torso_orientation(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active, desired)
    payload = result.to_json()

    assert result.status == "converged/with_residual"
    assert result.residual > 0.34
    np.testing.assert_allclose(result.chain_q[0], 0.0, atol=1e-9)
    assert payload["residual_parameterization"] == "so3_log"
    assert payload["residual_jacobian_source"] == "engine_relative_jacobian"
    assert payload["task_gradient_inf_norm"] < 1e-8
    assert payload["active_limit_kkt"]["satisfied"]


def test_fixed_torso_rank_zero_demand_has_rank_zero_evidence_and_no_history(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_fixed_torso_model(tmp_path))
    sites = _torso_sites(adapter)

    result = project_torso_orientation(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        [],
        so3_exp(np.array([0.0, 0.2, 0.0])),
    )
    payload = result.to_json()

    assert result.status == "unreachable/rank_zero"
    assert payload["rank_zero_reason"] == "no_active_coordinates_nonzero_demand"
    assert payload["continuation_history"] == []
    assert payload["active_limit_kkt"]["rank_zero"]
