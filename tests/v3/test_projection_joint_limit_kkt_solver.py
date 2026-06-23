from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.projection_solver import kkt_tolerances_for_scalar_dtype
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _limited_revolute_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "limited_revolute.xml",
        """
<mujoco model="limited_revolute">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <body name="link">
        <inertial pos="0.5 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="yaw" type="hinge" axis="0 0 1" range="-0.25 0.25"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.01"/>
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
            "LeftHand": {"body": "link", "local_position": [1.0, 0.0, 0.0]},
            "RightHand": {"body": "link", "local_position": [1.0, 0.0, 0.0]},
            "LeftFoot": "base",
            "RightFoot": "base",
        },
    )


def test_joint_limit_closest_point_reports_active_limit_kkt_evidence(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_limited_revolute_model(tmp_path))
    sites = _endpoint_sites(adapter)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    desired_angle = 0.9
    desired = np.array([np.cos(desired_angle), np.sin(desired_angle), 0.0])

    result = project_endpoint_position(adapter, adapter.neutral_q(), sites["Chest"], sites["LeftHand"], active, desired)
    payload = result.to_json()
    kkt = payload["active_limit_kkt"]
    expected_projected = np.array([np.cos(0.25), np.sin(0.25), 0.0])

    assert result.status == "converged/with_residual"
    np.testing.assert_allclose(result.chain_q[0], 0.25, atol=1e-10)
    np.testing.assert_allclose(result.projected, expected_projected, atol=1e-10)
    assert abs(result.residual - np.linalg.norm(expected_projected - desired)) < 1e-10
    assert kkt["active_lower"] == []
    assert kkt["active_upper"] == [0]
    assert kkt["multipliers"][0]["coordinate"] == 0
    assert kkt["multipliers"][0]["side"] == "upper"
    assert kkt["multipliers"][0]["value"] > 0.0
    assert kkt["stationarity_inf_norm"] <= kkt["stationarity_tolerance"]
    assert kkt["satisfied"]


def test_float32_backend_tolerance_is_explicitly_looser_than_float64():
    float32_tolerances = kkt_tolerances_for_scalar_dtype("float32")
    float64_tolerances = kkt_tolerances_for_scalar_dtype("float64")

    assert float32_tolerances["scalar_dtype"] == "float32"
    assert float64_tolerances["scalar_dtype"] == "float64"
    assert float32_tolerances["stationarity_tolerance"] > float64_tolerances["stationarity_tolerance"]
    assert float32_tolerances["active_bound_tolerance"] > float64_tolerances["active_bound_tolerance"]
    assert float32_tolerances["stationarity_tolerance"] == 5e-5
    assert float64_tolerances["stationarity_tolerance"] == 1e-7
