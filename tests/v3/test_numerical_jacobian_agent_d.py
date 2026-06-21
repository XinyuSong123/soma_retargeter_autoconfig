from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.numerical_jacobian import numerical_relative_jacobian
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def test_epsilon_halving_can_be_promoted_to_hard_gate(tmp_path: Path):
    model = tmp_path / "curved_column.xml"
    model.write_text(
        """
<mujoco model="curved_column">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="link">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="hinge_z" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "base", "Chest": {"body": "link", "local_position": [1, 0, 0]}})
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])

    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    assert jac.stability_gate_passed
    assert jac.unstable_columns == []

    with pytest.raises(FloatingPointError, match="epsilon-halving stability gate failed"):
        numerical_relative_jacobian(
            adapter,
            adapter.neutral_q(),
            sites["Hips"],
            sites["Chest"],
            active,
            stability_rtol=0.0,
            stability_atol=0.0,
            raise_on_unstable=True,
        )


def test_finite_difference_uses_velocity_tangent_for_ball_joint(tmp_path: Path):
    model = tmp_path / "ball_tip.xml"
    model.write_text(
        """
<mujoco model="ball_tip">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="ball_link">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="ball" type="ball"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "base", "Chest": {"body": "ball_link", "local_position": [0, 0, 1]}})
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)

    assert active == [0, 1, 2]
    assert np.linalg.matrix_rank(jac.rotation, tol=1e-6) == 3
    np.testing.assert_allclose(jac.rotation, np.eye(3), atol=1e-6)
