from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3.engine_jacobian import engine_relative_jacobian
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter
from soma_retargeter.robotics.v3.numerical_jacobian import numerical_relative_jacobian
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _cross_branch_model(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "cross_branch.xml",
        """
<mujoco model="cross_branch">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="ref" pos="0 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="ref_z" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
      <body name="target" pos="0 1 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="target_z" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )


def test_mujoco_engine_relative_jacobian_supports_cross_branch_site_offsets(tmp_path: Path):
    adapter = MuJoCoRuntimeModelAdapter(_cross_branch_model(tmp_path))
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": {"body": "ref", "local_position": [0.4, 0.0, 0.0]},
            "Chest": {"body": "target", "local_position": [0.0, 0.25, 0.0]},
            "LeftFoot": "ref",
            "RightFoot": "target",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])

    engine = engine_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    fd = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)

    assert engine.source == "mujoco.mj_jac"
    assert engine.finite
    np.testing.assert_allclose(engine.translation, fd.translation, atol=2e-5)
    np.testing.assert_allclose(engine.rotation, fd.rotation, atol=2e-5)


def test_mujoco_engine_relative_jacobian_preserves_zero_position_nonzero_rotation(tmp_path: Path):
    model = _write(
        tmp_path / "on_axis.xml",
        """
<mujoco model="on_axis">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="link">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="hinge_z" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "base", "Chest": "link", "LeftFoot": "base", "RightFoot": "base"})
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])

    engine = engine_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)

    np.testing.assert_allclose(engine.translation, np.zeros((3, 1)), atol=1e-12)
    np.testing.assert_allclose(engine.rotation[:, 0], [0.0, 0.0, 1.0], atol=1e-12)


def test_newton_engine_relative_jacobian_uses_eval_jacobian_not_fd_fallback(tmp_path: Path):
    newton = pytest.importorskip("newton")
    del newton
    adapter = NewtonRuntimeModelAdapter(_cross_branch_model(tmp_path), model_format="mjcf")
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": {"body": "ref", "local_position": [0.4, 0.0, 0.0]},
            "Chest": {"body": "target", "local_position": [0.0, 0.25, 0.0]},
            "LeftFoot": "ref",
            "RightFoot": "target",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])

    engine = engine_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    fd = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)

    assert engine.source == "newton.eval_jacobian"
    assert engine.finite
    np.testing.assert_allclose(engine.translation, fd.translation, atol=2e-4)
    np.testing.assert_allclose(engine.rotation, fd.rotation, atol=2e-4)
