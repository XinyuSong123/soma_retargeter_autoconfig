from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter
from soma_retargeter.robotics.v3.kinematic_paths import discover_paths
from soma_retargeter.robotics.v3.numerical_jacobian import numerical_relative_jacobian
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write_model(tmp_path: Path, name: str, xml: str) -> Path:
    path = tmp_path / name
    path.write_text(xml.strip())
    return path


def test_mujoco_free_joint_active_coordinates_drive_relative_jacobian(tmp_path: Path):
    model = _write_model(
        tmp_path,
        "free_tip.xml",
        """
<mujoco model="free_tip">
  <worldbody>
    <body name="floating">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="root_free" type="free"/>
      <geom type="sphere" size="0.01"/>
      <body name="tip" pos="1 0 0">
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
            "Hips": "world",
            "Chest": "tip",
        },
    )

    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    assert active == [0, 1, 2, 3, 4, 5]
    assert [adapter.coordinate(i).joint_type for i in active] == ["free"] * 6

    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    np.testing.assert_allclose(jac.translation[:, :3], np.eye(3), atol=1e-6)
    assert np.linalg.matrix_rank(jac.translation[:, 3:6], tol=1e-6) == 2
    assert np.linalg.matrix_rank(jac.rotation[:, 3:6], tol=1e-6) == 3


@pytest.mark.parametrize("adapter_cls", [MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter])
def test_fixed_intermediate_body_stays_in_body_path_without_active_fixed_coordinate(tmp_path: Path, adapter_cls):
    model = _write_model(
        tmp_path,
        "fixed_mid.xml",
        """
<mujoco model="fixed_mid">
  <worldbody>
    <body name="base">
      <geom type="sphere" size="0.01"/>
      <body name="fixed_mid" pos="0 0 1">
        <geom type="sphere" size="0.01"/>
        <body name="tip">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="tip_hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = adapter_cls(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": "tip",
        },
    )

    paths = discover_paths(adapter, sites)
    assert paths["torso"].body_path == ["base", "fixed_mid", "tip"]
    assert paths["torso"].coordinate_labels == ["tip_hinge"]
    assert paths["torso"].joint_types == ["revolute"]


def test_newton_free_joint_tangent_integration_normalizes_and_changes_fk(tmp_path: Path):
    model = _write_model(
        tmp_path,
        "newton_free.xml",
        """
<mujoco model="newton_free">
  <worldbody>
    <body name="floating">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="root_free" type="free"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = NewtonRuntimeModelAdapter(model)
    q0 = adapter.neutral_q()
    q0[3:7] *= 3.0
    delta = np.zeros(adapter.nv)
    delta[:3] = np.array([0.2, -0.1, 0.3])
    delta[3:6] = np.array([0.15, -0.05, 0.1])

    q1 = adapter.integrate(q0, delta)
    state0 = adapter.forward_kinematics(adapter.neutral_q())
    state1 = adapter.forward_kinematics(q1)

    assert abs(np.linalg.norm(q1[3:7]) - 1.0) < 1e-12
    assert not np.allclose(state1.body_xpos[0], state0.body_xpos[0])
    assert not np.allclose(state1.body_xmat[0], state0.body_xmat[0])


def test_newton_ball_joint_tangent_integration_normalizes_quaternion_branch():
    adapter = NewtonRuntimeModelAdapter.__new__(NewtonRuntimeModelAdapter)
    adapter.model = SimpleNamespace(joint_count=1)
    adapter.nq = 4
    adapter.nv = 3
    adapter._joint_type = np.array([2])
    adapter._joint_q_start = np.array([0])
    adapter._joint_qd_start = np.array([0])
    q0 = np.array([0.0, 0.0, 0.0, 2.0])
    delta = np.array([0.2, -0.1, 0.05])

    q1 = adapter.integrate(q0, delta)

    assert abs(np.linalg.norm(q1[:4]) - 1.0) < 1e-12
    assert not np.allclose(q1, q0 / np.linalg.norm(q0))


@pytest.mark.parametrize("adapter_cls", [MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter])
def test_explicit_model_site_reference_applies_site_local_position_offset(tmp_path: Path, adapter_cls):
    model = _write_model(
        tmp_path,
        "site_offset.xml",
        """
<mujoco model="site_offset">
  <worldbody>
    <body name="base">
      <geom type="sphere" size="0.01"/>
      <body name="link">
        <geom type="sphere" size="0.01"/>
        <site name="tip_site" pos="0.2 0 0" quat="0.7071067811865476 0 0 0.7071067811865475"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = adapter_cls(model)
    site_map = {
        "Hips": "base",
        "Chest": {"site": "tip_site", "local_position": [0.1, 0.0, 0.0]},
    }

    if adapter_cls is NewtonRuntimeModelAdapter:
        with pytest.warns(RuntimeWarning, match="does not expose compiled model sites"):
            sites = build_semantic_sites(adapter, site_map)
    else:
        sites = build_semantic_sites(adapter, site_map)

    assert sites["Chest"].body_name == "link"
    np.testing.assert_allclose(sites["Chest"].local_position, [0.2, 0.1, 0.0], atol=1e-12)
    assert sites["Chest"].reason == "explicit_model_site_offset"


def test_newton_world_reference_limitation_is_explicit_for_free_root(tmp_path: Path):
    model = _write_model(
        tmp_path,
        "newton_free_world.xml",
        """
<mujoco model="newton_free_world">
  <worldbody>
    <body name="floating">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="root_free" type="free"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = NewtonRuntimeModelAdapter(model)

    with pytest.warns(RuntimeWarning, match="do not expose a synthetic world body"):
        with pytest.raises(KeyError, match="unknown body 'world'"):
            adapter.resolve_body_name("world")
