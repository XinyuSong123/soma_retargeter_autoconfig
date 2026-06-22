from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.fd_uncertainty import coordinate_step
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.numerical_jacobian import numerical_relative_jacobian
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def test_fd_step_uses_backend_epsilon_one_third_and_chain_scale(tmp_path: Path):
    model = _write(
        tmp_path / "slide.xml",
        """
<mujoco model="slide">
  <worldbody>
    <body name="base">
      <body name="slider">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="slide_x" type="slide" axis="1 0 0" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "base", "Chest": "slider", "LeftFoot": "base", "RightFoot": "base"})
    dof = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])[0]

    step = coordinate_step(adapter, dof, q=adapter.neutral_q(), chain_length=3.0)

    assert step == np.clip(np.finfo(np.float64).eps ** (1.0 / 3.0) * 3.0, 1e-6, 5e-3)


def test_fd_report_records_selected_plateau_and_error_estimate(tmp_path: Path):
    model = _write(
        tmp_path / "hinge.xml",
        """
<mujoco model="hinge">
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

    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    column = jac.column_classifications[0]

    assert column["selected_plateau"] in {"h/h2", "2h/h"}
    assert column["error_estimate"] >= 0.0
    assert column["normalized_error"] is not None
