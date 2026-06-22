from __future__ import annotations

from pathlib import Path

from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.numerical_jacobian import numerical_relative_jacobian
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def test_engine_zero_column_is_classified_as_numerically_zero(tmp_path: Path):
    model = tmp_path / "irrelevant_branch.xml"
    model.write_text(
        """
<mujoco model="irrelevant_branch">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="irrelevant">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="irrelevant_z" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
      <body name="target" pos="1 0 0">
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
        + "\n"
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "base", "Chest": "target", "LeftFoot": "base", "RightFoot": "base"})

    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], [0])

    assert jac.column_classifications[0]["class"] == "numerically_zero"
    assert jac.column_classifications[0]["engine_norm"] == 0.0
