from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def test_endpoint_projection_normalizes_by_neutral_chain_length_not_joint_limit_sweep(tmp_path: Path):
    model = tmp_path / "two_segment.xml"
    model.write_text(
        """
<mujoco model="two_segment">
  <worldbody>
    <body name="base">
      <body name="mid" pos="1 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="hinge_z" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="sphere" size="0.01"/>
        <body name="tip" pos="0 2 0"><geom type="sphere" size="0.01"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
        + "\n"
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "base", "Chest": "tip", "LeftFoot": "base", "RightFoot": "base"})
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])

    result = project_endpoint_position(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active, np.array([1.0, 2.0, 0.0]))

    assert result.normalization_reference == "neutral_chain_length"
    assert result.normalization_scale == 3.0
