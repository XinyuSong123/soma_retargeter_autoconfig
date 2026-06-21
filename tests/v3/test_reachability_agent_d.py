from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.reachability import analyze_reachability
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def test_regular_rank_uses_local_ranks_not_concatenated_tangent_span(tmp_path: Path):
    model = tmp_path / "one_dof_circle.xml"
    model.write_text(
        """
<mujoco model="one_dof_circle">
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

    report = analyze_reachability(adapter, sites["Hips"], sites["Chest"], active, low_discrepancy_count=16)
    local_ranks = [sample["translation"]["local_rank"] for sample in report.sample_diagnostics]

    assert report.samples == 19
    assert report.regular_rank_translation == 1
    assert report.nominal_rank_translation == 1
    assert max(local_ranks) == 1
    assert report.epsilon_unstable_fraction == 0.0

    tangent_columns = []
    for sample in report.sample_diagnostics:
        sv = np.asarray(sample["translation"]["singular_values"])
        assert sv.size == 1
        tangent_columns.append(sv[0])
    assert max(tangent_columns) > 0.99
