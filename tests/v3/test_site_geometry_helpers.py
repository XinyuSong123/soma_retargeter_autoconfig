from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.site_geometry import (
    enforce_nonzero_origin,
    infer_distal_hand_site,
    infer_foot_sites,
)


def test_distal_hand_site_uses_geometry_bounds_not_body_origin(tmp_path: Path):
    model = tmp_path / "hand.xml"
    model.write_text(
        """
<mujoco model="hand">
  <worldbody>
    <body name="hand">
      <geom type="box" pos="0.08 0.01 0" size="0.04 0.02 0.01"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)

    site = infer_distal_hand_site(adapter, "hand", semantic_name="LeftHand")

    np.testing.assert_allclose(site.local_position, [0.12, 0.01, 0.0], atol=1e-12)
    assert site.source == "geometry_bounds"
    assert "distal_hand_axis_bounds" in site.evidence


def test_foot_site_helpers_generate_sole_toe_and_heel_with_nonzero_gate(tmp_path: Path):
    model = tmp_path / "foot.xml"
    model.write_text(
        """
<mujoco model="foot">
  <worldbody>
    <body name="foot">
      <geom type="box" pos="0.02 0 -0.03" size="0.12 0.04 0.02"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)

    sites = infer_foot_sites(adapter, "foot", side_prefix="Left")

    np.testing.assert_allclose(sites["LeftFoot"].local_position, [0.02, 0.0, -0.05], atol=1e-12)
    assert sites["LeftToe"].local_position[0] > sites["LeftHeel"].local_position[0]
    assert sites["LeftToe"].local_position[2] == sites["LeftFoot"].local_position[2]
    assert sites["LeftHeel"].local_position[2] == sites["LeftFoot"].local_position[2]


def test_nonzero_origin_gate_rejects_fake_distal_site():
    with pytest.raises(ValueError, match="body origin"):
        enforce_nonzero_origin("LeftHand", [0.0, 0.0, 0.0], source="body_name_only")
