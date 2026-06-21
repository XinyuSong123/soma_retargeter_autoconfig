from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.canonical_projection import project_canonical_targets
from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position, project_torso_orientation
from soma_retargeter.robotics.v3.kinematic_paths import discover_paths
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites
from soma_retargeter.robotics.v3.spatial import so3_exp, transform


def test_rank_zero_endpoint_preserves_nonzero_unreachable_demand(tmp_path: Path):
    model = tmp_path / "fixed_endpoint.xml"
    model.write_text(
        """
<mujoco model="fixed_endpoint">
  <worldbody>
    <body name="chest">
      <body name="hand" pos="0 0 0">
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Chest": "chest", "LeftHand": "hand"})

    result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        [],
        np.array([1.0, 0.0, 0.0]),
    )

    assert result.status == "unreachable/rank_zero"
    assert not result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.desired, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.projected, [0.0, 0.0, 0.0], atol=1e-12)
    assert result.residual == 1.0


def test_rank_zero_torso_preserves_nonzero_unreachable_rotation(tmp_path: Path):
    model = tmp_path / "fixed_torso.xml"
    model.write_text(
        """
<mujoco model="fixed_torso">
  <worldbody>
    <body name="hips">
      <body name="chest">
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "hips", "Chest": "chest"})

    result = project_torso_orientation(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        [],
        so3_exp([0.0, 0.0, 0.4]),
    )

    assert result.status == "unreachable/rank_zero"
    assert not result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.desired, [0.0, 0.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(result.projected, [0.0, 0.0, 0.0], atol=1e-12)
    assert abs(result.residual - 0.4) < 1e-12


def test_redundant_endpoint_projection_uses_deterministic_range_normalized_priors(tmp_path: Path):
    model = tmp_path / "redundant_slides.xml"
    model.write_text(
        """
<mujoco model="redundant_slides">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="slide_a">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="slide_a" type="slide" axis="1 0 0" range="-1 1"/>
        <body name="hand">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="slide_b" type="slide" axis="1 0 0" range="-1 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Chest": "chest", "LeftHand": "hand"})
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])

    neutral_result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.4, 0.0, 0.0]),
        neutral_prior_weight=1e-6,
        continuity_prior_weight=0.0,
    )
    np.testing.assert_allclose(neutral_result.chain_q, [0.2, 0.2], atol=1e-4)
    assert neutral_result.iterations > 0
    assert neutral_result.status == "converged"
    assert neutral_result.residual < 1e-6

    previous_q = adapter.set_velocity_coordinates(adapter.neutral_q(), active, np.array([0.35, 0.05]))
    continuity_result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.4, 0.0, 0.0]),
        previous_q=previous_q,
        neutral_prior_weight=0.0,
        continuity_prior_weight=1e-6,
    )
    np.testing.assert_allclose(continuity_result.chain_q, [0.35, 0.05], atol=5e-4)
    assert continuity_result.prior_residual_norm < neutral_result.prior_residual_norm + 1e-6


def test_canonical_projection_uses_motion_targets_not_neutral_as_desired(tmp_path: Path):
    model = tmp_path / "canonical_torso.xml"
    model.write_text(
        """
<mujoco model="canonical_torso">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(adapter, {"Hips": "hips", "Chest": "chest"})
    paths = discover_paths(adapter, sites)
    canonical_targets = {
        "torso_yaw": {
            "transforms": {
                "Hips": transform([0, 0, 0], np.eye(3)),
                "Chest": transform([0, 0, 0], so3_exp([0.0, 0.0, 0.25])),
            }
        }
    }

    projections = project_canonical_targets(adapter, sites, paths, canonical_targets)
    torso = projections["torso_yaw"]["torso"]

    assert not torso.neutral_as_desired
    assert torso.desired_source == "canonical_semantic_target_relative_transform"
    assert torso.result.status == "converged"
    np.testing.assert_allclose(torso.result.desired, [0.0, 0.0, 0.25], atol=1e-12)
    np.testing.assert_allclose(torso.result.chain_q, [0.25], atol=1e-6)
    assert torso.result.iterations > 0
