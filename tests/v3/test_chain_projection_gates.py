from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.canonical_projection import project_canonical_motion_sequence
from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position, project_torso_orientation
from soma_retargeter.robotics.v3.kinematic_paths import discover_paths
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites
from soma_retargeter.robotics.v3.spatial import so3_exp, transform
from soma_retargeter.robotics.v3.target_builder import SemanticTargets


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def test_rank_zero_endpoint_preserves_nonzero_demand_as_unreachable(tmp_path: Path):
    model = _write(
        tmp_path / "fixed_endpoint.xml",
        """
<mujoco model="fixed_endpoint">
  <worldbody>
    <body name="chest">
      <body name="hand"><geom type="sphere" size="0.01"/></body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "chest",
            "Chest": "chest",
            "LeftHand": "hand",
            "RightHand": "hand",
            "LeftFoot": "chest",
            "RightFoot": "chest",
        },
    )

    result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        [],
        np.array([0.1, 0.0, 0.0]),
    )

    assert result.status == "unreachable/rank_zero"
    assert not result.converged
    assert result.iterations == 0
    assert result.active_coordinates == []
    np.testing.assert_allclose(result.desired, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(result.projected, [0.0, 0.0, 0.0])
    assert abs(result.residual - 0.1) < 1e-12
    assert result.to_json()["status"] == "unreachable/rank_zero"


def test_rank_zero_torso_preserves_nonzero_rotation_demand_as_unreachable(tmp_path: Path):
    model = _write(
        tmp_path / "fixed_torso.xml",
        """
<mujoco model="fixed_torso">
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 1"><geom type="sphere" size="0.01"/></body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "hips",
            "Chest": "chest",
            "LeftFoot": "hips",
            "RightFoot": "hips",
        },
    )

    result = project_torso_orientation(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        [],
        so3_exp(np.array([0.0, 0.0, 0.2])),
    )

    assert result.status == "unreachable/rank_zero"
    assert not result.converged
    assert result.iterations == 0
    np.testing.assert_allclose(result.desired, [0.0, 0.0, 0.2], atol=1e-12)
    np.testing.assert_allclose(result.projected, [0.0, 0.0, 0.0], atol=1e-12)
    assert abs(result.residual - 0.2) < 1e-12


def test_projection_uses_range_normalized_continuity_prior_for_redundant_chain(tmp_path: Path):
    model = _write(
        tmp_path / "redundant_slides.xml",
        """
<mujoco model="redundant_slides">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="slide_a">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="slide_a" type="slide" axis="1 0 0" range="0 1"/>
        <body name="hand">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="slide_b" type="slide" axis="1 0 0" range="0 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
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
            "Hips": "chest",
            "Chest": "chest",
            "LeftHand": "hand",
            "RightHand": "hand",
            "LeftFoot": "chest",
            "RightFoot": "chest",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])
    previous_q = adapter.set_velocity_coordinates(adapter.neutral_q(), active, np.array([0.2, 0.0]))

    result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.2, 0.0, 0.0]),
        neutral_prior_weight=0.0,
        continuity_prior_weight=10.0,
        previous_q=previous_q,
    )

    assert result.status == "converged"
    assert result.iterations > 0
    assert result.active_coordinates == active
    assert result.prior_residual_norm < 1e-8
    np.testing.assert_allclose(result.projected, [0.2, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(result.chain_q[:2], [0.2, 0.0], atol=1e-8)


def test_canonical_projection_consumes_non_neutral_semantic_targets(tmp_path: Path):
    model = _write(
        tmp_path / "canonical_slide.xml",
        """
<mujoco model="canonical_slide">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="hand">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="reach_slide" type="slide" axis="1 0 0" range="0 0.3"/>
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
            "Hips": "chest",
            "Chest": "chest",
            "LeftHand": "hand",
            "RightHand": "hand",
            "LeftFoot": "chest",
            "RightFoot": "chest",
        },
    )
    paths = discover_paths(adapter, sites)
    neutral_transforms = {
        "Hips": transform(),
        "Chest": transform(),
        "LeftHand": transform(),
        "RightHand": transform(),
        "LeftFoot": transform(),
        "RightFoot": transform(),
    }
    reach_transforms = {name: value.copy() for name, value in neutral_transforms.items()}
    reach_transforms["LeftHand"] = transform([0.15, 0.0, 0.0])
    canonical_targets = {
        "neutral": SemanticTargets(neutral_transforms, {}, mode="neutral"),
        "arms_forward": SemanticTargets(reach_transforms, {}, mode="arms_forward"),
    }

    report = project_canonical_motion_sequence(
        adapter,
        sites,
        paths,
        canonical_targets,
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()

    left_hand = report["motions"]["arms_forward"]["tasks"]["left_hand"]
    np.testing.assert_allclose(left_hand["desired"], [0.15, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(left_hand["projected"], [0.15, 0.0, 0.0], atol=1e-8)
    assert left_hand["desired_source"] == "canonical_targets.transforms"
    assert left_hand["desired"] != report["motions"]["neutral"]["tasks"]["left_hand"]["desired"]
    assert report["target_source"] == "canonical_semantic_targets"
    assert report["failures"] == []
