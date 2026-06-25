from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.profile import compile_kinematic_profile_v3


def test_profile_projects_canonical_motion_targets_instead_of_neutral_fk(tmp_path: Path):
    model = tmp_path / "profile_projection.xml"
    model.write_text(
        """
<mujoco model="profile_projection">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 1">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
        <body name="left_hand" pos="0 0.4 0.1">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="left_reach" type="slide" axis="1 0 0" range="-1 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
        <body name="right_hand" pos="0 -0.4 0.1">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="right_reach" type="slide" axis="1 0 0" range="-1 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </body>
      <body name="left_foot" pos="0 0.2 -0.8">
        <geom type="sphere" size="0.01"/>
      </body>
      <body name="right_foot" pos="0 -0.2 -0.8">
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )

    profile = compile_kinematic_profile_v3(
        model,
        {
            "Hips": "hips",
            "Chest": "chest",
            "LeftHand": "left_hand",
            "RightHand": "right_hand",
            "LeftFoot": "left_foot",
            "RightFoot": "right_foot",
        },
        low_discrepancy_count=2,
    )
    reports = profile.projection_reports

    assert "torso_yaw" in reports
    assert "torso" in reports["torso_yaw"]
    assert "arms_forward" in reports
    assert "left_hand" in reports["arms_forward"]
    assert "left_hand" not in reports

    torso_yaw = reports["torso_yaw"]["torso"]
    assert torso_yaw["desired_source"] == "canonical_semantic_target_relative_transform"
    assert torso_yaw["neutral_as_desired"] is False
    assert abs(np.linalg.norm(torso_yaw["desired"]) - 0.25) < 1e-6
    assert torso_yaw["iterations"] > 0
    assert torso_yaw["active_coordinates"] == [0]

    mixed_torso = reports["mixed_torso_rotation"]["torso"]
    assert abs(np.linalg.norm(mixed_torso["desired"]) - 0.25) < 1e-6

    neutral_left = reports["neutral"]["left_hand"]["desired"]
    arms_left = reports["arms_forward"]["left_hand"]["desired"]
    assert not np.allclose(arms_left, neutral_left)
    assert reports["arms_forward"]["left_hand"]["desired_source"] == "canonical_semantic_target_relative_transform"
    assert reports["arms_forward"]["left_hand"]["neutral_as_desired"] is False
