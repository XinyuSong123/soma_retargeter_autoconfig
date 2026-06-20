from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position, project_torso_orientation
from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter
from soma_retargeter.robotics.v3.kinematic_paths import discover_paths
from soma_retargeter.robotics.v3.numerical_jacobian import (
    engine_translation_jacobian_crosscheck,
    matrix_rank_and_singular_values,
    numerical_relative_jacobian,
)
from soma_retargeter.robotics.v3.profile import compile_kinematic_profile_v3
from soma_retargeter.robotics.v3.reachability import analyze_reachability, deterministic_chain_samples
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites, load_robot_zoo_semantic_map
from soma_retargeter.robotics.v3.spatial import frame_from_yz, relative_transform, so3_exp, so3_log, transform, wahba_alignment
from soma_retargeter.robotics.v3.validation import REQUIRED_ARTIFACT_IDS, write_validation_artifacts


def test_so3_log_exp_round_trip():
    v = np.array([0.2, -0.3, 0.1])
    np.testing.assert_allclose(so3_log(so3_exp(v)), v, atol=1e-12)


def test_relative_transform_invariant_under_common_world_transform():
    reference = transform([0.2, -0.1, 0.5], so3_exp(np.array([0.1, 0.2, -0.3])))
    target = transform([0.8, 0.3, 0.2], so3_exp(np.array([-0.2, 0.1, 0.4])))
    common = transform([3.0, -2.0, 1.0], so3_exp(np.array([0.4, -0.1, 0.2])))
    np.testing.assert_allclose(
        relative_transform(reference, target),
        relative_transform(common @ reference, common @ target),
        atol=1e-12,
    )


def test_wahba_alignment_recovers_known_rotation():
    source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]])
    expected = so3_exp(np.array([0.2, -0.4, 0.1]))
    target = (expected @ source.T).T
    np.testing.assert_allclose(wahba_alignment(source, target), expected, atol=1e-12)


def test_frame_from_yz_reports_degenerate_parallel_hints():
    frame, confidence, reason = frame_from_yz(np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 0.0]))
    np.testing.assert_allclose(frame, np.eye(3), atol=1e-12)
    assert confidence == 0.0
    assert reason == "degenerate_yz"


def test_rpo_full_chain_paths_exclude_common_torso_coordinate():
    adapter = MuJoCoRuntimeModelAdapter("assets/robots/atom01/mjcf/atom01.xml")
    sites = build_semantic_sites(adapter, load_robot_zoo_semantic_map("roboparty_rpo_local"))
    paths = discover_paths(adapter, sites)

    assert paths["torso"].coordinate_labels == ["torso_joint"]
    assert paths["left_hand"].coordinate_labels == [
        "left_arm_pitch_joint",
        "left_arm_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
    ]
    assert paths["left_foot"].coordinate_labels == [
        "left_thigh_yaw_joint",
        "left_thigh_roll_joint",
        "left_thigh_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]


def test_rpo_torso_rotation_jacobian_rank_one():
    adapter = MuJoCoRuntimeModelAdapter("assets/robots/atom01/mjcf/atom01.xml")
    sites = build_semantic_sites(adapter, load_robot_zoo_semantic_map("roboparty_rpo_local"))
    paths = discover_paths(adapter, sites)
    jac = numerical_relative_jacobian(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        paths["torso"].active_velocity_coordinates,
    )
    rank, singular_values = matrix_rank_and_singular_values(jac.rotation)
    assert rank == 1
    assert singular_values[0] > 0.9


def test_newton_rpo_backend_matches_full_chain_contract():
    adapter = NewtonRuntimeModelAdapter("assets/robots/atom01/mjcf/atom01.xml")
    sites = build_semantic_sites(adapter, load_robot_zoo_semantic_map("roboparty_rpo_local"))
    paths = discover_paths(adapter, sites)

    assert adapter.nq == 30
    assert adapter.nv == 29
    assert paths["torso"].coordinate_labels == ["torso_joint"]
    assert paths["left_hand"].coordinate_labels == [
        "left_arm_pitch_joint",
        "left_arm_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
    ]
    jac = numerical_relative_jacobian(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        paths["torso"].active_velocity_coordinates,
    )
    rank, singular_values = matrix_rank_and_singular_values(jac.rotation)
    assert rank == 1
    assert singular_values[0] > 0.9
    cross = engine_translation_jacobian_crosscheck(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        paths["torso"].active_velocity_coordinates,
        jac.translation,
    )
    assert cross["available"]
    assert cross["max_abs_error"] < 1e-5


def test_non_world_axis_hinge_translation_jacobian(tmp_path: Path):
    model = tmp_path / "oblique.xml"
    model.write_text(
        """
<mujoco model="oblique">
  <worldbody>
    <body name="base">
      <body name="link" pos="0 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="hinge" type="hinge" axis="1 1 0" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
        <site name="tip" pos="0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": {"body": "link", "local_position": [0, 0, 1]},
            "LeftHand": "link",
            "RightHand": "link",
            "LeftFoot": "link",
            "RightFoot": "link",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    expected_axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    expected = np.cross(expected_axis, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(jac.translation[:, 0], expected, atol=1e-5)


def test_prismatic_numerical_jacobian(tmp_path: Path):
    model = tmp_path / "slide.xml"
    model.write_text(
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
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": "slider",
            "LeftHand": "slider",
            "RightHand": "slider",
            "LeftFoot": "slider",
            "RightFoot": "slider",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    np.testing.assert_allclose(jac.translation[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(jac.rotation[:, 0], np.zeros(3), atol=1e-8)


def test_prismatic_torso_profile_compiles_with_translation_rank_one(tmp_path: Path):
    model = tmp_path / "prismatic_torso.xml"
    model.write_text(
        """
<mujoco model="prismatic_torso">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="spine_slide_z" type="slide" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="sphere" size="0.01"/>
        <body name="left_hand" pos="0 0.4 0"><geom type="sphere" size="0.01"/></body>
        <body name="right_hand" pos="0 -0.4 0"><geom type="sphere" size="0.01"/></body>
      </body>
      <body name="left_foot" pos="0 0.12 -0.8"><geom type="sphere" size="0.01"/></body>
      <body name="right_foot" pos="0 -0.12 -0.8"><geom type="sphere" size="0.01"/></body>
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
        model_id="prismatic_torso",
        low_discrepancy_count=4,
    )
    assert profile.failures == []
    assert profile.capability_status == "full_humanoid_ready"
    assert profile.rank_stability["torso"].regular_rank_translation == 1
    assert profile.rank_stability["torso"].regular_rank_rotation == 0
    assert profile.neutral_jacobians["torso"]["engine_translation_crosscheck"]["available"]


def test_reachability_samples_are_centered_on_neutral_not_limit_midpoint(tmp_path: Path):
    model = tmp_path / "offset_neutral.xml"
    model.write_text(
        """
<mujoco model="offset_neutral">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="offset_yaw" type="hinge" axis="0 0 1" range="0 2" ref="0.3"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
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
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    samples = deterministic_chain_samples(adapter, active, low_discrepancy_count=0)
    values = [float(sample[0]) for sample in samples[:3]]
    np.testing.assert_allclose(values, [0.3, 0.1, 0.5], atol=1e-12)
    assert 1.0 not in values


def test_free_joint_tangent_integration_does_not_add_quaternion_components(tmp_path: Path):
    model = tmp_path / "free.xml"
    model.write_text(
        """
<mujoco model="free_body">
  <worldbody>
    <body name="floating">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="free" type="free"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    q0 = adapter.neutral_q()
    delta = np.zeros(adapter.nv)
    delta[3:6] = np.array([0.2, -0.1, 0.05])
    q1 = adapter.integrate(q0, delta)
    assert abs(np.linalg.norm(q1[3:7]) - 1.0) < 1e-12
    assert not np.allclose(q1[3:6] - q0[3:6], delta[3:6])


def test_ball_joint_tangent_integration_keeps_quaternion_normalized(tmp_path: Path):
    model = tmp_path / "ball.xml"
    model.write_text(
        """
<mujoco model="ball_body">
  <worldbody>
    <body name="base">
      <body name="ball_link">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="ball" type="ball"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    q0 = adapter.neutral_q()
    delta = np.array([0.2, -0.1, 0.05])
    q1 = adapter.integrate(q0, delta)
    assert abs(np.linalg.norm(q1[0:4]) - 1.0) < 1e-12


def test_one_dof_curved_workspace_remains_local_rank_one(tmp_path: Path):
    model = tmp_path / "curved.xml"
    model.write_text(
        """
<mujoco model="curved">
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
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": {"body": "link", "local_position": [1, 0, 0]},
            "LeftHand": "link",
            "RightHand": "link",
            "LeftFoot": "link",
            "RightFoot": "link",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    report = analyze_reachability(adapter, sites["Hips"], sites["Chest"], active, low_discrepancy_count=8)
    assert report.regular_rank_translation == 1
    assert report.nominal_rank_translation == 1


def test_multi_pose_rank_recovers_serial_arm_singularity(tmp_path: Path):
    model = tmp_path / "two_link.xml"
    model.write_text(
        """
<mujoco model="two_link">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="link1">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
        <geom type="capsule" size="0.01 0.2" fromto="0 0 0 1 0 0"/>
        <body name="link2" pos="1 0 0">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="elbow" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
          <geom type="capsule" size="0.01 0.2" fromto="0 0 0 1 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": {"body": "link2", "local_position": [1, 0, 0]},
            "LeftFoot": "base",
            "RightFoot": "base",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    neutral_jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    neutral_rank, _ = matrix_rank_and_singular_values(neutral_jac.translation)
    report = analyze_reachability(adapter, sites["Hips"], sites["Chest"], active, low_discrepancy_count=8)
    assert neutral_rank == 1
    assert report.regular_rank_translation == 2
    assert report.singularity_fraction_translation > 0.0


def test_rank_zero_torso_compiles(tmp_path: Path):
    model = tmp_path / "fixed_torso.xml"
    model.write_text(
        """
<mujoco model="fixed_torso">
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 1">
        <geom type="sphere" size="0.01"/>
      </body>
      <body name="left_foot" pos="0.1 0 -1"><geom type="sphere" size="0.01"/></body>
      <body name="right_foot" pos="-0.1 0 -1"><geom type="sphere" size="0.01"/></body>
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
            "LeftFoot": "left_foot",
            "RightFoot": "right_foot",
        },
        model_id="fixed_torso",
        low_discrepancy_count=1,
    )
    assert profile.failures == []
    assert profile.capability_status == "partial_humanoid"
    assert profile.rank_stability["torso"].regular_rank_rotation == 0
    assert profile.projection_reports["torso"]["status"] == "rank_zero"


def test_bounded_single_axis_torso_projection_respects_limits(tmp_path: Path):
    model = tmp_path / "single_axis_torso.xml"
    model.write_text(
        """
<mujoco model="single_axis">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-0.2 0.2"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
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
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    result = project_torso_orientation(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active, so3_exp([0, 0, 1.0]))
    q = result.chain_q
    info = adapter.coordinate(active[0])
    assert info.lower - 1e-9 <= q[info.qpos_adr] <= info.upper + 1e-9
    assert abs(q[info.qpos_adr] - 0.2) < 1e-5


def test_multi_axis_torso_projection_converges(tmp_path: Path):
    model = tmp_path / "multi_axis_torso.xml"
    model.write_text(
        """
<mujoco model="multi_axis">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="waist_yaw">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="yaw" type="hinge" axis="0 0 1" range="-1 1"/>
        <body name="waist_pitch">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="pitch" type="hinge" axis="0 1 0" range="-1 1"/>
          <body name="chest">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint name="roll" type="hinge" axis="1 0 0" range="-1 1"/>
            <geom type="sphere" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
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
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    result = project_torso_orientation(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        active,
        so3_exp(np.array([0.25, -0.2, 0.3])),
    )
    assert result.converged
    assert result.residual < 1e-6


def test_endpoint_projection_respects_position_joint_limit(tmp_path: Path):
    model = tmp_path / "limited_endpoint.xml"
    model.write_text(
        """
<mujoco model="limited_endpoint">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="hand">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="reach_slide" type="slide" axis="1 0 0" range="0 0.2"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
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
    result = project_endpoint_position(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([1.0, 0.0, 0.0]),
    )
    info = adapter.coordinate(active[0])
    assert result.converged
    assert abs(result.chain_q[info.qpos_adr] - 0.2) < 1e-6
    np.testing.assert_allclose(result.projected, [0.2, 0.0, 0.0], atol=1e-6)


def test_yaw_only_torso_uses_full_five_dof_no_wrist_hand_path(tmp_path: Path):
    model = tmp_path / "yaw_no_wrist.xml"
    model.write_text(
        """
<mujoco model="yaw_no_wrist">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
        <body name="left_shoulder">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="left_arm_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
          <body name="left_upper_arm">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint name="left_arm_roll" type="hinge" axis="1 0 0" range="-1 1"/>
            <body name="left_elbow">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <joint name="left_arm_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
              <body name="left_forearm">
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                <joint name="left_elbow_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
                <body name="left_hand" pos="0.4 0 0">
                  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                  <joint name="left_elbow_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
                  <geom type="sphere" size="0.01"/>
                </body>
              </body>
            </body>
          </body>
        </body>
        <body name="right_hand" pos="0 -0.4 0"><geom type="sphere" size="0.01"/></body>
      </body>
      <body name="left_foot" pos="0 0.12 -0.8"><geom type="sphere" size="0.01"/></body>
      <body name="right_foot" pos="0 -0.12 -0.8"><geom type="sphere" size="0.01"/></body>
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
            "LeftHand": {"body": "left_hand", "local_position": [0.1, 0, 0]},
            "RightHand": "right_hand",
            "LeftFoot": "left_foot",
            "RightFoot": "right_foot",
        },
        model_id="yaw_no_wrist",
        low_discrepancy_count=2,
    )
    assert profile.failures == []
    assert profile.rank_stability["torso"].regular_rank_rotation == 1
    assert profile.chains["left_hand"].coordinate_labels == [
        "left_arm_pitch",
        "left_arm_roll",
        "left_arm_yaw",
        "left_elbow_pitch",
        "left_elbow_yaw",
    ]


def test_three_dof_torso_and_wrist_path_are_discovered(tmp_path: Path):
    model = tmp_path / "torso_wrist.xml"
    model.write_text(
        """
<mujoco model="torso_wrist">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="waist_yaw">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
        <body name="waist_pitch">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="waist_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
          <body name="chest">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint name="waist_roll" type="hinge" axis="1 0 0" range="-1 1"/>
            <geom type="sphere" size="0.01"/>
            <body name="left_shoulder">
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
              <body name="left_wrist" pos="0.3 0 0">
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                <joint name="left_wrist_yaw" type="hinge" axis="0 0 1" range="-1 1"/>
                <body name="left_hand" pos="0.1 0 0"><geom type="sphere" size="0.01"/></body>
              </body>
            </body>
            <body name="right_hand" pos="0 -0.4 0"><geom type="sphere" size="0.01"/></body>
          </body>
        </body>
      </body>
      <body name="left_foot" pos="0 0.12 -0.8"><geom type="sphere" size="0.01"/></body>
      <body name="right_foot" pos="0 -0.12 -0.8"><geom type="sphere" size="0.01"/></body>
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
        model_id="torso_wrist",
        low_discrepancy_count=2,
    )
    assert profile.failures == []
    assert profile.chains["torso"].coordinate_labels == ["waist_yaw", "waist_pitch", "waist_roll"]
    assert profile.chains["left_hand"].coordinate_labels == ["left_shoulder_pitch", "left_wrist_yaw"]


def test_asymmetric_humanoid_records_bilateral_length_delta(tmp_path: Path):
    model = tmp_path / "asymmetric.xml"
    model.write_text(
        """
<mujoco model="asymmetric">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="sphere" size="0.01"/>
        <body name="left_hand" pos="0 0.45 0"><geom type="sphere" size="0.01"/></body>
        <body name="right_hand" pos="0 -0.45 0"><geom type="sphere" size="0.01"/></body>
      </body>
      <body name="left_foot" pos="0 0.12 -0.8"><geom type="sphere" size="0.01"/></body>
      <body name="right_foot" pos="0 -0.12 -1.1"><geom type="sphere" size="0.01"/></body>
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
        model_id="asymmetric",
        low_discrepancy_count=2,
    )
    assert profile.failures == []
    assert profile.rest_calibration.bilateral_symmetry["leg_length_abs_delta"] > 0.25


def test_lower_body_only_partial_humanoid_has_no_hand_chains(tmp_path: Path):
    model = tmp_path / "lower_body.xml"
    model.write_text(
        """
<mujoco model="lower_body">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 0.8"><geom type="sphere" size="0.01"/></body>
      <body name="left_thigh">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
        <body name="left_foot" pos="0 0.12 -0.8"><geom type="sphere" size="0.01"/></body>
      </body>
      <body name="right_thigh">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
        <body name="right_foot" pos="0 -0.12 -0.8"><geom type="sphere" size="0.01"/></body>
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
            "LeftFoot": "left_foot",
            "RightFoot": "right_foot",
        },
        model_id="lower_body",
        low_discrepancy_count=2,
    )
    assert profile.failures == []
    assert profile.capability_status == "partial_humanoid"
    assert {"torso", "left_foot", "right_foot"} == set(profile.chains)
    assert "left_hand" not in profile.rank_stability
    assert profile.rank_stability["left_foot"].regular_rank_translation == 1


def test_compile_rpo_profile_v3_smoke():
    profile = compile_kinematic_profile_v3(
        "assets/robots/atom01/mjcf/atom01.xml",
        load_robot_zoo_semantic_map("roboparty_rpo_local"),
        model_id="roboparty_rpo",
        low_discrepancy_count=1,
    )
    assert profile.failures == []
    assert profile.rank_stability["torso"].regular_rank_rotation == 1
    assert profile.rest_calibration.max_position_error < 0.001
    assert "neutral" in profile.canonical_targets
    assert profile.canonical_targets["arms_forward"]["transforms"] != profile.canonical_targets["neutral"]["transforms"]
    for target in profile.canonical_targets.values():
        for error in target["segment_length_errors"].values():
            assert abs(error) < 1e-9
    validation = profile.canonical_target_validation
    assert validation["neutral_reconstruction"]["max_position_error"] == 0.0
    assert validation["neutral_reconstruction"]["max_orientation_error"] == 0.0
    assert validation["root_translation_equivariant"]
    assert validation["global_root_yaw_equivariant"]
    assert validation["failures"] == []


def test_canonical_target_validation_records_motion_diagnostics():
    profile = compile_kinematic_profile_v3(
        "assets/robots/atom01/mjcf/atom01.xml",
        load_robot_zoo_semantic_map("roboparty_rpo_local"),
        model_id="roboparty_rpo",
        backend="newton",
        low_discrepancy_count=1,
    )
    per_motion = profile.canonical_target_validation["per_motion"]
    assert set(per_motion) == set(profile.canonical_targets)
    assert per_motion["root_translation"]["max_parent_frame_edge_delta_from_neutral"] == 0.0
    assert per_motion["global_root_yaw"]["max_parent_frame_edge_delta_from_neutral"] < 1e-9
    assert per_motion["arms_forward"]["max_parent_frame_edge_delta_from_neutral"] > 0.0


def test_profile_deterministic_hash_ignores_timing():
    kwargs = dict(
        model_path="assets/robots/atom01/mjcf/atom01.xml",
        semantic_map=load_robot_zoo_semantic_map("roboparty_rpo_local"),
        model_id="roboparty_rpo",
        backend="newton",
        low_discrepancy_count=1,
    )
    a = compile_kinematic_profile_v3(**kwargs).to_json()
    b = compile_kinematic_profile_v3(**kwargs).to_json()
    assert a["deterministic_hash"] == b["deterministic_hash"]


def test_compile_rpo_profile_v3_newton_backend():
    profile = compile_kinematic_profile_v3(
        "assets/robots/atom01/mjcf/atom01.xml",
        load_robot_zoo_semantic_map("roboparty_rpo_local"),
        model_id="roboparty_rpo",
        backend="newton",
        low_discrepancy_count=1,
    )
    assert profile.failures == []
    assert profile.model["backend"] == "newton"
    assert profile.rank_stability["torso"].regular_rank_rotation == 1


def test_validation_artifacts_include_required_reports(tmp_path: Path):
    summary = write_validation_artifacts(tmp_path / "artifacts", low_discrepancy_count=1)
    assert summary["manifest"]["model_count"] >= len(REQUIRED_ARTIFACT_IDS)
    assert len(summary["reports"]) == summary["manifest"]["model_count"]
    assert (tmp_path / "artifacts" / "environment.json").exists()
    reports = {}
    for report_id in REQUIRED_ARTIFACT_IDS:
        report_path = tmp_path / "artifacts" / "per_robot" / f"{report_id}.json"
        assert report_path.exists()
        import json

        reports[report_id] = json.loads(report_path.read_text())
        assert reports[report_id]["model"]["backend"] == "newton"
        assert reports[report_id]["status"] in {
            "passed",
            "partial_passed",
            "negative_control_passed",
            "source_unavailable",
            "model_load_failed",
            "algorithm_failed",
            "semantic_failed",
            "license_blocked",
        }

    rpo = reports["roboparty_rpo_local"]
    assert rpo["status"] == "passed"
    assert rpo["failures"] == []
    assert rpo["canonical_target_validation"]["failures"] == []
    assert rpo["canonical_target_validation"]["root_translation_equivariant"]
    assert rpo["canonical_target_validation"]["global_root_yaw_equivariant"]
    assert rpo["rank_stability"]["torso"]["regular_rank_rotation"] == 1
    np.testing.assert_allclose(np.ravel(rpo["neutral_jacobians"]["torso"]["rotation"]), [0.0, 0.0, 1.0], atol=1e-5)
    assert rpo["chains"]["left_hand"]["coordinate_labels"] == [
        "left_arm_pitch_joint",
        "left_arm_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
    ]
    assert rpo["chains"]["left_foot"]["coordinate_labels"] == [
        "left_thigh_yaw_joint",
        "left_thigh_roll_joint",
        "left_thigh_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    assert "canonical_projection_reports" in rpo


def test_urdf_mesh_paths_are_absolutized_for_g1_description():
    path = Path("/mnt/ssd1/song/.cache/robot_descriptions/unitree_ros/robots/g1_description/g1_29dof.urdf")
    if not path.exists():
        return
    adapter = MuJoCoRuntimeModelAdapter(path)
    assert adapter.nv == 29


def test_berkeley_lower_body_downgrades_to_partial_humanoid():
    path = Path("/mnt/ssd1/song/.cache/robot_descriptions/berkeley_humanoid_description/urdf/robot.urdf")
    if not path.exists():
        return
    adapter = MuJoCoRuntimeModelAdapter(path)
    from soma_retargeter.robotics.v3.semantic_sites import infer_semantic_map_from_body_names

    semantic_map = infer_semantic_map_from_body_names(adapter)
    profile = compile_kinematic_profile_v3(path, semantic_map, model_id="berkeley_humanoid", low_discrepancy_count=1)
    assert profile.failures == []
    assert profile.capability_status == "partial_humanoid"
    assert {"left_foot", "right_foot"} <= set(profile.chains)
