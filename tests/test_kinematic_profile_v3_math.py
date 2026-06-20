from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.chain_projection import project_endpoint_position, project_torso_orientation
from soma_retargeter.robotics.v3.numerical_jacobian import matrix_rank_and_singular_values, numerical_relative_jacobian
from soma_retargeter.robotics.v3.reachability import analyze_reachability, deterministic_chain_samples
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites
from soma_retargeter.robotics.v3.spatial import so3_exp


def _sites(adapter: MuJoCoRuntimeModelAdapter, reference: str, target: str, target_position=None):
    target_site = target if target_position is None else {"body": target, "local_position": target_position}
    return build_semantic_sites(
        adapter,
        {
            "Hips": reference,
            "Chest": target_site,
            "LeftHand": target,
            "RightHand": target,
            "LeftFoot": reference,
            "RightFoot": reference,
        },
    )


def test_unbounded_two_link_reachability_samples_recover_regular_rank(tmp_path: Path):
    model = tmp_path / "unbounded_two_link.xml"
    model.write_text(
        """
<mujoco model="unbounded_two_link">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <body name="link1">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="shoulder" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.01 0.2" fromto="0 0 0 1 0 0"/>
        <body name="link2" pos="1 0 0">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="elbow" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.01 0.2" fromto="0 0 0 1 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = _sites(adapter, "base", "link2", [1, 0, 0])
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])

    neutral = adapter.neutral_q()
    samples = deterministic_chain_samples(adapter, active, low_discrepancy_count=8)
    assert len(samples) > 1
    assert any(not np.allclose(sample, neutral) for sample in samples[1:])

    neutral_jac = numerical_relative_jacobian(adapter, neutral, sites["Hips"], sites["Chest"], active)
    neutral_rank, _ = matrix_rank_and_singular_values(neutral_jac.translation)
    report = analyze_reachability(adapter, sites["Hips"], sites["Chest"], active, low_discrepancy_count=8)

    assert neutral_rank == 1
    assert report.regular_rank_translation == 2
    assert report.singularity_fraction_translation > 0.0
    assert len(report.sample_diagnostics) == report.samples
    first = report.sample_diagnostics[0]
    assert first["translation"]["singular_values"]
    assert first["translation"]["local_rank"] == neutral_rank
    assert "conditioning" in first["translation"]


def test_epsilon_halving_records_small_analytic_prismatic_and_revolute_discrepancy(tmp_path: Path):
    prismatic_model = tmp_path / "slide.xml"
    prismatic_model.write_text(
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
    adapter = MuJoCoRuntimeModelAdapter(prismatic_model)
    sites = _sites(adapter, "base", "slider")
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    np.testing.assert_allclose(jac.translation[:, 0], [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(jac.rotation[:, 0], [0.0, 0.0, 0.0], atol=1e-8)
    assert jac.unstable_columns == []
    assert jac.epsilon_discrepancies[0]["translation_max_abs"] < 1e-4
    assert jac.epsilon_discrepancies[0]["rotation_max_abs"] < 1e-4

    revolute_model = tmp_path / "oblique.xml"
    revolute_model.write_text(
        """
<mujoco model="oblique">
  <worldbody>
    <body name="base">
      <body name="link">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="hinge" type="hinge" axis="1 1 0" range="-1 1"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    adapter = MuJoCoRuntimeModelAdapter(revolute_model)
    sites = _sites(adapter, "base", "link", [0, 0, 1])
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    jac = numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(jac.translation[:, 0], np.cross(axis, [0.0, 0.0, 1.0]), atol=1e-4)
    np.testing.assert_allclose(jac.rotation[:, 0], axis, atol=1e-4)
    assert jac.unstable_columns == []
    assert jac.epsilon_discrepancies[0]["translation_max_abs"] < 1e-4
    assert jac.epsilon_discrepancies[0]["rotation_max_abs"] < 1e-4


def _torso_model(path: Path, axis: str = "0 0 1", parent_euler: str = "0 0 0") -> Path:
    path.write_text(
        f"""
<mujoco model="torso">
  <compiler angle="radian"/>
  <worldbody>
    <body name="world_rot" euler="{parent_euler}">
      <body name="hips">
        <body name="chest">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="waist" type="hinge" axis="{axis}" range="-0.5 0.5"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()
    )
    return path


def test_torso_projection_accepts_compatible_yaw_rejects_pitch_roll_and_is_world_rotation_invariant(tmp_path: Path):
    base_model = _torso_model(tmp_path / "yaw.xml")
    rotated_model = _torso_model(tmp_path / "yaw_rotated.xml", parent_euler="0.4 -0.2 0.3")
    yaw_angle = 0.3
    pitch_roll = so3_exp(np.array([0.2, -0.2, 0.0]))

    yaw_solutions = []
    incompatible_responses = []
    for model in (base_model, rotated_model):
        adapter = MuJoCoRuntimeModelAdapter(model)
        sites = _sites(adapter, "hips", "chest")
        active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
        yaw = project_torso_orientation(
            adapter,
            adapter.neutral_q(),
            sites["Hips"],
            sites["Chest"],
            active,
            so3_exp(np.array([0.0, 0.0, yaw_angle])),
        )
        incompatible = project_torso_orientation(
            adapter,
            adapter.neutral_q(),
            sites["Hips"],
            sites["Chest"],
            active,
            pitch_roll,
        )
        info = adapter.coordinate(active[0])
        yaw_solutions.append(float(yaw.chain_q[info.qpos_adr]))
        incompatible_responses.append(float(abs(incompatible.chain_q[info.qpos_adr])))
        assert info.lower - 1e-9 <= yaw.chain_q[info.qpos_adr] <= info.upper + 1e-9
        assert abs(yaw.chain_q[info.qpos_adr] - yaw_angle) < 1e-6

    np.testing.assert_allclose(yaw_solutions[0], yaw_solutions[1], atol=1e-9)
    assert max(incompatible_responses) <= 0.05 * abs(yaw_angle)


def test_oblique_hinge_projection_preserves_axis_rotation_and_rejects_perpendicular_rotation(tmp_path: Path):
    model = _torso_model(tmp_path / "oblique.xml", axis="1 1 0")
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = _sites(adapter, "hips", "chest")
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    perpendicular = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)

    compatible = project_torso_orientation(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        active,
        so3_exp(axis * 0.25),
    )
    incompatible = project_torso_orientation(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        active,
        so3_exp(perpendicular * 0.25),
    )
    info = adapter.coordinate(active[0])
    assert abs(compatible.chain_q[info.qpos_adr] - 0.25) < 1e-6
    assert abs(incompatible.chain_q[info.qpos_adr]) <= 0.05 * abs(compatible.chain_q[info.qpos_adr])


def test_endpoint_projection_reports_normalized_residual_at_joint_limit(tmp_path: Path):
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
    assert abs(result.residual - 0.8) < 1e-6
    assert abs(result.normalization_scale - 0.2) < 1e-6
    assert abs(result.normalized_residual - 4.0) < 1e-6
