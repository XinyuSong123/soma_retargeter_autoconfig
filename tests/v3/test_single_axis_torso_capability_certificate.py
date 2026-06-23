from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.capability_projection import project_torso_orientation_with_certificate
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites
from soma_retargeter.robotics.v3.spatial import so3_exp


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _single_axis_torso_adapter(tmp_path: Path):
    model = _write(
        tmp_path / "single_axis_torso.xml",
        """
<mujoco model="single_axis_torso">
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
    return adapter, sites


def _project(adapter, sites, rotvec):
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    return project_torso_orientation_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Hips"],
        sites["Chest"],
        active,
        so3_exp(np.asarray(rotvec, dtype=float)),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()["capability_certificate"]


def test_single_axis_torso_certifies_yaw_as_reachable_and_pitch_as_rank_limited(tmp_path: Path):
    adapter, sites = _single_axis_torso_adapter(tmp_path)

    yaw = _project(adapter, sites, [0.0, 0.0, 0.15])
    pitch = _project(adapter, sites, [0.1, 0.0, 0.0])

    assert yaw["certificate_class"] == "exact_reachable"
    assert yaw["decomposition"]["rank"] == 1
    assert yaw["decomposition"]["compatible_retention_error_norm"] < 1e-8
    assert yaw["gates"]["compatible_demand_retained"] is True

    assert pitch["certificate_class"] == "capability_limited_rank"
    assert pitch["decomposition"]["rank"] == 1
    assert pitch["decomposition"]["rank_incompatible_residual_norm"] > 0.099
    assert pitch["decomposition"]["tangent_residual_norm"] < 1e-8
    assert pitch["gates"]["residual_explained"] is True


def test_single_axis_torso_certifies_yaw_limit_with_projected_gradient_kkt(tmp_path: Path):
    adapter, sites = _single_axis_torso_adapter(tmp_path)

    yaw_limit = _project(adapter, sites, [0.0, 0.0, 0.5])

    assert yaw_limit["certificate_class"] == "capability_limited_joint_limits"
    assert yaw_limit["decomposition"]["active_limit_residual_norm"] > 0.299
    assert yaw_limit["decomposition"]["rank_incompatible_residual_norm"] < 1e-8
    assert yaw_limit["gates"]["projected_gradient_kkt"] is True
    assert yaw_limit["active_limits"]["upper"] == [0]
