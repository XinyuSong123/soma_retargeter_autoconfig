from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3 import MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter
from soma_retargeter.robotics.v3.kinematic_paths import discover_paths
from soma_retargeter.robotics.v3.model_fingerprint import model_fingerprint_payload
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def test_mujoco_ball_joint_uses_tangent_integration(tmp_path: Path):
    model = _write(
        tmp_path / "ball.xml",
        """
<mujoco model="ball">
  <worldbody>
    <body name="base">
      <body name="tip">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="ball" type="ball"/>
        <geom type="sphere" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    q0 = adapter.neutral_q()
    delta = np.array([0.2, -0.15, 0.1])

    q1 = adapter.integrate(q0, delta)

    assert q1.shape == q0.shape
    assert abs(np.linalg.norm(q1[:4]) - 1.0) < 1e-12
    assert not np.allclose(q1[:3], q0[:3] + delta)


@pytest.mark.parametrize("adapter_cls", [MuJoCoRuntimeModelAdapter, NewtonRuntimeModelAdapter])
def test_cross_branch_path_keeps_fixed_bodies_and_uses_both_lca_branches(tmp_path: Path, adapter_cls):
    model = _write(
        tmp_path / "cross_branch.xml",
        """
<mujoco model="cross_branch">
  <worldbody>
    <body name="base">
      <body name="left_fixed" pos="0 0.2 0">
        <body name="left_tip">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="left_hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </body>
      <body name="right_fixed" pos="0 -0.2 0">
        <body name="right_tip">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <joint name="right_hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = adapter_cls(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "left_tip",
            "Chest": "right_tip",
            "LeftHand": "left_tip",
            "RightHand": "right_tip",
            "LeftFoot": "left_tip",
            "RightFoot": "right_tip",
        },
    )

    paths = discover_paths(adapter, sites)

    torso = paths["torso"]
    assert torso.lca_body == "base"
    assert torso.body_path == ["left_tip", "left_fixed", "base", "right_fixed", "right_tip"]
    assert torso.reference_branch_bodies == ["left_tip", "left_fixed"]
    assert torso.target_branch_bodies == ["right_tip", "right_fixed"]
    assert torso.coordinate_labels == ["left_hinge", "right_hinge"]
    assert torso.joint_types == ["revolute", "revolute"]


def test_fingerprint_records_primary_include_asset_loader_and_compiled_summary(tmp_path: Path):
    _write(tmp_path / "part.xml", "<mujoco><worldbody><body name='included'/></worldbody></mujoco>")
    _write(tmp_path / "mesh.stl", "solid mesh\nendsolid mesh")
    main = _write(
        tmp_path / "main.xml",
        """
<mujoco model="fingerprint">
  <include file="part.xml"/>
  <asset><mesh name="m" file="mesh.stl"/></asset>
</mujoco>
""",
    )

    payload = model_fingerprint_payload(
        main,
        backend="mujoco",
        model_format="xml",
        loader_name="mujoco",
        loader_version="test-loader",
        conversion_settings={"canonical_format": "mjcf"},
        loader_provenance={"patches": []},
        compiled_summary={"nq": 0, "nv": 0},
    )

    roles = {(entry["role"], Path(entry["path"]).name) for entry in payload["files"]}
    assert ("primary_model", "main.xml") in roles
    assert ("resolved_include", "part.xml") in roles
    assert ("referenced_asset", "mesh.stl") in roles
    assert payload["loader"]["version"] == "test-loader"
    assert payload["conversion_settings"] == {"canonical_format": "mjcf"}
    assert payload["compiled_summary"] == {"nq": 0, "nv": 0}
    assert len(payload["sha256"]) == 64


def test_newton_model_site_frame_records_mujoco_compiled_crosscheck(tmp_path: Path):
    model = _write(
        tmp_path / "site.xml",
        """
<mujoco model="site">
  <worldbody>
    <body name="base">
      <body name="link">
        <geom type="sphere" size="0.01"/>
        <site name="tip_site" pos="0.1 0.2 0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
    )
    adapter = NewtonRuntimeModelAdapter(model)

    with pytest.warns(RuntimeWarning, match="MuJoCo compiled site cross-check"):
        body, pos, quat = adapter.model_site_frame("tip_site")

    assert body == "link"
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])
    assert adapter.site_frame_provenance["tip_site"]["source"] == "mujoco_compiled_site_crosscheck"
    assert adapter.site_frame_provenance["tip_site"]["trusted_runtime_site"]
    assert adapter.fingerprint_details["compiled_summary"]["nv"] == adapter.nv
