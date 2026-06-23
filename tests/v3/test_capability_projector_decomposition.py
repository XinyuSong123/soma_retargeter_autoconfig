from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.capability_projection import project_endpoint_position_with_certificate
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _single_slide_adapter(tmp_path: Path, *, upper: float = 1.0):
    model = _write(
        tmp_path / "single_slide.xml",
        f"""
<mujoco model="single_slide">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest">
      <body name="hand">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint name="reach_x" type="slide" axis="1 0 0" range="0 {upper}"/>
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
    return adapter, sites


def test_single_axis_endpoint_decomposes_reachable_and_orthogonal_demand(tmp_path: Path):
    adapter, sites = _single_slide_adapter(tmp_path)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])

    result = project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.2, 0.3, 0.0]),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    )

    payload = result.to_json()
    cert = payload["capability_certificate"]
    decomposition = cert["decomposition"]

    assert payload["status"] == "converged/with_residual"
    assert cert["certificate_class"] == "capability_limited_rank"
    assert decomposition["rank"] == 1
    assert abs(decomposition["compatible_demand_norm"] - 0.2) < 1e-8
    assert decomposition["compatible_retention_error_norm"] < 1e-8
    assert decomposition["tangent_residual_norm"] < 1e-8
    assert decomposition["rank_incompatible_residual_norm"] > 0.299
    assert cert["gates"]["compatible_demand_retained"] is True
    assert cert["gates"]["residual_explained"] is True
    assert cert["gates"]["projected_gradient_kkt"] is True
    assert cert["gates"]["seed_consensus"] is True
    assert cert["gates"]["no_orthogonal_leakage"] is True


def test_endpoint_certificate_is_deterministic_and_contains_no_local_path_leakage(tmp_path: Path):
    adapter, sites = _single_slide_adapter(tmp_path)
    active = adapter.active_velocity_coordinates(sites["Chest"], sites["LeftHand"])

    first = project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.2, 0.3, 0.0]),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()["capability_certificate"]
    second = project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        active,
        np.array([0.2, 0.3, 0.0]),
        neutral_prior_weight=0.0,
        continuity_prior_weight=0.0,
    ).to_json()["capability_certificate"]

    assert first["deterministic_digest"] == second["deterministic_digest"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert str(tmp_path) not in json.dumps(first, sort_keys=True)
