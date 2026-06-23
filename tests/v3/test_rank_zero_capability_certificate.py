from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.capability_projection import project_endpoint_position_with_certificate
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _fixed_endpoint_adapter(tmp_path: Path):
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
    return adapter, sites


def test_rank_zero_zero_demand_is_exact_reachable_not_unsupported(tmp_path: Path):
    adapter, sites = _fixed_endpoint_adapter(tmp_path)

    result = project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        [],
        np.array([0.0, 0.0, 0.0]),
    ).to_json()

    cert = result["capability_certificate"]
    assert result["status"] == "rank_zero"
    assert cert["certificate_class"] == "exact_reachable"
    assert cert["decomposition"]["rank"] == 0
    assert cert["decomposition"]["residual_norm"] == 0.0
    assert cert["gates"]["residual_explained"] is True


def test_rank_zero_nonzero_demand_is_unsupported_with_no_fake_projection(tmp_path: Path):
    adapter, sites = _fixed_endpoint_adapter(tmp_path)

    result = project_endpoint_position_with_certificate(
        adapter,
        adapter.neutral_q(),
        sites["Chest"],
        sites["LeftHand"],
        [],
        np.array([0.1, 0.0, 0.0]),
    ).to_json()

    cert = result["capability_certificate"]
    assert result["status"] == "unreachable/rank_zero"
    assert cert["certificate_class"] == "unsupported_rank_zero"
    assert cert["decomposition"]["rank"] == 0
    assert cert["decomposition"]["residual_norm"] > 0.099
    assert cert["decomposition"]["rank_incompatible_residual_norm"] > 0.099
    assert cert["gates"]["compatible_demand_retained"] is False
    assert cert["gates"]["residual_explained"] is True
    assert cert["gates"]["no_orthogonal_leakage"] is True
