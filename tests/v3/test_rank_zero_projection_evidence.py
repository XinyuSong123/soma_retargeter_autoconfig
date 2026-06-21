from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.audit_retargeting_v3_step2 import _audit_rank_zero_false_pass
from soma_retargeter.robotics.v3.chain_projection import ProjectionResult, project_endpoint_position
from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.profile import compile_kinematic_profile_v3
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def test_rank_zero_zero_demand_endpoint_serializes_explicit_evidence(tmp_path: Path):
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
        np.array([0.0, 0.0, 0.0]),
    ).to_json()

    assert result["status"] == "rank_zero"
    assert result["residual"] == 0.0
    assert result["active_coordinates"] == []
    assert result["demand_residual"] == 0.0
    assert result["unreachable_demand"] is False
    assert result["rank_zero_reason"] == "no_active_coordinates_zero_demand"


def test_rank_zero_profile_projection_report_contains_zero_demand_evidence(tmp_path: Path):
    model = _write(
        tmp_path / "fixed_torso.xml",
        """
<mujoco model="fixed_torso">
  <worldbody>
    <body name="hips">
      <body name="chest" pos="0 0 1"><geom type="sphere" size="0.01"/></body>
      <body name="left_foot" pos="0.1 0 -1"><geom type="sphere" size="0.01"/></body>
      <body name="right_foot" pos="-0.1 0 -1"><geom type="sphere" size="0.01"/></body>
    </body>
  </worldbody>
</mujoco>
""",
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
    torso = profile.projection_reports["neutral"]["torso"]

    assert torso["status"] == "rank_zero"
    assert abs(torso["residual"]) < 1e-12
    assert torso["active_coordinates"] == []
    assert abs(torso["demand_residual"]) < 1e-12
    assert torso["unreachable_demand"] is False
    assert torso["rank_zero_reason"] == "no_active_coordinates_zero_demand"


def test_rank_zero_to_json_backfills_evidence_for_manual_results():
    result = ProjectionResult(
        np.zeros(3),
        np.zeros(3),
        np.zeros(0),
        0.0,
        0.0,
        1.0,
        True,
        "rank_zero",
        active_coordinates=[],
    ).to_json()

    assert result["demand_residual"] == 0.0
    assert result["unreachable_demand"] is False
    assert result["rank_zero_reason"] == "no_active_coordinates_zero_demand"


def test_rank_zero_audit_remains_strict_for_bare_zero_report():
    bare_report = {
        "projection_reports": {
            "torso": {
                "status": "rank_zero",
                "residual": 0.0,
                "desired": [0.0, 0.0, 0.0],
                "projected": [0.0, 0.0, 0.0],
            }
        }
    }
    findings = _audit_rank_zero_false_pass({"bare_bot": bare_report})
    assert [finding.gate for finding in findings] == ["rank0_false_pass"]

    evidenced_report = {
        "projection_reports": {
            "torso": {
                **bare_report["projection_reports"]["torso"],
                "demand_residual": 0.0,
                "unreachable_demand": False,
                "rank_zero_reason": "no_active_coordinates_zero_demand",
            }
        }
    }
    assert _audit_rank_zero_false_pass({"evidenced_bot": evidenced_report}) == []
