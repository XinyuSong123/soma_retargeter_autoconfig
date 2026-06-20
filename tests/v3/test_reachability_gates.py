from __future__ import annotations

from pathlib import Path

import numpy as np

from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3.numerical_jacobian import matrix_rank_and_singular_values, numerical_relative_jacobian
from soma_retargeter.robotics.v3.reachability import analyze_reachability, deterministic_chain_samples
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def test_regular_rank_uses_local_sample_ranks_not_concatenated_tangent_union(tmp_path: Path):
    model = _write(
        tmp_path / "curved.xml",
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
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": {"body": "link", "local_position": [1.0, 0.0, 0.0]},
            "LeftFoot": "base",
            "RightFoot": "base",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    samples = deterministic_chain_samples(adapter, active, low_discrepancy_count=8)
    local_jacobians = [
        numerical_relative_jacobian(adapter, q, sites["Hips"], sites["Chest"], active).translation
        for q in samples
    ]

    local_ranks = [matrix_rank_and_singular_values(jac)[0] for jac in local_jacobians]
    concatenated_rank, _ = matrix_rank_and_singular_values(np.column_stack(local_jacobians))
    report = analyze_reachability(adapter, sites["Hips"], sites["Chest"], active, low_discrepancy_count=8)

    assert max(local_ranks) == 1
    assert concatenated_rank == 2
    assert report.regular_rank_translation == 1
    assert report.rank_method == "per-sample-local-rank-regular-fraction"
    assert report.regular_rank_fraction_threshold == 0.20
    assert report.epsilon_stability_gate_passed
    assert report.epsilon_unstable_columns == []
    assert all(sample["epsilon_stability_gate_passed"] for sample in report.sample_diagnostics)
