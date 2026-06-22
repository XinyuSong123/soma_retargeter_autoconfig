from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma_retargeter.robotics.v3.model_adapter import MuJoCoRuntimeModelAdapter
from soma_retargeter.robotics.v3 import numerical_jacobian as nj
from soma_retargeter.robotics.v3.semantic_sites import build_semantic_sites


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _slide_model(tmp_path: Path) -> tuple[MuJoCoRuntimeModelAdapter, dict, list[int]]:
    model = _write(
        tmp_path / "slide.xml",
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
""",
    )
    adapter = MuJoCoRuntimeModelAdapter(model)
    sites = build_semantic_sites(
        adapter,
        {
            "Hips": "base",
            "Chest": "slider",
            "LeftFoot": "base",
            "RightFoot": "base",
        },
    )
    active = adapter.active_velocity_coordinates(sites["Hips"], sites["Chest"])
    return adapter, sites, active


def test_tangent_coordinate_central_difference_uses_adapter_integrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter, sites, active = _slide_model(tmp_path)
    calls: list[np.ndarray] = []
    original_integrate = adapter.integrate

    def record_integrate(q, delta_v):
        calls.append(np.asarray(delta_v, dtype=float).copy())
        return original_integrate(q, delta_v)

    monkeypatch.setattr(adapter, "integrate", record_integrate)

    jac = nj.numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)

    assert jac.stability_gate_passed
    assert len(calls) == 6
    assert all(delta.shape == (adapter.nv,) for delta in calls)
    assert {float(delta[active[0]]) for delta in calls} == {
        nj.coordinate_epsilon(adapter, active[0]),
        -nj.coordinate_epsilon(adapter, active[0]),
        2.0 * nj.coordinate_epsilon(adapter, active[0]),
        -2.0 * nj.coordinate_epsilon(adapter, active[0]),
        0.5 * nj.coordinate_epsilon(adapter, active[0]),
        -0.5 * nj.coordinate_epsilon(adapter, active[0]),
    }


def test_epsilon_halving_gate_records_and_can_hard_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter, sites, active = _slide_model(tmp_path)
    base_epsilon = nj.coordinate_epsilon(adapter, active[0])

    def unstable_column(adapter, q, reference, target, dof, epsilon):
        multiplier = 1.0 if epsilon == base_epsilon else 1.5
        return np.array([multiplier, 0.0, 0.0]), np.zeros(3)

    monkeypatch.setattr(nj, "_column", unstable_column)

    jac = nj.numerical_relative_jacobian(adapter, adapter.neutral_q(), sites["Hips"], sites["Chest"], active)
    assert not jac.stability_gate_passed
    assert jac.unstable_columns == active
    assert jac.to_json()["stability_gate_passed"] is False
    assert jac.epsilon_discrepancies[0]["stable"] is False
    assert jac.column_classifications[0]["class"] == "engine_fd_mismatch"

    with pytest.raises(FloatingPointError, match="stability gate failed"):
        nj.numerical_relative_jacobian(
            adapter,
            adapter.neutral_q(),
            sites["Hips"],
            sites["Chest"],
            active,
            raise_on_unstable=True,
        )
