from __future__ import annotations

import numpy as np

from soma_retargeter.robotics.v3 import canonical_projection as cp
from soma_retargeter.robotics.v3.chain_projection import ProjectionResult
from soma_retargeter.robotics.v3.kinematic_paths import KinematicPath
from soma_retargeter.robotics.v3.model_adapter import SemanticSite
from soma_retargeter.robotics.v3.spatial import transform
from soma_retargeter.robotics.v3.target_builder import SemanticTargets


class _Adapter:
    def neutral_q(self) -> np.ndarray:
        return np.array([0.0])


def _path() -> KinematicPath:
    return KinematicPath("left_hand", "Chest", "LeftHand", "Chest", "LeftHand", "Chest", [], [], [], [0], ["q"], ["slide"])


def _targets() -> dict[str, SemanticTargets]:
    return {
        "neutral": SemanticTargets({"Chest": transform([0, 0, 0]), "LeftHand": transform([1, 0, 0])}, {}, "neutral"),
        "arms_forward": SemanticTargets({"Chest": transform([0, 0, 0]), "LeftHand": transform([2, 0, 0])}, {}, "arms_forward"),
    }


def test_canonical_projection_does_not_pass_previous_q_by_default(monkeypatch):
    calls: list[tuple[np.ndarray, np.ndarray | None, float]] = []

    def fake_project(adapter, q_seed, reference, target, active, desired, **kwargs):
        calls.append((q_seed.copy(), kwargs.get("previous_q"), kwargs.get("continuity_prior_weight")))
        return ProjectionResult(desired, desired, np.array([len(calls)], dtype=float), 0.0, 0.0, 1.0, True, "converged")

    monkeypatch.setattr(cp, "project_endpoint_position", fake_project)
    cp.project_canonical_motion_sequence(
        _Adapter(),
        {"Chest": SemanticSite("Chest", "Chest", np.zeros(3), np.array([0, 0, 0, 1])), "LeftHand": SemanticSite("LeftHand", "LeftHand", np.zeros(3), np.array([0, 0, 0, 1]))},
        {"left_hand": _path()},
        _targets(),
        motion_order=["neutral", "arms_forward"],
    )

    assert [call[0].tolist() for call in calls] == [[0.0], [0.0]]
    assert [call[1] for call in calls] == [None, None]
    assert [call[2] for call in calls] == [0.0, 0.0]


def test_temporal_sequences_enable_continuity_prior(monkeypatch):
    calls: list[np.ndarray | None] = []

    def fake_project(adapter, q_seed, reference, target, active, desired, **kwargs):
        calls.append(kwargs.get("previous_q"))
        return ProjectionResult(desired, desired, np.array([len(calls)], dtype=float), 0.0, 0.0, 1.0, True, "converged")

    monkeypatch.setattr(cp, "project_endpoint_position", fake_project)
    reports = cp.project_temporal_motion_sequences(
        _Adapter(),
        {"Chest": SemanticSite("Chest", "Chest", np.zeros(3), np.array([0, 0, 0, 1])), "LeftHand": SemanticSite("LeftHand", "LeftHand", np.zeros(3), np.array([0, 0, 0, 1]))},
        {"left_hand": _path()},
        _targets(),
    )

    assert reports["arms_overhead_return"]["continuity_prior_enabled"] is True
    assert calls[0] is None
    assert any(call is not None for call in calls[1:])
