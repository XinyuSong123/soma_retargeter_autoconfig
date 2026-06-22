from __future__ import annotations

import numpy as np

from soma_retargeter.robotics.v3.reachability_metrics import principal_angles, projector_distance


def test_projector_distance_and_principal_angles_for_identical_subspaces():
    u = np.eye(3)[:, :2]

    assert projector_distance(u, u) == 0.0
    assert principal_angles(u, u) == [0.0, 0.0]
