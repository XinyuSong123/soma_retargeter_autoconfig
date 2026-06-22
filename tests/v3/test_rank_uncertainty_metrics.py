from __future__ import annotations

import numpy as np

from soma_retargeter.robotics.v3.reachability_metrics import rank_with_uncertainty


def test_rank_threshold_includes_jacobian_noise_norm():
    matrix = np.diag([1.0, 1e-3])

    rank, singular_values, threshold = rank_with_uncertainty(matrix, noise_norm=2e-4, k_uncertainty=10.0)

    assert singular_values.tolist() == [1.0, 1e-3]
    assert threshold == 2e-3
    assert rank == 1
