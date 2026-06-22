# Numerical Agent C Handoff

Reasoning strength: xhigh

Scope: reachability rank, subspace metrics, and statistical gates.

Final files:

- `soma_retargeter/robotics/v3/reachability.py`
- `soma_retargeter/robotics/v3/reachability_metrics.py`
- `tests/v3/test_rank_uncertainty_metrics.py`
- `tests/v3/test_subspace_stability_metrics.py`
- `tests/v3/test_reachability_gates.py`

Validation:

- `PYTHONPATH=. pytest -q tests/v3/test_rank_uncertainty_metrics.py tests/v3/test_subspace_stability_metrics.py tests/v3/test_reachability_gates.py`

Result: PASS. Reachability uses engine Jacobian rank as primary evidence, FD as uncertainty validator, task-specific blocks, uncertainty-aware rank thresholds, rank agreement, projector distance, principal-angle diagnostics, and 95% statistical gates.

Related detailed note: `numerical_agent_c_reachability.md`.
