# Numerical Agent B Handoff

Reasoning strength: xhigh

Scope: backend-aware multi-scale finite-difference uncertainty.

Final files:

- `soma_retargeter/robotics/v3/fd_uncertainty.py`
- `soma_retargeter/robotics/v3/numerical_jacobian.py`
- `tests/v3/test_fd_uncertainty_backend_steps.py`
- `tests/v3/test_zero_jacobian_columns_engine.py`
- `tests/v3/test_numerical_jacobian_gates.py`

Validation:

- `PYTHONPATH=. pytest -q tests/v3/test_fd_uncertainty_backend_steps.py tests/v3/test_zero_jacobian_columns_engine.py tests/v3/test_numerical_jacobian_gates.py`

Result: PASS. FD uses backend epsilon to the one-third power, samples `2h/h/h2`, selects the better Richardson plateau, records error estimates, and classifies zero, roundoff, nonsmooth, nonfinite, and engine/FD mismatch columns without robot-name branches.

Related detailed note: `numerical_agent_b_fd_uncertainty.md`.
