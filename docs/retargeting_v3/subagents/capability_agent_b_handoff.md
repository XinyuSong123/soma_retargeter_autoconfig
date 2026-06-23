# Agent B Handoff: Deterministic Bounded Continuation Solver

Date: 2026-06-23

## Changed Files

- `soma_retargeter/robotics/v3/projection_solver.py`
- `soma_retargeter/robotics/v3/chain_projection.py`
- `tests/v3/test_projection_continuation_solver.py`
- `tests/v3/test_projection_seed_consensus_solver.py`
- `tests/v3/test_projection_joint_limit_kkt_solver.py`
- `docs/retargeting_v3/subagents/capability_agent_b_handoff.md`

## Implementation Summary

- Added a robot-agnostic bounded least-squares continuation solver with deterministic seed enumeration, residual-space homotopy subdivision, per-seed histories, deterministic tie-breaking, and additive JSON evidence.
- Wired `project_endpoint_position()` and `project_torso_orientation()` through residual/Jacobian callbacks. MuJoCo engine Jacobians are used first, with finite-difference fallback only if engine Jacobian evaluation fails.
- Added SO(3) log residual handling for orientation tasks using the convention `left_perturbation_log_current_transpose_target`.
- Added task-only gradient and active-limit KKT evidence. Priors remain part of the optimization residual, but KKT evidence is computed from the task residual/Jacobian only.
- Added explicit scalar dtype KKT tolerances: float64 stationarity `1e-7`, active-bound `1e-9`; float32 stationarity `5e-5`, active-bound `1e-5`.
- Kept projection result additions backward-compatible by serializing new fields additively.

## Synthetic Fixtures Covered

- Reachable revolute endpoint.
- Unreachable target beyond link length.
- Joint-limit closest point.
- Yaw-only torso with pitch demand.
- Fixed torso rank-zero demand.
- Near-singular two-link arm.
- Two-local-minimum two-link chain.
- Nonidentity orientation target.
- Float32 backend tolerance oracle.
- Deterministic rerun of solution and history.

## Commands Run

- `PYTHONPATH=. pytest -q tests/v3/test_projection_continuation_solver.py tests/v3/test_projection_seed_consensus_solver.py tests/v3/test_projection_joint_limit_kkt_solver.py`
  - PASS: 10 passed.
- `PYTHONPATH=. pytest -q tests/v3/test_chain_projection_gates.py tests/v3/test_rank_zero_projection_evidence.py tests/v3/test_chain_length_normalization.py`
  - PASS: 10 passed.
- `PYTHONPATH=. pytest -q tests/v3/test_engine_jacobian.py tests/v3/test_numerical_jacobian_gates.py`
  - PASS: 5 passed.
- `PYTHONPATH=. pytest -q tests/v3/test_projection_*.py tests/v3/test_capability_*.py tests/v3/test_rank_zero_capability_certificate.py tests/v3/test_single_axis_torso_capability_certificate.py`
  - FAIL: 36 passed, 2 failed because `artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json` is missing.

## Numeric Choices

- Continuation step bound: normalized task-space increment <= `0.25`.
- Solver tolerances: `xtol=1e-14`, `ftol=1e-14`, `gtol=1e-12`, `max_nfev_per_step=400`.
- Exact projection status threshold remains `1e-8` absolute task residual.
- Rank-zero demand threshold remains `1e-10`.
- Direct projection neutral prior default is `1e-12`, matching the canonical projection path and avoiding measurable closest-point bias from a tie-breaker prior.

## Risks

- The solver uses SciPy TRF least-squares, so exact per-step iteration counts are deterministic on this environment but may vary across SciPy versions.
- The SO(3) Jacobian assumes the existing relative Jacobian convention is a left perturbation of the current relative rotation.
- Finite-difference Jacobian fallback is present for robustness, but the intended and tested path is engine-backed Jacobian evaluation.

## Unfinished Items

- The broader capability ledger tests require the missing committed artifact `artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json`; this was not in Agent B's owned file set.
- No branch was created, no push was attempted, and unrelated workspace changes were not reverted.
