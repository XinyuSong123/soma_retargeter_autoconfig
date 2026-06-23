# Capability Agent C Handoff

Scope: capability decomposition and projection certificates.

## Changed Files

- `soma_retargeter/robotics/v3/capability_projection.py`
  - Added tangent-space residual decomposition.
  - Added endpoint and torso projection wrappers with deterministic seed-consensus evidence.
  - Exposed `decompose_task_residual` for the current chain projection solver integration.
- `soma_retargeter/robotics/v3/projection_certificate.py`
  - Added deterministic certificate classes:
    `exact_reachable`, `capability_limited_rank`,
    `capability_limited_joint_limits`, `capability_limited_mixed`,
    `unsupported_rank_zero`, `solver_failed`, `numerical_invalid`,
    `invalid_target_geometry`.
  - Added certificate gates for compatible-demand retention, residual explanation,
    projected-gradient/KKT, seed consensus, and orthogonal leakage.
- `soma_retargeter/robotics/v3/reachability_metrics.py`
  - Added `backend_rank_abs_floor()` to reuse Step 2.1 backend rank floors.
- `tests/v3/test_capability_projector_decomposition.py`
- `tests/v3/test_projection_certificate_decisions.py`
- `tests/v3/test_rank_zero_capability_certificate.py`
- `tests/v3/test_single_axis_torso_capability_certificate.py`
- `docs/retargeting_v3/subagents/capability_agent_c_handoff.md`

`canonical_projection.py` was not edited by Agent C. In the current worktree,
canonical reports receive `capability_certificate` through `ProjectionResult.to_json()`
from the chain projection path.

## Commands And Results

- Initial red run:
  - `PYTHONPATH=. pytest -q tests/v3/test_capability_projector_decomposition.py tests/v3/test_projection_certificate_decisions.py tests/v3/test_rank_zero_capability_certificate.py tests/v3/test_single_axis_torso_capability_certificate.py`
  - Result: expected collection failure, missing `capability_projection`.
- Focused Agent C tests:
  - Same command after implementation.
  - Result: `8 passed in 15.97s`.
- Existing projection regressions:
  - `PYTHONPATH=. pytest -q tests/v3/test_chain_projection_gates.py tests/v3/test_rank_zero_projection_evidence.py`
  - Result: `9 passed in 3.53s`.
- Canonical/profile regressions:
  - `PYTHONPATH=. pytest -q tests/v3/test_canonical_independence_temporal.py tests/v3/test_profile_canonical_projection_contract.py`
  - Result: `3 passed in 13.55s`.
- Combined owned-pattern/projection run:
  - `PYTHONPATH=. pytest -q tests/v3/test_capability_projector_*.py tests/v3/test_projection_certificate_*.py tests/v3/test_rank_zero_capability_*.py tests/v3/test_single_axis_torso_capability_*.py tests/v3/test_chain_projection_gates.py tests/v3/test_rank_zero_projection_evidence.py tests/v3/test_canonical_independence_temporal.py tests/v3/test_profile_canonical_projection_contract.py`
  - Result: `20 passed in 31.00s`.
- Neighboring solver compatibility:
  - `PYTHONPATH=. pytest -q tests/v3/test_projection_continuation_solver.py tests/v3/test_projection_joint_limit_kkt_solver.py tests/v3/test_projection_seed_consensus_solver.py`
  - Result: `10 passed in 13.47s`.
- Rank/subspace metrics:
  - `PYTHONPATH=. pytest -q tests/v3/test_rank_uncertainty_metrics.py tests/v3/test_subspace_stability_metrics.py`
  - Result: `2 passed in 0.14s`.
- Compile check:
  - `PYTHONPATH=. python -m compileall -q soma_retargeter/robotics/v3/capability_projection.py soma_retargeter/robotics/v3/projection_certificate.py soma_retargeter/robotics/v3/reachability_metrics.py`
  - Result: pass.

## Numeric Gates

- Rank convention: `rank_with_uncertainty()`.
  - Threshold is `max(abs_floor, rel_tol * sigma0, k_uncertainty * noise_norm)`.
  - Backend floors: MuJoCo/default `1e-9`, Newton `1e-5`.
  - No fixed `1e-8` rank rule was added.
- Certificate residual/component tolerance:
  - Default `1e-7 * max(1, scale, ||desired||, ||projected||)`.
  - Chain solver may pass its own `exact_threshold`.
- KKT gate:
  - Wrapper evidence uses projected-gradient norm with `1e-7` absolute and `1e-5` relative tolerance.
  - Chain solver evidence is accepted when supplied, including dtype-specific tolerances.
- Active-bound detection:
  - Wrapper evidence uses `1e-9` absolute and `1e-8` relative boundary tolerance.
  - Chain solver active-bound lists are preserved when supplied.
- Seed consensus:
  - Wrapper reruns deterministic seeds and requires max projected/residual delta `<= 1e-7`.
- No leakage:
  - Orthogonal retained-motion leakage must be within component/leakage tolerance.

## Risks

- The worktree contains concurrent edits outside Agent C scope, including
  `chain_projection.py`, `projection_solver.py`, profile/validation files, scripts,
  artifacts, and additional tests. Agent C did not revert or modify those files.
- The current unowned chain solver imports `decompose_task_residual` and passes a
  richer `ProjectionCertificateEvidence` payload. Agent C kept the owned API
  compatible with that shape.
- Rotation certificates decompose SO(3) tasks in log-vector coordinates. This is
  exact for the covered rank-zero/single-axis cases and appropriate as local
  tangent evidence, but large multi-axis nonlinear rotations still need full
  robot-zoo evidence.
- Full robot-zoo artifacts were not regenerated in this pass.

## Unfinished Items

- Run the full Step 2.3/robot-zoo capability artifact pipeline after all agents land.
- Audit final canonical projection artifacts for certificate coverage across all
  positive, partial, and failure-baseline models.
- Reconcile Agent C certificate fields with any final schema decisions from the
  concurrent projection-solver and validation agents.
