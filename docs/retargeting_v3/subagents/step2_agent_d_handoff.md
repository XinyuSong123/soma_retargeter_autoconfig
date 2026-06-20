# Agent D Handoff

- Agent: D-xhigh, Reachability, Canonical Chain Projection & Mathematical Gates.
- Reasoning requirement: satisfied as an xhigh-designated subagent run; no lower-reasoning subagent was used for this handoff.
- Branch/commit: `agent-b-xhigh-verified-semantics` / `63630cc feat: harden chain projection gates`.
- Prior worktree reviewed: `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_d`; used only as reference, not assumed correct.

## Files

- `soma_retargeter/robotics/v3/numerical_jacobian.py`
- `soma_retargeter/robotics/v3/reachability.py`
- `soma_retargeter/robotics/v3/chain_projection.py`
- `soma_retargeter/robotics/v3/canonical_projection.py`
- `tests/v3/test_numerical_jacobian_gates.py`
- `tests/v3/test_reachability_gates.py`
- `tests/v3/test_chain_projection_gates.py`
- `docs/retargeting_v3/subagents/step2_agent_d_handoff.md`

## Changes

- Added explicit `stability_gate_passed` to numerical relative Jacobians and optional `raise_on_unstable=True` hard gate for epsilon-halving failures.
- Carried epsilon stability into multi-pose reachability with aggregate/sample fractions, unstable coordinate list, and explicit rank method metadata.
- Kept regular rank based on per-sample local ranks, not concatenated sampled Jacobian rank.
- Extended `ProjectionResult` with `iterations`, `active_coordinates`, `prior_residual_norm`, and `solver_message`.
- Changed rank-zero projection semantics so nonzero demand returns `unreachable/rank_zero`, `converged=False`, and preserves desired/projected/residual.
- Added range-normalized neutral and continuity priors to endpoint and torso projection costs.
- Added deterministic multi-starts from seed, neutral, previous, and midpoint candidates with deterministic tie-breaking.
- Added `canonical_projection.py` to consume real `SemanticTargets.transforms` for per-motion torso/hand/foot projection instead of neutral FK placeholders.

## Commands

- `git status --short`
- `git rev-parse --abbrev-ref HEAD && git rev-parse HEAD`
- `git -C /mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_d diff --stat`
- `PYTHONPATH=. pytest -q tests/v3/test_numerical_jacobian_gates.py tests/v3/test_reachability_gates.py tests/v3/test_chain_projection_gates.py`
- `PYTHONPATH=. pytest -q tests/v3`
- `PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3_math.py tests/test_kinematic_profile_v3.py -k 'jacobian or reachability or projection or rank_zero or canonical'`
- `PYTHONPATH=. python -m py_compile soma_retargeter/robotics/v3/numerical_jacobian.py soma_retargeter/robotics/v3/reachability.py soma_retargeter/robotics/v3/chain_projection.py soma_retargeter/robotics/v3/canonical_projection.py tests/v3/test_numerical_jacobian_gates.py tests/v3/test_reachability_gates.py tests/v3/test_chain_projection_gates.py`
- `git diff --check -- soma_retargeter/robotics/v3/numerical_jacobian.py soma_retargeter/robotics/v3/reachability.py soma_retargeter/robotics/v3/chain_projection.py soma_retargeter/robotics/v3/canonical_projection.py tests/v3/test_numerical_jacobian_gates.py tests/v3/test_reachability_gates.py tests/v3/test_chain_projection_gates.py`

## Tests

- PASS: `PYTHONPATH=. pytest -q tests/v3/test_numerical_jacobian_gates.py tests/v3/test_reachability_gates.py tests/v3/test_chain_projection_gates.py`
  - Result: `7 passed in 1.05s`.
- PASS: `PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3_math.py tests/test_kinematic_profile_v3.py -k 'jacobian or reachability or projection or rank_zero or canonical'`
  - Result: `13 passed, 22 deselected in 9.07s`.
- PASS: `PYTHONPATH=. python -m py_compile ...`
- PASS: `git diff --check -- ...`
- BLOCKED: `PYTHONPATH=. pytest -q tests/v3`
  - Result before this handoff existed: `42 passed, 1 failed in 6.04s`.
  - Failing test: `tests/v3/test_acceptance_gates_audit.py::test_retargeting_v3_step2_acceptance_gates_are_clean`.
  - Blocking findings then included canonical targets without per-motion projection in existing artifacts, zero-offset RPO hand/sole scoped pass, arbitrary G1 equivalence, dirty artifact metadata, robot-name semantic special cases, missing Agent B handoff, and missing Agent D handoff.
- BLOCKED: `PYTHONPATH=. pytest -q tests/v3/test_acceptance_gates_audit.py`
  - Result after this handoff file exists: `1 failed in 0.06s`.
  - Remaining blockers: canonical targets without per-motion projection in existing artifacts, arbitrary G1 equivalence, dirty artifact metadata, two robot-name semantic special-case findings, and missing Agent B handoff.

## Numerical Results

- Tangent finite difference test observed exactly four calls to `adapter.integrate()` for one active coordinate: `+epsilon`, `-epsilon`, `+epsilon/2`, `-epsilon/2`; the perturbations were full `nv` tangent vectors.
- Forced epsilon-halving mismatch produced `stability_gate_passed=False`, `unstable_columns=[0]`, and `raise_on_unstable=True` raised `FloatingPointError`.
- One-DoF curved endpoint:
  - max local sampled translation rank: `1`;
  - concatenated sampled tangent rank: `2`;
  - reported regular translation rank: `1`;
  - epsilon gate: passed, unstable columns `[]`.
- Rank-zero endpoint nonzero demand:
  - desired `[0.1, 0.0, 0.0]`;
  - projected `[0.0, 0.0, 0.0]`;
  - residual `0.1`;
  - status `unreachable/rank_zero`;
  - iterations `0`.
- Rank-zero torso nonzero demand:
  - desired rotation log `[0.0, 0.0, 0.2]`;
  - projected rotation log `[0.0, 0.0, 0.0]`;
  - residual `0.2`;
  - status `unreachable/rank_zero`.
- Redundant continuity-prior endpoint:
  - desired/projected `[0.2, 0.0, 0.0]`;
  - selected chain coordinates `[0.2, 0.0]`;
  - prior residual norm `< 1e-8`;
  - status `converged`.
- Canonical non-neutral target flow:
  - `arms_forward.left_hand.desired` was `[0.15, 0.0, 0.0]`;
  - neutral desired differed from arms-forward desired;
  - projected `[0.15, 0.0, 0.0]`;
  - desired source recorded as `canonical_targets.transforms`.

## Assumptions

- Agent D ownership excludes `soma_retargeter/robotics/v3/profile.py`, so the new canonical projection sequence is implemented in `canonical_projection.py` for Integrator wiring.
- Existing broad tests require `PYTHONPATH=.` in this environment because the checkout is not installed as a package.
- Priors use runtime coordinate ranges where finite, `2*pi` for unbounded revolute coordinates, `1.0` for unbounded prismatic coordinates, and `pi` for tangent quaternion coordinates.

## Known Failures

- `compile_kinematic_profile_v3()` still builds `projection_reports` from neutral FK in `profile.py`; this remains outside Agent D ownership and must be rewired by Integrator to call `project_canonical_motion_sequence()`.
- Existing artifacts still fail acceptance audit for stale/dirty metadata and missing or incomplete full-zoo reports.
- Existing semantic and validation blockers remain outside this Agent D scope: zero-offset RPO endpoint scoped pass, G1 equivalence not strict, robot-name semantic special-case audit findings, and missing Agent B handoff.
- Full Robot Zoo and complete-model canonical projection were not rerun in this Agent D worktree.

## Integration Order

1. Integrator should import `project_canonical_motion_sequence()` in `profile.py` after Agent C target builder output is created.
2. Replace neutral-only `projection_reports[task]` with per-motion projection reports keyed by canonical motion and task.
3. Use the report failures list to gate `unreachable/rank_zero` nonzero source demand instead of treating it as success.
4. Regenerate artifacts from a clean worktree after Agent B/C/E/F blockers are closed.
5. Re-run `tests/v3` and full Step 2 validation after profile wiring.

## Review Risks

- Deterministic multi-start increases projection solve count by up to four starts per task. This is acceptable for offline Step 2 but should be measured on TALOS/full-zoo profiles.
- Ball/free coordinate priors are tangent-coordinate approximations around the seed; complete-model validation should check this against Agent A adapter behavior.
- `status="converged/with_residual"` is now emitted for bounded solves that converge numerically but cannot hit the desired target exactly. Downstream gates should use residual and status together.
