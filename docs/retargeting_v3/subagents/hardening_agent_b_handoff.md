# Agent B Handoff: Raw Certificate Evidence

Scope completed on current branch without creating or pushing a branch.

Owned production files changed:

- `soma_retargeter/robotics/v3/projection_solver.py`
- `soma_retargeter/robotics/v3/projection_certificate.py`
- `soma_retargeter/robotics/v3/capability_projection.py`
- `soma_retargeter/robotics/v3/chain_projection.py`
- `soma_retargeter/robotics/v3/canonical_projection.py`

Owned tests added:

- `tests/v3/test_certificate_raw_evidence_schema_v2.py`
- `tests/v3/test_certificate_independent_recompute_kkt.py`
- `tests/v3/test_seed_task_space_consensus_evidence.py`
- `tests/v3/test_continuation_evidence_alpha.py`

Implementation notes:

- Removed `canonical_projection.py::_red_team_kkt_certificate()` and stopped canonical task payloads from emitting synthetic `kkt_certificate`.
- `capability_certificate` now serializes schema version 2 fields, raw `audit_evidence`, independent KKT recompute results, seed consensus spread, continuation alpha evidence, joint-limit evidence, numerical evidence, and a top-level `passed` flag.
- KKT gates are recomputed from `q_active`, bounds, raw task Jacobian, residual, and gradients. Provided solver KKT is retained only under `kkt.raw` and checked for consistency.
- Solver/projection evidence now includes prior gradients, per-seed `final_task_vector`, `final_q_active`, normalized residual, active limits, and candidate certificate class.
- Continuation history now records per-step `q_active` and `within_bounds`; certificate continuation evidence requires alpha to start at 0, increase, and finish at 1.

Validation run:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_certificate_raw_evidence_schema_v2.py \
  tests/v3/test_certificate_independent_recompute_kkt.py \
  tests/v3/test_seed_task_space_consensus_evidence.py \
  tests/v3/test_continuation_evidence_alpha.py
# 10 passed

PYTHONPATH=. python -m pytest -q \
  tests/v3/test_projection_certificate_decisions.py \
  tests/v3/test_capability_projector_decomposition.py \
  tests/v3/test_rank_zero_capability_certificate.py \
  tests/v3/test_single_axis_torso_capability_certificate.py \
  tests/v3/test_projection_continuation_solver.py \
  tests/v3/test_projection_seed_consensus_solver.py \
  tests/v3/test_projection_joint_limit_kkt_solver.py \
  tests/v3/test_chain_projection_gates.py
# 23 passed
```

Integration notes:

- Profile/status policy still needs to consume the stronger schema-v2 gates; this handoff did not edit Agent C-owned policy code.
- Existing red-team acceptance tests outside Agent B ownership still refer to legacy `kkt_certificate`; integrator should decide whether those become schema-v2 certificate checks or compatibility tests.
- Other agents have unrelated worktree changes in artifacts, workflow, validation, profile, scripts, and non-Agent-B tests. They were not modified here.
