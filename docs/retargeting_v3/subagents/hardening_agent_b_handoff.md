# Hardening Agent B Handoff

Agent: B - Raw Certificate Evidence and Math Recompute Inputs

Status: FINAL PASS

## Scope

Agent B owned schema-v2 capability certificate evidence and independent recompute inputs.

## Files

- `soma_retargeter/robotics/v3/projection_solver.py`
- `soma_retargeter/robotics/v3/projection_certificate.py`
- `soma_retargeter/robotics/v3/capability_projection.py`
- `soma_retargeter/robotics/v3/chain_projection.py`
- `soma_retargeter/robotics/v3/canonical_projection.py`
- `tests/v3/test_certificate_raw_evidence_schema_v2.py`
- `tests/v3/test_certificate_independent_recompute_kkt.py`
- `tests/v3/test_seed_task_space_consensus_evidence.py`
- `tests/v3/test_continuation_evidence_alpha.py`

## Final Evidence

- Removed `canonical_projection.py::_red_team_kkt_certificate()`.
- Capability certificates use `schema_version=2`.
- Certificates serialize raw audit evidence including task vectors, Jacobians, active coordinates, `q_active`, bounds, task gradients, prior gradients, seed results, continuation history, scalar dtype, and normalization scale.
- KKT, primal feasibility, dual/sign checks, complementarity, prior cancellation, residual decomposition, seed task-space spread, and continuation completion are independently recomputed by `scripts/audit_retargeting_v3_capability.py`.
- The final live audit reports `status=passed`, `failure_count=0`.

## Verification

- Clean full pytest artifact: `365 passed, 10 skipped, 148 warnings in 5354.42s (1:29:14)`.
- Final capability audit: `PASS`.

Remaining blockers: none.
