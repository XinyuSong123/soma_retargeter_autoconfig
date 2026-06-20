# Step 2 Agent R Handoff - Rank-Zero Projection Evidence

## Scope

Agent: R-xhigh

Allowed files touched:

- `soma_retargeter/robotics/v3/chain_projection.py`
- `tests/v3/test_rank_zero_projection_evidence.py`
- `docs/retargeting_v3/subagents/step2_agent_r_handoff.md`

`soma_retargeter/robotics/v3/profile.py` was inspected but did not require an edit.

## Projection Report Structure

`compile_kinematic_profile_v3()` builds `projection_reports` in `profile.py` as a task-keyed dict. Each task value is the JSON form of a `ProjectionResult` from `project_torso_orientation()` or `project_endpoint_position()`.

For the legacy neutral report path, rank-zero tasks can have:

- `status: "rank_zero"`
- `residual: 0.0`
- `desired == projected`
- `active_coordinates: []`

Those reports are valid only when they also carry explicit no-demand evidence.

## Changes

- Added `rank_zero_reason` to `ProjectionResult`.
- `ProjectionResult.to_json()` now backfills rank-zero evidence for both `rank_zero` and `unreachable/rank_zero` statuses:
  - `demand_residual`
  - `unreachable_demand`
  - `rank_zero_reason`
- No-active-coordinate projection paths now explicitly set that evidence at construction time.
- Added tests that prove:
  - zero-demand rank-zero endpoint projection serializes evidence;
  - `compile_kinematic_profile_v3()` emits evidence in `projection_reports`;
  - manually constructed rank-zero results are backfilled by `to_json()`;
  - the acceptance audit still flags a bare rank-zero zero-residual report.

## Verification

Commands run:

```bash
PYTHONPATH=. pytest -q tests/v3/test_rank_zero_projection_evidence.py
PYTHONPATH=. pytest -q tests/v3/test_rank_zero_projection_evidence.py tests/v3/test_acceptance_gates_audit.py
```

Results:

```text
4 passed in 0.61s
5 passed in 0.78s
```

## Artifact Note

No mainline artifacts were refreshed by Agent R. The code path now emits the required evidence, and the acceptance audit passed against the current workspace artifact state during verification. If the shared mainline artifacts still contain the three previously reported bare `rank_zero` torso reports, refresh `artifacts/retargeting_v3_step2/per_robot/*.json` from the updated compiler so those serialized reports pick up the new fields.
