# Step 3.2 Solver-Backed Generic Runtime Smoke Handoff

## Branch

```text
retargeting-v3-step3-2-solver-backed-generic-smoke
```

## Baseline

Closed Step 3.1.1 final HEAD:

```text
26817de67bdda0cb315a1237b53c30e4d8199c78
```

Step 3.2 preserves the closed Step 3.1.1 artifact tree under
`artifacts/retargeting_v3_step3_runtime_quality/` and writes new evidence under:

```text
artifacts/retargeting_v3_step3_2_solver_backed_smoke/
```

## Implementation Commits

```text
f29ed3c Add Step 3.2 solver-backed generic smoke
e521b1e Fix Step 3.2 solver failure aggregation
35974fb Add Step 3.2 solver-backed smoke artifacts
f49c797 Install Warp in Step 3.2 CI
```

The artifact provenance records source commit:

```text
e521b1e069a94dc43e5d1693af2da15f3323f2eb
```

The strict audit uses live GitHub check-run evidence for the current branch HEAD
so final HEAD CI is not represented by stale self-referential committed metadata.

## Artifact Counts

From `artifacts/retargeting_v3_step3_2_solver_backed_smoke/quality_summary.json`:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9

solver_backed_smoke_attempted_count = 32
solver_backed_completed_count = 32
solver_backed_count = 32
residual_only_count = 0

runtime_quality_passed_count = 0
runtime_quality_warned_count = 23
runtime_quality_failed_count = 9
partial_runtime_passed_count = 3
negative_control_runtime_passed_count = 9

deterministic_compared_count = 44
deterministic_matched_count = 44
```

Interpretation:

```text
Step 3.2 PASS means generic solver-backed smoke evidence exists and is audited.
It does not mean all full humanoids pass runtime quality, visual motion quality is solved,
or production retargeting is ready.
```

## Commands And Evidence

Artifact generation command is recorded in:

```text
artifacts/retargeting_v3_step3_2_solver_backed_smoke/commands.txt
```

Focused/local test evidence is recorded in:

```text
artifacts/retargeting_v3_step3_2_solver_backed_smoke/test_results/pytest.txt
artifacts/retargeting_v3_step3_2_solver_backed_smoke/test_results/pytest_summary.json
artifacts/retargeting_v3_step3_2_solver_backed_smoke/test_results/junit.xml
```

Recorded pytest result:

```text
25 passed, 8 warnings
```

Local audit command:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py \
  --artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root .
```

Strict final-head audit command:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_2_solver_backed_smoke.py \
  --artifact-dir artifacts/retargeting_v3_step3_2_solver_backed_smoke \
  --source-root . \
  --require-final-head-ci
```

Both must report:

```text
status = PASS
blocking_count = 0
```

LFS evidence:

```text
git lfs fsck = OK
```

## GitHub Actions

Workflow:

```text
Retargeting V3 Step 3.2 Solver-Backed Generic Smoke
.github/workflows/retargeting_v3_step3_2_solver_backed_smoke.yml
```

Required jobs:

```text
step3-2-static-and-unit
step3-2-artifact-audit
step3-2-lfs-and-snapshot-smoke
step3-2-pipeline-controls-reference
step3-2-final-head-ci-evidence
```

The strict Step 3.2 audit queries GitHub check-runs for the current HEAD and
requires the core Step 3.2 jobs to be successful.

## Known Limitations

```text
runtime_quality_passed_count remains 0.
23 full humanoid rows are runtime_quality_warned.
9 full humanoid rows are runtime_quality_failed.
The solver smoke is intentionally bounded and generic; it is not production retargeting.
No contact, collision, temporal smoothing, visual quality, or Step 3.3 work is included.
```

## Next Direction

Do not start Step 3.3 automatically. A future step can broaden generic solver tasks,
improve global convergence behavior, and reduce residual and joint-limit warnings while
preserving the same audit semantics and avoiding robot-specific tuning.
