# Step 3.1 Full-Fleet Runtime Quality Acceptance

Final verdict:

```text
Step 3.1 Full-Fleet Runtime Quality Evaluation: PASS
```

Acceptance evidence:

- `artifacts/retargeting_v3_step3_runtime_quality/model_matrix.json`
- `artifacts/retargeting_v3_step3_runtime_quality/quality_summary.json`
- `artifacts/retargeting_v3_step3_runtime_quality/profile_resolution_matrix.json`
- `artifacts/retargeting_v3_step3_runtime_quality/target_stream_matrix.json`
- `artifacts/retargeting_v3_step3_runtime_quality/generic_smoke_matrix.json`
- `artifacts/retargeting_v3_step3_runtime_quality/pipeline_backed_matrix.json`
- `artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json`

The artifact audit must be run with:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_quality \
  --source-root . \
  --write-report artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

Strict Step 3.1.1 final evidence closure must additionally run:

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_quality \
  --source-root . \
  --require-final-head-ci
```

The strict audit verifies GitHub check-runs for the checked-out current HEAD so
the final CI evidence cannot go stale when documentation changes are committed.

PASS means the full-fleet evaluation evidence is complete and deterministic. It does not claim final visual motion quality or start optimization/tuning.
