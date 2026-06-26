# Step 3.1 Full-Fleet Runtime Quality Report

Status: `PASS`

Step 3.1 evaluates the full 44-row Robot Zoo/local asset scope with a runtime-local, non-production generic harness. The production `NewtonPipeline` remains opt-in and is used only for RPO/G1 pipeline-backed controls.

## Scope

- In-scope rows: 44
- Full humanoid profiles: 32
- Structured partial humanoids: 3
- Negative controls: 9
- Deferred assets remain out of scope: `berkeley_humanoid_urdf`, `romeo_urdf`

## Runtime Profile Closure

- `profile_match`: 32
- `structured_partial_supported`: 3
- `negative_control_rejected`: 9
- `runtime_local_profile_generated`: 0
- `runtime_local_profile_failed`: 0

No full/partial row silently falls back on fingerprint/source mismatch. Negative controls do not generate humanoid runtime-local profiles.

## Runtime Evaluation

- Full humanoids run required BVH core clips through V3 source-frame extraction and target generation.
- Full humanoids run generic runtime FK residual smoke on at least two core clips.
- Partial humanoids report and smoke only supported semantics.
- Negative controls run runtime load/rejection smoke and are not promoted to humanoid target streams.
- CSV/BVH motion files under `assets/motions` are inventoried; BVH clips provide SOMA source semantic frames.

## Artifacts

Primary artifact root:

```text
artifacts/retargeting_v3_step3_runtime_quality/
```

The artifact tree includes model/profile/clip/target/smoke/pipeline/quality/failure/determinism matrices plus per-model and per-clip evidence.

## Test Classification

Targeted Step 3.1 tests, the Step 3.1 audit, and V3 suite results are recorded in `test_results`.

Full repository pytest was attempted and interrupted after 31m15s with `41 passed, 1 warning` and no visible failures before interruption. It is classified as `full_repo_failed` in `acceptance_ledger.json` and is not reported as passed.

## Verdict

```text
Step 3.1 Full-Fleet Runtime Quality Evaluation: PASS
```
