# Step 3 Runtime Shadow Acceptance

Status: PASS

Step 3.0 runtime profile shadow integration is implemented behind the explicit
`v3_runtime_profile` config. Missing or disabled config remains a no-op.

## Evidence Summary

- Runtime smoke matrix: `2` robots x `2` clips x `3` modes, `max_frames=120`.
- Matrix result: `10` rows passed and `2` G1 override rows fail-closed on runtime/profile fingerprint mismatch.
- RPO shadow: V3 targets are computed through the Step 2 profile and `target_builder`; IK input targets and final outputs match disabled mode exactly.
- G1 shadow: runtime/profile mismatch is recorded, V3 target override is skipped, and disabled-output equality is still proven.
- RPO override: explicit `override_experimental` config produces finite experimental output. No quality or visual improvement claim is made.
- G1 override: fail-closed unless a matching runtime-local profile is generated.
- Artifact audit: no missing diagnostics, no NaN/Inf JSON diagnostics, no local absolute path leakage, no Step 2 artifact mutation, and CI workflow scaffolding is present.

## Required Artifacts

- `artifacts/retargeting_v3_step3_runtime_shadow/profile_resolution.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/shadow_summary.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/override_smoke_summary.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/smoke_matrix.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/deterministic_rerun.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/per_clip/**/target_deltas.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/per_clip/**/pipeline_summary.json`
- `artifacts/retargeting_v3_step3_runtime_shadow/test_results/`
- `artifacts/retargeting_v3_step3_runtime_shadow/acceptance_ledger.json`

## Commands

```bash
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_runtime_shadow_smoke \
  --artifact-root artifacts/retargeting_v3_step3_runtime_shadow \
  --profile-artifact-root artifacts/retargeting_v3_step2_capability \
  --robots roboparty_rpo unitree_g1 \
  --clips assets/motions/bvh/Neutral_walk_forward_002__A057.bvh assets/motions/bvh/wave_R_001__A428.bvh \
  --modes disabled shadow override_experimental \
  --max-frames 120 \
  --fail-on-blocked

PYTHONPATH=. python scripts/audit_retargeting_v3_step3_runtime_shadow.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_shadow \
  --source-root . \
  --output-json artifacts/retargeting_v3_step3_runtime_shadow/acceptance_ledger.json
```
