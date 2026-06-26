# Step 3.0 Runtime Profile Shadow Report

Status: PASS

Step 3.0 wires verified V3 runtime profiles into `NewtonPipeline` behind the
explicit `v3_runtime_profile` configuration block. The default path remains
unchanged when that block is absent or disabled.

## Implemented Surface

- `disabled`: no V3 runtime imports, no V3 target computation, no V3 diagnostics.
- `shadow`: load and fingerprint-check the Step 2 profile, compute V3 semantic targets for diagnostics when safe, and keep legacy IK inputs unchanged.
- `override_experimental`: explicit config only; RPO finite smoke is allowed after strict fingerprint match, while G1 fail-closes on runtime/profile mismatch.

## Runtime Smoke Result

Command:

```bash
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_runtime_shadow_smoke \
  --artifact-root artifacts/retargeting_v3_step3_runtime_shadow \
  --profile-artifact-root artifacts/retargeting_v3_step2_capability \
  --robots roboparty_rpo unitree_g1 \
  --clips assets/motions/bvh/Neutral_walk_forward_002__A057.bvh assets/motions/bvh/wave_R_001__A428.bvh \
  --modes disabled shadow override_experimental \
  --max-frames 120 \
  --fail-on-blocked
```

Matrix:

- `roboparty_rpo`: disabled/shadow/override rows passed for both required clips.
- `unitree_g1`: disabled/shadow rows passed for both required clips.
- `unitree_g1`: override rows fail-closed for both required clips because the current Newton runtime asset fingerprint does not match the committed Step 2 `unitree_g1_mjcf` profile.

The smoke matrix therefore records `10` passed rows and `2` acceptable
fail-closed rows.

## Audit Result

The live red-team audit checks the false-positive gates from `goal.md`:

- default output changed
- shadow mutates IK inputs
- override without explicit config
- silent fingerprint mismatch acceptance
- partial or negative profile override
- LFS pointer profile/snapshot
- missing diagnostics
- diagnostics NaN/Inf
- local absolute path leakage
- Step 2 artifact mutation
- missing CI
- stale Agent F verdict

Current result: PASS with zero blocking findings once final artifacts and this
document are regenerated together.

## Limits

- No IK weights, solver iterations, contact policy, smoothing, collision, or Step 2 artifacts were changed.
- RPO override is an experimental finite smoke only. It is not a visual-quality or behavior-quality claim.
- G1 override remains unavailable until a matching runtime-local profile is generated or the runtime asset matches the committed Step 2 profile.
