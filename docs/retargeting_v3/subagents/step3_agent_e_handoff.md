# Step 3 Agent E Handoff

## Verdict

PASS

Agent E owns the Step 3 runtime-shadow artifact runner, acceptance wrapper, and
artifact schema tests. The runner now executes the real NewtonPipeline runtime
matrix when real Step 2 schema-v3 profiles are present; synthetic unit-test
profiles still produce honest blocked placeholders.

## Files Changed

- `soma_retargeter/tools/run_v3_runtime_shadow_smoke.py`
- `scripts/run_step3_runtime_shadow_acceptance.sh`
- `tests/v3/test_step3_runtime_artifact_schema_contract.py`
- `tests/v3/test_step3_runtime_smoke_matrix_cli.py`
- `artifacts/retargeting_v3_step3_runtime_shadow/`
- `docs/retargeting_v3/STEP3_RUNTIME_SHADOW_REPORT.md`
- `docs/retargeting_v3/subagents/step3_agent_e_handoff.md`

## Runtime Matrix

- Robots: `roboparty_rpo`, `unitree_g1`
- Clips: `Neutral_walk_forward_002__A057`, `wave_R_001__A428`
- Modes: `disabled`, `shadow`, `override_experimental`
- Max frames: `120`
- Result: `10` passed rows, `2` fail-closed G1 override rows.

## Evidence

- RPO strict profile/runtime fingerprint matches `roboparty_rpo_local`.
- RPO shadow computes V3 diagnostics and proves disabled-output equality.
- G1 shadow records the runtime/profile mismatch and proves no-op output equality without enabling V3 target override.
- RPO override writes finite experimental smoke output.
- G1 override fail-closes on fingerprint mismatch.
- Deterministic diagnostics hash excludes volatile runtime timing fields.
- No local absolute path tokens are written into the artifact JSON/TXT/XML files.

## Commands

```bash
git lfs fsck
PYTHONPATH=. python -m pytest -q tests/v3/test_step3_runtime_smoke_matrix_cli.py tests/v3/test_step3_runtime_artifact_schema_contract.py
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_runtime_shadow_smoke \
  --artifact-root artifacts/retargeting_v3_step3_runtime_shadow \
  --profile-artifact-root artifacts/retargeting_v3_step2_capability \
  --robots roboparty_rpo unitree_g1 \
  --clips assets/motions/bvh/Neutral_walk_forward_002__A057.bvh assets/motions/bvh/wave_R_001__A428.bvh \
  --modes disabled shadow override_experimental \
  --max-frames 120 \
  --fail-on-blocked
```

No git branch was created. No Step 2 artifacts were modified.
