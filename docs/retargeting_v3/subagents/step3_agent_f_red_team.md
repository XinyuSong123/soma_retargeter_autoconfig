# Step 3 Agent F Red Team Handoff

Date: 2026-06-26

verdict = PASS

## Scope

Agent F owns the live red-team audit, CI scaffold, stale-verdict checks, and
false-positive gate coverage for Step 3 runtime shadow evidence.

## Final Evidence

- Runtime smoke matrix: `10` passed rows, `2` G1 override fail-closed rows.
- Shadow equality result: PASS. Shadow IK input targets and final outputs match disabled mode exactly for required RPO and G1 clips.
- RPO override result: PASS for finite experimental smoke output only; no quality claim is made.
- G1 override result: PASS by fail-closed fingerprint policy.
- Live audit result: PASS with `blocking_count=0` after runtime artifacts and verdict documents are current.
- Required CI jobs are present in `.github/workflows/retargeting_v3_step3_runtime_shadow.yml`.

Commit and workflow IDs are filled by the final source/artifact commit and CI
verification pass:

- final_head: pending final artifact commit
- source_code_commit: pending source-code commit
- artifact_commit: pending artifact commit
- workflow_run_id: pending GitHub Actions run

## Commands

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_step3_runtime_shadow_acceptance_audit.py
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

## Remaining Blockers

None for Step 3.0 runtime shadow acceptance. G1 override remains intentionally
disabled until a matching runtime-local profile is generated.
