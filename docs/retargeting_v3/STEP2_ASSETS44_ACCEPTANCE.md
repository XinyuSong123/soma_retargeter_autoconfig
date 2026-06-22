# Step 2.2 Assets44 Acceptance

Current result: PASS.

Acceptance requires `scripts/audit_retargeting_v3_assets44.py` to pass against:

```text
artifacts/retargeting_v3_step2_assets44/
```

The committed source tree currently satisfies these prerequisites:

- Frozen 44/2 scope in `robot_zoo_lock.json`.
- 38 committed snapshots with no vendored meshes.
- 5 fetch-only models kept out of Git.
- Verified semantic maps or structured partial expectations for every in-scope humanoid.
- Lock-aware validation code path for source resolution and summary matrices.

Final evidence:

- `scripts/audit_retargeting_v3_assets44.py` passes against `artifacts/retargeting_v3_step2_assets44/`.
- `summary.json` records `source_unavailable=0`, `model_load_failed=0`, and `semantic_failed=0`.
- Deterministic rerun records `deterministic_compared=33` and `deterministic_matched=33`.
- `failures/` contains concrete algorithm failure reports for 11 models.
