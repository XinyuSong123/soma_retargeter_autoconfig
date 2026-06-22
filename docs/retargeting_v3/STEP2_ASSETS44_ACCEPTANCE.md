# Step 2.2 Assets44 Acceptance

Current result: BLOCKED pending regenerated Assets44 artifacts.

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

Blocking items before final PASS:

- Generate and commit the Assets44 artifact tree.
- Include `environment.json`, `scope.json`, `source_inventory.json`, `load_matrix.json`, `semantic_matrix.json`, `deterministic_rerun.json`, `cross_format.json`, `deferred_snapshots.json`, `summary.json`, `per_robot/`, `failures/`, and `test_results/`.
- Confirm `source_unavailable=0`, `model_load_failed=0`, `semantic_failed=0`.
- Confirm deterministic rerun comparison counts are nonzero and matched.
- Record final algorithm failures with concrete projection/rank metrics.

