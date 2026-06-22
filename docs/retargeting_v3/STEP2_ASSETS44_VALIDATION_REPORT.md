# Step 2.2 Assets44 Validation Report

Status: in progress.

Scope is frozen by `assets/robot_zoo/robot_zoo_lock.json`:

- 46 public source entries resolved.
- 44 in-scope entries.
- 38 committed mesh-free kinematic snapshots.
- 5 fetch-only cached models.
- 1 project-local RPO model.
- 2 deferred snapshot IDs: `berkeley_humanoid_urdf`, `romeo_urdf`.

Current verified evidence:

- Asset lock and snapshot integrity pass against the committed lock and snapshots.
- In-scope load closure is proven by lock-aware loader probe with 44/44 loaded.
- Lock-aware semantic coverage has no missing unclassified map: verified full maps or structured partial expectations cover all in-scope positive/partial humanoids.
- Negative controls route to `negative_control_passed` without humanoid profile generation.

Known open validation work:

- `artifacts/retargeting_v3_step2_assets44/` must be regenerated from a clean worktree.
- Full validation and deterministic rerun must be committed in that artifact tree.
- Remaining algorithm failures must be reported from the regenerated per-robot reports with motion/task/metric/threshold details.

