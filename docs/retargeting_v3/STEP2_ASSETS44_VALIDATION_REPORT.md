# Step 2.2 Assets44 Validation Report

Status: PASS.

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

Final validation result:

- `source_unavailable=0`
- `model_load_failed=0`
- `semantic_failed=0`
- `profile_passed=21`
- `partial_passed=3`
- `negative_control_passed=9`
- `algorithm_failed=11`
- `deterministic_compared=33`
- `deterministic_matched=33`

The authoritative artifacts are committed under `artifacts/retargeting_v3_step2_assets44/`.
