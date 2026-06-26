# Step 3.1 Agent B Handoff

## Scope

Implemented runtime-local clip inventory and target stream scaffolding for Step 3.1.

Owned files touched:

- `soma_retargeter/runtime/v3/clip_inventory.py`
- `soma_retargeter/runtime/v3/target_stream.py`
- `tests/v3/test_step3_clip_inventory_runtime_quality.py`
- `tests/v3/test_step3_full_fleet_target_stream_runtime_quality.py`
- `tests/v3/test_step3_partial_target_stream_runtime_quality.py`
- `docs/retargeting_v3/subagents/step3_1_agent_b_handoff.md`

## Clip Inventory

`clip_inventory.inventory_motion_clips(repo_root)` scans every `.bvh` and `.csv` under `assets/motions`, records relative paths, sha256, byte count, frame count, sample rate, load status, and deterministic frame budget.

Required core clips are enforced:

- `assets/motions/bvh/Neutral_walk_forward_002__A057.bvh`
- `assets/motions/bvh/wave_R_001__A428.bvh`
- `assets/motions/bvh/body_stretch_1_004__A069.bvh`
- `assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh`

Missing or parse-unloadable core clips set inventory status to `blocked`; `assert_core_clips_available()` raises a structured gate error.

Frame budget policy is deterministic:

- `<= 120` frames: all frames, `full_short_clip`
- `121..300` frames: all frames, `full_mid_clip`
- `> 300` frames: fixed stride `ceil(frame_count / 300)`, capped to at most 300 selected frames, `strided_long_clip`

## Target Stream

`target_stream.generate_target_stream_for_clip(profile, clip_path, repo_root=...)` handles one profile/clip pair.

Full humanoid profiles:

- load BVH source clips only;
- apply the deterministic frame budget;
- build a sliced `AnimationBuffer`;
- call existing `extract_source_semantic_frames`;
- call existing `build_runtime_semantic_targets`;
- validate finite SE(3) output and report per-semantic compact metrics.

Partial humanoid profiles:

- default to report-only mode;
- resolve explicit `supported_semantics` from profile morphology/failure taxonomy or semantic expectation JSON;
- report only supported semantics;
- do not call target adapter for missing semantics;
- do not fabricate hand/foot targets not declared as supported.

Missing source semantics and unsupported profile/clip states return structured blocked results with `failure.code`; they are not silently filled.

## Tests

Targeted validation command:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_clip_inventory_runtime_quality.py \
  tests/v3/test_step3_full_fleet_target_stream_runtime_quality.py \
  tests/v3/test_step3_partial_target_stream_runtime_quality.py
```

The tests cover inventory completeness/core gating, deterministic frame budget, full target stream reuse of source frames plus target adapter, structured missing-semantic failure, and partial supported-only reporting.
