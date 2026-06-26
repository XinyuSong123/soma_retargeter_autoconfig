# Step 3 Agent B Handoff

## Scope

Implemented Source Semantic Frames and V3 Target Adapter support in the Agent B owned files only.

## Files Changed

- `soma_retargeter/runtime/v3/source_frames.py`
- `soma_retargeter/runtime/v3/target_adapter.py`
- `soma_retargeter/runtime/v3/diagnostics.py`
- `tests/v3/test_runtime_source_frames_semantics.py`
- `tests/v3/test_runtime_target_adapter_semantics.py`
- `tests/v3/test_runtime_target_diagnostics_metrics.py`
- `docs/retargeting_v3/subagents/step3_agent_b_handoff.md`

Concurrent unowned Step 3 files are present in the worktree, including `soma_retargeter/runtime/v3/__init__.py`, `soma_retargeter/runtime/v3/profile_loader.py`, `soma_retargeter/runtime/v3/override.py`, pipeline files, scripts, configs, and artifacts. I did not revert or edit those outside the Agent B scope.

## Implementation Summary

- Added `SourceSemanticFrameBatch` with semantic names, source joint names, per-semantic `[F, 4, 4]` float64 transforms, frame count, sample rate, source label, and joint lookup evidence.
- Added deterministic SOMA semantic joint lookup for `Hips`, `Chest`, `LeftHand`, `RightHand`, `LeftFoot`, and `RightFoot`.
- Added strict finite SE(3) validation for offset transforms, extracted source frames, profile rest-calibration frames, and target outputs.
- Added frame slicing via `frame_start`, `frame_stop`, and `max_frames`.
- Added runtime target building through `target_builder.build_targets_from_source_semantic_frames`; no duplicate scaler math was added.
- Added profile JSON to `RestCalibration` reconstruction for Step 2 per-robot artifacts and Agent A `RuntimeV3Profile.payload`.
- Added target conversion back to existing effector order, preserving legacy targets for nonsemantic effectors when provided.
- Added per-semantic target delta diagnostics with deterministic JSON output.
- Preserved the concurrently added pipeline compatibility method in `RuntimeV3TargetAdapter.compute_targets()` and tightened invalid offset handling to fail closed.

## Commands Run

```bash
conda run -n soma-retargeter-v2 pytest tests/v3/test_runtime_source_frames_semantics.py tests/v3/test_runtime_target_adapter_semantics.py tests/v3/test_runtime_target_diagnostics_metrics.py
```

Initial expected failing run before implementation: collection failed because runtime modules did not exist and, at that moment, unowned Agent A `profile_loader.py` was absent.

```bash
conda run -n soma-retargeter-v2 pytest tests/v3/test_runtime_source_frames_semantics.py tests/v3/test_runtime_target_adapter_semantics.py tests/v3/test_runtime_target_diagnostics_metrics.py
```

Result after implementation: `7 passed in 7.67s`.

```bash
conda run -n soma-retargeter-v2 pytest tests/v3/test_target_builder_calibrated_geometry.py tests/v3/test_rest_calibration_validation.py tests/v3/test_runtime_source_frames_semantics.py tests/v3/test_runtime_target_adapter_semantics.py tests/v3/test_runtime_target_diagnostics_metrics.py
```

Final result: `18 passed in 11.58s`.

```bash
conda run -n soma-retargeter-v2 python -c "import soma_retargeter.runtime.v3.source_frames as s; import soma_retargeter.runtime.v3.target_adapter as t; import soma_retargeter.runtime.v3.diagnostics as d; print(s.SourceSemanticFrameBatch.__name__); print(t.RuntimeV3TargetAdapter.__name__); print(hasattr(d, 'build_target_delta_diagnostics'))"
```

Result: normal package import succeeded and printed `SourceSemanticFrameBatch`, `RuntimeV3TargetAdapter`, `True`.

```bash
conda run -n soma-retargeter-v2 python -m py_compile soma_retargeter/runtime/v3/source_frames.py soma_retargeter/runtime/v3/target_adapter.py soma_retargeter/runtime/v3/diagnostics.py
```

Result: passed.

## Numerical Findings

- Source extraction synthetic check: with `frame_start=1`, `max_frames=2`, and `+10m` x offset, `Hips` positions were `[[11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]`; `Chest` positions were `[[11.0, 0.0, 1.0], [12.0, 0.0, 1.0]]`.
- Target adapter synthetic check: with root horizontal scale `2.0`, frame-1 source root x delta `0.25` produced V3 `Hips` target position `[0.5, 0.0, 0.0]`.
- Diagnostics synthetic check: `Hips` translation deltas produced mean `0.2`, max `0.3`, p95 `0.29`, finite count `2`, nan count `0`.
- Diagnostics synthetic check: `Chest` rotation delta mean was `pi / 2` radians for the configured 90 degree yaw delta.
- Deterministic diagnostics JSON write was byte-identical across write/read/write.

## Risks / Blockers

- Full NewtonPipeline shadow and override smoke were not run by Agent B; those are Agent C/D/E scopes.
- Runtime artifact generation was not run by Agent B; diagnostics helpers are ready for pipeline/artifact agents.
- The worktree contains many concurrent untracked/modified Step 3 files outside Agent B scope. Final integration should run the combined Step 3 test matrix after all agents finish.
- Target adapter override safety depends on Agent A profile loader correctly blocking partial, negative-control, and fingerprint-mismatched profiles before Agent C/D call into this adapter.
