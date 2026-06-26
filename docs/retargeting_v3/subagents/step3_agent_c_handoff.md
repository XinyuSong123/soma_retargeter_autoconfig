# Step 3 Agent C Handoff

## Scope

Agent C implemented the NewtonPipeline shadow no-op integration boundary only. No git branches were created.

## Files Changed

- `soma_retargeter/pipelines/newton_pipeline.py`
- `soma_retargeter/pipelines/v3_runtime_config.py`
- `tests/v3/test_newton_pipeline_v3_config_parser.py`
- `tests/v3/test_newton_pipeline_v3_shadow_noop_targets.py`
- `docs/retargeting_v3/subagents/step3_agent_c_handoff.md`

## Implementation Summary

- Added parsing for optional top-level `v3_runtime_profile` config.
- Missing config and `enabled=false` normalize to disabled no-op.
- Added `NewtonPipeline` V3 state fields:
  - `v3_runtime_config`
  - `v3_runtime_profile`
  - `v3_runtime_profile_resolution`
  - `v3_runtime_adapter`
  - `v3_runtime_diagnostics`
- Disabled mode does not import runtime V3 modules, compute targets, or write diagnostics.
- Active modes lazy-load the runtime profile loader and target adapter.
- Shadow mode calls an injected/loaded adapter for target computation but returns the legacy `HumanToRobotScaler` target stream unchanged.
- Adapter receives a copy of legacy effectors, so adapter-side mutation cannot change `self.input_targets` in shadow mode.
- Diagnostics are collected in memory during `add_input_motions`.
- `execute()` was only changed after output buffers are produced, to write the summary diagnostics helper.
- IK solver objective order, objective weights, and frame loop target assignment were not changed.

## Commands Run

```bash
conda run -n soma-retargeter-v2 python -m pytest -q \
  tests/v3/test_newton_pipeline_v3_config_parser.py \
  tests/v3/test_newton_pipeline_v3_shadow_noop_targets.py
```

Result:

```text
10 passed in 6.22s
```

```bash
conda run -n soma-retargeter-v2 python -m pytest -q tests/test_newton_contact_sources.py
```

Result:

```text
10 passed in 9.17s
```

Earlier red-state check before implementation:

```text
ImportError: cannot import name 'RuntimeV3Mode' from 'soma_retargeter.pipelines.v3_runtime_config'
```

## Coverage Notes

- Config tests cover missing config, `enabled=false`, shadow defaults, override config validation, invalid mode, and invalid diagnostics frame budget.
- Shadow tests prove:
  - missing config does not call the adapter;
  - shadow computes fake V3 targets through the adapter;
  - shadow `self.input_targets` equals disabled `self.input_targets`;
  - IK iteration and objective weight fields are unchanged by shadow target computation;
  - shadow diagnostics summary can be written;
  - disabled mode does not write V3 diagnostics.

## Risks and Blockers

- Agent B runtime files `source_frames.py`, `target_adapter.py`, and `diagnostics.py` were not present when this handoff was written. The pipeline uses a duck-typed adapter boundary and raises a clear initialization error if active config is used before the target adapter exists.
- Agent A `profile_loader.py` appeared during this work and was treated as external. Agent C aligned to its current `load_runtime_v3_profile(target_type, runtime_mjcf_path, config)` shape.
- Override behavior is only fail-closed/minimal at this layer. Agent D still needs to own configured semantic replacement semantics.
- Full runtime smoke, real profile resolution, and artifact matrix tests were not run by Agent C.
- The shared worktree contains unrelated/unowned changes from other agents; Agent C did not revert or edit them.
