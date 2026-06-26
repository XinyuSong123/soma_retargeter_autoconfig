# Step 3 Agent D Handoff

## Verdict

Agent D owned scope: PASS.

Full Step 3 runtime clip smoke remains pending outside Agent D because NewtonPipeline integration and the smoke runner are owned by other agents.

## Files Changed

- `soma_retargeter/runtime/v3/override.py`
  - Added explicit `override_experimental` config validation.
  - Added order-preserving semantic target replacement.
  - Added finite target summary helpers.
  - Added profile artifact loading for Step 2 per-robot JSON profiles.
  - Added RPO/G1 override profile policy checks and fail-closed fingerprint handling.
- `configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_shadow.json`
- `configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_override.json`
- `configs/unitree_g1/soma_to_g1_retargeter_config_v3_shadow.json`
- `configs/unitree_g1/soma_to_g1_retargeter_config_v3_override.json`
  - Added explicit top-level smoke configs in the requested owned `configs/...` layout.
  - No IK iteration, joint-limit weight, smooth weight, or IK map body changes between shadow and override configs.
- `tests/v3/test_runtime_override_mapping_agent_d.py`
- `tests/v3/test_runtime_rpo_override_smoke_agent_d.py`
- `tests/v3/test_runtime_g1_override_policy_agent_d.py`
  - Added failing tests first, then implementation.
  - Tests include a fallback direct load of `override.py` only when unowned `soma_retargeter.runtime.v3.__init__` dependencies are temporarily missing during concurrent work.

## Commands Run

- `conda run -n soma-retargeter-v2 python -m pytest -q tests/v3/test_runtime_override_mapping_agent_d.py tests/v3/test_runtime_g1_override_policy_agent_d.py tests/v3/test_runtime_rpo_override_smoke_agent_d.py`
  - Initial expected failure before implementation: `ModuleNotFoundError: No module named 'soma_retargeter.runtime.v3.override'`.
- `conda run -n soma-retargeter-v2 python -m pytest -q tests/v3/test_runtime_override_mapping_agent_d.py tests/v3/test_runtime_g1_override_policy_agent_d.py tests/v3/test_runtime_rpo_override_smoke_agent_d.py`
  - Result after implementation: `13 passed in 4.94s`.
- `conda run -n soma-retargeter-v2 python -m pytest -q tests/v3/test_runtime_override_mapping_*.py tests/v3/test_runtime_rpo_override_smoke_*.py tests/v3/test_runtime_g1_override_policy_*.py`
  - Final result: `13 passed in 17.54s`.
- `conda run -n soma-retargeter-v2 python -m py_compile soma_retargeter/runtime/v3/override.py`
  - Result: passed.
- `git diff --check -- soma_retargeter/runtime/v3/override.py configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_shadow.json configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_override.json configs/unitree_g1/soma_to_g1_retargeter_config_v3_shadow.json configs/unitree_g1/soma_to_g1_retargeter_config_v3_override.json tests/v3/test_runtime_override_mapping_agent_d.py tests/v3/test_runtime_rpo_override_smoke_agent_d.py tests/v3/test_runtime_g1_override_policy_agent_d.py`
  - Result: passed.
- `conda run -n soma-retargeter-v2 python -c "import soma_retargeter.runtime.v3.override as override; print('import ok')"`
  - Result: `import ok`.

## Test Results

- Agent D targeted tests: 13 passed.
- `override.py` syntax compile: passed.
- Whitespace diff check: passed.

## Numerical Findings

- RPO profile fingerprint used for strict-match test: `0208dff000ed332b8fa77527a1cc66ab4494e1fac167389da27f1b2f366f0bea`.
- G1 profile fingerprint used for mismatch policy test: `f71e69596e0cad36d779e2f1a04ee540797c56a57cd9927c43282259cd66a9c0`.
- RPO replacement smoke shape: 120 frames, 7 effectors, 7 coordinates.
- RPO replacement smoke finite summary: `nan_count=0`, `inf_count=0`, `finite=True`.
- G1 override mismatch policy:
  - Mismatch without runtime-local profile: fail-closed via `FingerprintMismatchError`.
  - Mismatch with `allow_runtime_recompile_on_mismatch=true` but no runtime-local profile: fail-closed.
  - Mismatch with allow flag and runtime-local profile generated: policy decision allows override and marks `runtime_local_profile_required=True`.

## Risks And Blockers

- Full NewtonPipeline clip smoke was not run in Agent D because pipeline integration and smoke-matrix execution are outside this owned file set.
- The added `configs/...` files are top-level because that is the path in Agent D's owned list. Existing runtime package configs under `soma_retargeter/configs/...` were read but not modified.
- The worktree contains concurrent unowned Step 3 files and tests. I did not revert or edit them.
- `soma_retargeter/runtime/v3/override.py` was modified concurrently during verification; the final test result above was rerun against the current on-disk file.
- No visual quality claims were made.
- No IK weights, solver iterations, contact logic, temporal smoothing, or Step 2 artifacts were changed.
