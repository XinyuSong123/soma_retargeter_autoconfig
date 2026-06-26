# Step 3 Agent A Handoff - Runtime Profile Loader and Fingerprint Gate

Date: 2026-06-26

## Scope

Implemented the Step 3.0 runtime profile loader and fingerprint/source gate for Agent A.
No git branch was created.

## Files Changed

- `soma_retargeter/runtime/v3/profile_loader.py`
- `soma_retargeter/runtime/v3/__init__.py`
- `soma_retargeter/pipelines/utils.py`
- `tests/v3/test_runtime_profile_loader_contract.py`
- `tests/v3/test_runtime_fingerprint_gate_policy.py`
- `docs/retargeting_v3/subagents/step3_agent_a_handoff.md`

`soma_retargeter/runtime/__init__.py` was present in the shared untracked runtime tree; Agent A did not change its contents.

## Implementation Summary

- Added `RuntimeV3Profile`, `RuntimeModelIdentity`, and `RuntimeProfileResolution` dataclasses.
- Added structured `RuntimeV3ProfileError` errors with `code`, `message`, and `details`.
- Added direct profile loading from `artifacts/retargeting_v3_step2_capability/per_robot/<id>.json`.
- Validates schema version, terminal status, semantic sites, rest calibration, canonical targets, and task certificate summary.
- Added Git LFS pointer detection for profile artifacts and runtime MJCF files.
- Added `get_default_runtime_v3_profile_id()` resolver helper:
  - `roboparty_rpo -> roboparty_rpo_local`
  - `unitree_g1 -> unitree_g1_mjcf`
- Runtime fingerprint/source gate uses `NewtonRuntimeModelAdapter` and primary file SHA256.
- RPO requires strict fingerprint and source match.
- G1 shadow mismatch returns explicit `fingerprint_mismatch` with override disallowed.
- G1 override mismatch raises fail-closed unless `allow_runtime_recompile_on_mismatch=true`, in which case it returns `runtime_local_profile_required` and still disallows override.
- `soma_retargeter.runtime.v3` now lazy-loads Agent B modules so Agent A imports do not fail while neighboring work is incomplete.
- No robot-specific math branch was added; robot-specific logic is limited to profile ID resolution and fingerprint policy.

## Commands Run

```bash
conda run -n soma-retargeter-v2 bash -lc 'PYTHONPATH=. python -m pytest -q tests/v3/test_runtime_profile_loader_contract.py tests/v3/test_runtime_fingerprint_gate_policy.py'
```

Initial expected failing result after writing tests: 2 collection errors from missing runtime loader/package exports.

```bash
conda run -n soma-retargeter-v2 bash -lc 'PYTHONPATH=. python -m pytest -q tests/v3/test_runtime_profile_loader_*.py tests/v3/test_runtime_fingerprint_gate_*.py'
```

Final result: `10 passed in 7.14s`.

```bash
conda run -n soma-retargeter-v2 bash -lc 'PYTHONPATH=. python -m py_compile soma_retargeter/runtime/v3/profile_loader.py soma_retargeter/runtime/v3/__init__.py soma_retargeter/runtime/__init__.py soma_retargeter/pipelines/utils.py'
```

Result: passed.

```bash
git lfs fsck
```

Result: `Git LFS fsck OK`.

Also ran a live RPO/G1 runtime profile resolution script through `conda run -n soma-retargeter-v2` using `pipeline_utils.get_robot_mjcf_path()` and `resolve_runtime_v3_profile_id()`.

## Numerical Findings

- `roboparty_rpo_local` profile:
  - schema version: `3`
  - status: `passed`
  - semantic sites: `6`
  - canonical targets: `15`
  - task certificates: `5`
  - profile/runtime fingerprint: `0208dff000ed332b8fa77527a1cc66ab4494e1fac167389da27f1b2f366f0bea`
  - profile/runtime source SHA256: `d9c91b9cb9e14581975d6a76433c6cbaf5a00e81939e2fbb74c5b31196be1fa1`
  - live resolution: `matched`, `fingerprint_match=True`, `source_hash_match=True`

- `unitree_g1_mjcf` committed profile:
  - schema version: `3`
  - status: `passed`
  - semantic sites: `6`
  - canonical targets: `15`
  - task certificates: `5`
  - profile fingerprint: `f71e69596e0cad36d779e2f1a04ee540797c56a57cd9927c43282259cd66a9c0`
  - profile source SHA256: `7853bdf532f9f9724ac12ab1c2294fa646aac54d541478ef34f5e460124537d9`

- `unitree_g1` live Newton runtime asset:
  - runtime fingerprint: `f7896a2ba6c307a67119e7db55e6e62dbaeb5bf7be2a28b50ea31bbf2c4365fb`
  - runtime source SHA256: `a58f9b51cf878e20d6490842eb1e03000d32129e076a3785d51edfae01f8bb84`
  - live shadow resolution: `fingerprint_mismatch`
  - `fingerprint_match=False`, `source_hash_match=False`, `override_allowed=False`

## Risks / Blockers

- G1 runtime override remains blocked until a runtime-local profile is generated or the runtime asset matches the committed Step 2 profile.
- The live G1 check refreshed the Newton cached `unitree_g1` asset, which confirms the runtime-local mismatch path is currently exercised.
- RPO strict match depends on preserving Step 2 fingerprint provenance. The loader passes repo-local runtime paths to the Newton adapter when possible so the fingerprint matches the committed profile while still hashing the materialized file.
- Full `tests/v3` and full repo pytest were not run by Agent A because this handoff is scoped to the Agent A files and the worktree contains concurrent Step 3 changes outside this ownership set.
- Current shared worktree contains unrelated modified/untracked files from other agents, including `soma_retargeter/pipelines/newton_pipeline.py`, `soma_retargeter/robot_registry_parser.py`, runtime adapter files, configs, artifacts, scripts, and other tests. Agent A did not revert or edit those files.
