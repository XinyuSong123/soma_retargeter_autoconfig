# Step 2 Agent A Handoff - Runtime Model Truth & Cross-Format Adapter

## Agent Context

- Agent: A-xhigh - Runtime Model Truth & Cross-Format Adapter.
- Branch/worktree: `agent-a-xhigh-runtime-model-truth` at `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_a_xhigh`.
- This run used `xhigh` reasoning per `goal.md`. Prior Agent A commits `ca4c19c`/`dc560a6` were reviewed as input and re-verified; they were not assumed correct.
- Scope respected: only Agent A owned files were edited.
- Production `NewtonPipeline` was not touched.

## Commits

- `ca4c19c refactor: harden runtime model adapter truth`
- `d28a3c4 test: record xacro fingerprint provenance`

## Changed Files

- `soma_retargeter/robotics/v3/model_adapter.py`
- `soma_retargeter/robotics/v3/kinematic_paths.py`
- `soma_retargeter/robotics/v3/model_fingerprint.py`
- `soma_retargeter/robotics/v3/model_conversion.py`
- `soma_retargeter/tools/convert_robot_model_v3.py`
- `tests/v3/test_model_adapter_runtime_truth.py`
- `tests/v3/test_cross_format_adapter_scaffolding.py`
- `docs/retargeting_v3/subagents/step2_agent_a_handoff.md`

## Implementation Summary

- Added deterministic model fingerprint payloads that include primary model files, resolved XML includes, referenced assets, loader name/version, conversion settings, loader patch provenance, and compiled runtime topology summaries.
- Added explicit xacro resolution provenance to fingerprints. Xacro sources now record `xacro_resolution` with resolver status and, when Python `xacro` is available, the resolved-content SHA-256 and byte count. If the resolver is unavailable or fails, the payload records a structured non-trusted status instead of silently treating raw xacro text as resolved model truth.
- Replaced shallow adapter fingerprints with cached fingerprint metadata for both MuJoCo and Newton adapters.
- Preserved MuJoCo `mj_integratePos` tangent integration and added direct regression coverage for ball joints.
- Preserved Newton tangent integration for free/ball joints and existing regression coverage.
- Recorded temporary URDF mesh absolutization as loader provenance and kept random temporary filenames out of deterministic fingerprints.
- Added MuJoCo compiled-site provenance for MuJoCo adapter sites.
- Changed Newton model-site lookup to prefer a MuJoCo compiled site cross-check for the same source file; raw XML site use is now explicitly marked untrusted fallback rather than runtime truth.
- Added LCA branch metadata to `KinematicPath`: `reference_branch_bodies` and `target_branch_bodies`, while keeping fixed bodies in `body_path` and excluding fixed joints from active coordinates.
- Added same-source URDF to canonical MJCF conversion scaffolding using MuJoCo compiled load plus `mj_saveLastXML`.
- Added strict same-source runtime comparison scaffolding for topology, coordinate signatures, and optional neutral semantic FK.
- Added `soma_retargeter.tools.convert_robot_model_v3` CLI for conversion/report generation.

## Commands and Results

- `PYTHONPATH=. pytest -q tests/v3/test_model_adapter_runtime_truth.py tests/v3/test_cross_format_adapter_scaffolding.py`
  - Result in Agent A-xhigh worktree: `8 passed in 4.35s`.
- `PYTHONPATH=. pytest -q tests/v3/test_model_adapter_runtime_truth.py tests/v3/test_cross_format_adapter_scaffolding.py`
  - Prior Agent A result before xacro provenance repair: `7 passed in 4.40s`.
- `PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3_adapter.py tests/test_kinematic_profile_v3_math.py`
  - Result: `13 passed in 4.69s`
- `PYTHONPATH=. pytest -q tests/v3/test_model_adapter_runtime_truth.py tests/v3/test_cross_format_adapter_scaffolding.py tests/test_kinematic_profile_v3_adapter.py tests/test_kinematic_profile_v3_math.py`
  - Result in Agent A-xhigh worktree: `21 passed in 5.01s`
- `PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3.py tests/test_kinematic_profile_v3_validation_artifacts.py`
  - Result: `35 passed in 100.43s`
- `PYTHONPATH=. pytest -q`
  - Result in Agent A-xhigh worktree: `87 passed, 10 skipped in 126.22s`
- `python -m compileall -q soma_retargeter/robotics/v3 soma_retargeter/tools/convert_robot_model_v3.py`
  - Result: passed with no output.

## Numeric Results

- New Agent A tests: 8/8 passed.
- Existing adapter/math compatibility tests: 13/13 passed.
- Broader v3 profile/artifact tests: 35/35 passed.
- Full local pytest suite from clean Agent A-xhigh worktree: 87 passed, 10 skipped.
- Same-source synthetic URDF to canonical MJCF strict comparison: `strict_equivalent == true` in test fixture.
- Cross-branch synthetic path: LCA `base`; body path includes two fixed intermediate bodies; active coordinates are exactly `left_hinge`, `right_hinge`.
- MuJoCo ball tangent integration test: quaternion norm error `< 1e-12`.
- Xacro fingerprint regression: resolved-content digest `sha256("<robot name='expanded'/>")` is recorded, and `xacro:include` is listed as a resolved include.

## Assumptions

- MuJoCo is the available compiled reference for same-source site cross-checks when Newton does not expose compiled site frames.
- Same-source strict equivalence means original URDF and generated canonical MJCF from that same URDF; it does not claim vendor URDF and Menagerie MJCF variants are strictly equivalent.
- The local environment does not have the Python `xacro` package installed. The resolver path is covered by a deterministic fake-resolver regression test; real xacro sources will record `resolver_unavailable` until the validation environment installs `xacro`.
- Fingerprint path strings may still be absolute for assets outside the repo, but file content hashes and loader/conversion settings are included. Reproduction commands should still avoid local absolute paths at higher validation layers.
- Full G1/H1/Booster/TALOS same-source reports are infrastructure-ready but not generated by this Agent A handoff.

## Failures and Risks

- Initial pytest without `PYTHONPATH=.` failed with `ModuleNotFoundError: No module named 'soma_retargeter'`; all reported passing runs used `PYTHONPATH=.`.
- The original checkout at `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig` had unrelated staged/unstaged Agent B/E/F and generated-test changes while this xhigh repair was running. Agent A created the clean worktree above and committed only Agent A-owned files there.
- A dirty full-suite run in the original checkout failed on unrelated uncommitted Step 2 acceptance/full-zoo tests. The independent clean Agent A-xhigh worktree full suite passed.
- `semantic_sites.py` still contains a known Agent B-owned issue: inferred depth can fall back to body-name string length when no world body exists. Agent A did not edit it.
- Cross-format conversion scaffolding does not yet prove full G1 strict equivalence; Agent E/D integration must run complete-model chain, rank, semantic FK, and projection comparisons.
- Xacro resolved content is fingerprinted by digest/size, not embedded in full. Downstream artifact storage can choose to archive expanded xacro text if needed.
- Newton raw XML site fallback still exists for cases where MuJoCo cannot compile the source; it is explicitly marked `trusted_runtime_site: false` and should be treated as a warning/gate input by downstream validation.

## Integration Notes

- Integrate Agent A before Agents B-D consume semantic sites, paths, Jacobians, or projection inputs.
- Downstream report schemas can read `adapter.fingerprint_details`, including `files`, `xacro_resolution`, `loader_provenance`, `compiled_summary`, and `adapter.site_frame_provenance`.
- `KinematicPath.to_json()` now emits `reference_branch_bodies` and `target_branch_bodies`; existing consumers using permissive JSON should remain compatible, but strict schema checks need updating.
- Agent B should use site provenance to reject untrusted raw XML sites for positive humanoid gates unless independently verified.
- Agent D can use branch metadata to audit cross-branch relative tasks and confirm common ancestors above the LCA are excluded.
- Agent E can call `python -m soma_retargeter.tools.convert_robot_model_v3 --input <urdf> --output <canonical.xml> --report <report.json> --compare --semantic-map <map.json>` for same-source conversion reports.
