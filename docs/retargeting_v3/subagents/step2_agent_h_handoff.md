# Step 2 Agent H Handoff - Verified Semantic Maps Coverage

## Scope

Agent: H-xhigh
Reasoning mode: xhigh requested and used for this audit. This handoff records the decision process as evidence/methodology and outcomes, not hidden chain-of-thought.

Owned files touched:

- `soma_retargeter/robotics/v3/semantic_validation.py`
- `tests/v3/test_semantic_map_coverage_manifest.py`
- `docs/retargeting_v3/subagents/step2_agent_h_handoff.md`

No semantic map JSON was added in this pass. I only count existing verified maps and missing maps where local sources are available. I did not fabricate maps from robot names.

## Inputs Read

- `goal.md`
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `assets/robot_zoo/robot_zoo_manifest.json`
- `assets/robot_zoo/semantic_maps/roboparty_rpo_local.json`
- `assets/robot_zoo/semantic_expectations/roboparty_rpo_local.json`
- existing semantic validation/site tests

## Reasoning Record

The Step 1 and goal contract require verified maps to bind semantic sites to actual model topology/geometry and fingerprinted model sources. Name-only inference, confidence=1.0 inference, and body-origin distal endpoints are false positives.

I treated the manifest as the robot list of record. For local availability, I used side-effect-free local cache inspection:

- local workspace paths for project-local sources;
- installed `robot_descriptions` module files parsed with `ast`, not imported, so missing assets are not fetched;
- local `mujoco_menagerie` cache candidates under Robot Zoo and robot_descriptions cache roots.

I counted only `robot_class=humanoid` and `expected_capability=positive` manifest entries. Unavailable sources are reported separately and are not backfilled with speculative maps.

## Coverage Counts

Current manifest positive humanoids: **37**

- Required positive humanoids: **32**
- Fetch-only positive humanoids: **5**
- Locally available positive sources: **25**
- Source-unavailable positive sources: **12**
- Verified semantic maps present and schema-valid: **1**
- Locally available positives missing verified maps: **24**
- Source-unavailable positives missing verified maps: **12**

Existing verified map:

- `roboparty_rpo_local`

Locally available positives missing verified maps:

- `unitree_g1_urdf`
- `unitree_g1_mjcf`
- `unitree_h1_urdf`
- `unitree_h1_mjcf`
- `unitree_h1_2_urdf`
- `unitree_h1_2_mjcf`
- `robotis_op3_mjcf`
- `booster_t1_urdf`
- `booster_t1_mjcf`
- `toddlerbot_urdf`
- `toddlerbot_2xc_mjcf`
- `toddlerbot_2xm_mjcf`
- `adam_lite_mjcf`
- `apptronik_apollo_mjcf`
- `berkeley_humanoid_urdf`
- `fourier_n1_urdf`
- `fourier_n1_mjcf`
- `jvrc_urdf`
- `jvrc_mjcf`
- `elf2_urdf`
- `elf2_mjcf`
- `ergocub_urdf`
- `pal_talos_mjcf_direct`
- `berkeley_humanoid_mjcf_direct`

Source-unavailable positives missing verified maps:

- `draco3_urdf`
- `sigmaban_urdf`
- `simple_humanoid_urdf`
- `atlas_drc_urdf`
- `atlas_v4_urdf`
- `romeo_urdf`
- `mujoco_humanoid_mjcf`
- `fourier_gr1_urdf`
- `jaxon_urdf`
- `talos_urdf`
- `valkyrie_urdf`
- `robonaut2_urdf`

## Implementation

Added `SemanticMapCoverageEntry`, `SemanticMapCoverageReport`, and `compute_verified_semantic_map_coverage()` in `semantic_validation.py`.

The helper:

- dynamically reads the Robot Zoo manifest;
- statically resolves local cache paths without triggering upstream fetch;
- validates existing semantic map payloads with the verified-map schema gate;
- partitions available missing maps from unavailable missing maps;
- serializes a structured JSON-style report for test failure output and future artifacts.

Added `tests/v3/test_semantic_map_coverage_manifest.py`.

The tests:

- verify manifest-positive coverage is dynamic and structurally partitioned;
- verify committed maps are valid verified payloads;
- hard-fail when locally available positive humanoids lack verified maps;
- confirm unavailable positives are reported separately from the local available-source hard gate.

## Test Results

Command:

```bash
python -m pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Result: **1 failed, 3 passed**.

Expected failure:

- `test_available_positive_humanoids_have_verified_semantic_maps`
- fails because **24** locally available positive humanoids are missing verified semantic maps.

Command:

```bash
python -m pytest -q tests/v3/test_semantic_sites_verified.py
```

Result: **9 passed**.

Bare command note:

```bash
pytest -q tests/v3/test_semantic_map_coverage_manifest.py
```

Result: collection import error for `soma_retargeter` in this shell. `python -m pytest` preserves the repo root on `sys.path` and was used for verification.

## Risks

1. Step 2 verified-semantics gate remains blocked: only 1 of 25 locally available positive sources has a verified map.
2. The missing maps include key hard-gate robots: G1, H1, OP3, Booster T1, TALOS direct, and Berkeley direct.
3. URDF maps require topology/geometry verification from actual URDF model structure; static body-name selection is not enough.
4. Direct Menagerie availability currently depends on local cache inspection; keep resolver behavior aligned with this coverage helper to avoid inconsistent artifact status.
5. The worktree had unrelated modified/untracked files outside Agent H ownership during this pass; I left them untouched.

## Next Integration Step

Use the failing coverage test as the semantic-map hard gate. Add maps only after per-model topology/geometry verification can justify each required body/site/local pose, especially distal hand/sole/toe/heel anchors and Booster/TALOS trunk-foot semantics.
