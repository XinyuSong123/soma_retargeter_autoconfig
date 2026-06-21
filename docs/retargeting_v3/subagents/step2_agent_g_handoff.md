# Agent G Handoff - Robot Zoo Source Resolution

- Branch/commit:
  - Branch: `agent-b-xhigh-verified-semantics`
  - Code commit: `6e2332222b5164d9724537151ff78692bdd8faba` (`feat: resolve robot zoo sources from local caches`)
  - Handoff commit: this document commit
- Reasoning strength: `xhigh`
- Files:
  - `soma_retargeter/robotics/v3/robot_zoo.py`
  - `tests/v3/test_robot_zoo_source_resolution_local_cache.py`
  - `docs/retargeting_v3/subagents/step2_agent_g_handoff.md`
- Scope:
  - Non-network source resolution only.
  - Manifest-scoped `robot_descriptions` and MuJoCo Menagerie cache lookup.
  - No private model scans and no per-robot `robot_descriptions.*` imports in no-fetch mode.

## Reasoning Summary

- The existing no-fetch resolver returned `source_unavailable` for every `robot_descriptions` entry because importing those modules can call `_clone_to_cache(...)` and fetch upstream repositories.
- The safe path is to treat installed `robot_descriptions` module files as static metadata, parse simple `REPOSITORY_PATH`, `PACKAGE_PATH`, `URDF_PATH`, `MJCF_PATH`, and `XACRO_PATH` assignments with `ast`, and never execute module code.
- The resolver now only evaluates the manifest entry's declared `description_name` or Menagerie directory. It does not recursively search arbitrary local model directories.
- Menagerie direct entries now check the local `ROBOT_ZOO_CACHE`, repository cache, default `~/.cache/robot_descriptions/mujoco_menagerie`, and workspace cache. Within a manifest-declared Menagerie directory, non-`scene*.xml` files are preferred over scene files.
- The Newton cache was inspected. It contains public Unitree G1 assets, but I did not use it as a generic resolver source because the manifest entries are already resolved through `robot_descriptions`/Menagerie cache metadata and the manifest does not declare Newton cache sources for extra models.
- `source_unavailable` remains the status for declared cache files that are not present locally.

## Commands

```bash
PYTHONPATH=. pytest -q \
  tests/v3/test_robot_zoo_source_resolution_local_cache.py \
  tests/v3/test_full_robot_zoo_validation.py
```

Result: `11 passed in 4.10s`.

```bash
PYTHONPATH=. python -m soma_retargeter.tools.sync_robot_zoo_v3 \
  --dry-run \
  --output /tmp/robot_zoo_inventory_after_resolver.json
```

Result: completed without fetch/import side effects.

```bash
PYTHONPATH=. python -m soma_retargeter.tools.validate_kinematic_profile_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --output-dir /tmp/robot_zoo_validation_after_resolver \
  --low-discrepancy-count 1
```

Result: interrupted after several minutes with `KeyboardInterrupt` while Newton/trimesh was loading a large cached MJCF mesh. This did not produce a complete summary and did not modify project artifacts.

## Numerical Results

- Starting blocker from current artifacts/prompt: `passed=1`, `source_unavailable=45`.
- No-fetch source inventory after resolver commit:
  - `available=30`
  - `source_unavailable=16`
- Broad no-fetch validation:
  - incomplete; partial `/tmp/robot_zoo_validation_after_resolver` reports were written before interruption.
  - This is not counted as algorithm pass/fail evidence.

## Available Source Coverage

Resolved from local cache:

```text
adam_lite_mjcf
apptronik_apollo_mjcf
berkeley_humanoid_mjcf_direct
berkeley_humanoid_urdf
booster_t1_mjcf
booster_t1_urdf
cassie_mjcf
elf2_mjcf
elf2_urdf
ergocub_urdf
fourier_n1_mjcf
fourier_n1_urdf
franka_panda_mjcf
jvrc_mjcf
jvrc_urdf
pal_talos_mjcf_direct
roboparty_rpo_local
robotis_op3_mjcf
stretch3_mjcf
toddlerbot_2xc_mjcf
toddlerbot_2xm_mjcf
toddlerbot_urdf
unitree_g1_mjcf
unitree_g1_urdf
unitree_go2_mjcf
unitree_go2_urdf
unitree_h1_2_mjcf
unitree_h1_2_urdf
unitree_h1_mjcf
unitree_h1_urdf
```

Still `source_unavailable`:

```text
atlas_drc_urdf
atlas_v4_urdf
bolt_urdf
cassie_urdf
draco3_urdf
fourier_gr1_urdf
jaxon_urdf
mujoco_humanoid_mjcf
rhea_urdf
robonaut2_urdf
romeo_urdf
sigmaban_urdf
simple_humanoid_urdf
talos_urdf
upkie_urdf
valkyrie_urdf
```

## Assumptions

- `available` in source inventory means only that a fixed local source file was found. It is not an algorithm pass.
- Installed `robot_descriptions` package source files are trusted as pinned metadata from package version `2.0.0`; per-robot modules are not imported in no-fetch mode.
- Cache paths in reports are display-normalized with `${ROBOT_DESCRIPTIONS_CACHE}` or `${NEWTON_CACHE}` to avoid local absolute path leakage.
- Missing declared files stay unavailable rather than falling back to unrelated local files.

## Known Failures And Risks

- Full validation remains blocked by runtime load cost/failures and missing verified semantics for many newly resolved models.
- The static evaluator supports the simple `robot_descriptions` path pattern used by current manifest entries: `_clone_to_cache(...)` and `_path.join(...)`. More dynamic modules would need explicit static evaluator support.
- Worktree contained unrelated changes outside Agent G ownership during this work; they were not staged or modified by this commit sequence.

## Integration Order

1. Integrate `robot_zoo.py` resolver commit.
2. Re-run no-fetch inventory and confirm `available=30`, `source_unavailable=16`.
3. Let Agent E/F decide whether broad validation should run with mesh-loading limits, verified semantic maps, or per-model batching.
4. Preserve the 16 unresolved IDs as true local source misses until their declared fixed sources are added to cache or policy changes.
