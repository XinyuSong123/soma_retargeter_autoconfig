# Step 2 Agent B Handoff — Verified Semantics & Distal Geometry Sites

Branch/worktree:

- Branch: `agent-b-verified-semantics`
- Worktree: `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_b`

Reasoning-effort limitation:

- Integrator updated `goal.md` after this Agent B instance was spawned to require `xhigh` reasoning for all subagents.
- This Agent B instance was already running with high effort, and tool/session parameters cannot be changed in place.
- Work continued with highest available diligence; any replacement or extra spawned agent should use `xhigh`.

Commits:

- `b449eb2 fix(v3): verify semantic site provenance`

Owned files changed:

- `soma_retargeter/robotics/v3/semantic_sites.py`
- `soma_retargeter/robotics/v3/site_geometry.py`
- `soma_retargeter/robotics/v3/semantic_validation.py`
- `assets/robot_zoo/semantic_maps/roboparty_rpo_local.json`
- `assets/robot_zoo/semantic_expectations/roboparty_rpo_local.json`
- `tests/v3/test_semantic_sites_verified.py`
- `tests/v3/test_site_geometry_helpers.py`

Summary:

- Replaced the RPO default semantic map with an asset-backed verified map.
- Added explicit source/confidence handling in `build_semantic_sites()`.
- Inferred body-name semantics now emit `source="inferred_body_name"` and confidence below 1.0.
- Removed the fake topology fallback that used body-name string length as depth when no `world` body exists.
- Added a nonzero-origin gate for distal sites when requested by validation.
- Added geometry helpers for distal hand sites and foot sole/toe/heel sites from compiled body geometry bounds.
- Added semantic validation helpers for inferred-confidence, verified-map evidence, and distal-origin checks.

Numeric results:

- Focused Agent B tests: `8 passed in 0.80s`.
- Existing v3 regression subset: `48 passed in 104.42s`.
- RPO verified distal offsets now loaded by `default_rpo_semantic_map()`:
  - `LeftHand`: `[0.0958936956, -0.0000755965, -0.0000229437]`
  - `RightHand`: `[0.0958936956, 0.0000755183, 0.0000228174]`
  - `LeftFoot` sole: `[0.01693716, 0.0025669, -0.11837637]`
  - `RightFoot` sole: `[0.01697911, -0.00251041, -0.1183174]`
  - toe/heel sites are present for both feet and pass nonzero local-position checks.
- Verified/confidence policy:
  - verified map confidence is capped at `0.99`;
  - inferred hips/chest confidence is `0.7`;
  - inferred limb confidence is `0.6`;
  - geometry-bound distal sites are `0.85`.

Commands run:

```bash
PYTHONPATH=. pytest -q tests/v3/test_semantic_sites_verified.py tests/v3/test_site_geometry_helpers.py
PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3.py tests/test_kinematic_profile_v3_math.py tests/test_kinematic_profile_v3_adapter.py tests/test_kinematic_profile_v3_validation_artifacts.py
```

Assumptions:

- Agent B does not own `model_adapter.py`, so evidence is represented through existing `SemanticSite` fields: `source`, `confidence`, and `reason`, plus verified-map JSON entries.
- MuJoCo compiled body geometry is available for `site_geometry.py`; Newton-only models without geometry exposure must use verified semantic-map entries or another adapter-provided geometry source.
- RPO hand distal direction and foot toe direction use positive local X from compiled bounds; this is recorded as geometry provenance, not robot-name math.

Failures / risks:

- Full positive-humanoid verified maps for all manifest entries are not complete in this commit; only RPO is added as an asset-backed verified map.
- `validation.py` still writes inferred semantic maps for non-RPO description models. Integrator/Agent E should switch validation dispatch to require/load verified maps before claiming positive humanoid pass.
- Mesh geometry bounds use compiled MuJoCo `geom_size` conservatively. This is sufficient for the focused gates and RPO offsets, but exact visual mesh bounds/hash lineage are still future integration work.
- The nonzero distal gate is opt-in through `require_distal_site_offsets=True`; Agent E/F should enable it for positive humanoid validation.

Integration notes:

- Merge this after Agent A's adapter interfaces are stable; `site_geometry.py` uses only current adapter methods plus MuJoCo compiled geometry fields.
- Agent E should read `assets/robot_zoo/semantic_maps/<model_id>.json` before falling back to inference.
- Agent F should add an acceptance gate that rejects positive humanoid reports with `source="inferred_body_name"` for required Hand/Foot semantics.
- If Agent A exposes compiled Newton geometry/sites, wire those into `site_geometry.py` rather than parsing raw XML.
