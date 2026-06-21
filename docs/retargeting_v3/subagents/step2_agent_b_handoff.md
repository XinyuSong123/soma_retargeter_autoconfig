# Step 2 Agent B-xhigh Handoff - Verified Semantics & Distal Geometry Sites

Branch/worktree:

- Branch: `agent-b-xhigh-verified-semantics`
- Worktree: `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig`
- Reasoning: this Agent B run used xhigh reasoning as required by `goal.md`.

Context read:

- `goal.md`
- `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
- `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
- `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
- `docs/retargeting_v3/CODE_AUDIT.md`
- `docs/retargeting_v3/known_limitations.md`

Prior Agent B input reviewed:

- `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_b` commit `b449eb2 fix(v3): verify semantic site provenance`
- `/mnt/ssd1/song/Desktop/soma_retargeter_autoconfig_agent_b` commit `c2551e2 docs(v3): add agent b semantic handoff`
- The old patch was used as input only. This xhigh run rechecked the geometry and corrected the RPO offsets to account for compiled `geom_quat`; the older handoff's non-xhigh limitation does not apply to this run.

Commits:

- `6d4488c fix(v3): verify semantic provenance and distal sites`

Owned files changed:

- `soma_retargeter/robotics/v3/semantic_sites.py`
- `soma_retargeter/robotics/v3/site_geometry.py`
- `soma_retargeter/robotics/v3/semantic_validation.py`
- `assets/robot_zoo/semantic_maps/roboparty_rpo_local.json`
- `assets/robot_zoo/semantic_expectations/roboparty_rpo_local.json`
- `tests/v3/test_semantic_sites_verified.py`
- `tests/v3/test_site_geometry_helpers.py`
- `docs/retargeting_v3/subagents/step2_agent_b_handoff.md`

Summary:

- `build_semantic_sites()` no longer rewrites all semantics to `explicit_semantic_override` with confidence `1.0`.
- Configured entries use capped provenance (`configured_body`, `configured_site`, `configured_model_site`, `configured_distal_offset`).
- Inferred body-name semantics now emit structured entries with `source="inferred_body_name"`, evidence, and confidence below `1.0`.
- Removed the string-length depth fallback from body-name inference; when topology depth is unavailable, the first matching candidate is returned without pretending name length is tree depth.
- Added an opt-in nonzero-origin gate for distal hand/foot/toe/heel sites.
- Added compiled-geometry helpers for body-local distal hand and sole/toe/heel sites. Mesh geoms prefer compiled mesh vertices and apply `geom_quat`.
- Added semantic validation hooks for fake explicit evidence, distal-origin checks, verified-map payloads, and partial/negative-control classification.
- Added an asset-backed verified RPO map with model fingerprint, evidence, core hand/sole offsets, and auxiliary toe/heel anchors.

Numeric results:

- RPO verified map binds `model_fingerprint`: `b6da329ae3ce0dd41fe155bfb44389875e61f951cb43ca3fab9376791dd83d69`.
- RPO core distal local positions:
  - `LeftHand`: `[0.1382499970094016, -1.302929279918541e-07, -2.3430852050254636e-10]`
  - `RightHand`: `[0.13824999699495474, 1.3112782564980718e-07, 8.738377729744506e-11]`
  - `LeftFoot` sole: `[0.024999998077943168, 0.009425001263645708, -0.04500000282002089]`
  - `RightFoot` sole: `[0.025000002393491115, -0.009425001304015428, -0.04500000287209733]`
- RPO auxiliary foot anchors:
  - `LeftToe`: `[0.10400000044240255, 0.009425001263645708, -0.04500000282002089]`
  - `LeftHeel`: `[-0.05400000428651622, 0.009425001263645708, -0.04500000282002089]`
  - `RightToe`: `[0.10400000537286047, -0.009425001304015428, -0.04500000287209733]`
  - `RightHeel`: `[-0.05400000058587824, -0.009425001304015428, -0.04500000287209733]`
- Confidence policy:
  - verified map maximum: `0.99`
  - inferred hips/chest: `0.7`
  - inferred limbs: `0.6`
  - geometry-bound distal sites: `0.85`

Commands/tests:

```bash
PYTHONPATH=. pytest -q tests/v3/test_semantic_sites_verified.py tests/v3/test_site_geometry_helpers.py
# 13 passed in 1.31s

PYTHONPATH=. pytest -q tests/v3/test_model_adapter_runtime_truth.py tests/v3/test_cross_format_adapter_scaffolding.py tests/v3/test_semantic_sites_verified.py tests/v3/test_site_geometry_helpers.py
# 21 passed in 5.13s

python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from soma_retargeter.robotics.v3.validation import write_validation_artifacts
with TemporaryDirectory() as d:
    summary = write_validation_artifacts(Path(d) / "artifacts", low_discrepancy_count=1)
    print(summary["status_counts"])
    print(summary["reports"]["roboparty_rpo_local"])
PY
# status_counts {'passed': 1, 'source_unavailable': 45}
# roboparty_rpo_local status passed, failures []

PYTHONPATH=. pytest -q tests/test_kinematic_profile_v3.py tests/test_kinematic_profile_v3_math.py tests/test_kinematic_profile_v3_adapter.py tests/test_kinematic_profile_v3_validation_artifacts.py
# 43 passed, 1 failed, 4 errors in 39.34s
# Remaining failures are stale tests expecting removed summary["compiled_count"] in schema v4 validation artifacts.

PYTHONPATH=. pytest -q tests/v3
# 42 passed, 1 failed in 5.80s before this handoff existed.
# Remaining failure is Agent F acceptance audit over known Step-2 blockers outside Agent B scope.
```

Assumptions:

- Agent B does not own `model_adapter.py`, so runtime `SemanticSite.to_json()` still exposes source/confidence/reason but not a full evidence list. Evidence is preserved in verified-map JSON and validation helpers.
- Agent B does not own Agent C rest calibration. Therefore RPO toe/heel anchors are marked `auxiliary_semantics`; `load_semantic_map()` returns core profile semantics by default and requires `include_auxiliary=True` when callers are ready to consume toe/heel anchors.
- Newton-only compiled geometry is not exposed through a backend-neutral adapter API in this branch. `site_geometry.py` currently uses MuJoCo-style compiled geometry fields and raises a structured error otherwise.

Failures / risks:

- Full verified semantic maps for every positive humanoid are not complete in this commit. RPO is the only asset-backed verified map added here.
- `validation.py` still falls back to inferred body-name maps for non-RPO entries. Agent E/F must reject positive-humanoid pass claims that rely on inferred required semantics.
- `tests/v3/test_acceptance_gates_audit.py` still fails on unrelated Step-2 blockers: canonical projection artifacts, G1 equivalence, dirty/stale artifacts, and missing other subagent/red-team handoffs.
- Core source still has the legacy public function `default_rpo_semantic_map()`. It now loads an asset-backed map rather than hard-coded offsets, but Agent F's broad string scan may still flag the function name as a robot-specific default until validation dispatch is fully generalized.

Integration notes:

- Integrate after Agent A's adapter truth work; this commit only requires existing body topology and MuJoCo compiled geometry fields.
- Agent C should either ignore `auxiliary_semantics` during rest calibration or add explicit source rest frames for toe/heel anchors before enabling `include_auxiliary=True` in profile compilation.
- Agent E should load `assets/robot_zoo/semantic_maps/<model_id>.json` for positive humanoids and enable the distal nonzero gate for required hand/foot sites.
- Agent F should audit positive humanoid reports for `source="inferred_body_name"` on required semantics and for zero local offsets on distal hand/sole/toe/heel anchors.
