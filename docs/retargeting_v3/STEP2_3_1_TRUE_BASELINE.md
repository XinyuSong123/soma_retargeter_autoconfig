# Step 2.3.1 True Baseline

Status: BLOCKED on current after artifacts, with true baseline extraction in place.

Agent A replaces the legacy fabricated before status with a pinned git-object
baseline from:

```text
5ad5a001c445c525d4c8bbaf6339dec5c5c2c719
```

The baseline source root is:

```text
artifacts/retargeting_v3_step2_assets44
```

## Generated Artifacts

- `artifacts/retargeting_v3_step2_capability/baseline_summary.json`
- `artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json`
- `artifacts/retargeting_v3_step2_capability/before_after.json`

The baseline summary and failure ledger use `source_access=git_object`.
Missing commits or missing baseline paths raise `BaselineGitObjectError`; the
builders do not fall back to the working tree.

## Baseline Counts

The immutable baseline counts are:

| Status | Count |
| --- | ---: |
| `passed` | 21 |
| `partial_passed` | 3 |
| `negative_control_passed` | 9 |
| `algorithm_failed` | 11 |

The 11 baseline algorithm failures are:

```text
atlas_drc_urdf
atlas_v4_urdf
booster_t1_mjcf
booster_t1_urdf
fourier_n1_mjcf
jaxon_urdf
mujoco_humanoid_mjcf
pal_talos_mjcf_direct
robotis_op3_mjcf
talos_urdf
valkyrie_urdf
```

## Current Before/After Truth

The current after summary still reports:

```text
passed                       9
capability_limited_passed   23
partial_passed               3
negative_control_passed      9
```

True baseline transitions are:

| Transition | Count |
| --- | ---: |
| `algorithm_failed->capability_limited_passed` | 10 |
| `algorithm_failed->passed` | 1 |
| `negative_control_passed->negative_control_passed` | 9 |
| `partial_passed->partial_passed` | 3 |
| `passed->capability_limited_passed` | 13 |
| `passed->passed` | 8 |

The 13 `passed->capability_limited_passed` rows are illegal baseline
transitions. Therefore `before_after.json` records:

```text
transition_validation.status = failed
final_count_validation.status = failed
```

This is intentional fail-closed evidence. The previous fabricated matrix hid
these rows by reporting baseline `passed` models as `partial_passed`.

## Commands

Regenerate the failure ledger:

```bash
PYTHONPATH=. python scripts/extract_capability_failure_ledger.py \
  --output artifacts/retargeting_v3_step2_capability/baseline_failure_ledger.json
```

Regenerate the baseline summary and before/after matrix:

```bash
PYTHONPATH=. python scripts/build_capability_before_after.py \
  --baseline-output artifacts/retargeting_v3_step2_capability/baseline_summary.json \
  --output artifacts/retargeting_v3_step2_capability/before_after.json
```

Check committed artifacts:

```bash
PYTHONPATH=. python scripts/extract_capability_failure_ledger.py --check
PYTHONPATH=. python scripts/build_capability_before_after.py --check
```

Use strict transition mode when the integration target is expected to be fully
accepted:

```bash
PYTHONPATH=. python scripts/build_capability_before_after.py --check --strict-transitions
```

With the current after artifacts, strict mode returns nonzero because the true
transition matrix is blocked.
