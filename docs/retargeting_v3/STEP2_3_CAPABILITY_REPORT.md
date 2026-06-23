# Step 2.3 Capability Status Integration Report

Status date: 2026-06-23

Authoritative artifact root:

```text
artifacts/retargeting_v3_step2_capability/
```

## Summary

The Step 2.3 implementation adds terminal `capability_limited_passed`
classification for positive humanoids whose residuals are explained by
rank/limit/KKT certificates. Structured partial humanoids remain terminal
`partial_passed` only when required semantics are unavailable by expectation.

Counts from `summary.json`:

| Status | Count |
|---|---:|
| `passed` | 9 |
| `capability_limited_passed` | 23 |
| `partial_passed` | 3 |
| `negative_control_passed` | 9 |

Terminal pass total is 44/44:

- full humanoid profile passes: 9
- capability-limited humanoid passes: 23
- structured partial humanoid passes: 3
- negative-control load passes: 9
- deterministic compared/matched: 44/44

The structured partial terminal passes are:

- `berkeley_humanoid_mjcf_direct`
- `sigmaban_urdf`
- `simple_humanoid_urdf`

## Evidence

- `before_after.json` records `partial_passed->capability_limited_passed: 23`.
- `before_after.json` records `partial_passed->partial_passed: 3`.
- Every per-robot report has `task_certificate_summary`.
- `status_reason` is specific for terminal statuses and no longer falls back to the raw status string.
- `cross_format.gates.same_source_strict.status` remains `passed`.
- `cross_format.gates.variant_compatibility.status` is `passed` with 7 eligible positive humanoid pairs and 2 not-eligible pairs.
- Negative-control pairs (`cassie`, `unitree_go2`) are `not_eligible`, not humanoid equivalence failures.

## Notes

The capability artifact is freshly regenerated from the locked 44-model scope
with deterministic rerun enabled. Residuals over the exact global thresholds are
accepted only when the per-task certificate explains the limitation.
