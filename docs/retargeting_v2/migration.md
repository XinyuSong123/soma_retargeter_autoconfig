# Retargeting v2 Migration

## New Robot Setup

1. Register the robot MJCF/XML in `params.py`.
2. Provide a minimal semantic `ik_map` for pelvis, torso, hands, and feet.
3. Run:

```bash
python -m soma_retargeter.tools.autoconfigure_robot --robot <robot> --force
```

4. Inspect the generated compiled profile warnings and confidence. Use `--dry-run --write-report` to produce a sidecar `*.autoconfig_report.json` without writing the compiled profile.
5. Run:

```bash
python -m soma_retargeter.tools.benchmark_retargeting --robots <robot> --motions assets/motions/bvh --compare legacy v2 --output artifacts/retargeting_v2
```

The autoconfig CLI accepts `--benchmark --strict` as a one-command validation path; it forwards the configured seed to the benchmark and returns exit code `4` when strict benchmark gates fail.

## Legacy Compatibility

- String `ik_map` entries are still accepted and expanded by the registry.
- Legacy scaler files are still readable.
- v2 runtime uses `compiled_retarget_profile` when available.
- Pose-pair optimization is no longer the default onboarding path; use it only as bounded refinement after the compiled profile validates.

## Safety Checks

- Reject or warn on non-positive legacy scales before using data in v2.
- Review low-confidence semantic mappings.
- Inspect `priority_scheduler_diagnostics`.
- Keep `contact_aware_foot_ik` as the main foot-stability path and use post-processing only as fallback.
