# Retargeting v2 Implementation Report

## Current Commits

- `371113c Use virtual foot sites for root ground metadata`
- `cdbf65e Infer missing semantics from body names`
- `e9f5bd0 Infer distal sites from child anchors`
- `3f4421c Prefer explicit MJCF sites for v2 distal sites`
- `bb91825 Use mesh bounds for v2 virtual sites`
- `9496fa9 Validate compiled profile cache fingerprints`
- `bcbdfad Validate v2 left right chain symmetry`
- `75ab6a8 Strengthen compiled profile health validation`
- `7d52b6a Record root ground provenance in v2 profiles`
- `c48ef2e Add optional runtime collision barriers`
- `1c1ebc1 Add compile-time self collision metadata`
- `9de10ae Add v2 benchmark artifact reporting`
- `3e1b6c9 Add contact lock release state machine`
- `05a78a8 Add optional ground height barrier`
- `2201779 Guard priority zero joint limit residuals`
- `d53e5bb Apply v2 priority bands to runtime weights`
- `0fa2955 Add normalized temporal IK regularizers`
- `c9b437a Add v2 pole-vector IK tasks`
- `272fadd Add range-normalized joint limit barrier`
- `c363e3e Add v2 direction IK objectives`
- `96bbea3 Use segment-local scaler for v2 runtime profiles`
- `2156b71 Add segment-local v2 target builder`

## Implemented Design Items

- Dataclass profile model: `soma_retargeter/robotics/retarget_profile.py`
- MJCF morphology extraction: `soma_retargeter/robotics/morphology.py`
- Reachability projection helpers: `soma_retargeter/robotics/reachability.py`
- v2 task compiler: `soma_retargeter/robotics/task_compiler.py`
- Conservative body-name semantic inference for missing `ik_map` entries: `soma_retargeter/robotics/task_compiler.py`
- Autoconfig CLI: `soma_retargeter/tools/autoconfigure_robot.py`
- Runtime registry integration: `soma_retargeter/robot_registry_parser.py`
- Segment-local target builder: `soma_retargeter/robotics/human_to_robot_scaler.py`
- MJCF explicit sites, distal child anchors, and primitive/STL mesh bounds for distal virtual sites and collision proxies: `soma_retargeter/robotics/morphology.py` and `soma_retargeter/robotics/task_compiler.py`
- Direction, pole-vector, temporal, joint-limit, contact, and ground-barrier objectives: `soma_retargeter/pipelines/ik_objectives.py`
- v2 runtime wiring and priority guard: `soma_retargeter/pipelines/newton_pipeline.py`
- Compile-level and bounded runtime benchmark artifact CLI: `soma_retargeter/tools/benchmark_retargeting.py`
- Compile-level self-collision proxy and high-risk pair metadata: `soma_retargeter/robotics/morphology.py` and `soma_retargeter/robotics/task_compiler.py`
- Root-motion scale and ground-height provenance metadata from semantic hips and virtual foot site rest positions: `soma_retargeter/robotics/task_compiler.py`
- Compiled profile health-gate validation, including left/right chain length mismatch thresholds: `soma_retargeter/robot_registry_parser.py`
- Compiled profile cache fingerprint invalidation for robot morphology, source config, and compiler version: `soma_retargeter/robot_registry_parser.py`

## Tests

Latest full test command:

```bash
conda run -n soma-retargeter-v2 pytest -q
```

Latest verified result: `76 passed`. The benchmark CLI tests are in `tests/test_benchmark_retargeting.py`.

## Benchmark Summary

Generated command:

```bash
python -m soma_retargeter.tools.benchmark_retargeting --robots roboparty_rpo unitree_g1 --motions assets/motions/bvh --max-motions 2 --max-frames 5 --compare legacy v2 --output artifacts/retargeting_v2 --force
```

Current artifact scope includes compile-level validation plus bounded runtime rollouts for high-motion 5-frame windows from the first two resolved BVH motions. The benchmark now runs legacy and v2 compare modes through separate runtime configs: legacy uses the raw string `ik_map` with default position/rotation weights and no compiled profile, while v2 uses the compiled morphology-aware profile, direction/pole tasks, segment-local scaler, and priority guard. Runtime metrics include task residual by type/priority when profile tasks are evaluable, torso reachable/unreachable residuals for projected torso tasks, joint-limit margin, foot slide or structured unavailability, penetration, root tilt, hand/foot position RMSE, velocity/acceleration p95, solver iterations, fallback counts, and priority-guard diagnostics. `benchmark_summary.json` records `resolved_motions`, and per-motion payloads record `source_frame_start` plus frame-selection mode. `benchmark_frames.csv` includes the per-motion BVH path for each motion-level metric row. `benchmark_gates.json` records report-only legacy/v2 acceptance gates by default; `--strict-gates` returns exit code 4 on gate failure. `registry_coverage.json` records coverage for RPO, generic Unitree G1, G1 23DoF, G1 29DoF, E3 v2, and OLI.

RoboParty RPO currently compiles with schema v2, `xyzw` quaternion order, confidence `1.0`, enabled sparse tasks, mesh-derived hand/foot virtual sites, virtual-foot-site root/ground metadata, mesh-derived self-collision metadata, morphology site-count metadata, matching cache fingerprints, and separated legacy/v2 bounded runtime metrics with profile task residuals, torso leakage residuals, and zero measured v2 penetration in the bounded sample.

Unitree G1 currently produces a v2 artifact and separated legacy/v2 bounded runtime metrics. Generic `unitree_g1` now uses Newton's built-in 29DoF MJCF fallback when no local XML is registered, so its compiled profile has real morphology fingerprints, non-placeholder chain lengths, reachable chain ranks, and available torso residual metrics. The bounded sample still needs runtime acceptance work because v2 output remains static under full priority-guard rollback.

Registry coverage currently reports RoboParty RPO and generic `unitree_g1` as `ready`; `unitree_g1_23dof`, `unitree_g1_29dof`, `e3_v2`, and `oli` remain `missing_registration`.

Current bounded gate status is `failed`: RoboParty RPO and generic Unitree G1 pass the bounded tracking, penetration, and residual availability gates on the selected high-motion windows, but both still fail runtime-overhead gates. RPO v2 and Unitree G1 v2 both record full priority-guard rollback on the bounded run (`rollback_count=40` for 20 checked frames across two environments), explaining their static output.

## Remaining Work

- Extend the bounded runtime benchmark into uncapped full-length legacy/v2 comparison loops.
- Add G1 23DoF, G1 29DoF, E3 v2, and OLI registry assets when available.
- Add runtime benchmark coverage for optional collision barriers on real motions.
- Strengthen semantic auto-detection beyond conservative body-name matching.
- Turn report-only benchmark gates into required CI/acceptance gates after full-length motion coverage and morphology gaps are resolved.
- Produce full-motion before/after metrics for acceptance thresholds.
