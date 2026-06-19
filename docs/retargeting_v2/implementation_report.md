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

Latest verified result: `87 passed`. The benchmark CLI tests are in `tests/test_benchmark_retargeting.py`.

## Benchmark Summary

Generated command:

```bash
python -m soma_retargeter.tools.benchmark_retargeting --robots roboparty_rpo unitree_g1 --motions assets/motions/bvh --max-motions 2 --max-frames 80 --compare legacy v2 --output artifacts/retargeting_v2 --force
```

Current artifact scope includes compile-level validation plus bounded runtime rollouts for high-motion 80-frame windows from the first two resolved BVH motions. The benchmark now runs legacy and v2 compare modes through separate runtime configs: legacy uses the raw string `ik_map` with default position/rotation weights and no compiled profile, while v2 uses the compiled morphology-aware profile, direction/pole tasks, segment-local scaler, priority guard, an 8-iteration compiled-profile IK default, range-normalized actuated joint motion limiter, and compiled semantic-site offsets for position IK targets when the offset is reachable enough to be used as a solver target. Runtime metrics include task residual by type/priority when profile tasks are evaluable, torso reachable/unreachable residuals for projected torso tasks, joint-limit margin, foot slide or structured unavailability, penetration, root tilt, hand/foot position RMSE, per-axis tracking diagnostics (`axis_rmse`, `mean_error`, `p95_abs_error`), actuated velocity/acceleration p95, root velocity/acceleration diagnostics, solver iterations, fallback counts, and priority-guard diagnostics. Runtime profiling now records BVH load, pipeline construction, target setup, solve, metric evaluation, emitted frame count, and solve FPS separately. Tracking and profile residual metrics align target frames with the emitted runtime buffer by removing initialization and stabilization target frames before comparison. `benchmark_summary.json` records `resolved_motions`, and per-motion payloads record `source_frame_start` plus frame-selection mode. `benchmark_frames.csv` includes the per-motion BVH path for each motion-level metric row. `benchmark_gates.json` records report-only legacy/v2 acceptance gates by default; `--strict-gates` returns exit code 4 on gate failure. `registry_coverage.json` records coverage for RPO, generic Unitree G1, G1 23DoF, G1 29DoF, E3 v2, and OLI.

RoboParty RPO currently compiles with schema v2, `xyzw` quaternion order, confidence `1.0`, enabled sparse tasks, mesh-derived hand/foot virtual sites, virtual-foot-site root/ground metadata, mesh-derived self-collision metadata, morphology site-count metadata, matching cache fingerprints, and separated legacy/v2 bounded runtime metrics with profile task residuals, torso leakage residuals, and zero measured v2 penetration in the bounded sample.

Unitree G1 currently produces a v2 artifact and separated legacy/v2 bounded runtime metrics. Generic `unitree_g1` now uses Newton's built-in 29DoF MJCF fallback when no local XML is registered, so its compiled profile has real morphology fingerprints, non-placeholder chain lengths, reachable chain ranks, and available torso residual metrics. The bounded sample still needs runtime acceptance work because v2 output now moves but overshoots tracking gates.

Registry coverage currently reports RoboParty RPO and generic `unitree_g1` as `ready`; `unitree_g1_23dof`, `unitree_g1_29dof`, `e3_v2`, and `oli` remain `missing_registration`.

Current bounded gate status is `failed`: RoboParty RPO and generic Unitree G1 pass penetration, torso leakage, and actuated smoothness gates on the selected high-motion windows, and both record zero priority-guard rollbacks while protecting hard joint-limit penetration. The hard-penetration guard uses the CUDA graph-capture fast path, and the compiled-profile IK default uses 8 solver iterations instead of 24 to bound overfitting and runtime. The range-normalized joint motion limiter brings RPO actuated `velocity_p95/acceleration_p95` to `12.68/436.19` versus legacy `28.91/1626.89`, and Unitree G1 to `33.42/788.35` versus legacy `56.51/6388.52`. With reachable semantic-site offsets applied to position IK, Unitree G1 hand/foot RMSE now passes at `0.51/0.28` versus legacy `0.55/0.45`; its per-axis hand/foot RMSE diagnostics are `[0.33, 0.27, 0.27]` and `[0.17, 0.10, 0.18]`. RPO foot RMSE improves to `0.48` from the previous `0.80`, but RPO hand remains failed at `0.88` versus legacy `0.22` because its hand semantic site is an elbow-link distal extrapolation rather than a wrist-supported target. RPO hand error is dominated by the y/z axes (`axis_rmse [0.23, 0.72, 0.41]`, `mean_error [0.09, -0.60, 0.28]`), while RPO foot residuals are more evenly distributed (`axis_rmse [0.28, 0.25, 0.27]`). Unitree G1 now fails root tilt (`1.86` versus legacy `1.60`) and runtime, while RPO still fails hand/foot tracking and runtime. Runtime profiling shows the remaining overhead is solver dominated rather than BVH load or metric evaluation: RPO v2 solves at about `22.21` output frames/s versus legacy `193.06`, and Unitree G1 v2 solves at about `20.12` output frames/s versus legacy `175.72`; target setup contributes about `0.24s` per bounded two-motion run. The next acceptance work is RPO reach-limited hand tracking, Unitree root-tilt control after site-offset tracking improvements, and solver-kernel/runtime reduction rather than static-output rollback.

## Remaining Work

- Extend the bounded runtime benchmark into uncapped full-length legacy/v2 comparison loops.
- Add G1 23DoF, G1 29DoF, E3 v2, and OLI registry assets when available.
- Add runtime benchmark coverage for optional collision barriers on real motions.
- Strengthen semantic auto-detection beyond conservative body-name matching.
- Turn report-only benchmark gates into required CI/acceptance gates after full-length motion coverage and morphology gaps are resolved.
- Produce full-motion before/after metrics for acceptance thresholds.
