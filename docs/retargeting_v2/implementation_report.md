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
- `4224c20 Add tracking axis diagnostics`
- `36f97e3 Schedule v2 task weights by morphology`
- `6802c5c Weaken single-axis torso tracking`
- `d27828e Schedule v2 IK iterations by torso rank`
- `62323ed Use analytic direction Jacobians for multi-axis torso`
- `df44a5c Register Unitree G1 29DoF coverage alias`
- `1ab974c Gate analytic pole-vector Jacobians`
- `0421764 Validate pole-vector analytic Jacobian sign`
- `f84a787 Persist benchmark gate failure artifacts`
- `99b45c9 Report onboarding templates for missing robots`
- `1848a9f Complete autoconfig CLI reporting`
- `e42246a Report per-semantic tracking diagnostics`
- `248a1de Register Unitree G1 23DoF coverage`

## Implemented Design Items

- Dataclass profile model: `soma_retargeter/robotics/retarget_profile.py`
- MJCF morphology extraction: `soma_retargeter/robotics/morphology.py`
- Reachability projection helpers: `soma_retargeter/robotics/reachability.py`
- v2 task compiler: `soma_retargeter/robotics/task_compiler.py`
- Conservative body-name semantic inference for missing `ik_map` entries: `soma_retargeter/robotics/task_compiler.py`
- Autoconfig CLI with dry-run, report, seed, strict validation, and optional strict benchmark return-code handling: `soma_retargeter/tools/autoconfigure_robot.py`
- Runtime registry integration: `soma_retargeter/robot_registry_parser.py`
- Segment-local target builder: `soma_retargeter/robotics/human_to_robot_scaler.py`
- MJCF explicit sites, distal child anchors, and primitive/STL mesh bounds for distal virtual sites and collision proxies: `soma_retargeter/robotics/morphology.py` and `soma_retargeter/robotics/task_compiler.py`
- Direction, pole-vector, temporal, joint-limit, contact, and ground-barrier objectives: `soma_retargeter/pipelines/ik_objectives.py`
- v2 runtime wiring and priority guard: `soma_retargeter/pipelines/newton_pipeline.py`
- Compile-level and bounded runtime benchmark artifact CLI, including gate-failure JSON summaries: `soma_retargeter/tools/benchmark_retargeting.py`
- Compile-level self-collision proxy and high-risk pair metadata: `soma_retargeter/robotics/morphology.py` and `soma_retargeter/robotics/task_compiler.py`
- Root-motion scale and ground-height provenance metadata from semantic hips and virtual foot site rest positions: `soma_retargeter/robotics/task_compiler.py`
- Compiled profile health-gate validation, including left/right chain length mismatch thresholds: `soma_retargeter/robot_registry_parser.py`
- Compiled profile cache fingerprint invalidation for robot morphology, source config, and compiler version: `soma_retargeter/robot_registry_parser.py`
- Morphology-scheduled direction-objective analytic Jacobians and gated pole-vector analytic Jacobian support: `soma_retargeter/pipelines/ik_objectives.py`, `soma_retargeter/pipelines/newton_pipeline.py`, and `soma_retargeter/robot_registry_parser.py`
- Cache protection that preserves a valid compiled profile when a later regeneration attempt only produces incomplete missing-MJCF morphology: `soma_retargeter/robot_registry_parser.py`
- Runtime solver-objective complexity diagnostics in benchmark and gate-failure artifacts: `soma_retargeter/pipelines/newton_pipeline.py` and `soma_retargeter/tools/benchmark_retargeting.py`

## Tests

Latest full test command:

```bash
conda run -n soma-retargeter-v2 pytest -q
```

Latest verified result: `93 passed, 6 subtests passed`. The benchmark CLI tests are in `tests/test_benchmark_retargeting.py`.

## Benchmark Summary

Generated command:

```bash
python -m soma_retargeter.tools.benchmark_retargeting --robots roboparty_rpo unitree_g1 --motions assets/motions/bvh --max-motions 2 --max-frames 80 --compare legacy v2 v2_pole_analytic --output artifacts/retargeting_v2 --force
```

Current artifact scope includes compile-level validation plus bounded runtime rollouts for high-motion 80-frame windows from the first two resolved BVH motions. The benchmark now runs legacy and v2 compare modes through separate runtime configs: legacy uses the raw string `ik_map` with default position/rotation weights and no compiled profile, while v2 uses the compiled morphology-aware profile, direction/pole tasks, segment-local scaler, priority guard, an 8-iteration compiled-profile IK default, range-normalized actuated joint motion limiter, compiled semantic-site offsets for position IK targets when the offset is reachable enough to be used as a solver target, and task-normalized weights from the compiler instead of only priority-band weights. The optional `v2_pole_analytic` compare mode starts from v2 and forces pole-vector tasks onto the analytic Jacobian path for reproducible runtime/tracking tradeoff analysis without changing default v2 gates. Runtime metrics include task residual by type/priority when profile tasks are evaluable, torso reachable/unreachable residuals for projected torso tasks, joint-limit margin, foot slide or structured unavailability, penetration, root tilt, hand/foot position RMSE, per-axis tracking diagnostics (`axis_rmse`, `mean_error`, `p95_abs_error`), per-semantic hand/foot tracking breakdowns, actuated velocity/acceleration p95, root velocity/acceleration diagnostics, solver iterations, fallback counts, priority-guard diagnostics, and solver-objective complexity diagnostics. Runtime profiling now records BVH load, pipeline construction, target setup, solve, metric evaluation, emitted frame count, and solve FPS separately. `motion_benchmark.solver_objectives` records active objective counts, sparse residual dimension, analytic/autodiff direction and pole-vector counts, autodiff sparse residual dimension, IK iterations, batch size, and Jacobian mode; gate-failure payloads mirror these under `compare_solver_objectives`. Tracking and profile residual metrics align target frames with the emitted runtime buffer by removing initialization and stabilization target frames before comparison. `benchmark_summary.json` records `resolved_motions`, and per-motion payloads record `source_frame_start` plus frame-selection mode. `benchmark_frames.csv` includes the per-motion BVH path for each motion-level metric row. `benchmark_gates.json` records report-only legacy/v2 acceptance gates by default; failed gates now also write `failures/<robot>_gates.json` with failed gates, compare metrics, solver-objective diagnostics, profile metadata, warnings, and the reproduction command. `--strict-gates` returns exit code 4 on legacy/v2 gate failure. `registry_coverage.json` records coverage for RPO, generic Unitree G1, G1 23DoF, G1 29DoF, E3 v2, and OLI.

RoboParty RPO currently compiles with schema v2, `xyzw` quaternion order, confidence `1.0`, enabled sparse tasks, mesh-derived hand/foot virtual sites, virtual-foot-site root/ground metadata, mesh-derived self-collision metadata, morphology site-count metadata, matching cache fingerprints, and separated legacy/v2 bounded runtime metrics with profile task residuals, torso leakage residuals, and zero measured v2 penetration in the bounded sample. Single-axis projected torso rotation is treated as a weak style target instead of a hard tracking target so the RPO torso constraint does not dominate reach-limited hand and foot tracking.

Unitree G1 currently produces a v2 artifact and separated legacy/v2 bounded runtime metrics. Generic `unitree_g1` now uses Newton's built-in 29DoF MJCF fallback when no local XML is registered, so its compiled profile has real morphology fingerprints, non-placeholder chain lengths, reachable chain ranks, and available torso residual metrics. Multi-axis torso profiles now soften projected torso rotation and strengthen foot position tasks; this keeps Unitree hand/foot tracking passing while bringing root tilt back under the legacy gate.

Registry coverage currently reports RoboParty RPO, generic `unitree_g1`, explicit `unitree_g1_23dof`, and explicit `unitree_g1_29dof` targets as `ready`. `unitree_g1_23dof` compiles against Newton's built-in 23DoF MJCF fallback with a separate semantic map and generated profile, while `unitree_g1_29dof` resolves to the generic Unitree G1 registration because that path uses Newton's built-in 29DoF MJCF fallback. `e3_v2` and `oli` remain `missing_registration`; each missing target includes an onboarding report with required asset/config slots, `params.py` entries, a minimal semantic `ik_map` template, and validation/benchmark commands. This records the OLI fallback path without fabricating or downloading assets.

Current bounded gate status is `failed`: RoboParty RPO and generic Unitree G1 pass penetration, torso leakage, and actuated smoothness gates on the selected high-motion windows, and both record zero priority-guard rollbacks while protecting hard joint-limit penetration. The hard-penetration guard uses the CUDA graph-capture fast path. Compiled-profile runtime uses morphology-scheduled IK iteration defaults: single-axis torso profiles use 4 iterations, while multi-axis torso profiles keep 8 iterations to preserve Unitree tracking gates. Direction objectives now use an analytic Jacobian only for multi-axis torso profiles; single-axis torso profiles keep the autodiff direction objective because the analytic path changed bounded RPO tracking. Pole-vector objectives have an experimental analytic Jacobian implementation with expanded finite-difference validation, but compiled runtime configs keep it disabled until benchmark-safe activation because direct enablement regressed bounded hand/foot/root tracking. The range-normalized joint motion limiter brings RPO actuated `velocity_p95/acceleration_p95` to `21.90/554.54` versus legacy `28.91/1626.89`, and Unitree G1 to `36.89/863.50` versus legacy `56.51/6388.52`. With reachable semantic-site offsets and morphology-scheduled weights applied to position IK, Unitree G1 hand/foot RMSE passes at `0.57/0.44` versus legacy `0.55/0.45`, and root tilt now passes at `0.89` versus legacy `1.60`. Weak single-axis torso tracking plus the 4-iteration single-axis runtime default keeps RPO hand/foot RMSE at `0.40/0.43` versus legacy `0.22/0.18`, and keeps RPO root tilt below legacy at `1.57` versus `1.67`, but RPO hand/foot tracking still fails the report-only gates. RPO hand residuals remain relatively balanced (`axis_rmse [0.21, 0.21, 0.24]`, `mean_error [-0.10, -0.06, -0.03]`), while RPO foot residuals remain elevated, especially y/z (`axis_rmse [0.18, 0.26, 0.25]`). Unitree G1 now fails only runtime, while RPO still fails hand/foot tracking and runtime. These failures are saved in `failures/roboparty_rpo_gates.json` and `failures/unitree_g1_gates.json`. Runtime profiling shows the remaining overhead is solver dominated rather than BVH load or metric evaluation: RPO v2 runs in `mixed` Jacobian mode at about `40.46` output frames/s versus legacy `analytic` mode at `178.78`, and Unitree G1 v2 runs in `mixed` mode at about `35.88` output frames/s versus legacy `analytic` mode at `165.60`. Solver-objective diagnostics show v2 has fewer active objectives and lower sparse residual dimension than legacy (`24` objectives and `66` sparse residual dimensions versus legacy `30` and `84`), but still pays autodiff cost from sparse direction and pole-vector residuals: RPO v2 records `48` autodiff sparse residual dimensions from autodiff direction plus pole-vector tasks, and Unitree G1 records `24` from autodiff pole-vector tasks after multi-axis direction tasks use analytic Jacobians. The `v2_pole_analytic` diagnostic confirms why pole activation is not yet safe: RPO improves to about `73.91` output frames/s but root tilt regresses to `1.85` and hand/foot RMSE worsens to `0.44/0.45`; Unitree G1 reaches pure `analytic` mode with about `193.58` output frames/s and runtime `1.09s`, but hand/foot RMSE worsens to `0.80/0.58` and penetration becomes nonzero at `5.5e-7m`. The runtime Jacobian scheduler now selects pure `analytic` mode whenever every active objective supports analytic Jacobians, so future benchmark-safe pole-vector activation will not remain on a mixed/autodiff tape path. The next acceptance work is RPO reach-limited hand/foot tracking and solver-kernel/runtime reduction, especially replacing or damping pole-vector analytic activation so it preserves bounded tracking/root gates.

## Remaining Work

- Extend the bounded runtime benchmark into uncapped full-length legacy/v2 comparison loops.
- Add G1 23DoF, E3 v2, and OLI registry assets when available.
- Add runtime benchmark coverage for optional collision barriers on real motions.
- Strengthen semantic auto-detection beyond conservative body-name matching.
- Turn report-only benchmark gates into required CI/acceptance gates after full-length motion coverage and morphology gaps are resolved.
- Produce full-motion before/after metrics for acceptance thresholds.
