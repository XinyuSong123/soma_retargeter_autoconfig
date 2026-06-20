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
- Single-axis torso/low-rank no-wrist endpoint weight schedule: `soma_retargeter/robotics/task_compiler.py`

## Tests

Latest full test command:

```bash
conda run -n soma-retargeter-v2 pytest -q
```

Latest verified result: `110 passed, 6 subtests passed`. The benchmark CLI tests are in `tests/test_benchmark_retargeting.py`.

## Benchmark Summary

Generated command:

```bash
python -m soma_retargeter.tools.benchmark_retargeting --robots roboparty_rpo unitree_g1 e3_v2 --motions assets/motions/bvh --max-motions 2 --max-frames 80 --compare legacy v2 v2_pole_analytic --output artifacts/retargeting_v2 --force
```

Current artifact scope includes compile-level validation plus bounded runtime rollouts for high-motion 80-frame windows from the first two resolved BVH motions. The benchmark now runs legacy and v2 compare modes through separate runtime configs: legacy uses the raw string `ik_map` with default position/rotation weights and no compiled profile, while v2 uses the compiled morphology-aware profile, direction/pole tasks, segment-local scaler, priority guard, an 8-iteration compiled-profile IK default, range-normalized actuated joint motion limiter, compiled semantic-site offsets for position IK targets when the offset is reachable enough to be used as a solver target, and task-normalized weights from the compiler instead of only priority-band weights. Diagnostic compare modes start from v2 without changing default gates: `v2_pole_analytic` forces pole-vector tasks onto the analytic Jacobian path, `v2_pole_tangent_analytic` replaces each three-row pole normal residual with a two-row tangent-space analytic residual, `v2_pole_analytic_w<scale>` scales pole weights, `v2_pole_analytic_w<scale>_hand_w<weight>` combines analytic pole-vector scaling with hand position weight overrides, `v2_no_pole` removes pole-vector tasks, `v2_pos_projected` projects position objectives onto compiled translational bases, `v2_pole_keep_<selector>` keeps selected pole-vector semantic groups for runtime/tracking attribution, `v2_iter<N>` overrides IK iterations, and `v2_hand_w<weight>` overrides hand position weights for reach-limited hand sensitivity checks. Runtime metrics include task residual by type/priority when profile tasks are evaluable, torso reachable/unreachable residuals for projected torso tasks, joint-limit margin, foot slide or structured unavailability, penetration, root tilt, hand/foot position RMSE, per-axis tracking diagnostics (`axis_rmse`, `mean_error`, `p95_abs_error`), per-semantic hand/foot tracking breakdowns, actuated velocity/acceleration p95, root velocity/acceleration diagnostics, solver iterations, fallback counts, priority-guard diagnostics, contact-score source/coverage diagnostics, and solver-objective complexity diagnostics. Runtime profiling now records BVH load, pipeline construction, target setup, solve, metric evaluation, emitted frame count, and solve FPS separately. `motion_benchmark.solver_objectives` records active objective counts, sparse residual dimension, analytic/autodiff direction and pole-vector counts, autodiff sparse residual dimension, IK iterations, batch size, and Jacobian mode; gate-failure payloads mirror these under `compare_solver_objectives`. Tracking and profile residual metrics align target frames with the emitted runtime buffer by removing initialization and stabilization target frames before comparison. `benchmark_summary.json` records `resolved_motions`, and per-motion payloads record `source_frame_start` plus frame-selection mode. `benchmark_frames.csv` includes the per-motion BVH path for each motion-level metric row. `benchmark_gates.json` records report-only legacy/v2 acceptance gates by default; failed gates now also write `failures/<robot>_gates.json` with failed gates, compare metrics, solver-objective diagnostics, profile metadata, warnings, and the reproduction command. `--strict-gates` returns exit code 4 on legacy/v2 gate failure. `registry_coverage.json` records coverage for RPO, generic Unitree G1, G1 23DoF, G1 29DoF, E3 v2, and OLI. Compiled profiles now also record `semantic_edge_length` for each chain as an explanatory semantic-site distance; default segment-local scaling still uses chain `total_length` because bounded sensitivity showed substituting edge length directly regresses current tracking gates.

RoboParty RPO currently compiles with schema v2, `xyzw` quaternion order, confidence `1.0`, enabled sparse tasks, mesh-derived hand/foot virtual sites, virtual-foot-site root/ground metadata, mesh-derived self-collision metadata, morphology site-count metadata, matching cache fingerprints, and separated legacy/v2 bounded runtime metrics with profile task residuals, torso leakage residuals, and zero measured v2 penetration in the bounded sample. Single-axis projected torso rotation is treated as a weak style target instead of a hard tracking target so the RPO torso constraint does not dominate reach-limited hand and foot tracking.

Unitree G1 currently produces a v2 artifact and separated legacy/v2 bounded runtime metrics. Generic `unitree_g1` now uses Newton's built-in 29DoF MJCF fallback when no local XML is registered, so its compiled profile has real morphology fingerprints, non-placeholder chain lengths, reachable chain ranks, and available torso residual metrics. Multi-axis torso profiles now soften projected torso rotation and strengthen foot position tasks; this keeps Unitree hand/foot tracking passing while bringing root tilt back under the legacy gate.

E3 v2 coverage uses a small checked-in synthetic MJCF capability fixture, not a vendor asset. It validates the E3 minimal-map path, deterministic autoconfig, simplified hand mapping without wrist links, torso rank/basis reporting, and bounded benchmark artifact generation until licensed E3 assets are registered.

Registry coverage currently reports RoboParty RPO, generic `unitree_g1`, explicit `unitree_g1_23dof`, explicit `unitree_g1_29dof`, and synthetic-fixture `e3_v2` targets as `ready`. `unitree_g1_23dof` compiles against Newton's built-in 23DoF MJCF fallback with a separate semantic map and generated profile, while `unitree_g1_29dof` resolves to the generic Unitree G1 registration because that path uses Newton's built-in 29DoF MJCF fallback. `oli` remains `missing_registration` and includes an onboarding report with required asset/config slots, `params.py` entries, a minimal semantic `ik_map` template, and validation/benchmark commands. This records the OLI fallback path without fabricating or downloading assets.

Current bounded gate status is `failed`: generic Unitree G1 now passes the report-only bounded gates, while RoboParty RPO and synthetic E3 v2 still fail. RPO, Unitree, and E3 all pass penetration, torso leakage, and actuated smoothness gates on the selected high-motion windows, and each records zero priority-guard rollbacks while protecting hard joint-limit penetration. The hard-penetration guard uses the CUDA graph-capture fast path. Contact heuristic inference now accepts both Warp transforms and numpy transform rows, so the synthetic E3 runtime no longer disables contact locks due to transform type mismatch; its legacy and v2 contact configs also receive virtual sole anchor offsets. Compiled-profile runtime uses morphology-scheduled IK iteration defaults: single-axis torso profiles use 4 iterations, while multi-axis torso profiles keep 8 iterations. Direction objectives use an analytic Jacobian only for multi-axis torso profiles; single-axis torso profiles keep the autodiff direction objective because the analytic path changed bounded RPO tracking. Pole-vector objectives now use the tangent-space analytic residual only for morphology profiles with multi-axis torso rotation and rank-2 hand reach, which keeps Unitree on the pure analytic path without applying the unsafe analytic pole schedule to RPO or the low-rank synthetic E3 hand profile. The range-normalized joint motion limiter brings RPO actuated `velocity_p95/acceleration_p95` to `21.87/578.49` versus legacy `28.91/1626.89`, Unitree G1 to `36.64/837.03` versus legacy `56.51/6388.52`, and synthetic E3 v2 to `8.27/214.00` versus legacy `32.88/3464.22`. Unitree G1 now reaches pure `analytic` mode with sparse residual dimension `58`, hand/foot RMSE `0.54/0.31` versus legacy `0.55/0.45`, root tilt `1.33` versus `1.60`, and runtime `1.47s` versus legacy `1.29s`, so it passes the bounded gates. Synthetic E3 v2 still keeps autodiff pole-vector tasks because applying tangent analytic pole residuals caused penetration/root regressions in default gates; it passes foot/root/torso gates (`0.31` foot RMSE versus legacy `0.42`, root tilt `0.87` versus `1.14`) but fails hand tracking (`1.20` versus legacy `0.59`) and runtime. Weak single-axis torso tracking, the 4-iteration single-axis runtime default, and softened low-rank endpoint weights keep RPO hand/foot RMSE at `0.39/0.41` versus legacy `0.22/0.18`, and keep RPO root tilt slightly below legacy at `1.669` versus `1.673`, but RPO hand/foot tracking still fails the report-only gates. RPO hand residuals remain relatively balanced (`axis_rmse [0.21, 0.21, 0.23]`, `mean_error [-0.10, -0.06, -0.03]`), while RPO foot residuals remain elevated (`axis_rmse [0.18, 0.24, 0.22]`). The remaining failure artifacts are saved in `failures/roboparty_rpo_gates.json` and `failures/e3_v2_gates.json`. Runtime profiling shows the remaining overhead is solver dominated rather than BVH load or metric evaluation: RPO v2 runs in `mixed` Jacobian mode at about `40.50` output frames/s versus legacy `analytic` mode at `178.90`, Unitree G1 v2 runs in `analytic` mode at about `130.89` output frames/s versus legacy `analytic` mode at `125.32`, and synthetic E3 v2 runs in `mixed` mode at about `41.18` output frames/s versus legacy `analytic` mode at `192.08`. Solver-objective diagnostics show RPO still pays autodiff cost from sparse direction plus pole-vector tasks, E3 pays autodiff cost from pole-vector tasks, and Unitree no longer has autodiff sparse residuals in the bounded default path. The next acceptance work is reach-limited RPO/E3 hand tracking and solver-kernel/runtime reduction without reintroducing the E3 safety regressions seen under unconditional tangent pole activation.

Pole-vector subset diagnostics (`v2_pole_keep_distal` and `v2_pole_keep_proximal`) reduce autodiff residual dimension and improve solve FPS on the bounded sample, but they are not benchmark-safe defaults: RPO and Unitree G1 hand/foot tracking regress when only half of the pole-vector set is retained, and E3 v2 remains hand-limited. This keeps the next runtime target focused on a tracking-equivalent lower-cost pole formulation rather than semantic pole pruning.

Tangent-space pole diagnostics (`v2_pole_tangent_analytic`) validate a lower-dimensional pole objective that projects the pole-normal error into the target normal's tangent plane. On a warm-cache bounded run it reduces sparse residual dimension from `66` to `58`; Unitree G1 and synthetic E3 v2 can reach pure `analytic` mode, while RPO remains `mixed` because its direction objective is still autodiff. The accepted default schedule is narrower than the diagnostic mode: tangent pole residuals are enabled only for multi-axis torso profiles whose hand chains have at least rank-2 translational reach. This admits Unitree G1 and makes its bounded default gates pass, while keeping RPO and synthetic E3 on the safer autodiff pole path because unconditional tangent pole activation regressed RPO hand/foot tracking and introduced E3 penetration/root gate failures.

Pole analytic weight diagnostics show that low-weight analytic pole-vector tasks can remove the mixed/autodiff path and pass the Unitree runtime gate, but still miss tracking gates when used alone. In the bounded sweep, `v2_pole_analytic_w0.05` brought Unitree motion runtime below legacy tolerance while hand RMSE regressed beyond the gate, and RPO tracking regressed for all tested analytic weight scales. The benchmark now exposes `v2_pole_analytic_w<scale>_hand_w<weight>` to make combined pole/runtime and endpoint-weight experiments reproducible before any morphology-scheduled default is changed. A Unitree follow-up sweep with hand weights 150, 200, and 300 did not recover tracking, so endpoint weight amplification is not sufficient to make low-weight analytic pole-vector scheduling benchmark-safe.

Projected position diagnostics (`v2_pos_projected`) add runtime support for position residuals constrained to each compiled chain's translational basis and reduce sparse residual dimension for low-rank endpoints, but direct projection is not benchmark-safe as a default. On the bounded sample it improves some foot/root residuals, while RPO and Unitree hand tracking regress and synthetic E3 hand/foot/root tracking worsens versus standard v2. This validates the reachability projection infrastructure for solver experiments, but default activation still needs a safer blended or style-only formulation rather than replacing full position objectives outright.

## Remaining Work

- Extend the bounded runtime benchmark into uncapped full-length legacy/v2 comparison loops.
- Replace the synthetic E3 v2 fixture with licensed E3 assets when available, and add OLI registry assets when available.
- Add runtime benchmark coverage for optional collision barriers on real motions.
- Strengthen semantic auto-detection beyond conservative body-name matching.
- Turn report-only benchmark gates into required CI/acceptance gates after full-length motion coverage and morphology gaps are resolved.
- Produce full-motion before/after metrics for acceptance thresholds.
