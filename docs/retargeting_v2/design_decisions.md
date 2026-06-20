# Retargeting v2 Design Decisions

## Projected Targets vs Custom Relative Objectives

Problem: low-DoF torso chains cannot track full source quaternion targets.

Decision: project torso rotation targets into the compiled reachable basis before setting runtime rotation objectives.

Reason: projection prevents unreachable pitch/roll from leaking into hips, knees, root, and shoulders while preserving the existing analytic Newton rotation objective path.

Risk: projection quality depends on morphology rank estimation.

Verification: reachability projection unit tests remove unreachable pitch/roll for yaw-only fixtures.

## Priority Scheduler

Problem: weighted least squares can let low-priority tracking damage joint safety.

Decision: use compiled priority weight bands plus a high-priority residual guard for priority-0 hard joint-limit penetration.

Reason: it is implementable with the current Newton API and provides a concrete rollback mechanism without waiting for full HQP/nullspace support.

Risk: guard currently protects hard joint-limit penetration, not every priority-0 residual.

Verification: priority band validation and rollback helper tests.

## Actuated Joint Motion Limiter

Problem: v2 IK can jump between equivalent local solutions on a small number of frames, creating extreme actuated joint velocity and acceleration even when most frames are static.

Decision: compiled v2 runtime configs enable a range-normalized exported joint motion limiter for finite actuated coordinates. The limiter bounds per-frame delta and delta-change by each joint range and clip sample rate, while floating root coordinates remain outside the actuated q smoothness gate and are reported through separate root diagnostics.

Reason: small temporal objective weights were insufficiently predictable and large weights made runtime and tracking worse. A command-space limiter directly bounds the exported trajectory without changing legacy configs.

Risk: limiting can increase hand/foot tracking error when the IK solution jumps far from the previous command; tracking gates therefore remain active and still fail in the current bounded artifacts.

Verification: unit tests cover root masking and range/dt delta limiting, and bounded benchmark artifacts show RPO and generic Unitree G1 actuated velocity/acceleration gates passing while tracking/runtime gates remain report-only failures.

## Direction Jacobian Schedule

Problem: direction objectives are solver dominated when evaluated through the autodiff Jacobian path, but changing their Jacobian path can change bounded tracking behavior.

Decision: use an analytic direction-objective Jacobian for multi-axis torso profiles, and keep the autodiff direction objective for single-axis torso profiles.

Reason: generic Unitree G1 has enough torso rotational rank to benefit from the faster analytic path while preserving tracking gates. RoboParty RPO's single-axis torso remains more sensitive, so it keeps the previously validated bounded tracking behavior.

Risk: single-axis torso profiles still pay the autodiff cost, and pole-vector objectives remain on the autodiff Jacobian path.

Verification: sparse-objective tests cover the analytic direction tuple wiring, registry tests assert the torso-rank schedule, and bounded artifacts show Unitree runtime improving while RPO tracking stays consistent with the single-axis baseline.

## Single-Axis Tracking Weight Schedule

Problem: single-axis torso, no-wrist profiles over-constrain a reach-limited robot when endpoint tracking keeps the same hand/foot weights as higher-DoF morphology.

Decision: soften low-rank no-wrist hand position tasks at compile time, independent of torso rank. For profiles whose torso rotational rank is at most one, also soften foot position tasks. Multi-axis torso profiles keep the stronger foot weighting that preserves Unitree G1 tracking/root gates.

Reason: bounded RPO experiments showed that high foot weights improve foot RMSE only by sacrificing hand tracking, while direct hand virtual-site offsets are not reachable and substantially regress hand RMSE. A morphology schedule gives the single-axis profile more balanced tracking without a robot-name branch.

Risk: RPO hand/foot gates still fail; this is an incremental improvement, not an acceptance pass.

Verification: runtime config tests assert the low-rank no-wrist hand weight, the single-axis foot weight, and multi-axis G1 foot weights. Compiled profile artifacts record compiler `2.1.6`; bounded artifacts keep RPO hand/foot RMSE at roughly `0.39/0.41`, keep Unitree G1 hand/foot/root gates passing, and keep synthetic E3 foot/root gates passing while hand tracking remains a blocker.

## Synthetic E3 Capability Fixture

Problem: the goal requires E3 v2 integration evidence, but the current workspace does not contain licensed E3 robot assets.

Decision: register a small checked-in `e3_v2` MJCF capability fixture with a minimal semantic map, and mark it as `asset_kind: synthetic_fixture` in registry coverage artifacts.

Reason: this exercises the same autoconfig, deterministic profile, simplified no-wrist hand map, torso rank, runtime benchmark, and failure-artifact paths without fabricating a vendor asset or downloading unlicensed files.

Risk: synthetic benchmark metrics are not hardware acceptance metrics and must be replaced when real E3 assets become available.

Verification: `e3_v2` autoconfig writes a schema v2 profile with valid cache diagnostics, registry coverage reports it as a ready synthetic fixture, and bounded benchmark artifacts include `per_robot/e3_v2.json` plus `failures/e3_v2_gates.json`.

## Pole Jacobian Gate

Problem: pole-vector objectives still keep v2 on a mixed/autodiff Jacobian path, but directly enabling the validated analytic pole Jacobian changes bounded optimizer behavior.

Decision: implement pole-vector analytic Jacobian support behind a per-task `analytic_jacobian` flag, and emit compiled runtime configs with the flag disabled by default.

Reason: bounded benchmark experiments show direct analytic pole enablement lowers runtime substantially but regresses RPO, Unitree, and E3 hand/foot/root metrics. Preserving tracking correctness is more important than using the faster path before benchmark-safe activation.

Risk: runtime remains above the report-only acceptance gate while pole objectives use autodiff.

Verification: sparse-objective tests cover the pole task flag wiring, finite-difference tests validate the analytic kernel for parent-only, middle-only, child-only, partial-chain, and full-chain body masks with mixed linear/angular spatial velocity, and model-topology tests validate the generated ancestor DOF masks for parent/middle/child links. Registry tests assert the disabled reason, and bounded artifacts record pole tasks as disabled pending benchmark-safe activation. The benchmark exposes `v2_pole_analytic`, `v2_pole_analytic_w<scale>`, `v2_pole_analytic_w<scale>_hand_w<weight>`, and `v2_no_pole` as explicit diagnostic compare modes; current artifacts and local sensitivity runs show analytic, weighted analytic, or removed pole tasks improve runtime but regress RPO/Unitree/E3 tracking or root metrics, so they remain outside the default v2 path. Low-weight analytic pole tasks can pass the Unitree runtime gate but still miss hand tracking until endpoint scheduling is solved. The runtime Jacobian scheduler now selects pure analytic mode whenever every active objective supports analytic Jacobians, so a later safe pole-vector activation does not remain on a mixed/autodiff tape path.

## Per-Environment Contact Weight

Problem: shared contact weights break multi-env parity.

Decision: use `IKObjectivePerEnvWeightedPosition` and per-anchor/per-env lock state.

Reason: stance/swing weights and locked targets must be independent for each clip in a batch.

Risk: more per-frame CPU updates.

Verification: contact regression tests cover per-env weight setting, warmup label alignment, minimum duration, and release blending.

## Contact Transform Input Compatibility

Problem: contact heuristic inference can receive global transforms as either Warp transform objects or numpy transform rows, depending on the animation buffer path.

Decision: normalize each transform through a small translation extractor that first tries `wp.transform_get_translation` and then falls back to the first three numeric row values.

Reason: contact-aware IK should not silently disable locks just because a buffer returns `[x, y, z, qx, qy, qz, qw]` rows instead of Warp transform objects.

Risk: nonstandard transform row layouts still need explicit adapters.

Verification: contact inference tests cover numpy transform rows, and current bounded E3 artifacts run without the previous transform-type contact warning.

## Virtual Site Inference

Problem: robots without wrists should not track a full hand orientation at an elbow-yaw link origin.

Decision: compiled profiles mark hand orientation unsupported when no wrist is present, generate hand position tasks, and place distal hand/foot sites from explicit MJCF sites, distal child anchors, or geom/mesh bounds.

Reason: sparse task definitions are closer to actual morphology and avoid impossible 6D targets.

Risk: virtual-site placement still depends on registered geometry quality; non-STL mesh formats and rotated/fromto primitive bounds are limited.

Verification: RPO profile disables hand orientation and tests assert rotational rank zero for no-wrist hand fixtures, explicit MJCF site preference, child-anchor fallback, and geom/STL distal bounds.

## Symmetry Tying

Problem: legacy scaler could learn negative and asymmetric scales from small pose-pair sets.

Decision: v2 profiles derive segment lengths from morphology and validate legacy scaler values instead of using them as initial truth.

Reason: morphology-derived positive lengths are deterministic and safer than unconstrained optimization.

Risk: asymmetric robots need explicit evidence before symmetry assumptions should be relaxed.

Verification: legacy scaler validation rejects non-positive scale.

## Collision Proxy

Problem: self-collision should be a priority-0 feasibility term.

Decision: compile sphere proxy metadata and high-risk non-adjacent pair lists into the profile, then instantiate optional analytic sphere-pair barriers when `collision_weight > 0`.

Reason: the sphere-pair barrier is lightweight, deterministic, and can be disabled by default for backward compatibility while preserving a clear path to feasibility-pass collision costs.

Risk: motions can still self-intersect when `collision_weight` is left at its legacy default of `0.0`; tuning default activation still needs motion-level benchmark evidence.

Verification: unit tests cover geom-derived proxy generation, pair generation, safe disabled behavior when proxies are unavailable, optional runtime objective creation, and the sphere overlap residual kernel.

## Cache

Problem: compiled profiles must be deterministic and reusable.

Decision: generated profile paths are derived from robot config paths, validated before reuse, and invalidated when the cached robot fingerprint, source config hash, or compiler version differs from the current registry state. If regeneration only produces incomplete missing-MJCF morphology while an existing cached profile is structurally valid and complete, preserve the existing cache instead of overwriting it.

Reason: the runtime can opt into v2 without requiring manual profile path wiring, and accidental execution in an environment without optional robot assets should not replace a usable compiled profile with a placeholder morphology.

Risk: fingerprint coverage is only as complete as the morphology analyzer and registered config inputs; unregistered external dependencies still need explicit registration before they can affect cache invalidation.

Verification: profile deterministic serialization tests, registry validation path, stale cache fingerprint diagnostics, and regression coverage for preserving a valid cache when a missing-MJCF regeneration attempt returns incomplete morphology.
