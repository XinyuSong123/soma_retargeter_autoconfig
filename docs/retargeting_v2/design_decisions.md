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

## Per-Environment Contact Weight

Problem: shared contact weights break multi-env parity.

Decision: use `IKObjectivePerEnvWeightedPosition` and per-anchor/per-env lock state.

Reason: stance/swing weights and locked targets must be independent for each clip in a batch.

Risk: more per-frame CPU updates.

Verification: contact regression tests cover per-env weight setting, warmup label alignment, minimum duration, and release blending.

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

Decision: generated profile paths are derived from robot config paths, validated before reuse, and invalidated when the cached robot fingerprint, source config hash, or compiler version differs from the current registry state.

Reason: the runtime can opt into v2 without requiring manual profile path wiring.

Risk: fingerprint coverage is only as complete as the morphology analyzer and registered config inputs; unregistered external dependencies still need explicit registration before they can affect cache invalidation.

Verification: profile deterministic serialization tests, registry validation path, and stale cache fingerprint diagnostics.
