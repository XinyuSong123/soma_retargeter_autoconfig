# Retargeting v2 Design Decisions

## Projected Targets vs Custom Relative Objectives

Problem: low-DoF torso chains cannot track full source quaternion targets.

Decision: project torso rotation targets into the compiled reachable basis before setting runtime rotation objectives.

Reason: projection prevents unreachable pitch/roll from leaking into hips, knees, root, and shoulders while preserving the existing analytic Newton rotation objective path.

Risk: projection quality depends on morphology rank estimation.

Verification: reachability projection unit tests remove unreachable pitch/roll for yaw-only fixtures.

## Priority Scheduler

Problem: weighted least squares can let low-priority tracking damage joint safety.

Decision: use compiled priority weight bands plus a high-priority residual guard for priority-0 joint-limit margin residuals.

Reason: it is implementable with the current Newton API and provides a concrete rollback mechanism without waiting for full HQP/nullspace support.

Risk: guard currently protects joint-limit margin, not every priority-0 residual.

Verification: priority band validation and rollback helper tests.

## Per-Environment Contact Weight

Problem: shared contact weights break multi-env parity.

Decision: use `IKObjectivePerEnvWeightedPosition` and per-anchor/per-env lock state.

Reason: stance/swing weights and locked targets must be independent for each clip in a batch.

Risk: more per-frame CPU updates.

Verification: contact regression tests cover per-env weight setting, warmup label alignment, minimum duration, and release blending.

## Virtual Site Inference

Problem: robots without wrists should not track a full hand orientation at an elbow-yaw link origin.

Decision: compiled profiles mark hand orientation unsupported when no wrist is present and generate hand position tasks instead.

Reason: sparse task definitions are closer to actual morphology and avoid impossible 6D targets.

Risk: current virtual-site placement is still conservative and mostly uses semantic body origins.

Verification: RPO profile disables hand orientation and tests assert rotational rank zero for no-wrist hand fixtures.

## Symmetry Tying

Problem: legacy scaler could learn negative and asymmetric scales from small pose-pair sets.

Decision: v2 profiles derive segment lengths from morphology and validate legacy scaler values instead of using them as initial truth.

Reason: morphology-derived positive lengths are deterministic and safer than unconstrained optimization.

Risk: asymmetric robots need explicit evidence before symmetry assumptions should be relaxed.

Verification: legacy scaler validation rejects non-positive scale.

## Collision Proxy

Problem: self-collision should be a priority-0 feasibility term.

Decision: compile sphere proxy metadata and high-risk non-adjacent pair lists into the profile first; keep runtime barrier execution disabled and explicitly reported as `not_implemented`.

Reason: proxy extraction and pair filtering can be made deterministic now, while runtime collision Jacobians need separate solver work.

Risk: motions can still self-intersect until runtime barriers consume the compiled pairs.

Verification: unit tests cover geom-derived proxy generation, pair generation, and safe disabled behavior when proxies are unavailable.

## Cache

Problem: compiled profiles must be deterministic and reusable.

Decision: generated profile paths are derived from robot config paths and validated before reuse.

Reason: the runtime can opt into v2 without requiring manual profile path wiring.

Risk: stale profiles can remain if source assets change without force regeneration.

Verification: profile deterministic serialization tests and registry validation path.
