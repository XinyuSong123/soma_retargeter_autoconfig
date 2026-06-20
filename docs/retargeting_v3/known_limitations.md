# Retargeting v3 Step 2 Known Limitations

This file records issues discovered while implementing the offline kinematic compiler without expanding Step 2 into asset synchronization or production IK integration.

## Current validation status

- All required Step-2 report IDs currently compile in `artifacts/retargeting_v3_step2/per_robot/`.
- `berkeley_humanoid` is reported with `capability_status: partial_humanoid`: lower-body chains compile, while hand tasks are intentionally unavailable instead of fabricated.
- `unitree_g1_urdf` uses an adapter-side URDF loading patch that rewrites mesh filenames to absolute paths in a temporary load copy. The upstream cached asset is not modified.
- `artifacts/retargeting_v3_step2/validation_checks.json` records complete-model cross-checks that are not hard profile gates.

## Covered focused tests

- SO(3) log/exp round trip.
- Relative transform invariance under a common world transform.
- Semantic frame construction degeneracy reporting.
- Free and ball joint tangent integration without quaternion component addition.
- Prismatic, revolute and non-world-axis hinge numerical Jacobians.
- Prismatic torso profile compilation with translation-only torso rank.
- Neutral-away-from-limit-midpoint deterministic sampling.
- Epsilon stability reporting through central-difference halving.
- One-DoF curved workspace local rank.
- Multi-pose rank recovery from a singular neutral serial-arm pose.
- Fixed/rank-zero torso projection.
- Bounded single-axis and multi-axis torso projection, plus endpoint projection limit handling.
- Synthetic yaw-only torso with full five-DoF no-wrist hand path.
- Synthetic three-DoF torso with wrist path discovery.
- Synthetic asymmetric humanoid bilateral length diagnostics.
- Synthetic lower-body-only partial humanoid downgrade without fabricated hand chains.
- Wahba/Kabsch-style vector alignment.
- Deterministic profile hash.
- Newton engine translation-Jacobian cross-check for LCA-relative semantic chains.
- Canonical target validation for neutral reconstruction, segment-length preservation, root-translation equivariance and global-yaw equivariance.

## Implementation boundaries

- The v3 package is an offline compiler. It is not wired into `NewtonPipeline` or production whole-body IK.
- The formal validation artifacts are compiled with the Newton backend. Newton `eval_jacobian` is used as an engine translation cross-check where the semantic reference is the kinematic LCA. The MuJoCo backend remains available as a portable fallback and for `mj_jac` comparison where useful.
- Complete-model semantic maps beyond explicit RPO are inferred from body names for validation convenience. Capability, paths, ranks and Jacobians still come from compiled runtime topology and FK, not model-name branches.
- Canonical target generation emits deterministic semantic target motions and chain-only projection summaries. These artifacts validate target mathematics only; they are not production whole-body IK results.
- G1 MJCF/URDF strict equivalence is currently a documented limitation, not a passed claim. The cached URDF and MJCF reports differ in ordered hand-chain topology and at least one torso translation-rank summary. The equivalence artifact records exact topology/rank tolerances and the observed differences.
- RPO distal hand validation proves full shoulder-to-current-endpoint chain use for the public model's elbow-yaw endpoint. It does not prove an independent palm or fingertip mesh anchor because no such semantic hand site is exposed by the current public RPO model.
- ROBOTIS OP3 chest demand is represented as a rank-zero torso projection in the current complete-model artifact. The validation check confirms no leg coordinates are assigned to the OP3 torso task; it does not claim a production whole-body IK policy has been exercised.
