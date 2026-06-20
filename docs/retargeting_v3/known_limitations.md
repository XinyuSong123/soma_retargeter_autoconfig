# Retargeting v3 Step 2 Known Limitations

This file records issues discovered while implementing the offline kinematic compiler without expanding Step 2 into asset synchronization or production IK integration.

## Current Step 2 status

- Step 2 is **BLOCKED**. It must not be reported as complete or `PASS`.
- Formal pytest artifact:
  `artifacts/retargeting_v3_step2/test_results/pytest.txt` reports
  `209 passed, 10 skipped`.
- Acceptance audit artifact:
  `artifacts/retargeting_v3_step2/acceptance_ledger.json` reports
  `status="BLOCKED"` with `blocking_count=3`.
- The current acceptance blockers are:
  `arbitrary_g1_equivalence`,
  `cross_format.gates.same_source_strict`, and
  `cross_format.gates.variant_compatibility`.
- All current, continued, or newly launched Step 2 subagents must use `xhigh`
  reasoning strength and record that fact in handoff artifacts.

## Current validation status

- All required Step-2 report IDs currently compile in `artifacts/retargeting_v3_step2/per_robot/`.
- Current Robot Zoo status counts are `passed=7`, `negative_control_passed=4`,
  `algorithm_failed=14`, `semantic_failed=3`, `model_load_failed=2`, and
  `source_unavailable=16`.
- `deterministic_rerun.status="passed"`, but deterministic evidence does not
  close the G1/cross-format acceptance blockers.
- `berkeley_humanoid` is reported with `capability_status: partial_humanoid`: lower-body chains compile, while hand tasks are intentionally unavailable instead of fabricated.
- `unitree_g1_urdf` uses an adapter-side URDF loading patch that rewrites mesh filenames to absolute paths in a temporary load copy. The upstream cached asset is not modified.
- `artifacts/retargeting_v3_step2/validation_checks.json` records complete-model cross-checks that are not hard profile gates. Its G1 same-source entry is currently `status="incomplete"`, not a strict equivalence pass.

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
- G1 MJCF/URDF strict equivalence is currently a documented limitation, not a passed claim. Gate A remains `incomplete` because semantic FK, active-chain, rank-summary, and canonical-projection evidence are unavailable or missing. Gate B variant compatibility remains `blocked` because common-chain, rank, DoF-difference, and projection evidence for independently passing variants has not been produced.
- RPO distal hand validation proves full shoulder-to-current-endpoint chain use for the public model's elbow-yaw endpoint. It does not prove an independent palm or fingertip mesh anchor because no such semantic hand site is exposed by the current public RPO model.
- ROBOTIS OP3 chest demand is represented as a rank-zero torso projection in the current complete-model artifact. The validation check confirms no leg coordinates are assigned to the OP3 torso task; it does not claim a production whole-body IK policy has been exercised.
