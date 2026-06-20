# Retargeting v3 — Three-Step Roadmap

## Step 1 — Mathematical contract and complete-model review

Status: **completed in this branch**

Deliverables:

- `STEP1_MATHEMATICAL_SPEC.md`;
- `STEP1_COMPLETE_MODEL_CROSSCHECK.md`;
- corrected definitions of full-chain capability, relative Jacobians, multi-pose rank, rest calibration, reachable target projection and normalization;
- explicit rejection of the main v2 failure modes.

No Codex implementation is counted as part of Step 1.

## Step 2 — Offline kinematic compiler

Branch: `retargeting-v3-step2-kinematic-core`

Scope:

- runtime model adapter;
- full-chain numerical Jacobians;
- deterministic multi-pose capability;
- semantic rest-frame calibration;
- legacy-offset-free target builder;
- chain-only closest-reachable torso/endpoint projection;
- complete-model reports for RPO, G1, H1, OP3, Booster T1, TALOS and Berkeley Humanoid.

Explicitly out of scope:

- replacing production NewtonPipeline;
- contact integration;
- whole-body priority solver;
- collision;
- temporal smoothing;
- full Robot Zoo download;
- weight tuning.

Exit condition: offline neutral/canonical tests and complete-model capability reports pass.

## Step 3 — Runtime integration and full Robot Zoo validation

Created only after Step 2 passes.

Scope:

- minimal normalized whole-body IK;
- true relative torso objective;
- per-environment contact anchors;
- staged feasibility/tracking solve;
- independent semantic benchmark;
- full permissive/fetch-only Robot Zoo matrix;
- RPO and G1 runtime acceptance gates;
- CI and performance work.

No Step-3 implementation should be started while Step-2 hard gates are failing.
