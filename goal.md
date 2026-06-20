# Goal — Step 2: Offline Runtime-Kinematics and Target Compiler

> **Codex execution contract**
>
> Current branch: `retargeting-v3-step2-kinematic-core`  
> Parent planning branch: `retargeting-v3-core-robot-zoo`  
> Clean code lineage: `dev` at `aed9f9fbf09a1bf2816d8bfa5c8423ea2300cf2c`  
> Step-1 specification is already complete in:
>
> - `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
> - `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
> - `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
>
> Read all three files before touching code. They are normative. This task is deliberately smaller than the previous v2/v3 master plan.

---

## 1. Single objective

Implement and validate an **offline kinematic compiler** that converts a complete robot model plus a minimal semantic mapping into:

1. runtime-true semantic sites;
2. full-chain active-coordinate paths;
3. numerical relative position/orientation Jacobians;
4. deterministic multi-pose capability reports;
5. source/robot rest-frame calibration;
6. an offset-free morphology target builder;
7. chain-only closest-reachable torso and endpoint targets;
8. neutral and canonical-motion validation artifacts.

Do not integrate this into the production whole-body Newton retargeting pipeline in Step 2.

---

## 2. Why this branch starts from clean `dev`

The v2 branch is an exploration archive, not a safe implementation base. It contains useful conclusions and a few reusable utilities, but its core path has known design defects:

- static XML morphology treated as kinematic truth;
- immediate semantic parent used instead of full endpoint chain;
- one-rest-pose rank;
- legacy scaler offsets retained in v2 target generation;
- world Chest quaternion projected instead of true Hips-relative rotation;
- no-wrist virtual hand offset not used by position IK;
- priority/normalization fields not fully connected to runtime;
- benchmark target mismatch.

Starting from clean `dev` avoids silently preserving those defects. You may inspect `retargeting-v2-design`, but may copy code only after proving it satisfies the Step-1 contract.

Do not merge the v2 branch.

---

## 3. Hard scope boundary

### In scope

- Newton runtime model introspection;
- URDF/MJCF model-loading adapter needed by selected validation robots;
- FK-based semantic transforms;
- full-chain active DoF discovery;
- tangent-coordinate finite differences;
- multi-pose rank and conditioning;
- rest-frame calibration;
- virtual distal hand and sole sites;
- offset-free target generation;
- small chain-only nonlinear projection;
- complete-model validation reports;
- tests and deterministic artifacts.

### Out of scope

- replacing `NewtonPipeline`;
- production IK objective wiring;
- contact state machine changes;
- whole-body staged solver;
- collision objectives;
- temporal filtering;
- full Robot Zoo synchronization;
- UI work;
- teacher refinement;
- pose-pair optimization;
- hand/foot/torso weight sweeps;
- optimizing real-time speed beyond recording a baseline.

If an out-of-scope issue is discovered, record it in `known_limitations.md`; do not expand the implementation.

---

## 4. Required architecture

Create a parallel v3 offline package. Do not force unfinished code through legacy classes.

Suggested layout:

```text
soma_retargeter/robotics/v3/
  __init__.py
  model_adapter.py
  semantic_sites.py
  kinematic_paths.py
  numerical_jacobian.py
  reachability.py
  rest_frames.py
  target_builder.py
  chain_projection.py
  profile.py
  validation.py

soma_retargeter/tools/
  compile_kinematic_profile_v3.py
  validate_kinematic_profile_v3.py
```

File names may change, but responsibilities must remain separate.

The output schema for this step is `KinematicProfileV3`, not the production runtime profile.

---

## 5. Runtime model adapter

Implement a typed adapter whose truth comes from the robot model actually loaded by Newton.

Minimum interface:

```python
class RuntimeRobotModelAdapter(Protocol):
    model_format: str
    nq: int
    nv: int

    def neutral_q(self) -> np.ndarray: ...
    def integrate(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray: ...
    def forward_kinematics(self, q: np.ndarray) -> RobotKinematicState: ...
    def body_transform(self, state, body_name: str) -> np.ndarray: ...
    def site_transform(self, state, site: SemanticSite) -> np.ndarray: ...
    def active_velocity_coordinates(self, reference: SemanticSite, target: SemanticSite) -> list[int]: ...
```

Requirements:

1. free/ball joints are perturbed through velocity/tangent integration;
2. no naive quaternion-component addition;
3. body and joint labels are normalized without losing originals;
4. joint q/qd indices come from runtime arrays;
5. fixed bodies remain in the semantic path but add no active coordinate;
6. the adapter exposes a deterministic model fingerprint;
7. model resources are released cleanly;
8. CPU tests must run when CUDA is unavailable.

Raw XML parsing may supply provenance or explicit site hints, but cannot override runtime FK.

---

## 6. Semantic sites

Step 2 accepts a minimal explicit semantic map for hard-gate robots. Automatic semantic discovery is not the primary goal yet.

Required semantics:

- `Hips`;
- `Chest`;
- `LeftHand`;
- `RightHand`;
- `LeftFoot`;
- `RightFoot`.

Optional intermediate landmarks:

- shoulders;
- elbows;
- hips;
- knees;
- head.

A semantic endpoint is a body plus a local transform.

Position-site offsets must always be honored independently of orientation capability.

Site inference order:

1. explicit model site;
2. explicit semantic override;
3. distal child anchor;
4. safe geometry bound;
5. symmetric mirror;
6. segment extrapolation.

Each site records source, confidence, local pose and reason.

---

## 7. Full-chain path discovery

For every relative semantic task, compute the kinematic path using runtime topology.

Required chains:

```text
Hips -> Chest
Chest -> LeftHandSite
Chest -> RightHandSite
Hips -> LeftSoleSite
Hips -> RightSoleSite
```

The hand chain must contain every proximal arm coordinate that affects the distal site. The foot chain must contain every hip/knee/ankle coordinate that affects the sole.

Do not use semantic `ForeArm` as the reachability root for Hand.

Save:

- reference body;
- target body;
- LCA;
- body path;
- active velocity-coordinate indices and labels;
- joint type per coordinate;
- limit source.

---

## 8. Numerical relative Jacobians

Implement the exact formulas in the Step-1 specification.

### Translation

\[
J_p[:,i]=\frac{p(q^+)-p(q^-)}{2\epsilon_i}
\]

where position is expressed in the semantic reference frame.

### Rotation

\[
J_R[:,i]=\frac{\operatorname{Log}(R^+(R^-)^T)}{2\epsilon_i}.
\]

Requirements:

- central difference;
- adaptive epsilon by coordinate type/range;
- epsilon-halving stability check;
- quaternion sign safety;
- separate translation and rotation matrices;
- engine Jacobian cross-check where available;
- finite and shape validation;
- deterministic output.

---

## 9. Multi-pose reachability

Sample each chain at:

1. neutral;
2. ±10% safe-range per active coordinate;
3. 32 deterministic low-discrepancy configurations inside central finite ranges.

For each sample save singular values, local rank and conditioning.

Compute:

- `regular_rank`;
- `nominal_rank`;
- `singularity_fraction`;
- `epsilon_unstable_fraction`;
- conditioning percentiles.

Do not use concatenated-Jacobian rank as simultaneous controllable rank.

Do not produce a fixed endpoint projector from the aggregate sample set.

---

## 10. Rest-frame calibration

Implement semantic frames and per-edge alignment without loading any legacy scaler config.

Hard rule:

```text
No v3 module may import or read v1 optimized joint_scales or joint_offsets.
```

The calibration must produce:

- source rest semantic frames;
- robot neutral semantic frames;
- edge alignment rotations;
- robot segment lengths;
- bilateral symmetry report;
- confidence and fallbacks.

Neutral exactness gate:

```text
max semantic position error < 0.001 m
max relative orientation error < 0.01 rad
```

---

## 11. Offset-free target builder

Build desired robot semantic targets recursively in robot parent frames using calibrated edge alignment and robot segment length.

Required properties:

1. global source translation does not change local limb articulation;
2. global source rotation rotates the root trajectory but does not corrupt local edge mapping;
3. source neutral reconstructs robot neutral;
4. output segment lengths equal robot lengths;
5. left/right targets respect verified symmetry;
6. no legacy offsets;
7. no per-robot weight table;
8. source zero-length edges use a recorded fallback.

Implement a NumPy reference path first. A Warp optimization is optional and must match the reference.

---

## 12. Chain-only closest-reachable projection

Implement an offline deterministic bounded solver for:

### Torso relative orientation

Solve the Step-1 nonlinear relative-orientation problem over torso-chain coordinates.

Must support:

- rank 0 welded torso;
- one-axis waist not aligned with world Z;
- multi-axis serial waist;
- joint limits;
- continuity prior;
- deterministic initialization.

### Endpoint position

For underactuated or infeasible hand/foot targets, solve chain-only normalized position projection and return:

- desired target;
- feasible projected target;
- chain q;
- residual;
- convergence status.

This is an offline reference implementation. Do not integrate it into production whole-body IK in this step.

---

## 13. Complete robot validation set

Hard-gate complete models:

1. RoboParty RPO repository MJCF;
2. Unitree G1 29DoF MJCF;
3. Unitree G1 URDF equivalent;
4. Unitree G1 23DoF real fixed model;
5. Unitree H1 MJCF;
6. ROBOTIS OP3 MJCF;
7. Booster T1 MJCF.

Additional compile/correctness gates:

8. PAL TALOS MJCF;
9. Berkeley Humanoid MJCF/URDF entry.

Synthetic fixtures are still mandatory, but cannot replace these complete models.

Use fixed upstream refs from `assets/robot_zoo/robot_zoo_manifest.json`. Do not download floating `main`.

Do not commit large visual meshes in Step 2. Cache full assets outside Git and save hashes/provenance.

---

## 14. Model-specific expectations are tests, not branches

RPO assertions:

- torso regular rotational rank 1;
- actual torso axis recovered in pelvis frame;
- full shoulder-to-distal-hand path;
- full hip-to-sole path;
- distal hand position site used despite limited orientation capability.

OP3 assertion:

- zero-DoF or effectively fixed torso semantics must compile cleanly;
- source chest demand is reported unreachable instead of leaking into another chain.

G1 assertion:

- richer torso and arm capability is discovered from the model;
- URDF and MJCF semantic results are equivalent within tolerance.

Berkeley assertion:

- missing full upper-body semantics downgrade to partial humanoid instead of fabricating Hand tasks.

Do not implement these with `if robot_name` in core code.

---

## 15. Required tests

### Unit mathematics

- SO(3) log/exp round trip;
- relative-transform invariance under common world transform;
- tangent integration for ball/free joints;
- prismatic numerical Jacobian;
- revolute numerical Jacobian;
- non-world-axis hinge;
- epsilon stability;
- one-DoF curved workspace local-rank test;
- multi-pose singularity recovery;
- LCA path;
- fixed-body path;
- full hand/foot path;
- frame construction degeneracy;
- Kabsch/Wahba alignment;
- neutral exactness;
- source global-transform equivariance;
- bounded single-axis torso projection;
- rank-zero torso;
- multi-axis torso projection;
- endpoint projection and limit handling;
- deterministic JSON/hash.

### Synthetic models

- yaw-only torso + 5DoF no-wrist arm;
- 3DoF torso + wrist;
- prismatic torso;
- rotated base and oblique joint axis;
- neutral pose away from limit midpoint;
- asymmetric humanoid;
- lower-body-only partial humanoid.

### Complete models

Every model in Section 13 must execute:

```text
load -> semantic sites -> path -> neutral FK -> Jacobians -> multi-pose rank
     -> rest calibration -> neutral target -> canonical target projection -> report
```

No crash or silent skip.

---

## 16. Canonical offline motions

The target compiler must be tested on generated source semantic motions:

1. neutral;
2. root translation;
3. global root yaw;
4. torso pitch;
5. torso roll;
6. torso yaw;
7. mixed torso rotation;
8. arms forward;
9. elbow bend;
10. overhead reach;
11. squat;
12. single step target.

These motions test target mathematics only; do not run production whole-body IK.

---

## 17. Artifacts

Write:

```text
artifacts/retargeting_v3_step2/
  environment.json
  commands.txt
  summary.json
  failures/
  per_robot/
    roboparty_rpo.json
    unitree_g1_mjcf.json
    unitree_g1_urdf.json
    unitree_g1_23dof.json
    unitree_h1.json
    robotis_op3.json
    booster_t1.json
    pal_talos.json
    berkeley_humanoid.json
```

Each report includes:

- model source/ref/hash;
- runtime model dimensions;
- semantic sites;
- body paths;
- active coordinates;
- numerical Jacobians and singular values;
- rank statistics;
- rest calibration;
- neutral errors;
- canonical target projection residuals;
- timing;
- warnings/failures;
- reproduction command.

Do not commit huge raw matrices for every sample. Save compact summaries plus selected diagnostic matrices.

---

## 18. Subagent plan

The main Codex agent is the Integrator and owns scope.

### Agent A — Runtime model adapter

Owns:

- runtime introspection;
- tangent integration;
- FK state;
- path/coordinate extraction;
- model fingerprint;
- adapter tests.

### Agent B — Mathematical core

Owns:

- SO(3)/SE(3) helpers;
- numerical Jacobians;
- multi-pose ranks;
- chain projection;
- synthetic mathematical tests.

### Agent C — Calibration and targets

Owns:

- semantic rest frames;
- virtual sites;
- edge alignment;
- target builder;
- neutral/canonical tests.

### Agent D — Complete-model validator and red team

Owns:

- fixed asset resolution;
- complete-model reports;
- URDF/MJCF comparison;
- checking for robot-name branches, legacy offsets and false rank-zero passes.

### Collaboration rules

1. Agents use isolated worktrees/branches.
2. Parallel heavy agents: maximum 3.
3. File ownership must not overlap without Integrator approval.
4. Each agent submits small commits and a handoff report.
5. Integrator merges in order A → B → C → D.
6. Agent D cannot weaken acceptance gates.
7. No agent edits production `newton_pipeline.py` except a tiny import-free diagnostic hook explicitly approved by Integrator.

---

## 19. Acceptance gates

### Adapter

- runtime FK deterministic;
- tangent integration valid;
- active paths match synthetic truth;
- no static XML FK used as final truth.

### Jacobians

- synthetic analytic comparison error < `1e-4` relative where conditioned;
- epsilon-halving discrepancy below documented tolerance;
- non-finite values: zero.

### Reachability

- yaw-only torso regular rank 1;
- welded torso rank 0;
- no-wrist full arm hand-position rank uses all proximal coordinates;
- full leg path includes hip, knee and ankle;
- one-DoF curved-workspace test remains local rank 1.

### Calibration

- neutral semantic position max error < `1 mm`;
- neutral relative rotation max error < `0.01 rad`;
- no v1 offsets read;
- segment lengths positive and finite.

### Torso projection

For a yaw-only synthetic and RPO chain:

- pure source pitch/roll produces projected hinge motion ≤ 5% of pure compatible-axis response;
- compatible-axis rotation is preserved within joint limits;
- result invariant to common world rotation.

### Complete models

- all Section-13 models produce a report;
- hard-gate models pass load/path/Jacobian/calibration/neutral checks;
- partial morphology returns structured status;
- no hard-gate model silently falls back to a synthetic model.

### Determinism

Two runs with the same model hashes and seed produce semantically identical profile JSON and rank results.

---

## 20. Explicitly forbidden

- merge `retargeting-v2-design`;
- production IK integration;
- hand/foot/chest weight tuning;
- robot-name special cases in mathematical modules;
- static XML body transforms as runtime truth;
- semantic immediate parent as endpoint-chain root;
- concatenated-Jacobian rank as controllable rank;
- legacy scaler offsets;
- world Chest quaternion projection;
- position site offset conditioned on orientation support;
- rank-zero residual reported as successful zero error;
- large unpinned asset downloads;
- claiming complete-model validation when only synthetic fixtures ran;
- adding direction/pole/collision objectives;
- weakening gates to finish.

---

## 21. Recommended commit sequence

1. `test: add v3 synthetic runtime kinematics fixtures`
2. `feat: add Newton runtime model adapter`
3. `feat: add full-chain path discovery`
4. `feat: add numerical relative Jacobians`
5. `feat: add deterministic multi-pose reachability`
6. `feat: add semantic sites and rest frames`
7. `feat: add offset-free target builder`
8. `feat: add bounded chain target projection`
9. `test: validate complete RPO and G1 models`
10. `test: validate H1 OP3 Booster TALOS Berkeley models`
11. `docs: publish step2 implementation report and artifacts`

Do not make dozens of parameter-sweep commits.

---

## 22. Completion definition

Step 2 is complete only when:

- [ ] runtime model adapter exists and is tested;
- [ ] full-chain active-coordinate discovery passes;
- [ ] numerical relative Jacobians pass synthetic checks;
- [ ] deterministic multi-pose rank passes;
- [ ] rest-frame calibration passes neutral exactness;
- [ ] v3 target builder reads no legacy offsets;
- [ ] true relative torso chain projection passes;
- [ ] endpoint chain projection passes;
- [ ] all hard-gate complete robots produce passing reports;
- [ ] TALOS and Berkeley produce structured reports;
- [ ] URDF/MJCF G1 semantic equivalence is reported;
- [ ] no production IK integration was added;
- [ ] tests were actually executed;
- [ ] artifacts and reproduction commands are committed;
- [ ] red-team review has no blocking finding;
- [ ] all commits are pushed to this branch.

When this list passes, stop. Do not begin Step 3 on this branch.
