# Step 1 — Mathematical Specification for Morphology-Aware Retargeting

Status: **completed design specification**  
Applies to: SOMA source skeleton → arbitrary humanoid/biped robot  
Implementation target: Step 2 branch goal in `/goal.md`

This document fixes the mathematical contract before any new runtime IK code is written. It deliberately separates:

1. model truth;
2. semantic calibration;
3. kinematic capability;
4. target construction;
5. nonlinear projection to a reachable target;
6. whole-body IK, which is deferred to Step 3.

The central conclusion is that the previous v2 design mixed these layers. Static XML parsing, a single rest-pose Jacobian, legacy scaler offsets, and one weighted IK solve cannot provide a morphology-independent retargeter.

---

## 1. Coordinate and transform conventions

A transform

\[
{}^A T_B =
\begin{bmatrix}
{}^A R_B & {}^A p_B\\
0 & 1
\end{bmatrix}
\]

maps coordinates from frame \(B\) into frame \(A\).

For a semantic site \(S\), the runtime model returns

\[
{}^W T_S(q).
\]

For two semantic sites \(A\) and \(B\), the relative pose is

\[
{}^A T_B(q)=({}^W T_A(q))^{-1}{}^W T_B(q).
\]

All capability calculations for a limb are expressed in a semantic reference frame:

- torso: `Hips → Chest`;
- left/right arm: `Chest → HandSite`;
- left/right leg: `Hips → SoleSite`;
- head: `Chest → Head`.

This removes floating-base motion and common ancestor motion from limb capability.

Quaternion storage remains `xyzw`, but all derivations are on \(SO(3)\). Quaternion signs must be canonicalized before interpolation or finite differences.

---

## 2. The compiled runtime model is the kinematic truth

The algorithm must obtain kinematic facts from the model actually used by Newton after loading or conversion:

- body and joint topology;
- joint type;
- q-position and velocity-coordinate layout;
- joint limits and neutral position;
- body/site local transforms;
- forward kinematics;
- integration of tangent increments for free and ball joints.

Raw URDF/MJCF is provenance and input, not final kinematic truth.

This is required because MJCF supports inherited default classes, several orientation representations, multiple joints on one body, nonzero joint positions, free/ball/slide/hinge joints, and compiler processing. MuJoCo itself defines joint positions and axes in body coordinates and exposes compiled end-effector Jacobians; the same principle applies when Newton is the runtime engine.

### 2.1 Required adapter interface

```python
class RuntimeRobotModelAdapter(Protocol):
    model_format: Literal["urdf", "mjcf"]
    nq: int
    nv: int

    def neutral_q(self) -> np.ndarray: ...
    def integrate(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray: ...
    def forward_kinematics(self, q: np.ndarray) -> RobotKinematicState: ...
    def body_transform(self, state, body_name: str) -> np.ndarray: ...
    def site_transform(self, state, site: SemanticSite) -> np.ndarray: ...
    def relative_transform(self, state, reference: SemanticSite, target: SemanticSite) -> np.ndarray: ...
    def active_velocity_coordinates(self, reference: SemanticSite, target: SemanticSite) -> list[int]: ...
    def coordinate_limits(self) -> CoordinateLimits: ...
```

### 2.2 Perturb velocity coordinates, not quaternion components

Finite differences operate in tangent/velocity coordinates \(v\in\mathbb R^{n_v}\):

\[
q^+ = \operatorname{Integrate}(q,+\epsilon e_i),\qquad
q^- = \operatorname{Integrate}(q,-\epsilon e_i).
\]

Naively adding to quaternion q-position components is forbidden.

---

## 3. Semantic sites and frames

A semantic site is not just a body name:

```python
@dataclass(frozen=True)
class SemanticSite:
    semantic_name: str
    body_name: str
    local_position: np.ndarray
    local_rotation_xyzw: np.ndarray
    source: str
    confidence: float
```

Position-site usage is independent of orientation capability. A robot may have no wrist joint but still requires a distal hand site for position tracking.

### 3.1 Site inference order

1. explicit model site/frame;
2. distal child-joint anchor;
3. collision/visual primitive bounds;
4. mirrored site from a verified symmetric side;
5. segment extrapolation.

Each inferred site records source, confidence, hash lineage, and whether it is used by runtime or only by diagnostics.

### 3.2 Semantic rest frames

For each source and robot, build a right-handed rest frame for:

- pelvis;
- chest;
- upper arm;
- forearm;
- thigh;
- shin;
- sole.

A frame should use at least two independent geometric directions where possible.

Example pelvis frame:

\[
y = \operatorname{normalize}(p_{LHip}-p_{RHip}),
\]

\[
z' = p_{Chest}-p_{Hips},\qquad
z=\operatorname{normalize}(z'-y(y^Tz')),
\]

\[
x=\operatorname{normalize}(y\times z).
\]

If the source convention requires the opposite forward direction, apply one global, explicit convention transform and record it. Do not hide it in per-joint offsets.

### 3.3 Degeneracy handling

When two frame-defining directions are nearly parallel:

1. use another semantic direction;
2. use the previous valid frame for dynamic source motion;
3. use the rest frame;
4. reduce confidence and record the fallback.

No silent identity fallback is allowed.

---

## 4. Full-chain active degrees of freedom

The active coordinates for a relative transform are all velocity coordinates on the kinematic path between the reference and target sites.

For the normal humanoid ancestor case:

- `Chest → LeftHandSite` includes every shoulder, elbow, forearm and wrist coordinate;
- `Hips → LeftSoleSite` includes every hip, knee and ankle coordinate;
- `Hips → Chest` includes the full waist/torso chain.

The semantic parent of `Hand` is not the kinematic root for hand capability. Using `ForeArm → Hand` would discard shoulder and elbow control.

For the general case where neither site body is an ancestor of the other, find the lowest common ancestor. The relative transform is affected by coordinates on both branches from the LCA to the two sites. Common ancestors above the LCA cancel.

Floating-base coordinates are excluded from limb-relative capability but are included in world-root tasks.

---

## 5. Numerical relative Jacobians

Correctness takes priority over an analytic implementation in Step 2.

Let

\[
{}^A T_B(q)=
\begin{bmatrix}
R(q)&p(q)\\0&1
\end{bmatrix}.
\]

For active velocity coordinate \(i\), compute integrated states \(q^+\) and \(q^-\).

### 5.1 Translational Jacobian

The position is already expressed in reference frame \(A\):

\[
J_p[:,i]=\frac{p(q^+)-p(q^-)}{2\epsilon_i}.
\]

For scale-independent rank diagnostics use

\[
\bar J_p = J_p/L,
\]

where \(L\) is the full reference-to-site chain length or a robust characteristic length.

### 5.2 Rotational Jacobian

Let \(R^+=R(q^+)\), \(R^-=R(q^-)\). The spatial angular increment expressed in reference frame \(A\) is

\[
J_R[:,i]=\frac{\operatorname{Log}(R^+(R^-)^T)}{2\epsilon_i}.
\]

This is preferable to subtracting Euler angles or quaternion components.

### 5.3 Adaptive epsilon

For a scalar revolute coordinate:

\[
\epsilon_i=\operatorname{clip}(10^{-4}\max(1,\operatorname{range}_i),10^{-6},10^{-3})\;\text{rad}.
\]

For a prismatic coordinate use the analogous value in metres. Continuous coordinates use a fixed safe tangent increment.

Every Jacobian is recomputed with \(\epsilon/2\). If the relative difference exceeds tolerance, mark the column numerically unstable.

### 5.4 Engine Jacobian cross-check

Where the runtime engine exposes a site/body Jacobian, compare it with the numerical result. The finite-difference implementation remains the portable reference; engine Jacobians are an optimization and validation source.

---

## 6. Reachability is not one static projector

Three concepts must remain separate.

### 6.1 Local controllable rank

At configuration \(q_k\):

\[
r_p(q_k)=\operatorname{rank}(J_p(q_k)),\qquad
r_R(q_k)=\operatorname{rank}(J_R(q_k)).
\]

Use SVD threshold

\[
\sigma_j > \max(\tau_{abs},\tau_{rel}\sigma_1).
\]

Rank confidence also requires stability under epsilon changes.

### 6.2 Structural regular rank

Sample deterministic safe configurations and collect local ranks. The structural rank is the highest rank observed on a non-negligible fraction of samples, not the rank of a horizontally concatenated Jacobian.

Why: a one-DoF endpoint moving on a circle has local rank one, while tangents sampled around the circle may span a two-dimensional vector space. Concatenated-SVD rank would incorrectly claim two simultaneous controls.

Recommended rule:

```text
regular_rank = largest r with fraction(local_rank >= r) >= 0.20
nominal_rank = median(local_rank)
singularity_fraction = fraction(local_rank < regular_rank)
```

The exact fractions are testable constants, not per-robot tuning parameters.

### 6.3 Local conditioning

For runtime diagnostics:

\[
\kappa(q)=\sigma_1/\sigma_r.
\]

A chain can structurally support 3D hand position while being locally near-singular in a T-pose. Do not disable a task globally because one rest pose is singular.

### 6.4 Static linear projection is restricted

A fixed rest-pose basis may be used only for exact one-parameter or fixed-axis subspaces proven by the model. It must not replace nonlinear reachable-set projection for a general arm or leg.

---

## 7. Deterministic configuration sampling

For each chain:

1. neutral configuration;
2. individual ±10% safe-range perturbation for every active coordinate;
3. deterministic Halton or Sobol samples inside the central 60% of finite ranges;
4. mirrored samples for verified bilateral chains.

Do not sample near hard limits by default.

Step 2 minimum:

- neutral;
- 2 per-coordinate perturbations;
- 32 low-discrepancy chain samples.

The random seed, samples and model hash are saved in the report.

---

## 8. Rest-pose calibration without legacy offsets

The v3 path never reads optimized v1 scales or joint offsets.

For semantic edge \(e=(P,C)\), construct source and robot edge frames in their respective parent frames:

\[
{}^{P_h}R_{E_h,0},\qquad {}^{P_r}R_{E_r,0}.
\]

The source-parent-vector to robot-parent-vector alignment is

\[
A_e = {}^{P_r}R_{E_r,0}({}^{P_h}R_{E_h,0})^T.
\]

At motion frame \(t\), express the source direction in the moving source parent frame:

\[
u_h^{P_h}(t)=({}^W R_{P_h}(t))^T
\frac{{}^W p_{C_h}(t)-{}^W p_{P_h}(t)}
{\|{}^W p_{C_h}(t)-{}^W p_{P_h}(t)\|}.
\]

Map it into the desired robot parent frame:

\[
u_r^{P_r}(t)=A_e u_h^{P_h}(t).
\]

Then recursively construct the child target:

\[
{}^W p_{C_r}^*(t) = {}^W p_{P_r}^*(t)
+{}^W R_{P_r}^*(t)L_{e,r}u_r^{P_r}(t).
\]

This is frame-equivariant: a global rotation applied to the source does not alter local limb articulation.

### 8.1 Neutral exactness

For the source calibration pose, recursive targets must reproduce robot neutral semantic sites:

```text
max semantic position error < 1 mm
max relative rotation error < 0.01 rad
```

If this fails, dynamic retargeting is not allowed to start.

### 8.2 Symmetry

If left/right topology, ranges and rest geometry pass a symmetry test, use averaged segment lengths and mirrored frames. Otherwise preserve true asymmetry and report it.

---

## 9. Root translation and orientation decomposition

Let the source pelvis rest position be \(p_{h,0}\). Express horizontal displacement in the source rest pelvis frame:

\[
\Delta p_h^{local}(t)=({}^W R_{Hips,h,0})^T(p_h(t)-p_{h,0}).
\]

With leg-length scale

\[
s_L=L_{leg,r}/L_{leg,h},
\]

the robot horizontal root displacement is

\[
\Delta p_{r,xy}=s_L A_{root}\Delta p_{h,xy}^{local}.
\]

Vertical crouch is based on relative pelvis-to-support-foot height, not overall mesh height.

Root orientation is decomposed into:

- heading/yaw target;
- upright or limited tilt feasibility target;
- optional style tilt.

Floating-base roll/pitch is not forced to copy human pelvis roll/pitch as a high-priority task.

---

## 10. Exact relative torso target

Let source relative chest orientation be

\[
R_h^{rel}(t)=({}^W R_{Hips,h}(t))^T{}^W R_{Chest,h}(t).
\]

Rest delta:

\[
\Delta R_h(t)=(R_{h,0}^{rel})^T R_h^{rel}(t).
\]

Map source torso coordinates into robot pelvis coordinates with calibration rotation \(A_T\):

\[
\widehat{\Delta R}_r(t)=A_T\Delta R_h(t)A_T^T.
\]

The robot torso-chain forward map is

\[
F_T(q_T)=(R_{r,0}^{rel})^T R_r^{rel}(q_T).
\]

The closest reachable torso target is defined by the bounded problem

\[
q_T^*(t)=\arg\min_{q_T\in\mathcal Q_T}
\frac12\|\operatorname{Log}(F_T(q_T)^T\widehat{\Delta R}_r(t))\|^2
+\frac{\lambda}{2}\|D(q_T-q_T^*(t-1))\|^2.
\]

The emitted target is

\[
R_{r,target}^{rel}(t)=R_{r,0}^{rel}F_T(q_T^*(t)).
\]

### 10.1 Single hinge

For a one-axis waist this becomes projection onto a one-parameter subgroup, with joint limits. It is exact for that chain and does not assume the axis is world Z.

### 10.2 Zero-DoF torso

If the torso is welded to the pelvis, the target is the robot rest relative orientation and source torso demand is reported as unreachable style demand. No compensating hip/root rotation is introduced.

### 10.3 Multi-axis torso

For two or three serial axes, solve the small bounded problem. A fixed linear projector is only a local approximation and is not the primary target generator.

---

## 11. Endpoint target and nonlinear reachability projection

The recursive segment construction produces a morphology-scaled desired hand/foot target \(p_d\). It is geometrically consistent with robot segment lengths, but joint-axis constraints and limits can still make it unreachable.

For a deficient or questionable chain, define a chain-only bounded projection:

\[
q_C^*=\arg\min_{q_C\in\mathcal Q_C}
\frac12\left\|\frac{p_C(q_C)-p_d}{L_C}\right\|^2
+\frac{\lambda_n}{2}\|D(q_C-q_{neutral})\|^2
+\frac{\lambda_t}{2}\|D(q_C-q_{prev})\|^2.
\]

Optional orientation is added only when structurally supported.

The feasible site target is

\[
p_C^*=p_C(q_C^*).
\]

Report both:

- desired morphology target \(p_d\);
- projected feasible target \(p_C^*\);
- projection residual.

Do not project endpoint errors onto a rest-pose linear basis and claim that the remaining error is zero.

---

## 12. Capability-driven task selection

For every semantic task save:

- active velocity coordinates;
- regular and nominal local ranks;
- singularity fraction;
- condition statistics;
- site confidence;
- target-generation mode;
- disabled reason.

Default decisions:

### Torso orientation

- rank 0: fixed rest relative orientation;
- rank 1–3: nonlinear closest-reachable relative orientation.

### Hand position

- regular rank 3 with acceptable projection residual: full distal-site position;
- lower rank or frequent projection failure: chain-projected position;
- missing arm: task unavailable, not zero-error.

### Hand orientation

- determined by full chain rotational behavior at the distal site;
- not determined by whether a link name contains `wrist`;
- rank 0: disabled;
- partial rank: closest-reachable chain projection.

### Foot

- sole-site position always uses the inferred local offset;
- stance anchors are contact tasks;
- foot orientation is represented through sole anchors or closest-reachable chain projection.

---

## 13. Residual normalization contract for Step 3

Although Step 2 does not integrate whole-body IK, its outputs define the later residuals.

Position:

\[
e_p=\frac{p(q)-p^*}{\max(L,\epsilon)\sqrt 3}.
\]

Rotation:

\[
e_R=\frac{\operatorname{Log}((R^*)^TR(q))}{\sqrt 3}.
\]

Joint prior:

\[
e_q=D(q-q_0),\qquad D_{ii}=1/\max(q_i^{max}-q_i^{min},\epsilon).
\]

Weights are residual gains, so squared cost scales with their square. Extremely large values such as 500–1500 are not the default after normalization.

A `robust_loss` field may only exist when the runtime actually evaluates it.

---

## 14. Step-1 cross-model conclusions

The complete-model review establishes that the algorithm must support all of these cases with the same equations:

| Model family | Morphology stress | Required conclusion |
|---|---|---|
| RoboParty RPO | one-axis torso, reduced arm, no wrist-rich end chain | exact one-parameter torso projection; distal hand position independent of orientation support; full shoulder-to-hand chain |
| Unitree G1 29DoF | multi-axis waist and richer distal arm | same nonlinear torso formulation naturally expands; hand orientation may become available |
| Unitree G1 23DoF | reduced variant of the same family | capability comes from compiled coordinates, never model-name assumptions |
| Unitree H1 | reduced upper-body chain and narrow waist capability | one-axis/low-rank behavior is common, not an RPO exception |
| ROBOTIS OP3 | compact humanoid with little or no articulated waist depending semantic split | torso rank zero must be valid; source chest demand must not leak to legs |
| Booster T1 | different naming and axis conventions | topology/FK must dominate name heuristics |
| PAL TALOS | large multi-axis torso, arms and wrists | full relative transforms, compiled defaults and high-DoF chains must work without special cases |
| Berkeley Humanoid | lower-body-centric reduced humanoid | partial humanoid must downgrade semantics safely rather than fabricate hands |

Therefore Step 2 must use full robot models, not only synthetic fixtures. Synthetic fixtures prove individual invariants; complete robots prove that adapters, paths, neutral states and semantics compose correctly.

---

## 15. Rejected alternatives

### Rejected: one rest-pose SVD basis for every frame

It confuses singularity with structural capability and ignores nonlinear reachable manifolds.

### Rejected: concatenate Jacobians and use the union rank as controllable rank

A one-DoF curved trajectory can produce a multi-dimensional union of tangents while retaining only one instantaneous control.

### Rejected: map Hand relative to ForeArm for capability

It omits proximal controls and produced the v2 RPO rank error.

### Rejected: apply virtual hand offset only when orientation is supported

Position and orientation capabilities are independent.

### Rejected: project a world Chest quaternion

Torso motion is relative to pelvis, calibrated at rest, and constrained by the torso chain.

### Rejected: keep v1 quaternion offsets as initialization

They are not identifiable separately from target-frame calibration and may encode pose-pair overfitting.

### Rejected: solve all design uncertainty with task weights

Weights cannot make an impossible target physically reachable.

---

## 16. Step-2 implementation boundary

Step 2 implements only:

1. runtime model adapter;
2. semantic sites needed by the selected complete models;
3. full-chain numerical Jacobians;
4. multi-pose capability reports;
5. rest-frame calibration;
6. offset-free target builder;
7. torso and endpoint chain-only closest-reachable projection;
8. neutral and canonical-motion validation.

Step 2 does **not** replace the production Newton pipeline and does not tune whole-body IK weights. That integration is Step 3.

---

## 17. Primary references

- MuJoCo XML Reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- MuJoCo API Jacobian functions: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
- Modern Robotics, Space Jacobian: https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-1-1-space-jacobian/
- Modern Robotics, Exponential Coordinates: https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-3-exponential-coordinates-of-rigid-body-motion/
- MuJoCo Menagerie pinned model set: https://github.com/google-deepmind/mujoco_menagerie/tree/feadf76d42f8a2162426f7d226a3b539556b3bf5
- robot_descriptions 2.0.0 catalog: https://github.com/robot-descriptions/robot_descriptions.py/tree/v2.0.0
