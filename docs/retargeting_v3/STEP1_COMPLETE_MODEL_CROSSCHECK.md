# Step 1 — Complete Robot Model Cross-Check

Status: **design review completed; executable assertions assigned to Step 2**

The mathematical specification was checked against complete robot families, not only synthetic chains. The purpose is not to hard-code each robot's expected rank. The purpose is to identify morphology cases that one universal implementation must represent correctly.

The model files are consumed at fixed upstream versions through the Robot Zoo manifest. Step 2 must record the actual result from the compiled runtime model and compare it with the qualitative expectations below.

---

## 1. Mandatory complete-model set for Step 2

| ID | Source | Role |
|---|---|---|
| `roboparty_rpo_local` | repository MJCF | primary single-axis torso/no-wrist regression |
| `unitree_g1_mjcf` | pinned Menagerie/robot_descriptions | rich humanoid baseline |
| `unitree_g1_urdf` | pinned robot_descriptions | URDF/MJCF equivalence check |
| `unitree_g1_23dof` | existing Newton asset or fixed snapshot | reduced capability variant |
| `unitree_h1_mjcf` | pinned model | independent low-DoF upper-body case |
| `robotis_op3_mjcf` | pinned Menagerie model | compact/fixed-torso case |
| `booster_t1_mjcf` | pinned Menagerie model | different naming and axis layout |
| `pal_talos_mjcf_direct` | pinned Menagerie model | high-DoF, multi-axis torso and arm case |
| `berkeley_humanoid_mjcf_direct` | pinned Menagerie model | partial/reduced humanoid capability case |

The first six are hard gates. TALOS and Berkeley are required compile/capability gates even if full canonical motion projection is slower.

---

## 2. RoboParty RPO

Known complete-model characteristics from the repository model and existing configurations:

- floating root;
- one articulated torso hinge;
- shoulder and elbow chain without a conventional wrist-rich distal chain;
- six-coordinate leg chains ending at ankle/foot links;
- existing human Hand mapping historically points to a distal elbow/forearm body.

Step-2 assertions:

1. `Hips → Chest` active chain contains exactly the true torso coordinate(s), not root coordinates.
2. torso regular rotational rank is one.
3. the recovered axis is expressed in the pelvis reference frame and is invariant to a global root rotation.
4. `Chest → HandSite` contains all proximal shoulder/elbow coordinates on that side.
5. hand position rank is computed from the full chain, not from the final link alone.
6. the distal hand site local offset is used for position calibration even when hand orientation tracking is unavailable.
7. `Hips → SoleSite` includes hip, knee and ankle coordinates.
8. source pure chest pitch/roll projects to negligible torso-hinge response; source twist around the actual hinge axis produces a response.

Any RPO report that lists only `elbow_yaw` as the hand position chain is a hard failure.

---

## 3. Unitree G1 29DoF

This is the rich-humanoid baseline.

Step-2 assertions:

1. torso-chain active coordinates are discovered from the compiled model, not a hard-coded DoF count.
2. the torso relative orientation regular rank is greater than the single-axis RPO case.
3. hand chains include shoulder, elbow and available distal arm/wrist coordinates.
4. leg chains include the full pelvis-to-sole path.
5. neutral semantic reconstruction passes position and relative-rotation gates.
6. the URDF and MJCF descriptions produce semantically equivalent chain topology and capability within documented conversion differences.
7. no `robot_type == unitree_g1` branch is present in core mathematics.

---

## 4. Unitree G1 23DoF

The reduced G1 model verifies that family names cannot define capability.

Step-2 assertions:

1. the actual missing/locked coordinates are identified from the compiled model.
2. capability differs from the 29DoF result only where the actual model differs.
3. the same source semantic map can be compiled with capability-aware task decisions.
4. no G1-23-specific weight or axis table is introduced.

If a real 23DoF model is unavailable in the local environment, a fixed upstream model is required. Locking coordinates in the 29DoF model is permitted only as an additional controlled ablation, not as the sole validation.

---

## 5. Unitree H1

H1 is an independent test of reduced upper-body and torso capability.

Step-2 assertions:

1. the adapter discovers the true waist chain and arm chain without borrowing G1 assumptions.
2. low torso rank is handled by the same closest-reachable relative-orientation formulation as RPO.
3. distal hand/arm site position remains independent of orientation support.
4. no wrist-name heuristic determines rotational capability.

---

## 6. ROBOTIS OP3

OP3 tests a compact morphology where the chest may be effectively fixed relative to the pelvis under the selected semantics.

Step-2 assertions:

1. a welded or zero-active-coordinate `Hips → Chest` chain is valid.
2. torso rotational rank zero does not cause an exception.
3. the emitted torso target remains at robot rest relative orientation.
4. source torso demand is reported as unreachable demand, not converted into root/leg target motion.
5. compact limb geometry does not break frame construction.

---

## 7. Booster T1

Booster T1 tests different body/joint naming and axis conventions.

Step-2 assertions:

1. active paths come from topology.
2. axes come from runtime finite differences or runtime screw data.
3. semantic name scoring can propose sites, but capability remains independent of names.
4. neutral reconstruction and chain reports pass without a Booster-specific core branch.

---

## 8. PAL TALOS

TALOS stresses high DoF, multi-axis torso/arms and a larger compiled model.

Step-2 assertions:

1. model loading resolves all includes/defaults through the runtime loader.
2. the runtime adapter's body/joint counts and neutral FK are stable.
3. multi-axis torso projection solves the nonlinear bounded closest-orientation problem.
4. high-DoF hand chains do not cause rank overestimation through concatenated Jacobians.
5. numerical Jacobian cost is reported and bounded for offline compilation.

TALOS is not a performance gate for real-time Step 2; it is a correctness and scalability gate.

---

## 9. Berkeley Humanoid

This model tests partial-humanoid semantics.

Step-2 assertions:

1. missing or inadequate arm semantics produce a structured capability status.
2. the compiler does not fabricate left/right hand tasks merely because the model is categorized as humanoid.
3. available pelvis/leg/sole semantics still compile and validate.
4. the report distinguishes `partial_humanoid` from `full_humanoid_ready`.

---

## 10. Cross-format equivalence

For robot families with both URDF and MJCF descriptions, compare:

- semantic body graph;
- active chain coordinates;
- joint types and finite limits;
- neutral semantic site positions after frame alignment;
- regular local ranks;
- segment lengths;
- closest-reachable canonical targets.

The files need not have identical body names or inertial parameters. The semantic kinematic result should agree within explicit tolerances.

Minimum pairs:

- G1 URDF vs G1 MJCF;
- H1 URDF vs H1 MJCF when both fixed sources are available;
- Booster T1 URDF vs MJCF when both fixed sources are available.

---

## 11. Required report per complete model

```json
{
  "model": {},
  "runtime_adapter": {},
  "semantic_sites": {},
  "neutral_fk": {},
  "chains": {
    "torso": {},
    "left_hand": {},
    "right_hand": {},
    "left_foot": {},
    "right_foot": {}
  },
  "sampling": {},
  "rank_stability": {},
  "rest_calibration": {},
  "canonical_targets": {},
  "failures": [],
  "reproduction_command": "..."
}
```

For every chain, save:

- reference and target site;
- LCA and body path;
- active velocity-coordinate labels;
- joint types;
- neutral Jacobians;
- per-sample singular values/ranks;
- regular rank, nominal rank and singularity fraction;
- epsilon-stability diagnostics;
- characteristic length;
- site provenance;
- target projection residuals.

---

## 12. What the complete-model review proves

The review supports these design decisions:

1. single-axis waists are not exceptional;
2. zero-axis torso chains must also be first-class;
3. hand capability cannot be inferred from the final joint or the word `wrist`;
4. endpoint capability requires the entire relative kinematic path;
5. static rest-pose rank is insufficient;
6. cross-format conversion must be validated semantically;
7. partial humanoids require graceful downgrade;
8. model-specific weights cannot substitute for correct target construction.

It does **not** prove runtime whole-body IK quality. That is intentionally deferred until the offline kinematic compiler passes all Step-2 gates.
