# G1 Teacher-guided AutoConfig Refinement — Design Outline

> Target repo: `XinyuSong123/soma_retargeter_autoconfig`  
> Goal: improve retargeting quality for non-G1 robots, especially robots with fewer than 29 DoF, yaw-only waist, simplified ankles, or unstable foot details, while keeping the user-facing workflow almost unchanged.

---

## 0. One-sentence design

Use Unitree G1 29DoF as a **canonical teacher**, not as a joint-angle transfer target: generate a base config as before, run a teacher-guided refinement stage that adjusts tracking priorities, IK weights, foot anchors, and capability-aware fallbacks, then automatically accept the refined config only if it improves held-out validation metrics without hurting robot feasibility.

---

## 1. Current repo assumptions

The current project is already a good base for this feature because it has:

- SOMA/BVH input to robot CSV output.
- Robot registration through `params.py`.
- Retargeter configs under `soma_retargeter/configs/<robot>/`.
- Runtime config generation from minimal robot registration.
- Human-to-robot proportional scaling.
- Multi-objective Newton IK.
- Optional feet stabilization.
- Joint limit clamping.
- A config optimizer that can start from T-pose and later add paired poses.

Relevant current files:

```text
params.py
app/bvh_to_csv_converter.py
app/pose_optimizer_ui.py
app/optimize_scaler_config.py
soma_retargeter/robot_registry_parser.py
soma_retargeter/pipelines/newton_pipeline.py
soma_retargeter/robotics/human_to_robot_scaler.py
soma_retargeter/configs/
```

The feature should integrate into this existing structure instead of creating a separate retargeter.

---

## 2. Product rule

The user should not need to understand the teacher stage.

User-facing workflow should stay close to the current one:

```bash
python ./app/pose_optimizer_ui.py --viewer gl --robot my_robot
```

or:

```bash
python ./app/bvh_to_csv_converter.py --viewer null --robot my_robot
```

New optional flags can exist, but the default behavior should be simple:

```bash
python ./app/pose_optimizer_ui.py --viewer gl --robot my_robot --auto-refine
```

Recommended default in the long term:

```text
If robot_capability exists and teacher refinement resources are available, run teacher refinement automatically.
```

---

## 3. Do not implement naive G1 joint transfer

Do **not** do this:

```text
SOMA motion -> G1 joint angles -> target robot joint angles
```

Reason:

- Joint names are different.
- Joint axes may differ.
- Joint limits differ.
- Leg/torso/ankle structures differ.
- Low-DoF robots cannot reproduce G1 torso behavior.
- Copying G1 joint-level behavior can over-constrain the target robot and worsen feet.

Instead, do this:

```text
SOMA motion
    -> G1 teacher motion
    -> G1 FK semantic trajectories
    -> DoF-aware target projection
    -> target robot IK/config refinement
```

G1 should act as a **motion-quality teacher**, not as an intermediate joint format.

---

## 4. New high-level pipeline

```text
Input:
    robot URDF/MJCF
    current link mapping / retargeter config
    optional pose pairs
    optional robot capability profile

Stage 1: Base AutoConfig
    Generate the same base retargeter/scaler config as the current pipeline.

Stage 2: Capability Resolution
    Load robot_capability.yaml/json.
    Validate user-declared joints and links against URDF/MJCF/Newton model.
    Infer missing fields if marked as auto.
    Build a capability profile.

Stage 3: Virtual Sole Anchor Generation
    Generate soft foot anchors from collision mesh, visual mesh, bounding box, or user-provided foot size.
    Save generated anchor offsets to debug config.

Stage 4: Teacher Motion Generation
    Run calibration motions on G1 using the known-good G1 config.
    FK G1 motion to semantic link trajectories and contact states.

Stage 5: Target Motion Evaluation
    Run the same calibration motions on the target robot using base config.
    Compute validation metrics.

Stage 6: Teacher-guided Refinement
    Adjust IK weights, tracking priority, offset/scaler values, and foot-anchor weights.
    Respect capability-aware constraints.

Stage 7: Accept / Reject
    Evaluate base vs refined config on held-out validation motions.
    Accept refined config only if it passes hard gates and improves weighted score.

Output:
    soma_retargeter/configs/<robot>/soma_to_<robot>_retargeter_config.json
    debug/base_autoconfig.json
    debug/g1_teacher_refined.json
    debug/refine_report.json
```

---

## 5. Robot capability profile

### 5.1 Goal

Avoid fragile automatic search of every joint. Let users declare the important capability facts manually, but still validate them automatically.

### 5.2 Recommended file

```text
soma_retargeter/configs/<robot>/robot_capability.yaml
```

or JSON if the repo prefers JSON-only configs:

```text
soma_retargeter/configs/<robot>/robot_capability.json
```

YAML is more user-friendly, but JSON is easier to keep consistent with current config files.

### 5.3 Minimal schema

```yaml
schema_version: 1
robot: my_robot

capability_mode: user_declared_with_auto_validation

waist:
  yaw: waist_yaw_joint        # string / false / auto
  pitch: false                # string / false / auto
  roll: false                 # string / false / auto

ankle:
  left_pitch: left_ankle_pitch_joint
  left_roll: left_ankle_roll_joint
  right_pitch: right_ankle_pitch_joint
  right_roll: right_ankle_roll_joint

toe:
  left: false                 # string / false / auto
  right: false

links:
  pelvis: pelvis_link
  chest: torso_link           # optional; false if absent
  left_foot: left_foot_link
  right_foot: right_foot_link
  left_hand: left_hand_link   # optional
  right_hand: right_hand_link # optional

foot:
  sole_anchor_mode: auto      # auto / mesh / bbox / manual / disabled
  forward_axis: auto          # auto / +x / -x / +y / -y
  up_axis: auto               # auto / +z / -z
  manual_size:
    left:
      length: auto
      width: auto
      sole_z: auto
    right:
      length: auto
      width: auto
      sole_z: auto

tracking_policy:
  mode: auto                  # auto / full_body / low_dof_safe / foot_priority
```

### 5.4 Semantics

```text
string:
    User declares exact joint/link name. Must exist; otherwise error.

false:
    User declares this capability does not exist. Do not auto-search it.

auto:
    System tries to infer it from URDF/MJCF/Newton model.
    If multiple candidates are found, warn or ask user to fill explicitly.
```

### 5.5 Resolved internal profile

The loader should convert the user-facing file into a resolved object:

```json
{
  "robot": "my_robot",
  "dof_class": "low_dof",
  "waist_type": "yaw_only",
  "ankle_type": "pitch_roll",
  "toe_type": "none",
  "has_chest_link": true,
  "has_hand_links": true,
  "foot_anchor_mode": "bbox",
  "torso_tracking_mode": "yaw_only_or_weak",
  "priority_template": "low_dof_safe"
}
```

---

## 6. Capability-aware tracking policy

### 6.1 Normal 29DoF-like robot

```yaml
tracking_priority:
  feet_contact: high
  pelvis_position: high
  root_heading: high
  chest_orientation: medium_high
  hand_position: medium
  hand_orientation: medium_low
  foot_orientation: high
```

### 6.2 Yaw-only waist robot

```yaml
tracking_priority:
  feet_contact: very_high
  pelvis_position: high
  root_heading: high
  chest_yaw: medium
  chest_pitch_roll: low
  hand_position: medium
  hand_orientation: low
  foot_orientation: high
```

### 6.3 No waist robot

```yaml
tracking_priority:
  feet_contact: very_high
  pelvis_position: high
  root_heading: high
  chest_yaw: low_medium
  chest_pitch_roll: very_low
  hand_position: medium_low
  hand_orientation: low
  foot_orientation: high
```

### 6.4 Simplified ankle robot

```yaml
tracking_priority:
  feet_contact: high
  foot_position: high
  foot_roll_pitch: medium_or_low
  toe_heel_anchor: medium
  pelvis_position: high
  joint_limit_margin: high
```

---

## 7. Virtual sole anchors

### 7.1 Naming

Prefer the name:

```text
virtual sole anchors
```

Do not call them real toes. They are soft reference points used for IK/contact/stability evaluation.

### 7.2 Anchors

For each foot:

```text
sole_center
heel
toe
inner_edge
outer_edge
```

Optional:

```text
toe_inner
toe_outer
heel_inner
heel_outer
```

### 7.3 Generation strategy

Fallback order:

```text
1. collision mesh support polygon
2. visual mesh bounding box
3. primitive geometry bbox
4. user-provided foot length/width/sole_z
5. disabled, use foot link only
```

### 7.4 Recommended generated config

```json
{
  "virtual_sole_anchors": {
    "enabled": true,
    "source": "bbox",
    "left": {
      "sole_center": [0.02, 0.00, -0.035],
      "toe": [0.10, 0.00, -0.035],
      "heel": [-0.07, 0.00, -0.035],
      "inner_edge": [0.02, -0.035, -0.035],
      "outer_edge": [0.02, 0.035, -0.035]
    },
    "right": {
      "sole_center": [0.02, 0.00, -0.035],
      "toe": [0.10, 0.00, -0.035],
      "heel": [-0.07, 0.00, -0.035],
      "inner_edge": [0.02, -0.035, -0.035],
      "outer_edge": [0.02, 0.035, -0.035]
    }
  }
}
```

### 7.5 Important rule

Virtual sole anchors are **soft constraints**, not hard constraints.

Recommended dynamic weights:

```text
stance phase:
    toe/heel/sole anchors: high
    foot orientation: high
    horizontal slip: high penalty

swing phase:
    toe/heel/sole anchors: low
    foot link position: medium/high
    toe clearance: medium
```

---

## 8. G1 teacher data

### 8.1 Teacher output should be semantic, not joint-level

Store the G1 teacher output as:

```text
pelvis trajectory
chest/torso trajectory
left foot trajectory
right foot trajectory
left hand trajectory
right hand trajectory
optional knee/elbow hint points
foot contact states
foot velocity
foot height
root heading
```

### 8.2 Recommended data structure

```json
{
  "motion_name": "walk_forward",
  "fps": 50,
  "frames": [
    {
      "t": 0.0,
      "links": {
        "pelvis": {"pos": [0,0,0], "quat_xyzw": [0,0,0,1]},
        "chest": {"pos": [0,0,0], "quat_xyzw": [0,0,0,1]},
        "left_foot": {"pos": [0,0,0], "quat_xyzw": [0,0,0,1]},
        "right_foot": {"pos": [0,0,0], "quat_xyzw": [0,0,0,1]}
      },
      "contacts": {
        "left_foot": true,
        "right_foot": false
      },
      "sole_anchors": {
        "left_foot": {
          "toe": [0,0,0],
          "heel": [0,0,0],
          "sole_center": [0,0,0]
        }
      }
    }
  ]
}
```

---

## 9. Calibration motion sets

Use three splits.

### 9.1 Train set

Used for refinement.

```text
walk_forward
walk_turn
small_squat
arm_swing
```

### 9.2 Validation set

Used to decide whether refined config replaces base config.

```text
side_step
deep_squat
foot_lift
turn_in_place
```

### 9.3 Stress set

Used only for report/debug, not for automatic accept/reject.

```text
high_knee
bend_down
kick
fast_walk
large_arm_motion
```

### 9.4 Why split motions

Do not accept refined config based on the same motions used to optimize it. Otherwise the config may overfit a small calibration set.

---

## 10. Refinement targets

The teacher refinement stage may adjust:

```text
IK tracking weights
body-part priority weights
foot anchor weights
stance/swing foot weights
chest orientation weights
hand orientation weights
joint scale values, if safe
joint offset translation, if safe
joint offset rotation, if safe
feet stabilization parameters
```

It should not blindly change:

```text
robot joint limits
URDF/MJCF structure
robot link names
physical geometry
```

---

## 11. Refinement loss

Recommended conceptual loss:

```text
L_total =
    w_foot     * L_foot
  + w_contact  * L_contact
  + w_body     * L_teacher_link
  + w_smooth   * L_smooth
  + w_limit    * L_joint_limit
  + w_reg      * L_config_regularization
```

### 11.1 Foot loss

```text
L_foot =
    foot link position error
  + foot orientation error
  + toe/heel/sole anchor error during stance
```

### 11.2 Contact loss

```text
L_contact =
    stance foot horizontal slip
  + stance foot floating
  + foot penetration
```

### 11.3 Teacher link loss

```text
L_teacher_link =
    pelvis/chest/hands/feet semantic trajectory error
```

Use capability-aware weights. For yaw-only waist robots, do not heavily penalize chest pitch/roll mismatch.

### 11.4 Smoothness loss

```text
L_smooth =
    joint velocity penalty
  + joint acceleration penalty
  + optional jerk penalty
```

### 11.5 Joint limit loss

```text
L_joint_limit =
    limit violation penalty
  + near-limit usage penalty
```

### 11.6 Config regularization

```text
L_config_regularization =
    deviation from base config
```

Do not allow the optimizer to destroy the base config to chase a few teacher frames.

---

## 12. Evaluation metrics

### 12.1 Hard gates

Refined config must be rejected if any of these become worse beyond tolerance:

```text
joint limit violation
IK fail frames
foot penetration
stance foot slip
root/pelvis drift
severe jitter
```

### 12.2 Weighted score

Only after hard gates pass:

```text
score =
    0.35 * foot_score
  + 0.20 * contact_score
  + 0.15 * body_teacher_score
  + 0.10 * smoothness_score
  + 0.10 * joint_limit_score
  + 0.10 * robot_feasibility_score
```

### 12.3 Acceptance rule

Recommended first version:

```python
accept = (
    refined.hard_gate_passed
    and refined.total_score > base.total_score * 1.05
    and refined.foot_score > base.foot_score * 1.08
    and refined.foot_slip <= base.foot_slip * 1.02
    and refined.joint_limit_violation <= base.joint_limit_violation * 1.02
)
```

### 12.4 Report

Save:

```text
soma_retargeter/configs/<robot>/debug/refine_report.json
```

Example:

```json
{
  "decision": "use_teacher_refined",
  "reason": "refined improves validation score by 12.4% and reduces foot slip by 31.8%",
  "base_score": 0.742,
  "refined_score": 0.834,
  "improvement": 0.124,
  "metrics": {
    "foot_slip_m": {"base": 0.043, "refined": 0.029},
    "foot_penetration_m": {"base": 0.012, "refined": 0.006},
    "joint_limit_usage": {"base": 0.81, "refined": 0.74},
    "smoothness": {"base": 0.66, "refined": 0.71},
    "teacher_link_error_m": {"base": 0.094, "refined": 0.078}
  }
}
```

---

## 13. File structure proposal

```text
soma_retargeter/
  teacher_refinement/
    __init__.py
    capability_schema.py
    capability_loader.py
    capability_resolver.py
    sole_anchor_generator.py
    teacher_motion_builder.py
    contact_detector.py
    metrics.py
    evaluator.py
    refiner.py
    accept_reject.py
    report.py

app/
  teacher_refine_config.py
```

Optional CLI:

```bash
python ./app/teacher_refine_config.py --robot my_robot --teacher unitree_g1
```

Integration CLI:

```bash
python ./app/pose_optimizer_ui.py --robot my_robot --auto-refine
python ./app/bvh_to_csv_converter.py --robot my_robot --auto-refine
```

---

## 14. Implementation order for Codex

### Phase 1: Config and capability only

- Add `robot_capability.json` schema.
- Add loader and validator.
- Validate declared joint/link names.
- Generate resolved capability profile.
- Print readable warnings.

Deliverable:

```text
resolved_capability.json
```

### Phase 2: Virtual sole anchors

- Parse foot link geometry.
- Generate bbox-based anchors.
- Add manual fallback.
- Save generated anchors into debug config.

Deliverable:

```text
generated_sole_anchors.json
```

### Phase 3: Metrics evaluator

- Run base config on validation motions.
- Compute foot slip, penetration, floating, smoothness, joint limit usage, teacher link error if teacher data exists.
- Save `base_eval_report.json`.

Deliverable:

```text
metrics.py
evaluator.py
```

### Phase 4: Teacher motion builder

- Run G1 config on calibration motions.
- FK to semantic link trajectories.
- Detect contact states.
- Save teacher trajectories.

Deliverable:

```text
teacher_cache/<motion>.json
```

### Phase 5: Refinement

- Adjust tracking weights and foot weights first.
- Do not optimize too many variables initially.
- Keep scale/offset refinement as a later optional step.

Deliverable:

```text
g1_teacher_refined.json
```

### Phase 6: Accept/reject

- Compare base and refined configs on validation motions.
- If accepted, write final config to existing expected path.
- Save base and refined under debug.

Deliverable:

```text
retargeter_config.json
refine_report.json
```

---

## 15. Minimum viable version

For the first implementation, do not attempt full optimizer complexity.

MVP should implement:

```text
1. robot_capability profile
2. bbox virtual sole anchors
3. capability-aware tracking weight templates
4. base vs refined validation metrics
5. accept/reject report
```

MVP refinement can simply choose between preset templates:

```text
normal_29dof_template
low_dof_yaw_only_waist_template
no_waist_template
simplified_ankle_template
foot_priority_template
```

This avoids unstable continuous optimization in the first version.

---

## 16. Main risks

### 16.1 Bad geometry in URDF/MJCF

Mitigation:

- Use manual foot size fallback.
- Save generated anchors visually/debuggably.
- Keep anchors soft.

### 16.2 G1 teacher bias

Mitigation:

- Compare on validation motions.
- Use capability-aware projection.
- Reject if feasibility gets worse.

### 16.3 Low-DoF overconstraint

Mitigation:

- Lower chest pitch/roll tracking for yaw-only waist.
- Prioritize feet, pelvis, root heading.

### 16.4 Overfitting calibration motions

Mitigation:

- Split train/validation/stress motions.
- Accept only based on validation.

---

## 17. Success definition

The feature is successful if:

```text
For non-G1 robots, especially low-DoF robots, the final config reduces foot slip, foot penetration/floating, and joint-limit abuse on held-out motions, without requiring the user to manually tune many weights.
```

User-facing promise:

```text
Users still only bind key links/joints. The system internally decides what should be tracked strongly, what should be weakened, and whether G1 teacher refinement improves the final config.
```

---

## 18. Recommended README wording

```text
For robots with different DoF layouts, especially yaw-only waist or simplified ankles, the tool can optionally use Unitree G1 as a canonical teacher. The teacher stage does not copy G1 joint angles. Instead, it extracts semantic link trajectories, foot contacts, and foot stability cues from G1, then uses them to refine the target robot's IK weights, virtual sole anchors, and tracking priorities. The refined config is accepted only when it improves held-out validation metrics without increasing joint-limit violations, foot sliding, or penetration.
```

---

## 19. Sources checked while drafting

- Project README: https://github.com/XinyuSong123/soma_retargeter_autoconfig
- `params.py`: https://raw.githubusercontent.com/XinyuSong123/soma_retargeter_autoconfig/main/params.py
- `app/optimize_scaler_config.py`: https://raw.githubusercontent.com/XinyuSong123/soma_retargeter_autoconfig/main/app/optimize_scaler_config.py
