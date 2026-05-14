# G1 Teacher-guided AutoConfig Refinement

This document describes a proposed extension for `soma_retargeter_autoconfig`.

The goal is to improve retargeting quality when transferring SOMA motions to robots that are not Unitree G1 29DoF, especially robots with fewer DoFs, yaw-only waist, simplified ankles, missing toe joints, or unstable foot details.

The key idea is simple:

> Use G1 as a canonical motion-quality teacher, not as an intermediate joint-angle format.

The refined config should replace the base config only when objective validation metrics show that it is better.

---

## Why this feature is needed

The current retargeter works well when the target robot has a structure close to the expected full-body humanoid setup. However, users may report degraded results when:

- the robot has fewer than 29 DoF,
- the waist only has yaw,
- the robot has no waist pitch/roll,
- the foot or ankle structure is simpler,
- the foot link is too coarse,
- toe/heel details are not represented,
- IK targets over-constrain the robot and force compensation through the legs or feet.

A good G1 result already contains useful robot-like motion priors:

- stable foot contact timing,
- feasible humanoid pelvis motion,
- cleaner swing/stance foot behavior,
- robot-like whole-body coordination,
- smoother trajectories than raw human skeleton targets.

However, directly copying G1 joint angles to another robot is not robust. The correct use of G1 is to extract semantic trajectories and contact cues, then adapt them to the target robot according to the target robot's capability.

---

## What this feature does

The proposed feature adds an internal refinement stage:

```text
SOMA motion
    -> base target robot config
    -> G1 teacher motion
    -> semantic teacher trajectories and contact states
    -> capability-aware target projection
    -> refined target robot config
    -> validation-based accept/reject
```

If the refined config improves held-out validation metrics, it becomes the final config. Otherwise, the system keeps the original base config.

---

## What this feature does not do

It does **not** do this:

```text
G1 joint angle -> target robot joint angle
```

This is intentionally avoided because:

- joint names differ,
- joint axes differ,
- joint limits differ,
- robot proportions differ,
- low-DoF robots cannot reproduce all G1 torso/ankle behavior,
- joint-level transfer can make feet worse.

---

## User-facing workflow

The user should still only provide simple robot information:

1. robot URDF/MJCF path,
2. retargeter link mapping,
3. T-pose or pose pairs,
4. optional robot capability profile.

Example command:

```bash
python ./app/pose_optimizer_ui.py --viewer gl --robot my_robot --auto-refine
```

Batch mode:

```bash
python ./app/bvh_to_csv_converter.py --viewer null --robot my_robot --auto-refine
```

Optional standalone command:

```bash
python ./app/teacher_refine_config.py --robot my_robot --teacher unitree_g1
```

---

## Robot capability profile

A capability profile lets the user declare important robot structure without forcing the program to guess every joint from names.

Recommended path:

```text
soma_retargeter/configs/<robot>/robot_capability.yaml
```

Example:

```yaml
schema_version: 1
robot: my_robot

waist:
  yaw: waist_yaw_joint
  pitch: false
  roll: false

ankle:
  left_pitch: left_ankle_pitch_joint
  left_roll: left_ankle_roll_joint
  right_pitch: right_ankle_pitch_joint
  right_roll: right_ankle_roll_joint

toe:
  left: false
  right: false

links:
  pelvis: pelvis_link
  chest: torso_link
  left_foot: left_foot_link
  right_foot: right_foot_link
  left_hand: left_hand_link
  right_hand: right_hand_link

foot:
  sole_anchor_mode: auto
  forward_axis: auto
  up_axis: auto
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
  mode: auto
```

Field semantics:

```text
string:
    Use the exact declared joint/link name. Validate it exists.

false:
    The robot does not have this capability. Do not auto-search it.

auto:
    Try to infer it from URDF/MJCF/Newton model. Warn if ambiguous.
```

The system resolves this into an internal profile:

```json
{
  "dof_class": "low_dof",
  "waist_type": "yaw_only",
  "ankle_type": "pitch_roll",
  "toe_type": "none",
  "foot_anchor_mode": "bbox",
  "tracking_template": "low_dof_safe"
}
```

---

## Capability-aware tracking templates

### Full humanoid / G1-like robot

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

### Yaw-only waist robot

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

### No-waist robot

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

### Simplified ankle robot

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

## Virtual sole anchors

The feature may generate virtual foot anchors to improve foot stability.

Use the term:

```text
virtual sole anchors
```

not real toes.

Generated anchors:

```text
sole_center
heel
toe
inner_edge
outer_edge
```

Generation fallback order:

```text
1. collision mesh support polygon
2. visual mesh bounding box
3. primitive geometry bounding box
4. user-provided foot length/width/sole_z
5. disabled, use foot link only
```

Example generated config:

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

Important rule:

> Virtual sole anchors must be soft constraints, not hard constraints.

During stance phase, anchor weights should be high. During swing phase, anchor weights should be lower.

---

## Teacher data format

The G1 teacher output should store semantic trajectories, not joint angles.

Recommended data:

```text
pelvis pose
chest/torso pose
left foot pose
right foot pose
left hand pose
right hand pose
optional knee/elbow hint points
left/right foot contact states
left/right foot velocity
left/right foot height
root heading
virtual sole anchor world positions
```

Example:

```json
{
  "motion_name": "walk_forward",
  "fps": 50,
  "frames": [
    {
      "t": 0.0,
      "links": {
        "pelvis": {"pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
        "chest": {"pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
        "left_foot": {"pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
        "right_foot": {"pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]}
      },
      "contacts": {
        "left_foot": true,
        "right_foot": false
      }
    }
  ]
}
```

---

## Calibration motions

Use three motion splits.

### Train motions

Used to tune/refine config.

```text
walk_forward
walk_turn
small_squat
arm_swing
```

### Validation motions

Used to decide whether refined config should replace base config.

```text
side_step
deep_squat
foot_lift
turn_in_place
```

### Stress motions

Used only for report/debug.

```text
high_knee
bend_down
kick
fast_walk
large_arm_motion
```

Do not accept the refined config based only on the motions used for refinement.

---

## Evaluation metrics

### Hard gates

Reject refined config if it makes these worse beyond tolerance:

```text
joint limit violation
IK fail frames
foot penetration
stance foot slip
root/pelvis drift
severe jitter
```

### Weighted score

After hard gates pass:

```text
score =
    0.35 * foot_score
  + 0.20 * contact_score
  + 0.15 * body_teacher_score
  + 0.10 * smoothness_score
  + 0.10 * joint_limit_score
  + 0.10 * robot_feasibility_score
```

### Acceptance rule

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

If accepted:

```text
write refined config to final retargeter_config path
```

If rejected:

```text
keep base config as final retargeter_config path
```

---

## Output files

Recommended output structure:

```text
soma_retargeter/configs/<robot>/
  soma_to_<robot>_retargeter_config.json       # final config used by normal pipeline
  robot_capability.yaml                        # optional user-edited profile
  debug/
    base_autoconfig.json
    g1_teacher_refined.json
    resolved_capability.json
    generated_sole_anchors.json
    base_eval_report.json
    refined_eval_report.json
    refine_report.json
```

The user should only need the final config. Debug files exist for developers and issue reports.

---

## Proposed modules

```text
soma_retargeter/teacher_refinement/
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

---

## Recommended implementation phases

### Phase 1: Capability profile

Implement:

- schema,
- loader,
- validation,
- resolved capability export,
- warnings for missing/ambiguous joints.

### Phase 2: Virtual sole anchors

Implement:

- bbox-based anchor generation,
- manual fallback,
- disabled fallback,
- debug export.

### Phase 3: Metrics evaluator

Implement:

- stance foot slip,
- foot penetration,
- foot floating,
- joint limit usage,
- smoothness,
- teacher link error.

### Phase 4: G1 teacher cache

Implement:

- run G1 on calibration motions,
- FK semantic links,
- contact detection,
- teacher JSON cache.

### Phase 5: Template-based refinement MVP

Start with template-based weight adjustment, not full continuous optimization.

Templates:

```text
normal_29dof_template
low_dof_yaw_only_waist_template
no_waist_template
simplified_ankle_template
foot_priority_template
```

### Phase 6: Accept/reject and final overwrite

Implement:

- base/refined validation comparison,
- hard gate,
- weighted score,
- final config overwrite if accepted,
- debug report.

---

## Minimal viable product

The MVP should include:

```text
1. robot_capability profile
2. bbox virtual sole anchors
3. capability-aware tracking templates
4. validation metrics
5. accept/reject report
6. final config overwrite if accepted
```

Do not start with a complex gradient optimizer for all weights. First make the acceptance/evaluation system reliable.

---

## Final behavior

The final behavior should be:

```text
1. Generate base config.
2. Resolve robot capability.
3. Generate soft sole anchors.
4. Apply capability-aware tracking template.
5. Run G1 teacher-guided refinement.
6. Evaluate base vs refined on held-out motions.
7. If refined improves quality and does not hurt feasibility, overwrite final config.
8. Otherwise keep base config.
```

User-facing summary:

```text
Users still only bind key links/joints. The system internally decides what should be tracked strongly, what should be weakened, how the feet should be stabilized, and whether the G1 teacher-refined config is actually better.
```

---

## Notes for Codex

When implementing, preserve existing behavior unless `--auto-refine` is enabled or the project later decides to enable it by default.

Do not break current robot registration in `params.py`.

Do not require all users to create `robot_capability.yaml`. Missing capability profile should fall back to current behavior or a conservative auto mode.

Keep all generated debug files under `soma_retargeter/configs/<robot>/debug/`.

The first implementation should prefer simple, testable modules over a monolithic optimizer.
