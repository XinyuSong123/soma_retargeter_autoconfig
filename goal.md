# Goal — Step 3.1：Full-Fleet Runtime Quality Evaluation & Runtime-Local Profile Closure

> **Codex 强制执行契约——严禁自由发挥**
>
> 当前分支：`retargeting-v3-step3-runtime-quality-eval`  
> 基线分支：`retargeting-v3-step3-runtime-profile-shadow`  
> 基线提交：`1b0d223841909bec3fb9b850dee5c4fad1956be1`  
> 当前阶段：**Step 3.1 Full-Fleet Runtime Quality Evaluation**
>
> 用户明确要求：**不要只做 RPO/G1，要全量。**  
> 因此本轮必须让 **44 个 in-scope Robot Zoo / local assets 全部进入 runtime evaluation matrix**。RPO/G1 只作为 pipeline-backed 对照，不再是唯一范围。
>
> 本轮不是“调参让视频好看”。本轮只做：
>
> 1. 对 44 个 in-scope 模型建立完整 runtime eligibility / profile resolution / target-stream / smoke-evaluation 矩阵；
> 2. 为所有 fingerprint mismatch 或 runtime source mismatch 的模型生成 **runtime-local V3 profile**，不得静默跳过；
> 3. 对所有 full humanoid profile 运行 generic runtime V3 target + IK/solver smoke；
> 4. 对 partial humanoid 运行 supported semantics smoke，不伪造手部能力；
> 5. 对 negative controls 验证它们不会被提升成 humanoid runtime profile；
> 6. 建立质量指标体系和 artifacts，为下一轮真正优化提供依据。
>
> **禁止重写生产 retargeter，禁止调 IK/objective 权重，禁止 contact/collision/temporal smoothing，禁止修改 Step 2.1/2.3 thresholds，禁止按机器人名称加数学特例，禁止宣称最终动作效果已经解决。**

---

## 0. 开始前必须执行

```bash
conda activate soma-retargeter-v2

git fetch origin --prune
git checkout retargeting-v3-step3-runtime-quality-eval
git pull --ff-only

git status --short
git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 1b0d223841909bec3fb9b850dee5c4fad1956be1 HEAD

git lfs install
git lfs pull
git lfs fsck

which python
python --version
python - <<'PY'
import mujoco, newton, warp, numpy, scipy
print('mujoco', mujoco.__version__)
print('newton', getattr(newton, '__version__', 'unknown'))
print('warp', getattr(warp, '__version__', 'unknown'))
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
PY
```

必须确认：

```text
branch = retargeting-v3-step3-runtime-quality-eval
worktree clean
base commit is an ancestor
Git LFS fsck OK
snapshot XML materialized, not pointer-only
```

环境不满足时，只修环境或 checkout，不得修改算法绕过。

---

## 1. 必须先阅读的文件

按顺序完整阅读：

1. `/goal.md`（本文件）
2. `docs/retargeting_v3/STEP2_3_1_CAPABILITY_ACCEPTANCE.md`
3. `docs/retargeting_v3/subagents/capability_agent_f_red_team.md`
4. `docs/retargeting_v3/STEP3_RUNTIME_SHADOW_ACCEPTANCE.md`
5. `docs/retargeting_v3/subagents/step3_agent_f_red_team.md`
6. `artifacts/retargeting_v3_step2_capability/summary.json`
7. `artifacts/retargeting_v3_step2_capability/before_after.json`
8. `artifacts/retargeting_v3_step2_capability/environment.json`
9. `artifacts/retargeting_v3_step2_capability/lfs_state.json`
10. `artifacts/retargeting_v3_step3_runtime_shadow/acceptance_ledger.json`
11. `artifacts/retargeting_v3_step3_runtime_shadow/smoke_matrix.json`
12. `artifacts/retargeting_v3_step3_runtime_shadow/override_smoke_summary.json`
13. `soma_retargeter/pipelines/newton_pipeline.py`
14. `soma_retargeter/pipelines/utils.py`
15. `soma_retargeter/runtime/v3/profile_loader.py`
16. `soma_retargeter/runtime/v3/source_frames.py`
17. `soma_retargeter/runtime/v3/target_adapter.py`
18. `soma_retargeter/runtime/v3/override.py`
19. `soma_retargeter/robotics/v3/profile.py`
20. `soma_retargeter/robotics/v3/validation.py`
21. `soma_retargeter/robotics/v3/capability_status.py`
22. `assets/robot_zoo/robot_zoo_lock.json`
23. `assets/robot_zoo/robot_zoo_manifest.json`
24. all current Step 3.0 runtime tests under `tests/v3/test_runtime_*` and `tests/v3/test_step3_runtime_*`

不得只读 acceptance summary 后直接扩展 pipeline。

---

## 2. 当前事实基线

### 2.1 Step 2.3.1 离线能力基线

当前可信离线 profile/certificate 结果：

```text
44 in-scope assets
source/load/semantic failure = 0
passed = 32
partial_passed = 3
negative_control_passed = 9
algorithm_failed = 0
deterministic rerun = 44/44 matched
Git LFS fsck = OK
full pytest artifact = 365 passed, 10 skipped
```

注意：这只是离线 profile 和 capability certificate 通过，不代表 runtime motion quality 已解决。

### 2.2 Step 3.0 runtime shadow 基线

当前 Step 3.0 结果：

```text
Final HEAD = 1b0d223841909bec3fb9b850dee5c4fad1956be1
CI run = 28234721577, all 3 jobs success
Step 3 targeted suite = 62 passed
Tests/v3 = 347 passed
Runtime smoke wrapper = passed
Live audit = PASS, blocking_count=0
Determinism diagnostics hash = 6c033defc32895342c3c0ca8ad7ab44bd4d352ae13118a2489e884288341454d
```

Known caveat from Step 3.0：

```text
full repo pytest was attempted but interrupted after 14m54s;
only 37 tests had completed;
full repo pytest was not claimed as passed.
```

This caveat must remain visible in Step 3.1 until full repo pytest is either completed or explicitly classified.

### 2.3 Step 3.0 runtime scope limitation

Step 3.0 only did RPO/G1 smoke:

```text
roboparty_rpo: shadow + override smoke passed
unitree_g1: shadow/fingerprint logic existed, override failed closed on fingerprint mismatch
```

This is no longer enough. Step 3.1 must cover the full 44 in-scope matrix.

### 2.4 Existing runtime pipeline constraint

`NewtonPipeline` currently has production config and runtime path mainly for built-in/registered robots. Full Robot Zoo models do **not** all have production retargeter configs. Therefore Step 3.1 must implement a separate **full-fleet runtime evaluation harness** instead of pretending every robot is already a production pipeline target.

Do not force every Robot Zoo model into `NewtonPipeline` by generating fake production configs. Production `NewtonPipeline` remains minimal and opt-in.

---

## 3. Definitions and exact scope

### 3.1 In-scope total

The full matrix must be read from:

```text
assets/robot_zoo/robot_zoo_lock.json
artifacts/retargeting_v3_step2_capability/summary.json
```

Expected total:

```text
in_scope_total = 44
```

Do not hardcode a separate list in code. Tests may freeze the count and compare IDs to artifacts, but runtime code must derive from lock/profile artifacts.

### 3.2 Categories

The matrix must classify each model into one of:

```text
full_humanoid_profile      # expected positive humanoid with passed profile
partial_humanoid_profile   # structured partial passed
negative_control           # negative_control_passed
```

Expected counts:

```text
full_humanoid_profile = 32
partial_humanoid_profile = 3
negative_control = 9
```

### 3.3 Required modes per category

#### full_humanoid_profile, 32 models

Required evaluations:

```text
runtime_profile_resolution
runtime_local_profile_if_needed
target_stream_generation
generic_shadow_quality
generic_override_smoke
runtime_determinism
```

#### partial_humanoid_profile, 3 models

Required evaluations:

```text
runtime_profile_resolution
runtime_local_profile_if_needed
supported_semantic_target_stream
partial_generic_shadow_quality
no_full_override
runtime_determinism
```

#### negative_control, 9 models

Required evaluations:

```text
runtime_model_load_or_policy_status
no_humanoid_profile_promotion
no_target_stream_override
negative_control_runtime_smoke_if_loadable
runtime_determinism_of_rejection
```

### 3.4 Pipeline-backed vs generic harness

Two evaluation pathways must exist:

```text
pipeline_backed   # existing NewtonPipeline with retarget config; required for RPO and G1 if eligible
generic_harness   # new full-fleet evaluator using runtime model + V3 profile, required for all 44 categories as applicable
```

RPO/G1 remain required pipeline-backed controls. All 44 must still be represented in the generic full-fleet matrix.

---

## 4. Hard boundaries

### 4.1 This round includes

- Full 44-model runtime eligibility matrix;
- runtime-local profile generation for fingerprint/source mismatches;
- generic runtime target stream for all full/partial humanoids;
- generic short/mid clip runtime smoke for all full humanoids;
- partial humanoid supported-semantics smoke;
- negative-control rejection smoke;
- RPO/G1 pipeline-backed regression retained;
- quality metric computation;
- deterministic rerun;
- clean artifacts;
- CI/red-team.

### 4.2 This round forbids

- Changing production default behavior;
- enabling V3 runtime by default;
- removing existing RPO/G1 pipeline-backed tests;
- pretending all Robot Zoo models are production `NewtonPipeline` targets;
- adding fake production configs just to pass tests;
- tuning IK weights/objective weights;
- changing exact residual thresholds;
- changing numerical epsilon/rank/subspace gates;
- adding robot-ID mathematical special cases;
- visual-only pass/fail;
- manual screenshot inspection as evidence;
- contact/collision/self-penetration runtime solving;
- foot locking / temporal smoothing;
- teacher/policy/refinement;
- modifying Step 2 artifacts;
- using private or company assets;
- re-adding deferred `berkeley_humanoid_urdf` or `romeo_urdf` into scope.

---

## 5. Runtime-local profile closure

Step 3.0 exposed G1 fingerprint mismatch. Step 3.1 must solve this generally.

### 5.1 New concept

Add runtime-local profile generation:

```text
runtime model source
+ existing verified semantic map / semantic expectation
+ Step 2.3 compiler
→ runtime-local profile artifact
```

Output root:

```text
artifacts/retargeting_v3_step3_runtime_quality/runtime_local_profiles/<model_id>.json
```

### 5.2 Rules

For each of 44 in-scope entries:

1. Load committed Step 2.3 capability profile.
2. Resolve runtime source path from lock/snapshot/fetch-only cache/local existing source.
3. Compute runtime fingerprint.
4. Compare runtime fingerprint to committed profile fingerprint.
5. If strict match, reuse committed profile.
6. If mismatch and model is full/partial humanoid, generate runtime-local profile from the actual runtime source.
7. Runtime-local profile must pass Step 2.3 status gates or be terminal `runtime_local_profile_failed`.
8. Negative controls must not generate humanoid profile.
9. The runtime-local profile must record source profile, runtime source, semantic map, hashes, environment and deterministic hash.
10. Do not modify Step 2 artifacts.

### 5.3 Fail-closed policy

No row may silently fall back from mismatch to committed profile.

Allowed resolution statuses:

```text
profile_match
runtime_local_profile_generated
runtime_local_profile_failed
structured_partial_supported
negative_control_rejected
source_or_cache_unavailable
runtime_model_load_failed
```

For PASS, source/cache/load failures must be zero for all 44; runtime-local profile failures may remain only if the row is reported as `runtime_quality_failed` with exact reason, not as pass.

---

## 6. Full-fleet generic runtime harness

Add module:

```text
soma_retargeter/runtime/v3/fleet_harness.py
```

The generic harness is not production `NewtonPipeline`. It is an evaluation harness that uses the same runtime model truth and V3 target stream to run per-frame smoke checks.

### 6.1 Inputs

```python
@dataclass
class FleetRuntimeCase:
    model_id: str
    category: str
    runtime_source_path: Path
    model_format: str
    profile_path: Path
    semantic_map_path: Path | None
    clips: list[Path]
    modes: list[str]
    max_frames: int
```

### 6.2 Modes

```text
target_stream_only
generic_shadow_quality
generic_override_smoke
negative_control_rejection
partial_supported_smoke
```

### 6.3 Required behavior

For each full humanoid model:

- load runtime model;
- load/generate runtime-local profile;
- extract source semantic frames from each clip;
- generate V3 target stream;
- evaluate target stream finite/SE(3)/smoothness metrics;
- run generic per-frame semantic IK/chain solve smoke or equivalent runtime solver check;
- compute residual and joint-limit metrics;
- write per-frame compact stats;
- never tune weights per robot.

For each partial humanoid:

- only supported semantics are evaluated;
- missing hands are not fabricated;
- no full override is attempted;
- metrics clearly marked partial.

For each negative control:

- load if applicable;
- prove no humanoid semantic target stream/override generated;
- record rejection status.

### 6.4 Generic smoke solver

Allowed implementation choices:

```text
A. Newton IK if a generated generic semantic-body objective map is valid;
B. Step 2 chain projection per semantic task as runtime frame-by-frame smoke;
C. MuJoCo/Newton FK residual evaluation against target stream.
```

Integrator must choose the simplest reliable option, but must document which was used.

Prohibited:

- generating fake pass without moving through runtime model FK;
- using only offline canonical targets and calling it runtime;
- using screenshot/video appearance;
- tuning robot-specific weights.

---

## 7. Motion clips and frame budgets

### 7.1 Clip inventory

Use all public clips already committed under:

```text
assets/motions/bvh/*.bvh
assets/motions/csv/*.csv
```

Do not use private clips. Do not download new motion assets.

### 7.2 Required core clips

At minimum every full/partial humanoid must run on:

```text
assets/motions/bvh/Neutral_walk_forward_002__A057.bvh
assets/motions/bvh/wave_R_001__A428.bvh
assets/motions/bvh/body_stretch_1_004__A069.bvh
assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh
```

If any of these are missing or unloadable, Step 3.1 is BLOCKED.

### 7.3 Extended clips

All other committed BVH/CSV clips must be included in inventory. The full matrix may use capped frames for runtime cost:

```text
short_clip_max_frames = 120
mid_clip_max_frames = 300
long_clip_sample_stride = deterministic, recorded
```

Do not randomly sample clips or frames. Any truncation/subsampling must be deterministic and recorded.

### 7.4 Required coverage

PASS requires:

```text
full humanoid 32 models × required core clips × target_stream_only
full humanoid 32 models × at least 2 core clips × generic_override_smoke
partial humanoid 3 models × required core clips × supported target stream
negative controls 9 models × rejection/load smoke
pipeline-backed RPO/G1 controls retained
```

If cost is too high, do not reduce model count. Reduce frames deterministically and record the frame budget.

---

## 8. Quality metrics

Add module:

```text
soma_retargeter/runtime/v3/quality_metrics.py
```

Metrics must be objective and numeric. No visual claims.

### 8.1 Target-stream metrics

Per model/clip/semantic:

```text
finite_count
nan_count
se3_orthogonality_error_max
translation_delta_vs_legacy_mean/p95/max if legacy exists
rotation_delta_vs_legacy_mean/p95/max if legacy exists
frame_to_frame_translation_velocity_p95/max
frame_to_frame_rotation_velocity_p95/max
target_jump_count
root_horizontal_scale
support_height_policy
capability_status
```

### 8.2 Runtime smoke output metrics

Per model/clip/mode:

```text
output_frame_count
joint_coord_count
nan_count
inf_count
joint_limit_violation_count
max_joint_limit_violation
joint_position_abs_p95/max
joint_velocity_p95/max
joint_acceleration_p95/max
task_residual_mean/p95/max
normalized_task_residual_mean/p95/max
solver_success_fraction
solver_iteration_mean/p95/max
runtime_seconds
```

### 8.3 Contact/foot quality diagnostics only

Do not solve contact. Only measure:

```text
support_height_min/max
foot_height_below_ground_count
foot_sliding_proxy_if_contact_scores_available
stance_width_p95/max
```

### 8.4 Regression metrics

For pipeline-backed RPO/G1 only:

```text
shadow output == disabled baseline
shadow target stream finite
override finite
RPO override diff from disabled recorded
G1 override pass or fail-closed recorded
```

For generic harness, do not claim equivalence to old pipeline if old pipeline does not exist.

---

## 9. Artifact schema

Output root:

```text
artifacts/retargeting_v3_step3_runtime_quality/
```

Required files:

```text
environment.json
commands.txt
source_inventory.json
model_matrix.json
profile_resolution_matrix.json
runtime_local_profile_summary.json
clip_inventory.json
target_stream_matrix.json
generic_smoke_matrix.json
pipeline_backed_matrix.json
quality_summary.json
failure_matrix.json
deterministic_rerun.json
acceptance_ledger.json
per_model/<model_id>/profile_resolution.json
per_model/<model_id>/clip_matrix.json
per_model/<model_id>/quality_metrics.json
per_model/<model_id>/failures.json
per_clip/<model_id>/<clip_id>/target_deltas.json
per_clip/<model_id>/<clip_id>/smoke_summary.json
runtime_local_profiles/*.json
test_results/pytest.txt
test_results/junit.xml
test_results/pytest_summary.json
```

### 9.1 `model_matrix.json`

Must include all 44 rows:

```text
model_id
category
profile_status
expected_capability
runtime_source_status
runtime_profile_resolution_status
target_stream_status
generic_smoke_status
pipeline_backed_status
negative_control_status
final_step3_1_status
```

Allowed final statuses:

```text
runtime_quality_passed
runtime_quality_warned
runtime_quality_failed
partial_runtime_passed
negative_control_runtime_passed
blocked_source_or_profile
```

For PASS, all 44 must be terminal and none may be `blocked_source_or_profile`.

Runtime quality failures may remain only if fully quantified and not infrastructure blockers. The overall Step 3.1 can PASS as an evaluation milestone with `runtime_quality_failed` rows, provided all evidence is complete. Do not report algorithm success as 44/44 if quality failures remain.

### 9.2 `quality_summary.json`

Must report:

```text
in_scope_total = 44
full_humanoid_total = 32
partial_total = 3
negative_total = 9
profile_match_count
runtime_local_profile_generated_count
runtime_local_profile_failed_count
target_stream_success_count
generic_smoke_success_count
generic_smoke_failed_count
pipeline_backed_success_count
pipeline_backed_fail_closed_count
quality_failed_count
deterministic_compared_count
deterministic_matched_count
```

### 9.3 `failure_matrix.json`

Every failure row must include:

```text
model_id
clip_id
mode
stage
failure_type
message
numeric_context
reproduction_command
next_action
```

No generic `failed` strings.

---

## 10. Six xhigh subagents

Each agent must:

- use an independent worktree/branch;
- record reasoning strength = `xhigh`;
- write failing tests first;
- only edit owned files;
- make small commits;
- write handoff with commit, files, commands, tests, numeric results, blockers and risks;
- not claim unrun tests.

### Agent A — Fleet Inventory, Runtime Source Resolution and Runtime-Local Profiles

**Goal**: Make every one of the 44 in-scope models resolvable as runtime source, and generate runtime-local profiles for mismatches.

Owned files:

```text
soma_retargeter/runtime/v3/fleet_inventory.py
soma_retargeter/runtime/v3/runtime_local_profile.py
soma_retargeter/runtime/v3/profile_loader.py
soma_retargeter/tools/compile_runtime_local_profiles_v3.py
artifacts/retargeting_v3_step3_runtime_quality/profile_resolution_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/runtime_local_profile_summary.json
artifacts/retargeting_v3_step3_runtime_quality/runtime_local_profiles/
tests/v3/test_step3_fleet_inventory_*.py
tests/v3/test_step3_runtime_local_profile_*.py
docs/retargeting_v3/subagents/step3_1_agent_a_handoff.md
```

Hard requirements:

1. Derive 44 IDs from lock/profile artifacts, not a separate hardcoded list.
2. Verify category counts 32/3/9.
3. Resolve source path for snapshots, fetch-only cache and local RPO.
4. Detect LFS pointer-only files.
5. Generate runtime-local profile when fingerprint mismatches.
6. Fail closed for negative controls.
7. Runtime-local profile must include source profile ID, runtime source hash, semantic map hash, deterministic hash.
8. No Step 2 artifact mutation.

### Agent B — Full-Fleet Target Stream and Source Motion Coverage

**Goal**: Generate deterministic V3 target streams for all full/partial humanoids across required clips.

Owned files:

```text
soma_retargeter/runtime/v3/source_frames.py
soma_retargeter/runtime/v3/target_adapter.py
soma_retargeter/runtime/v3/clip_inventory.py
soma_retargeter/runtime/v3/target_stream.py
artifacts/retargeting_v3_step3_runtime_quality/clip_inventory.json
artifacts/retargeting_v3_step3_runtime_quality/target_stream_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/per_clip/**/target_deltas.json
tests/v3/test_step3_clip_inventory_*.py
tests/v3/test_step3_full_fleet_target_stream_*.py
tests/v3/test_step3_partial_target_stream_*.py
docs/retargeting_v3/subagents/step3_1_agent_b_handoff.md
```

Hard requirements:

1. Use all required core clips.
2. Inventory all committed BVH/CSV clips.
3. Deterministic frame caps/stride.
4. Full humanoid target streams finite for required semantics.
5. Partial models evaluate only supported semantics.
6. Missing semantic = structured failure, not fallback.
7. Reuse Step 2 target builder/rest calibration.
8. No duplicate scaler math.
9. SE(3) validation for every transform.

### Agent C — Generic Runtime Harness and Smoke Solver

**Goal**: Run a generic full-fleet runtime smoke for all 32 full humanoids and structured smoke/rejection for the rest.

Owned files:

```text
soma_retargeter/runtime/v3/fleet_harness.py
soma_retargeter/runtime/v3/generic_smoke.py
soma_retargeter/runtime/v3/generic_ik.py        # only if needed; no production pipeline changes
soma_retargeter/runtime/v3/runtime_status.py
artifacts/retargeting_v3_step3_runtime_quality/generic_smoke_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/per_model/*/quality_metrics.json
artifacts/retargeting_v3_step3_runtime_quality/per_clip/**/smoke_summary.json
tests/v3/test_step3_generic_harness_*.py
tests/v3/test_step3_full_humanoid_smoke_*.py
tests/v3/test_step3_negative_control_rejection_*.py
docs/retargeting_v3/subagents/step3_1_agent_c_handoff.md
```

Hard requirements:

1. All 32 full humanoids enter generic smoke.
2. 3 partial humanoids enter partial supported smoke.
3. 9 negatives are rejected, not promoted.
4. Runtime model FK/solver must be used; not offline-only canonical targets.
5. No per-robot weights or thresholds.
6. Smoke can fail quality, but must produce numeric evidence.
7. No NaN/Inf accepted as pass.
8. Joint limit violations measured and classified.
9. Runtime results deterministic.

### Agent D — Quality Metrics, Comparators and Pipeline-Backed Controls

**Goal**: Measure runtime quality objectively and preserve RPO/G1 pipeline-backed regressions.

Owned files:

```text
soma_retargeter/runtime/v3/quality_metrics.py
soma_retargeter/runtime/v3/comparators.py
soma_retargeter/tools/run_v3_runtime_shadow_smoke.py
soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py
artifacts/retargeting_v3_step3_runtime_quality/pipeline_backed_matrix.json
artifacts/retargeting_v3_step3_runtime_quality/quality_summary.json
tests/v3/test_step3_quality_metrics_*.py
tests/v3/test_step3_pipeline_backed_controls_*.py
tests/v3/test_step3_g1_runtime_local_profile_policy_*.py
docs/retargeting_v3/subagents/step3_1_agent_d_handoff.md
```

Hard requirements:

1. RPO pipeline-backed disabled/shadow/override still pass.
2. G1 pipeline-backed mismatch is either runtime-local profile resolved or fail-closed with explicit reason.
3. Shadow no-op remains true for pipeline-backed controls.
4. Quality metrics include target residuals, joint stats, smoothness, finite counts.
5. No visual claims.
6. No IK weight changes.
7. No production default behavior changes.

### Agent E — Full-Fleet Orchestration, Clean Artifacts and Determinism

**Goal**: Orchestrate the full matrix and produce clean, reproducible artifacts.

Owned files:

```text
scripts/run_step3_full_fleet_runtime_quality.sh
soma_retargeter/tools/run_v3_full_fleet_runtime_quality.py
artifacts/retargeting_v3_step3_runtime_quality/
tests/v3/test_step3_full_fleet_artifact_schema_*.py
tests/v3/test_step3_full_fleet_determinism_*.py
docs/retargeting_v3/STEP3_1_FULL_FLEET_RUNTIME_QUALITY_REPORT.md
docs/retargeting_v3/subagents/step3_1_agent_e_handoff.md
```

Hard requirements:

1. Clean source commit / artifact commit protocol.
2. All 44 rows in model matrix.
3. Required artifacts present.
4. Deterministic rerun covers all terminal rows.
5. Source worktree clean before/after.
6. No local absolute paths.
7. LFS state recorded.
8. Full tests or documented full-repo pytest classification.
9. If full repo pytest is interrupted or too slow again, classify with exact last test and reason; do not claim pass.

### Agent F — Independent Red Team, CI and Final Verdict

**Goal**: Attack the full-fleet claims and decide PASS/BLOCKED.

Owned files:

```text
scripts/audit_retargeting_v3_step3_full_fleet.py
.github/workflows/retargeting_v3_step3_full_fleet.yml
tests/v3/test_step3_full_fleet_acceptance_*.py
docs/retargeting_v3/STEP3_1_FULL_FLEET_RUNTIME_QUALITY_ACCEPTANCE.md
docs/retargeting_v3/subagents/step3_1_agent_f_red_team.md
artifacts/retargeting_v3_step3_runtime_quality/test_results/
artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

Must fail on:

1. Matrix has fewer than 44 rows.
2. Any full humanoid omitted from target stream.
3. Any full humanoid omitted from generic smoke without structured blocker.
4. Any negative control promoted to humanoid target stream.
5. RPO/G1 pipeline controls removed.
6. Default or shadow mode mutates outputs.
7. Runtime profile mismatch silently ignored.
8. Runtime-local profile generated but not validated.
9. Quality metrics missing numeric fields.
10. NaN/Inf hidden or counted as pass.
11. Joint limit violations not reported.
12. Local absolute path leakage.
13. LFS pointer-only model/profile.
14. Step 2 artifacts modified.
15. CI only tests RPO/G1.
16. Agent handoff claims unrun tests.
17. Full-repo pytest caveat omitted.

Final handoff must include:

```text
verdict = PASS or BLOCKED
final_head
source_code_commit
artifact_commit
workflow_run_id
44-row matrix summary
profile resolution summary
target stream summary
generic smoke summary
pipeline-backed control summary
quality failure count
pytest summary
full-repo pytest classification
remaining blockers
```

---

## 11. Integrator fixed execution order

Do not change this order.

### Wave 0 — Freeze scope and red-team tests

1. Start 6 xhigh agents.
2. Agent F writes failing acceptance tests for 44-row matrix and RPO/G1-only cheating.
3. Agent A freezes fleet inventory schema.
4. Agent B freezes target stream schema.
5. Integrator freezes artifact schema.

### Wave 1 — Source/profile closure

Merge order:

```text
Agent A only
```

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_step3_fleet_inventory_*.py tests/v3/test_step3_runtime_local_profile_*.py
```

### Wave 2 — Target streams

Merge order:

```text
Agent B
```

Run:

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_step3_clip_inventory_*.py tests/v3/test_step3_full_fleet_target_stream_*.py tests/v3/test_step3_partial_target_stream_*.py
```

### Wave 3 — Generic smoke

Merge order:

```text
Agent C
```

Run generic harness tests. Do not touch NewtonPipeline.

### Wave 4 — Quality and pipeline controls

Merge order:

```text
Agent D
```

Verify RPO/G1 Step 3.0 controls still pass.

### Wave 5 — Full orchestration

Merge order:

```text
Agent E
```

Generate development artifacts. Run Agent F first live audit. Return blockers to owners.

### Wave 6 — Clean artifacts and final red-team

1. Freeze code commit `C`.
2. Push `C`.
3. Clean detached worktree from `C`.
4. `git lfs pull && git lfs fsck`.
5. Run full matrix.
6. Run targeted tests and tests/v3.
7. Attempt full repo pytest; record result honestly.
8. Run audit.
9. Commit artifacts `A`.
10. Push.
11. CI green.
12. Agent F final PASS/BLOCKED.

---

## 12. Required commands

### 12.1 Targeted tests

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_step3_fleet_inventory_*.py \
  tests/v3/test_step3_runtime_local_profile_*.py \
  tests/v3/test_step3_clip_inventory_*.py \
  tests/v3/test_step3_full_fleet_target_stream_*.py \
  tests/v3/test_step3_partial_target_stream_*.py \
  tests/v3/test_step3_generic_harness_*.py \
  tests/v3/test_step3_full_humanoid_smoke_*.py \
  tests/v3/test_step3_negative_control_rejection_*.py \
  tests/v3/test_step3_quality_metrics_*.py \
  tests/v3/test_step3_pipeline_backed_controls_*.py \
  tests/v3/test_step3_full_fleet_artifact_schema_*.py \
  tests/v3/test_step3_full_fleet_determinism_*.py \
  tests/v3/test_step3_full_fleet_acceptance_*.py
```

### 12.2 V3 suite

```bash
PYTHONPATH=. python -m pytest -q tests/v3
```

### 12.3 Full repo suite

```bash
PYTHONPATH=. python -m pytest -q
```

If full repo suite is too slow/interrupted, save:

```text
start time
stop time
duration
last completed test count
last visible test/path
reason
whether any failure occurred before interruption
```

Do not report full repo suite as passed unless it completes with exit code 0.

### 12.4 Full fleet command

```bash
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_full_fleet_runtime_quality \
  --artifact-root artifacts/retargeting_v3_step3_runtime_quality \
  --step2-profile-root artifacts/retargeting_v3_step2_capability \
  --step3-shadow-root artifacts/retargeting_v3_step3_runtime_shadow \
  --lock assets/robot_zoo/robot_zoo_lock.json \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --clip-root assets/motions \
  --required-core-clips \
    assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
    assets/motions/bvh/wave_R_001__A428.bvh \
    assets/motions/bvh/body_stretch_1_004__A069.bvh \
    assets/motions/bvh/item_pick_up_standing_R_001__A410.bvh \
  --short-max-frames 120 \
  --mid-max-frames 300 \
  --deterministic-rerun
```

### 12.5 Audit

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_full_fleet.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_quality \
  --source-root . \
  --write-report artifacts/retargeting_v3_step3_runtime_quality/acceptance_ledger.json
```

---

## 13. CI requirements

New workflow:

```text
.github/workflows/retargeting_v3_step3_full_fleet.yml
```

Required jobs:

```text
full-fleet-static-and-unit
full-fleet-artifact-audit
lfs-and-snapshot-smoke
pipeline-backed-regression
```

CI must:

- checkout with LFS;
- run `git lfs pull && git lfs fsck`;
- install package;
- run targeted tests that do not require private caches;
- audit committed full-fleet artifacts;
- verify model matrix has 44 rows;
- verify CI is not RPO/G1-only;
- upload compact artifacts if generated.

Full 44 runtime recomputation may be local/self-hosted due fetch-only cache and runtime cost, but committed artifacts must record the local clean run and CI must audit them.

---

## 14. Final acceptance gates

### Scope

- [ ] 44 in-scope model rows.
- [ ] Counts 32 full, 3 partial, 9 negative.
- [ ] No deferred assets reintroduced.
- [ ] No model skipped just because it is not RPO/G1.

### Runtime profile

- [ ] Runtime source resolved for all 44.
- [ ] Runtime-local profile generated for every full/partial mismatch.
- [ ] No silent fingerprint mismatch.
- [ ] Negative controls not promoted.
- [ ] Runtime-local profile failures are explicit and quantified.

### Target streams

- [ ] Full humanoid target streams generated for required clips.
- [ ] Partial target streams generated for supported semantics only.
- [ ] SE(3)/finite checks pass or structured failure.
- [ ] Target stream metrics written.

### Generic smoke

- [ ] 32 full humanoids enter generic smoke.
- [ ] 3 partials enter partial smoke.
- [ ] 9 negatives enter rejection smoke.
- [ ] No NaN/Inf hidden.
- [ ] Joint-limit metrics written.
- [ ] Quality failures are quantified.

### Pipeline controls

- [ ] RPO disabled/shadow/override retained.
- [ ] G1 pipeline-backed policy retained.
- [ ] Shadow no-op still true.
- [ ] No production default behavior change.

### Reproducibility

- [ ] Deterministic rerun covers all terminal rows.
- [ ] Clean source/artifact commits.
- [ ] LFS OK.
- [ ] No absolute paths/private assets.
- [ ] Tests saved.
- [ ] Full repo pytest result classified honestly.
- [ ] CI green.
- [ ] Agent F final PASS/BLOCKED.

---

## 15. What PASS means

Final report must be exactly one of:

```text
Step 3.1 Full-Fleet Runtime Quality Evaluation: PASS
```

or:

```text
Step 3.1 Full-Fleet Runtime Quality Evaluation: BLOCKED
```

PASS means:

```text
44-row matrix complete
all infrastructure blockers resolved or explicitly failed with numeric evidence
RPO/G1 pipeline controls intact
all full/partial/negative categories evaluated
artifacts clean and deterministic
CI/audit pass
```

PASS does **not** mean every robot's motion looks good. If quality failures remain, they must be counted and summarized. The point of this round is to know exactly where runtime quality fails across the full fleet.

If any model is omitted, if the matrix is RPO/G1-only, if evidence is missing, or if Codex hides failures by status renaming, final verdict must be BLOCKED.

Even if PASS, stop after Step 3.1. Do not start optimization/tuning automatically.
