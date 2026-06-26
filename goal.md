# Goal — Step 3.0：V3 Profile Runtime Shadow Integration（最小运行时闭环）

> **Codex 强制执行契约——禁止自由发挥**
>
> 当前分支：`retargeting-v3-step3-runtime-profile-shadow`  
> 基线分支：`retargeting-v3-step2-capability-acceptance-hardening`  
> 基线提交：`50a52b2ffbeab03a5636327da021f656a2f044f5`  
> 当前阶段：**Step 3.0 Runtime Shadow Integration**  
>
> 本轮只做 **V3 离线 profile 到现有 `NewtonPipeline` 的最小 runtime 接线**。默认行为必须完全不变。实验功能必须通过显式 config 打开。
>
> 本轮必须一次创建并持续高强度使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator，负责接口冻结、冲突解决、全量测试、clean artifacts 和最终诚实判定。
>
> **禁止重写生产 retargeter，禁止重写 whole-body IK，禁止调 IK 权重，禁止 contact/collision/temporal smoothing，禁止修改 Step 2.1/2.3 thresholds，禁止添加 robot-specific 数学分支，禁止宣称动作效果已经最终变好。**

---

## 0. 开始前必须执行

```bash
conda activate soma-retargeter-v2

git fetch origin --prune
git checkout retargeting-v3-step3-runtime-profile-shadow
git pull --ff-only

git status --short
git branch --show-current
git rev-parse HEAD
git merge-base --is-ancestor 50a52b2ffbeab03a5636327da021f656a2f044f5 HEAD

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
branch = retargeting-v3-step3-runtime-profile-shadow
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
4. `artifacts/retargeting_v3_step2_capability/summary.json`
5. `artifacts/retargeting_v3_step2_capability/environment.json`
6. `artifacts/retargeting_v3_step2_capability/lfs_state.json`
7. `artifacts/retargeting_v3_step2_capability/per_robot/roboparty_rpo_local.json`
8. `artifacts/retargeting_v3_step2_capability/per_robot/unitree_g1_mjcf.json`
9. `artifacts/retargeting_v3_step2_capability/per_robot/unitree_g1_urdf.json`
10. `soma_retargeter/pipelines/newton_pipeline.py`
11. `soma_retargeter/pipelines/utils.py`
12. `soma_retargeter/robotics/human_to_robot_scaler.py`
13. `soma_retargeter/robotics/v3/target_builder.py`
14. `soma_retargeter/robotics/v3/rest_frames.py`
15. `soma_retargeter/robotics/v3/capability_status.py`
16. `soma_retargeter/robotics/v3/validation.py`
17. `robot_registry_parser.py`
18. existing `configs/**/soma_to_*_retargeter_config.json`

不得只读 summary 后直接改 `NewtonPipeline`。

---

## 2. 当前事实基线

### 2.1 Step 2.3.1 已完成的离线基础

当前可信基线：

```text
44 in-scope Robot Zoo / local assets
source/load/semantic failure = 0
passed = 32
partial_passed = 3
negative_control_passed = 9
algorithm_failed = 0
deterministic rerun = 44/44 matched
Git LFS fsck = OK
full pytest artifact = 365 passed, 10 skipped
```

这只是 **离线 profile / capability certificate** 通过，不代表生产 demo 已经变好。

### 2.2 当前 runtime pipeline 事实

`NewtonPipeline` 当前：

- 在 `__init__` 中通过 `pipeline_utils.get_robot_mjcf_path(self.target_type)` 加载 runtime MJCF；
- 使用 `HumanToRobotScaler` 从 source animation buffer 计算 `buffer_effectors`；
- 再根据 `ik_map` 取 `target_effector_indices`；
- 在 `execute()` 中把每帧 target 设置到 `position_objectives` / `rotation_objectives`；
- 现有 output、post-processing、foot stabilization、virtual grounding 均基于旧 target stream。

本轮只能在这个路径中 **加可选 V3 runtime target stream**，不能拆掉现有 pipeline。

### 2.3 当前最小风险策略

本轮分两层：

```text
shadow mode   = 计算V3 targets + 写diagnostics，但绝不改变IK输入和输出
override mode = 显式config开启，只替换RPO/G1语义目标stream，用于短clip smoke，不宣称最终动作质量
```

默认必须等价于当前 pipeline。

---

## 3. 本轮唯一目标

建立以下最小 runtime 数据流：

```text
AnimationBuffer frame
→ source SOMA global semantic frames
→ load verified V3 runtime profile
→ target_builder/rest_calibration 生成 robot semantic target transforms
→ map to existing IK effector order
→ shadow diagnostics, default no-op
→ optional experimental override for RPO/G1 smoke
→ runtime artifact matrix
→ CI/red-team
```

最终必须能回答：

1. 不开启 V3 时，`NewtonPipeline` 输出是否 bitwise 或 tolerance 等价于基线？
2. Shadow mode 是否计算出 finite V3 semantic targets 且不改变 IK targets？
3. RPO runtime model 与 V3 profile fingerprint/source 是否匹配？
4. G1 runtime model 与 committed profile 是否匹配；若不匹配，是否 fail closed 或生成独立 runtime-local profile？
5. Override mode 在 RPO/G1 的短 clip 上是否 finite、deterministic、within limits？
6. V3 target stream 与旧 scaler target stream 的差异在哪里，是否有 per-frame/per-semantic diagnostics？

---

## 4. 严格范围边界

### 4.1 本轮包含

- 新增 V3 runtime profile loader；
- profile/fingerprint/source validation；
- source SOMA semantic frame extraction；
- runtime target adapter；
- `NewtonPipeline` shadow-mode integration；
- explicit experimental override mode；
- RPO + G1 short-clip smoke tests；
- no-op default regression；
- diagnostics artifacts；
- CI/red-team。

### 4.2 本轮禁止

- 默认启用 V3；
- 改变现有默认 output；
- 删除或绕过 `HumanToRobotScaler`；
- 重写 `NewtonPipeline.execute()` 主循环；
- 重写 Newton IK solver；
- 调 `ik_iterations`、joint limit weight、smooth weight 或 body weights；
- contact-aware foot IK；
- collision/self-penetration；
- temporal smoothing；
- foot locking；
- policy/teacher/refinement；
- 全 44 runtime demo；
- 修改 Step 2 artifacts；
- 修改 Step 2 numerical/capability thresholds；
- robot ID 数学分支；
- 以 screenshot/目测作为 pass；
- 进入 Step 3.1 全量 runtime quality 或 Step 4。

---

## 5. 新配置契约

在 retarget config 中新增一个顶层可选字段：

```json
{
  "v3_runtime_profile": {
    "enabled": false,
    "mode": "disabled",
    "profile_artifact_root": "artifacts/retargeting_v3_step2_capability",
    "profile_model_id": null,
    "target_policy": "shadow_only",
    "fail_on_fingerprint_mismatch": true,
    "allow_runtime_recompile_on_mismatch": false,
    "semantic_tasks": ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"],
    "override_tasks": [],
    "diagnostics_enabled": true,
    "diagnostics_max_frames": 240,
    "diagnostics_output_dir": "artifacts/retargeting_v3_step3_runtime_shadow",
    "write_per_frame_debug": false
  }
}
```

Allowed modes：

```text
disabled
shadow
override_experimental
```

### 5.1 Default behavior

如果字段缺失或：

```json
{"enabled": false}
```

则 pipeline必须完全沿用旧逻辑，不读取 profile，不写 diagnostics，不改变 outputs。

### 5.2 Shadow mode

```json
{
  "enabled": true,
  "mode": "shadow",
  "target_policy": "shadow_only"
}
```

要求：

- 计算 V3 targets；
- 记录 target deltas；
- 不修改 `self.input_targets`；
- 不修改 IK objective targets；
- output CSV/frames 与 disabled mode 在 tolerance 内相同。

### 5.3 Override experimental mode

```json
{
  "enabled": true,
  "mode": "override_experimental",
  "target_policy": "replace_configured_semantics",
  "override_tasks": ["Hips", "Chest", "LeftHand", "RightHand", "LeftFoot", "RightFoot"]
}
```

要求：

- 只替换 `override_tasks` 对应的 existing IK target stream；
- 不改变 IK map body mapping；
- 不改变 IK objective weights；
- 不新增 contact/temporal/collision；
- 只允许 RPO 和 G1 smoke；
- 失败必须 fail closed，不得 fallback 成“看似通过”。

---

## 6. Robot / profile 映射策略

默认映射：

```text
roboparty_rpo -> roboparty_rpo_local
unitree_g1    -> unitree_g1_mjcf preferred, fallback unitree_g1_urdf only for diagnostic comparison
```

必须实现：

```python
resolve_runtime_v3_profile_id(target_type, runtime_mjcf_path, config) -> RuntimeProfileResolution
```

规则：

1. 如果 config 显式给 `profile_model_id`，使用显式值；
2. 否则按上表；
3. 加载 `per_robot/<profile_model_id>.json`；
4. 检查状态必须为 `passed` 或 `capability_limited_passed`；
5. 检查 semantic sites、rest calibration、canonical targets、capability summary存在；
6. 检查 runtime model fingerprint/source hash；
7. RPO 必须 strict match；
8. G1 若 runtime Newton-downloaded model 与 committed snapshot fingerprint 不同：
   - shadow mode 可记录 `fingerprint_mismatch` 并跳过 target override；
   - override mode 必须 fail closed，除非显式 `allow_runtime_recompile_on_mismatch=true` 并生成 runtime-local profile artifact；
9. 不得用 `robot_type` 名称绕过 fingerprint。

---

## 7. Runtime source semantic frames

必须新增模块：

```text
soma_retargeter/runtime/v3/source_frames.py
```

职责：

- 从 `AnimationBuffer` 每帧计算 source global transforms；
- 将 SOMA skeleton joints 映射到 semantic names：
  - `Hips`
  - `Chest`
  - `LeftHand`
  - `RightHand`
  - `LeftFoot`
  - `RightFoot`
- 使用现有 skeleton joint names，不能写死只对某个 BVH 文件有效；
- 若 skeleton 缺 semantic，fail closed；
- 支持 frame range / max frames；
- 支持 offset transform；
- 输出 4x4 numpy transforms，float64；
- 保留 source joint name evidence。

必须新增：

```python
@dataclass
class SourceSemanticFrameBatch:
    semantic_names: list[str]
    joint_names: dict[str, str]
    transforms: dict[str, np.ndarray]  # [F,4,4]
    frame_count: int
    sample_rate: float
    source: str
```

不得使用 OCR、viewer 或可视化截图作为 source truth。

---

## 8. Runtime V3 target adapter

必须新增模块：

```text
soma_retargeter/runtime/v3/target_adapter.py
```

职责：

```text
SourceSemanticFrameBatch + RuntimeV3Profile
→ per-frame SemanticTargets
→ existing effector order transforms
→ diagnostics
```

必须：

1. 复用 Step 2 的 `build_targets_from_source_semantic_frames()` 和 rest calibration；
2. 不重新实现一套 scaler；
3. 保持 parent-frame rotation transfer；
4. 保持 root horizontal/vertical policy；
5. 对每个 semantic输出：
   - desired target transform；
   - old scaler transform；
   - translation delta；
   - rotation delta；
   - target source；
   - capability status；
6. 输出必须可映射回 `HumanToRobotScaler.effector_names()`；
7. 对没有 capability 的 task，不得伪造 target；
8. 对 partial/negative model不得进入 override。

---

## 9. NewtonPipeline 集成方式

只能对 `NewtonPipeline` 做最小、安全改动。

### 9.1 `__init__`

允许新增：

```python
self.v3_runtime_config
self.v3_runtime_profile
self.v3_runtime_adapter
self.v3_runtime_diagnostics
```

要求：

- 默认 disabled；
- disabled 时不 import heavyweight v3 runtime modules，或 import 不改变行为；
- profile load错误在 disabled mode 不影响 pipeline；
- shadow/override mode 初始化失败必须给清晰错误。

### 9.2 `add_input_motions`

现有代码：

```python
buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(...)
self.input_targets.append(buffer_effectors[:, self.target_effector_indices, :])
```

新增逻辑必须是：

```python
legacy_buffer_effectors = compute_effectors_from_buffer(...)
if v3 disabled:
    buffer_effectors = legacy_buffer_effectors
elif v3 shadow:
    v3_targets = compute_v3_targets(...)
    record diagnostics
    buffer_effectors = legacy_buffer_effectors
elif v3 override_experimental:
    v3_targets = compute_v3_targets(...)
    buffer_effectors = replace_configured_semantic_targets(legacy_buffer_effectors, v3_targets)
    record diagnostics
```

不得改变 contact inference、initialization frames、offset semantics、sample rate。

### 9.3 `execute`

本轮不允许重写 `execute()` 主循环。

允许：

- 在输出后写 summary diagnostics；
- 保存 per-run target diagnostics；
- 添加 read-only metadata。

禁止：

- 改 IK solver objective order；
- 改 objective weight；
- 改 frame stepping；
- 改 graph capture logic；
- 改 feet stabilizer逻辑。

---

## 10. Diagnostics artifact schema

新目录：

```text
artifacts/retargeting_v3_step3_runtime_shadow/
```

必须生成：

```text
environment.json
commands.txt
profile_resolution.json
shadow_summary.json
override_smoke_summary.json
per_clip/<robot>/<clip>/target_deltas.json
per_clip/<robot>/<clip>/pipeline_summary.json
test_results/pytest.txt
test_results/junit.xml
test_results/pytest_summary.json
acceptance_ledger.json
```

### 10.1 `profile_resolution.json`

必须包含：

```text
robot_type
profile_model_id
profile_status
profile_artifact_path
runtime_mjcf_path
runtime_fingerprint
profile_fingerprint
fingerprint_match
source_hash_match
strict_match_required
resolution_status
warnings/errors
```

### 10.2 `target_deltas.json`

必须包含每个 clip：

```text
robot_type
mode
clip_name
frame_count
semantic_names
legacy_target_available
v3_target_available
per_semantic:
  translation_delta_mean/max/p95
  rotation_delta_mean/max/p95
  finite_count
  nan_count
  skipped_reason
root_policy:
  horizontal_scale
  support_height_policy
capability_policy:
  exact/capability_limited/unsupported
```

### 10.3 `pipeline_summary.json`

必须包含：

```text
mode
output_frame_count
joint_coord_count
nan_count
inf_count
joint_limit_violation_count
max_joint_limit_violation
output_equal_to_disabled_baseline  # only for shadow mode
output_diff_max
runtime_seconds
```

---

## 11. Test clips and runtime smoke scope

只允许使用仓库已有公开 motion assets。

最小 clips：

```text
assets/motions/bvh/Neutral_walk_forward_002__A057.bvh
assets/motions/bvh/wave_R_001__A428.bvh
```

若某 clip load失败，不能换私有资产。必须记录 failure。

### 11.1 Required robots

```text
roboparty_rpo
unitree_g1
```

### 11.2 Required modes

For both robots：

```text
disabled baseline
shadow
```

Override：

```text
roboparty_rpo override_experimental required
unitree_g1 override_experimental only if runtime fingerprint policy passes or runtime-local profile is generated
```

### 11.3 Frame budget

Default smoke：

```text
max_frames = 120
```

Longer clips are not required this round.

---

## 12. Acceptance gates

### 12.1 No-op / default behavior

- [ ] Missing `v3_runtime_profile` config = old behavior;
- [ ] `enabled=false` = old behavior;
- [ ] disabled output equals pre-change output within exact numeric tolerance;
- [ ] disabled mode does not write Step 3 diagnostics;
- [ ] existing tests pass.

### 12.2 Shadow behavior

- [ ] shadow computes finite V3 targets for RPO;
- [ ] shadow computes finite V3 targets or explicit fingerprint-skip for G1;
- [ ] shadow does not modify `self.input_targets`;
- [ ] shadow output equals disabled output within tolerance;
- [ ] diagnostics exist and are deterministic;
- [ ] no NaN/Inf in diagnostics.

### 12.3 Override smoke behavior

- [ ] RPO override runs at least 2 clips / 120 frames each;
- [ ] output finite;
- [ ] no joint limit violations after clamper beyond existing tolerance;
- [ ] output frame count correct;
- [ ] diagnostics written;
- [ ] mode clearly labeled experimental;
- [ ] G1 override either passes same gates or is fail-closed with fingerprint reason;
- [ ] no claim of visual quality.

### 12.4 Profile/fingerprint safety

- [ ] RPO strict match;
- [ ] G1 mismatch not silently ignored;
- [ ] no robot-name-only acceptance;
- [ ] profile status must be terminal pass;
- [ ] partial/negative profile cannot override;
- [ ] LFS materialized.

### 12.5 Source frame / target correctness

- [ ] source semantic frame extraction tested on synthetic skeleton;
- [ ] 4x4 transforms valid SE(3);
- [ ] root translation policy matches Step 2 target_builder tests;
- [ ] target adapter reuses Step 2 target_builder;
- [ ] old scaler target order preserved;
- [ ] missing semantic fail closed.

### 12.6 CI / provenance

- [ ] clean source commit;
- [ ] clean artifact commit;
- [ ] pytest/junit saved;
- [ ] GitHub Actions green;
- [ ] Agent F PASS;
- [ ] no local absolute paths;
- [ ] no private assets;
- [ ] no Step 2 artifact mutation except read-only references.

---

## 13. 六个 xhigh Subagents

每个 agent 必须：

- 独立 worktree/branch；
- reasoning strength=`xhigh`；
- 先写 failing tests；
- 只修改 owned files；
- 小 commits；
- handoff包含 commit、files、commands、tests、数值结果、风险、未完成项；
- 不得声明未运行测试通过。

### Agent A — Runtime Profile Loader and Fingerprint Gate

**唯一目标**：安全加载 Step 2.3 profile，不让 runtime 用错模型。

Owned files：

```text
soma_retargeter/runtime/v3/profile_loader.py
soma_retargeter/runtime/v3/__init__.py
soma_retargeter/runtime/__init__.py
soma_retargeter/pipelines/utils.py                    # 仅允许增加 profile-id resolver helper
tests/v3/test_runtime_profile_loader_*.py
tests/v3/test_runtime_fingerprint_gate_*.py
docs/retargeting_v3/subagents/step3_agent_a_handoff.md
```

必须实现：

1. `RuntimeV3Profile` dataclass；
2. 从 `artifacts/retargeting_v3_step2_capability/per_robot/<id>.json` 加载；
3. 校验 schema/status/semantic sites/rest calibration/canonical capability；
4. resolver：`roboparty_rpo -> roboparty_rpo_local`，`unitree_g1 -> unitree_g1_mjcf`；
5. runtime MJCF fingerprint check；
6. RPO strict match required；
7. G1 mismatch fail-closed unless config explicitly allows runtime-local profile;
8. LFS pointer detection；
9. structured error messages；
10. no robot ID math branch。

### Agent B — Source Semantic Frames and V3 Target Adapter

**唯一目标**：从 runtime animation frames 生成 V3 robot semantic target stream。

Owned files：

```text
soma_retargeter/runtime/v3/source_frames.py
soma_retargeter/runtime/v3/target_adapter.py
soma_retargeter/runtime/v3/diagnostics.py
tests/v3/test_runtime_source_frames_*.py
tests/v3/test_runtime_target_adapter_*.py
tests/v3/test_runtime_target_diagnostics_*.py
docs/retargeting_v3/subagents/step3_agent_b_handoff.md
```

必须实现：

1. `SourceSemanticFrameBatch`；
2. SOMA semantic joint lookup with evidence；
3. finite SE(3) validation；
4. frame slicing/max frames；
5. target_builder reuse；
6. target -> effector order conversion；
7. per-semantic delta metrics；
8. missing semantic fail closed；
9. no duplicate scaler math；
10. deterministic diagnostics JSON。

### Agent C — NewtonPipeline Shadow No-op Integration

**唯一目标**：把 V3 target stream 接入 pipeline，但 shadow/default不改变输出。

Owned files：

```text
soma_retargeter/pipelines/newton_pipeline.py
soma_retargeter/pipelines/v3_runtime_config.py         # 可新增
tests/v3/test_newton_pipeline_v3_shadow_noop_*.py
tests/v3/test_newton_pipeline_v3_config_*.py
docs/retargeting_v3/subagents/step3_agent_c_handoff.md
```

必须实现：

1. parse `v3_runtime_profile` config；
2. disabled/missing config no-op；
3. shadow mode target computation but no IK input mutation；
4. diagnostics collection；
5. no changes to IK solver objective order/weights；
6. no changes to execute loop except post-run diagnostics write；
7. unit tests prove `self.input_targets` equality in shadow；
8. default output equality regression。

### Agent D — Experimental Override Smoke for RPO/G1

**唯一目标**：显式 override mode 的最小 smoke，不调质量。

Owned files：

```text
soma_retargeter/runtime/v3/override.py
configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_shadow.json      # if config layout allows
configs/roboparty_rpo/soma_to_rpo_retargeter_config_v3_override.json    # if config layout allows
configs/unitree_g1/soma_to_g1_retargeter_config_v3_shadow.json          # if config layout allows
configs/unitree_g1/soma_to_g1_retargeter_config_v3_override.json        # if config layout allows
tests/v3/test_runtime_override_mapping_*.py
tests/v3/test_runtime_rpo_override_smoke_*.py
tests/v3/test_runtime_g1_override_policy_*.py
docs/retargeting_v3/subagents/step3_agent_d_handoff.md
```

必须实现：

1. replace only configured semantic targets;
2. preserve old effector order;
3. override requires explicit mode;
4. RPO smoke required;
5. G1 override fail-closed on fingerprint mismatch unless runtime-local profile generated;
6. output finite tests;
7. no visual-quality claims;
8. no weight tuning。

### Agent E — Runtime Artifacts, Scripts and Reproducible Smoke Matrix

**唯一目标**：生成可复现 runtime evidence。

Owned files：

```text
soma_retargeter/tools/run_v3_runtime_shadow_smoke.py
scripts/run_step3_runtime_shadow_acceptance.sh
artifacts/retargeting_v3_step3_runtime_shadow/
tests/v3/test_step3_runtime_artifact_schema_*.py
tests/v3/test_step3_runtime_smoke_matrix_*.py
docs/retargeting_v3/STEP3_RUNTIME_SHADOW_REPORT.md
docs/retargeting_v3/subagents/step3_agent_e_handoff.md
```

必须实现：

1. command-line smoke runner;
2. robots/clips/modes matrix;
3. max_frames support;
4. save diagnostics schema;
5. save pytest/junit/summary;
6. no local absolute paths;
7. deterministic rerun of diagnostics;
8. fail if shadow changes output。

### Agent F — Red Team, CI and Final Verdict

**唯一目标**：独立验证 Step 3.0，没有参与 core修改。

Owned files：

```text
scripts/audit_retargeting_v3_step3_runtime_shadow.py
.github/workflows/retargeting_v3_step3_runtime_shadow.yml
tests/v3/test_step3_runtime_shadow_acceptance_*.py
docs/retargeting_v3/STEP3_RUNTIME_SHADOW_ACCEPTANCE.md
docs/retargeting_v3/subagents/step3_agent_f_red_team.md
artifacts/retargeting_v3_step3_runtime_shadow/test_results/
artifacts/retargeting_v3_step3_runtime_shadow/acceptance_ledger.json
```

Must test/fail on：

1. default output changed；
2. shadow mutates IK inputs；
3. override enabled without explicit config；
4. fingerprint mismatch silently accepted；
5. partial/negative profile used for override；
6. LFS pointer profile/snapshot；
7. missing diagnostics；
8. diagnostics contain NaN/Inf；
9. local absolute path leakage；
10. old Step 2 artifacts mutated；
11. no CI；
12. Agent F stale PASS/FAIL mismatch。

Final handoff must include：

```text
verdict = PASS or BLOCKED
final_head
source_code_commit
artifact_commit
workflow_run_id
pytest summary
smoke matrix summary
shadow equality result
RPO override result
G1 override/fail-closed result
remaining blockers
```

---

## 14. Integrator 固定执行顺序

不得改变顺序。

### Wave 0 — Tests and interfaces

1. 启动6个 xhigh agents；
2. Agent F写 failing acceptance tests；
3. Agent A/B冻结 dataclasses；
4. Integrator冻结 config schema；
5. 此时不得改 pipeline。

### Wave 1 — Pure runtime modules

合并顺序：

```text
A profile loader → B source/target adapter
```

运行：

```bash
PYTHONPATH=. python -m pytest -q tests/v3/test_runtime_profile_loader_*.py tests/v3/test_runtime_source_frames_*.py tests/v3/test_runtime_target_adapter_*.py
```

### Wave 2 — Pipeline shadow

1. 合并 C；
2. disabled no-op tests；
3. shadow no-op tests；
4. 禁止 override 代码先合并。

### Wave 3 — Override smoke

1. 合并 D；
2. RPO override smoke；
3. G1 policy/fingerprint tests；
4. 若 G1 mismatch，不修模型资产，只记录 fail-closed。

### Wave 4 — Artifacts and audit

1. 合并 E；
2. 生成 runtime artifacts；
3. 合并 F audit；
4. Red-team blockers返给owner。

### Wave 5 — Clean artifact generation

1. code freeze commit `C`；
2. push `C`；
3. clean detached worktree；
4. LFS pull/fsck；
5. pytest；
6. smoke matrix；
7. audit；
8. artifact commit `A`；
9. CI；
10. Agent F final PASS/BLOCKED。

---

## 15. Required tests

### 15.1 Targeted

```bash
PYTHONPATH=. python -m pytest -q \
  tests/v3/test_runtime_profile_loader_*.py \
  tests/v3/test_runtime_fingerprint_gate_*.py \
  tests/v3/test_runtime_source_frames_*.py \
  tests/v3/test_runtime_target_adapter_*.py \
  tests/v3/test_newton_pipeline_v3_shadow_noop_*.py \
  tests/v3/test_newton_pipeline_v3_config_*.py \
  tests/v3/test_runtime_override_mapping_*.py \
  tests/v3/test_step3_runtime_artifact_schema_*.py \
  tests/v3/test_step3_runtime_shadow_acceptance_*.py
```

### 15.2 Full v3

```bash
PYTHONPATH=. python -m pytest -q tests/v3
```

### 15.3 Full repo

```bash
PYTHONPATH=. python -m pytest -q
```

### 15.4 Smoke command

```bash
PYTHONPATH=. python -m soma_retargeter.tools.run_v3_runtime_shadow_smoke \
  --artifact-root artifacts/retargeting_v3_step3_runtime_shadow \
  --profile-artifact-root artifacts/retargeting_v3_step2_capability \
  --robots roboparty_rpo unitree_g1 \
  --clips assets/motions/bvh/Neutral_walk_forward_002__A057.bvh assets/motions/bvh/wave_R_001__A428.bvh \
  --modes disabled shadow override_experimental \
  --max-frames 120
```

### 15.5 Audit

```bash
PYTHONPATH=. python scripts/audit_retargeting_v3_step3_runtime_shadow.py \
  --artifact-dir artifacts/retargeting_v3_step3_runtime_shadow \
  --source-root . \
  --write-report artifacts/retargeting_v3_step3_runtime_shadow/acceptance_ledger.json
```

---

## 16. CI requirements

New workflow：

```text
.github/workflows/retargeting_v3_step3_runtime_shadow.yml
```

Must include jobs：

```text
runtime-shadow-unit-tests
runtime-shadow-lfs-smoke
runtime-shadow-artifact-audit
```

CI must：

- checkout with LFS;
- `git lfs pull && git lfs fsck`;
- install package;
- run targeted tests;
- run artifact audit;
- not require private assets;
- not run full long smoke unless workflow_dispatch/self-hosted;
- upload compact artifacts if generated。

If CI cannot run full RPO/G1 smoke on GitHub runner, final local artifacts must contain smoke results and CI must at least audit committed artifacts.

---

## 17. Final artifacts requirements

```text
artifacts/retargeting_v3_step3_runtime_shadow/
  environment.json
  commands.txt
  profile_resolution.json
  shadow_summary.json
  override_smoke_summary.json
  smoke_matrix.json
  deterministic_rerun.json
  acceptance_ledger.json
  per_clip/<robot>/<clip>/target_deltas.json
  per_clip/<robot>/<clip>/pipeline_summary.json
  test_results/pytest.txt
  test_results/junit.xml
  test_results/pytest_summary.json
```

`environment.json` must include：

```text
source_code_commit
source_code_commit_remote_resolvable=true
source_code_commit_is_artifact_commit_ancestor=true
source_worktree_clean_before_run=true
source_worktree_clean_after_run=true
git_status_short=""
python/mujoco/newton/warp/numpy/scipy versions
```

---

## 18. Final acceptance checklist

### Safety/no-op

- [ ] default config output unchanged;
- [ ] disabled mode output unchanged;
- [ ] shadow mode output unchanged;
- [ ] override requires explicit config;
- [ ] no Step 2 artifact mutation;
- [ ] no threshold changes;
- [ ] no IK weight changes。

### Runtime target stream

- [ ] source semantic frames finite;
- [ ] V3 targets finite;
- [ ] target order mapping correct;
- [ ] diagnostics deterministic;
- [ ] missing semantic fail closed。

### RPO/G1

- [ ] RPO strict profile match;
- [ ] RPO shadow pass;
- [ ] RPO override smoke pass;
- [ ] G1 shadow pass or explicit fingerprint-skip;
- [ ] G1 override pass or explicit fail-closed;
- [ ] no private assets。

### Artifacts/CI

- [ ] all required artifacts exist;
- [ ] pytest/junit saved;
- [ ] smoke matrix saved;
- [ ] audit PASS;
- [ ] CI PASS;
- [ ] Agent F PASS;
- [ ] worktree clean;
- [ ] LFS OK。

---

## 19. Completion definition

Final report must be exactly one of：

```text
Step 3.0 Runtime Profile Shadow: PASS
```

or：

```text
Step 3.0 Runtime Profile Shadow: BLOCKED
```

PASS requires all：

```text
default no-op verified
shadow no-op verified
RPO profile strict matched
RPO shadow diagnostics finite
RPO override smoke finite/deterministic
G1 shadow finite or fingerprint-skip recorded
G1 override pass or fail-closed recorded
full tests pass
runtime artifacts clean
CI green
Agent F PASS
no Step 3.1 quality claims
```

If any hard gate fails, write BLOCKED and list blockers. Do not hide failure by changing mode to disabled or deleting diagnostics.

**即使 PASS，也必须在此停止。不要自动开始 Step 3.1 full runtime quality / visual evaluation。**
