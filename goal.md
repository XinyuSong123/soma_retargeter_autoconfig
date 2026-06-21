# Goal — Step 2 Finalize：把离线运动学编译器真正接通，并完成全量 Robot Zoo 验收

> **Codex 执行契约**
>
> 当前分支：`retargeting-v3-step2-finalize`  
> 起始分支：`retargeting-v3-step2-integrated-local`  
> 起始提交：`a51ea3d49ba1f61ef0a79153896c148407e3cfaa`  
> 当前阶段：**Step 2 最终闭环**  
>
> 本轮必须使用 **6 个高强度专业 subagents**，由主 Codex 作为 Integrator。六个 agent 必须全部真实工作、提交独立 commits、保存 handoff，并由 Agent F 最终红队复核。
>
> **不得开始 Step 3，不得修改生产 `NewtonPipeline`，不得进行 whole-body IK、contact runtime、collision、temporal 或权重搜索。**

---

## 0. 开始前必须阅读

1. `/goal.md`（本文件）
2. `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
3. `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
4. `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
5. `docs/retargeting_v3/STEP2_AGENT_MERGE_PLAN.md`
6. `docs/retargeting_v3/subagents/step2_agent_{a,b,c,d,e}_handoff.md`
7. `docs/retargeting_v3/subagents/step2_agent_f_red_team.md`
8. `STEP2_IMPLEMENTATION_REPORT.md`
9. `assets/robot_zoo/robot_zoo_manifest.json`

不得只读 handoff 摘要后开始改代码；必须检查当前实现与测试。

---

## 1. 当前真实基线

### 1.1 当前分支已有的有效成果

- runtime FK、q/nv metadata、tangent integration 已从运行时模型获取；
- full-chain/LCA 路径基础已建立；
- numerical relative Jacobian 与 multi-pose reachability 已有测试；
- RPO 已有非零 distal hand、sole、toe、heel sites；
- rest calibration 不再直接把误差硬编码为零；
- rotation delta 已使用 `A ΔR Aᵀ` 共轭；
- root horizontal displacement 已按腿长比例缩放；
- chain projection 已支持 neutral/continuity prior、rank-zero unreachable demand；
- manifest-driven Robot Zoo 骨架、acceptance audit 和 CI 骨架已经存在。

### 1.2 当前环境中已执行的组合测试

```text
49 passed
2 failed
```

失败分别是：

1. `test_source_reference_target_reconstruction_is_global_transform_equivariant`
   - 旧测试要求人体世界平移原样复制；
   - 新设计要求水平位移按腿长比例缩放、垂直位移按支撑高度变化；
   - 应重写测试表达新数学契约，不能回退算法。

2. `test_retargeting_v3_step2_acceptance_gates_are_clean`
   - 当前读取的是旧 `artifacts/retargeting_v3_step2`；
   - 旧 artifacts 导致 104 个 blockers；
   - 必须先完成主路径与全量验证，再从 clean source commit 重新生成 artifacts。

### 1.3 当前仍真实存在的核心阻断

1. `profile.py` 仍把 robot neutral FK 当作 projection desired，执行 neutral→neutral 假投影；
2. `canonical_projection.py::project_canonical_targets()` 已存在，但没有接入 compiler 主路径；
3. `target_builder.py` 的 zero-length source edge 分支引用未定义的 `global_delta/_apply_global_delta`；
4. 只有 RPO 有 asset-backed verified semantic map；
5. 非 RPO positive humanoid 仍可通过 body-name inference 进入 `passed`；
6. 完整 Robot Zoo 没有真正执行，之前 no-fetch 结果是 `1 passed + 45 source_unavailable`；
7. same-source G1 URDF→canonical MJCF strict equivalence 没有运行；
8. vendor URDF 与 Menagerie MJCF 的 variant compatibility 没有完成；
9. 真实 G1 23DoF 闭环验证未完成；
10. deterministic two-run 没有实现；
11. 旧 artifacts、绝对路径、dirty metadata、旧 summary 仍在仓库；
12. CI 没有真实通过记录；
13. Agent D 原分支没有自写 handoff，当前文件只是 integration reconstruction。

---

## 2. 本轮唯一目标

将当前离线模块连成一个可审计的完整编译流程：

```text
fixed model source
→ runtime model truth
→ fingerprint-bound verified semantics/sites
→ full-chain paths
→ numerical Jacobians
→ multi-pose reachability
→ independent rest calibration
→ canonical SOMA semantic motions
→ morphology target construction
→ per-motion torso/hand/foot closest-reachable projection
→ capability-aware validation
→ full Robot Zoo matrix
→ deterministic rerun
→ clean artifacts
→ Agent F audit PASS
```

Step 2 完成后应能回答：

- 对某个完整机器人，腰部、手臂、腿实际能实现哪些 semantic demands？
- canonical torso/hand/foot target 被投影成什么？
- 不可达 demand 有多大？
- site、chain、rank、calibration、projection 结果能否在第二次运行中复现？

本轮仍不回答“生产 demo 的最终动作是否好看”，那属于 Step 3 whole-body integration。

---

## 3. 严格范围边界

### 3.1 本轮包含

- `soma_retargeter/robotics/v3/` 离线编译器；
- `compile_kinematic_profile_v3()` 主路径；
- canonical chain projection；
- calibration/target bug 修复；
- verified semantic/site maps；
- Robot Zoo source resolution；
- full manifest validation；
- same-source cross-format；
- deterministic rerun；
- acceptance audit；
- CI、JUnit、compact artifacts、implementation report。

### 3.2 本轮禁止

- 修改 `soma_retargeter/pipelines/newton_pipeline.py`；
- 把 v3 接入当前 demo；
- whole-body IK；
- contact runtime；
- foot locking；
- collision；
- temporal smoothing；
- direction/pole-vector；
- task weight tuning；
- viewer/UI；
- teacher refinement；
- pose-pair optimizer；
- 根据机器人名称写数学特例；
- 为通过 audit 修改 audit 去忽略真实问题；
- 开始 Step 3。

---

## 4. 六个高强度 Subagents

主 Codex 是 **Integrator**，负责接口冻结、冲突解决、合并、全量测试和最终诚实结论。

启动后必须一次创建六个 agent，全部使用最高可用 reasoning effort。每个 agent：

- 使用独立 worktree/branch；
- 只修改自己的 owned files；
- 提交小 commits；
- 保存真实 handoff；
- handoff 包含：commit、files、commands、tests、数值结果、假设、未完成项、风险；
- 不得宣称未运行的测试通过；
- 不得降低 acceptance gates。

### Agent A — Compiler Wiring & Projection Contract

**目标**：将已经存在的 canonical target 和 chain projection 正确接入 `KinematicProfileV3` 主路径。

**Owned files**

```text
soma_retargeter/robotics/v3/profile.py
soma_retargeter/robotics/v3/canonical_projection.py
soma_retargeter/robotics/v3/chain_projection.py
soma_retargeter/robotics/v3/profile_validation.py          # 可新增
tests/v3/test_profile_canonical_projection_*.py
tests/v3/test_projection_contract_*.py
docs/retargeting_v3/subagents/finalize_agent_a_handoff.md
```

**必须实现**

1. 删除/隔离 compiler 中 neutral→neutral `projection_reports`；
2. 正确顺序：

```text
adapter/sites/paths
→ Jacobian/reachability
→ source rest/calibration
→ canonical motion targets
→ project_canonical_targets
→ validation/status
```

3. `projection_reports` 顶层必须按 motion 组织：

```json
{
  "neutral": {"torso": {}, "left_hand": {}, ...},
  "torso_pitch": {...},
  "arms_forward": {...}
}
```

4. 每项保存：
   - desired；
   - projected；
   - residual/normalized residual；
   - chain q；
   - iterations；
   - active coordinates；
   - prior residual；
   - solver status/message；
   - `desired_source="canonical_semantic_target_relative_transform"`；
   - `neutral_as_desired=false`（除 neutral motion 外也必须明确 false）；
5. profile status 不能只依据 `failures=[]`；必须检查 canonical coverage、projection convergence、rank-zero evidence；
6. adapter 使用 `try/finally` 关闭；
7. projection deterministic；
8. RPO torso pitch/roll/yaw、arms-forward、overhead、squat、single-step 有实际 projection evidence；
9. 不允许生成只有 neutral projection 的 profile；
10. 修正 Agent D handoff，保存真实命令和测试结果。

### Agent B — Calibration, Target Policy & Regression Repair

**目标**：关闭 calibration/target builder 的剩余数学与回归问题。

**Owned files**

```text
soma_retargeter/robotics/v3/source_rest.py
soma_retargeter/robotics/v3/rest_frames.py
soma_retargeter/robotics/v3/target_builder.py
soma_retargeter/robotics/v3/calibration_validation.py
tests/test_kinematic_profile_v3_calibration_targets.py
tests/v3/test_rest_calibration_*.py
tests/v3/test_target_builder_*.py
docs/retargeting_v3/subagents/finalize_agent_b_handoff.md
```

**必须实现**

1. 修复 undefined `global_delta/_apply_global_delta` 路径；
2. zero-length source edge 使用明确 fallback：
   - previous valid source direction；
   - source rest direction；
   - robot neutral relative transform；
   - 保存 fallback reason/confidence；
3. 不允许 silent identity；
4. 重写旧 global-transform test：
   - common source yaw 保持局部 articulation；
   - horizontal root displacement 按 `root_horizontal_scale`；
   - vertical root displacement 按 support-height delta；
   - 不能再要求世界平移原样复制；
5. 保留并加强 `A ΔR Aᵀ` 测试；
6. neutral validation 必须独立于 runtime neutral target copy；
7. 增加非中立几何 oracle：torso rotation、arm redirect、single-step；
8. calibration errors 可精确为零，但必须附独立计算 evidence，audit 不得仅因数值为零误报；
9. source rest 加载失败时：
   - positive humanoid 验收不能用 robot-neutral proxy 后仍标 passed；
   - 只能 `algorithm_failed` 或明确 blocked；
10. 不读取 legacy scaler/offset/pose-pair。

### Agent C — Semantic Core & Anchor Robot Verified Maps

**目标**：建立通用 verified-map 流程，并完成最关键完整机器人的语义/site ground truth。

**Owned files**

```text
soma_retargeter/robotics/v3/semantic_sites.py
soma_retargeter/robotics/v3/site_geometry.py
soma_retargeter/robotics/v3/semantic_validation.py
soma_retargeter/robotics/v3/verified_semantic_maps.py      # 可新增
assets/robot_zoo/semantic_maps/<Agent-C shard>
assets/robot_zoo/semantic_expectations/<Agent-C shard>
tests/v3/test_semantic_core_*.py
tests/v3/test_anchor_robot_semantics_*.py
docs/retargeting_v3/subagents/finalize_agent_c_handoff.md
```

**Agent C shard**

- `roboparty_rpo_local`
- `unitree_g1_urdf`
- `unitree_g1_mjcf`
- 真实 `unitree_g1_23dof` source
- `unitree_h1_urdf`
- `unitree_h1_mjcf`
- `unitree_h1_2_urdf`
- `unitree_h1_2_mjcf`
- `robotis_op3_mjcf`
- `booster_t1_urdf`
- `booster_t1_mjcf`
- `pal_talos_mjcf_direct`
- `berkeley_humanoid_urdf`
- `berkeley_humanoid_mjcf_direct`

**必须实现**

1. 通用 map lookup：`assets/robot_zoo/semantic_maps/<model_id>.json`；
2. 删除 validation 中 `if entry.id == "roboparty_rpo_local"` 专用分支；
3. positive humanoid 无 verified map 时必须 `semantic_failed`，不能自动 inferred→passed；
4. verified map 必须绑定 model fingerprint/source hash；
5. map schema 保存：body/site/local pose/confidence/evidence/verification status；
6. inference 只产生 candidate report，不能作为 final ground truth；
7. distal Hand 与 orientation capability 解耦；
8. Foot 至少含 sole center；模型几何允许时含 toe/heel；
9. TALOS foot 必须 distal；
10. Booster Hips/Chest 必须正确暴露 waist chain；
11. OP3 fixed/zero-DoF torso 是合法结果；
12. RPO 已有 offsets 必须通过 compiled geometry/site cross-check；
13. 不允许把 `[0,0,0]` site 自动视为 distal；
14. verified maps 左右 side、topology、neutral geometry 全部验证。

### Agent D — Robot Zoo Semantic Shard A

**目标**：为第一批其余 positive humanoids建立完整 verified maps 与 semantic expectations。

**Owned files**

```text
assets/robot_zoo/semantic_maps/<Agent-D shard>
assets/robot_zoo/semantic_expectations/<Agent-D shard>
tests/v3/test_robot_zoo_semantics_shard_a.py
docs/retargeting_v3/subagents/finalize_agent_d_handoff.md
```

**Agent D shard**

- `toddlerbot_urdf`
- `toddlerbot_2xc_mjcf`
- `toddlerbot_2xm_mjcf`
- `adam_lite_mjcf`
- `apptronik_apollo_mjcf`
- `fourier_n1_urdf`
- `fourier_n1_mjcf`
- `jvrc_urdf`
- `jvrc_mjcf`
- `elf2_urdf`
- `elf2_mjcf`

**要求**

- 每个 source 固定 ref/hash；
- 每个 positive humanoid verified map完整；
- 若 URDF/MJCF 是不同 variant，明确记录差异；
- 每个 Hand/Foot site 都有 evidence；
- map 必须由 runtime topology/FK/compiled geometry验证；
- 不修改 semantic core；
- 不能以文件名或机器人类别代替实际验证。

### Agent E — Robot Zoo Semantic Shard B & Morphology Controls

**目标**：完成剩余 positive、fetch-only 与 negative-control 的语义/形态证据。

**Owned files**

```text
assets/robot_zoo/semantic_maps/<Agent-E positive shard>
assets/robot_zoo/semantic_expectations/<Agent-E all shard>
tests/v3/test_robot_zoo_semantics_shard_b.py
tests/v3/test_robot_zoo_negative_controls.py
docs/retargeting_v3/subagents/finalize_agent_e_handoff.md
```

**Positive shard**

- `ergocub_urdf`
- `draco3_urdf`
- `sigmaban_urdf`
- `simple_humanoid_urdf`
- `atlas_drc_urdf`
- `atlas_v4_urdf`
- `romeo_urdf`
- `mujoco_humanoid_mjcf`

**Fetch-only positive**

- `fourier_gr1_urdf`
- `jaxon_urdf`
- `talos_urdf`
- `valkyrie_urdf`
- `robonaut2_urdf`

**Negative controls**

- `cassie_urdf`
- `cassie_mjcf`
- `upkie_urdf`
- `bolt_urdf`
- `rhea_urdf`
- `unitree_go2_urdf`
- `unitree_go2_mjcf`
- `franka_panda_mjcf`
- `stretch3_mjcf`

**要求**

- positive 有 verified maps；
- negative controls 不生成伪 humanoid profile；
- biped-no-arms 可结构化 partial/unsupported；
- Franka Panda 只指公开 Franka Emika Panda；
- 禁止任何公司私有资产或 `cxxx_190`；
- fetch-only 固定 ref、许可证和获取状态；
- source unavailable 不算 algorithm pass。

### Agent F — Full Validation, Cross-Format, Artifacts, CI & Red Team

**目标**：把 A–E 输出接入统一验证，生成最终证据，并独立决定 PASS/BLOCKED。

**Owned files**

```text
soma_retargeter/robotics/v3/robot_zoo.py
soma_retargeter/robotics/v3/validation.py
soma_retargeter/tools/compile_kinematic_profile_v3.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
soma_retargeter/tools/sync_robot_zoo_v3.py
soma_retargeter/tools/convert_robot_model_v3.py
scripts/audit_retargeting_v3_step2.py
.github/workflows/retargeting_v3_step2.yml
tests/v3/test_full_robot_zoo_*.py
tests/v3/test_acceptance_*.py
artifacts/retargeting_v3_step2/
docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md
docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md
docs/retargeting_v3/STEP2_KNOWN_LIMITATIONS.md
docs/retargeting_v3/subagents/finalize_agent_f_red_team.md
```

**必须实现**

1. manifest 46 个 entry 全部生成结构化结果；
2. required source 不允许 silent `source_unavailable`；
3. 使用 verified maps，不允许 positive inferred→passed；
4. full canonical projection coverage gate；
5. same-source G1 URDF→canonical MJCF strict equivalence；
6. vendor URDF vs Menagerie MJCF variant compatibility；
7. 真实 G1 23DoF 验证；
8. 所有可比较 URDF/MJCF families 输出 pair report；
9. deterministic two-run 真正运行并比较；
10. clean-source artifact generation；
11. reproduction commands 不含本机绝对路径；
12. audit 区分“精确零但有独立 evidence”和“硬编码零”；
13. audit 检查当前代码和新 artifacts，不被旧文件污染；
14. CI 安装/使用正确依赖，不能只安装 `pytest` 后期待 Newton/MuJoCo 测试运行；
15. 最终红队报告只能是 `PASS` 或 `BLOCKED`。

---

## 5. Agent 协作和集成节奏

### Wave 1 — 六个 agent 同时审计

- A：定位 profile 主路径接线与 schema；
- B：定位 calibration/target/test 语义冲突；
- C：冻结 verified map schema；
- D/E：完成各自 source inventory 和候选 maps，不在 schema 冻结前提交最终 maps；
- F：先写/更新能重现当前阻断的 failing gates。

Integrator 冻结以下接口：

```text
VerifiedSemanticMapV3
SemanticSiteV3
RestCalibrationV3
CanonicalSemanticMotion
CanonicalProjectionReport
KinematicProfileV3
RobotValidationResultV3
CrossFormatReportV3
DeterminismReportV3
```

### Wave 2 — 并行实现

- A、B、C 实现核心；
- D、E 按冻结 schema 完成 maps/expectations；
- F 完成验证 orchestration、audit、CI。

### Wave 3 — 严格依赖集成

```text
B calibration/targets
→ C semantic core/maps
→ D/E map shards
→ A compiler/projection wiring
→ F full validation/audit
```

若 A 需要 C/B 接口，可先 rebase，不得复制临时兼容层长期保留。

### Wave 4 — 红队修复循环

- F 输出 blockers；
- Integrator 将每个 blocker 指派给 owner；
- owner 修复并提交；
- F 重新运行；
- 直到 zero blockers 或诚实标记 BLOCKED。

---

## 6. Compiler 主路径的强制数据契约

### 6.1 禁止 neutral placeholder

旧代码：

```python
neutral_rel = adapter.relative_transform(...)
project_*(..., desired=neutral_rel)
```

必须删除或仅保留为显式 `neutral_projection_smoke`，不能写入最终 canonical projection evidence。

### 6.2 正确主路径

```python
adapter = ...
sites = verified_semantic_sites(...)
paths = discover_paths(...)
jacobians = ...
reachability = ...
source_rest = ...
calibration = ...
canonical_targets = canonical_motion_targets(calibration)
canonical_projections = project_canonical_targets(
    adapter,
    sites,
    paths,
    canonical_targets,
)
validation = validate_profile(...)
```

### 6.3 Canonical coverage

至少包括：

1. neutral
2. root_translation
3. global_root_yaw
4. torso_pitch
5. torso_roll
6. torso_yaw
7. mixed_torso_rotation
8. arms_forward
9. elbow_bend
10. overhead_reach
11. squat
12. single_step_target

可增加 asymmetric/cross-body/limit stress，但不得删除上述项来减少失败。

### 6.4 每个可用 task 都投影

- torso；
- left/right hand；
- left/right foot。

缺失 task 必须有 capability reason，不得静默 absence。

---

## 7. Verified semantics 通过标准

### 7.1 Positive humanoid

必须有 asset-backed verified map：

```text
assets/robot_zoo/semantic_maps/<model_id>.json
```

并至少包含：

- schema version；
- model id；
- source/ref/hash；
- model fingerprint；
- verification status；
- Hips/Chest；
- Left/Right Hand（若形态存在）；
- Left/Right Foot；
- local pose；
- source/confidence/evidence；
- capability notes。

### 7.2 Inference 的角色

Inference 只能：

- 生成候选；
- 保存 alternatives；
- 提供 evidence；
- 帮助人工/程序验证。

Inference 不能：

- 直接获得 `verification_status=verified`；
- confidence=1；
- 使 positive humanoid status 变成 passed。

### 7.3 Distal sites

- Hand：末端几何、explicit site、distal child anchor 或有证据 extrapolation；
- Foot：sole center；有几何时 toe/heel；
- body origin 只有在证明确为 endpoint 时可用；
- orientation support 与 position site 独立。

---

## 8. Robot Zoo 全量验收

Manifest 是唯一模型列表。每个 entry 必须输出：

```text
artifacts/retargeting_v3_step2/per_robot/<id>.json
```

允许状态：

```text
passed
partial_passed
negative_control_passed
algorithm_failed
semantic_failed
model_load_failed
source_unavailable
license_blocked
```

### 8.1 Required positive

必须：

```text
source available
+ model load
+ verified semantics/sites
+ full paths
+ Jacobians/reachability
+ calibration
+ canonical targets
+ canonical projection
+ deterministic rerun
```

`source_unavailable` 对 `required=true` 是硬失败。

### 8.2 Negative controls

必须加载或明确 source failure，并验证：

- 不伪装成 full humanoid；
- 不生成虚假 Hips/Chest/Hands/Feet；
- morphology classification正确；
- 结果为 `negative_control_passed` 或明确 failure。

### 8.3 Fetch-only

- 固定 source/ref/license；
- 允许 cache/network 获取；
- 不提交受限模型；
- unavailable 不能计入 algorithm pass；
- 保存复现命令和原因。

---

## 9. G1 Cross-Format 三层验收

### Gate A — same-source strict equivalence

同一个固定 G1 URDF：

1. 直接加载 URDF；
2. deterministic 转 canonical MJCF；
3. 使用同一 verified semantic map；
4. 比较：
   - topology；
   - joint types/axis/limits；
   - neutral semantic FK；
   - active chains；
   - rank；
   - canonical projection；
5. 必须 strict pass。

### Gate B — vendor/Menagerie variant compatibility

- 自动识别额外/缺失 DoF；
- 比较共同 semantic capabilities；
- 不宣称 strict equivalence；
- 两个模型分别通过自己的 algorithm gates。

### Gate C — real 23DoF

- 使用固定真实 23DoF source；
- 不得只锁定 29DoF 代替；
- locked-29DoF 只可作为额外 ablation；
- 保存真实 chain/rank/projection 报告。

---

## 10. RPO 专项验收

1. torso chain 只含真实腰部 coordinate；
2. torso regular rotational rank=1；
3. axis在pelvis frame表达；
4. common world rotation不改变兼容轴结论；
5. pure incompatible pitch/roll 对 hinge 响应 ≤ compatible rotation 的5%；
6. compatible yaw/twist 正确保留且受limit限制；
7. full shoulder/elbow hand chain；
8. nonzero distal hand sites；
9. full hip/knee/ankle foot chain；
10. sole/toe/heel sites；
11. arms_forward/elbow_bend/overhead 真正投影；
12. squat/single_step 真正投影；
13. residual非零时诚实报告；
14. no-wrist orientation capability 不影响 hand position site。

---

## 11. 测试要求

### 11.1 必须保留并通过

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/v3
python -m pytest -q \
  tests/test_kinematic_profile_v3_adapter.py \
  tests/test_kinematic_profile_v3_math.py \
  tests/test_kinematic_profile_v3_calibration_targets.py
```

### 11.2 旧 global-transform 测试

必须重写，不得删除。新断言：

- global yaw equivariance；
- horizontal displacement scale；
- support-height vertical policy；
- local semantic articulation invariance。

### 11.3 Full suite

```bash
python -m pytest -q
```

若仓库全量测试包含与 Step 2 无关的外部依赖失败，必须分类并保存；Step 2 自身 tests 不允许失败。

### 11.4 Audit

```bash
python scripts/audit_retargeting_v3_step2.py \
  --artifact-dir artifacts/retargeting_v3_step2 \
  --source-root . \
  --output-json /tmp/step2_audit.json \
  --junit-xml /tmp/step2_audit.xml
```

最终：

```text
blocking_count == 0
status == PASS
```

不得通过删除 gate 名或把错误字段改名绕过。

---

## 12. Deterministic rerun

对所有 required positive models：

- 相同 source hash；
- 相同 map hash；
- 相同 seed；
- 运行两次；
- profile deterministic hash相同；
- semantic sites相同；
- path/rank相同；
- projection status相同；
- q/residual在容差内；
- warnings/fallbacks相同。

输出：

```text
artifacts/retargeting_v3_step2/deterministic_rerun.json
```

不能再写 `not_run` 或 `scaffolded` 后算完成。

---

## 13. Clean Artifacts Protocol

旧 artifacts 必须移除或整体重建，不得混合新旧 schema。

推荐流程：

1. 所有核心代码、maps、tests 先 commit；
2. 在该 source-code commit 建一个全新 clean detached worktree；
3. 在 clean worktree 中运行 tests 和 full zoo；
4. artifacts 输出到临时目录；
5. 记录：
   - `source_code_commit`；
   - `source_worktree_clean_before_run=true`；
   - environment/package/GPU；
   - manifest/map/source hashes；
6. 将 compact artifacts复制回主 worktree；
7. commit artifacts/docs；
8. artifact commit之后不得修改核心代码；如修改，全部重跑。

`environment.json` 不应要求 artifact commit 自己等于 source code commit；应验证 source code commit 是当前 artifact commit 的祖先，并且之后没有 core-code diff。

### 13.1 必须生成

```text
artifacts/retargeting_v3_step2/
  environment.json
  commands.txt
  source_inventory.json
  summary.json
  acceptance_ledger.json
  cross_format.json
  deterministic_rerun.json
  per_robot/*.json
  failures/*.json
  test_results/pytest_summary.json
```

JUnit 可以作为 GitHub Actions artifact，不必用 Git LFS 提交大文件。

### 13.2 禁止

- `/mnt/ssd1/song/...` 等本机绝对路径；
- dirty source worktree；
- 旧 HEAD；
- 旧 `compiled` 状态；
- `1 pass + 45 source_unavailable` 被称为全量验证；
- 旧 RPO zero-offset report；
- stale Agent F audit结果。

---

## 14. CI 要求

修复 `.github/workflows/retargeting_v3_step2.yml`：

- 不得只安装 pytest；
- 必须安装项目及测试依赖；
- CPU math/static/audit job可在GitHub-hosted runner；
- Newton/full-zoo如果必须使用项目环境，可使用明确的self-hosted/manual job；
- required tests不能因依赖缺失而静默skip；
- 上传compact test/audit artifacts；
- 当前分支至少有可见workflow run；
- CI失败时不得宣称Step2通过。

---

## 15. 私有资产规则

- 只使用 manifest 中公开来源与仓库已有 RPO；
- Franka Panda 是公开 negative control，可使用；
- 禁止 `cxxx_190`；
- 禁止任何用户公司内部模型、路径、名称或资产；
- 不扫描 manifest 外本地资产目录；
- cache 中 manifest 外文件必须忽略。

---

## 16. Acceptance Ledger

### Compiler

- [ ] canonical projection 接入主路径；
- [ ] 无 neutral placeholder；
- [ ] 每个 motion/task有真实报告；
- [ ] rank0 demand保留；
- [ ] adapter正确关闭；
- [ ] projection deterministic。

### Calibration/targets

- [ ] zero-length bug修复；
- [ ] global-transform测试表达新策略；
- [ ] `A ΔR Aᵀ`；
- [ ] horizontal/vertical root policy；
- [ ] independent neutral evidence；
- [ ] no legacy offsets。

### Semantics

- [ ] positive required全部verified maps；
- [ ] maps绑定fingerprint/hash；
- [ ] distal Hand/Foot evidence；
- [ ] TALOS/Booster修正；
- [ ] inference不能直接pass；
- [ ] negative controls正确降级。

### Robot Zoo

- [ ] 46个entry都有新schema结果；
- [ ] required source无unavailable；
- [ ] required positive全部通过；
- [ ] negative controls通过；
- [ ] fetch-only诚实分类；
- [ ] 无静默skip。

### Cross-format

- [ ] same-source G1 strict pass；
- [ ] variant compatibility完成；
- [ ] real G1 23DoF完成；
- [ ] 其他可比较pairs有结果。

### Reproducibility

- [ ] two-run deterministic；
- [ ] clean source worktree；
- [ ] 无绝对路径；
- [ ] artifacts为最终schema；
- [ ] tests通过；
- [ ] audit blockers=0；
- [ ] CI有结果；
- [ ] 六个finalize handoffs完整；
- [ ] Agent F red-team PASS。

---

## 17. 明确禁止的捷径

以下任何一种都算失败：

- 只将 `project_canonical_targets` import 但不实际调用；
- 把 neutral projection 重命名成 canonical；
- 只跑 RPO；
- 只写 46 个 `source_unavailable` JSON；
- 自动 inference map后标 verified；
- 为每个机器人加专用数学 if/else；
- 删除失败 canonical motion；
- rank0 demand置零；
- 修改 audit 忽略 TALOS/Booster/G1/RPO 问题；
- 删除旧 test而不写新契约 test；
- 使用 synthetic G1 23DoF代替真实模型；
- 提交 dirty artifacts；
- 把 no-fetch scaffold当全量结果；
- 没有 Agent F PASS 就进入Step3；
- 修改生产 pipeline。

---

## 18. 推荐提交顺序

1. `test: expose final Step 2 compiler and target blockers`
2. `fix: repair zero-length target fallback and root policy tests`
3. `feat: add generic fingerprint-bound verified semantic maps`
4. `data: verify anchor robot semantic sites`
5. `data: verify robot zoo semantic shard a`
6. `data: verify robot zoo semantic shard b and controls`
7. `feat: wire canonical projections into kinematic profiles`
8. `feat: add capability-aware profile validation`
9. `feat: run same-source and variant cross-format gates`
10. `feat: run full manifest validation and deterministic rerun`
11. `test: close Step 2 acceptance audit blockers`
12. `ci: run Step 2 validation workflows`
13. `docs: publish six handoffs and final implementation report`
14. `artifacts: publish clean final Step 2 evidence`

禁止再次压成一个不可审查的大提交。

---

## 19. Codex 启动步骤

1. `git status --short`
2. `git branch --show-current`
3. `git rev-parse HEAD`
4. 确认分支为 `retargeting-v3-step2-finalize`
5. 确认起始祖先包含 `a51ea3d`
6. 激活正确环境：`soma-retargeter-v2`
7. 记录 Python/MuJoCo/Newton/Warp/NumPy/SciPy 版本
8. 一次创建六个高强度subagents
9. 运行当前49-pass/2-fail基线并保存
10. Wave 1审计与接口冻结
11. 按ownership实现
12. 按依赖顺序集成
13. 运行full tests
14. 运行full Robot Zoo with source fetch/cache
15. deterministic rerun
16. clean artifact generation
17. Agent F red-team
18. blocker返修循环
19. push所有commits
20. 到此停止，不进入Step3

---

## 20. 完成定义

只有同时满足以下条件，才能在最终报告写：

```text
Step 2: PASS
```

- [ ] 当前49-pass/2-fail基线中的两项失败均被正确关闭；
- [ ] canonical projection真正接入compiler；
- [ ] target builder zero-length路径无bug；
- [ ] required positive humanoids全部verified semantics/sites；
- [ ] full manifest运行；
- [ ] required entries无source unavailable；
- [ ] RPO专项通过；
- [ ] G1 same-source strict pass；
- [ ] G1 variant compatibility完成；
- [ ] real G1 23DoF完成；
- [ ] deterministic rerun通过；
- [ ] full tests通过；
- [ ] audit blocking_count=0；
- [ ] clean final artifacts；
- [ ] CI有通过结果或明确可信self-hosted结果；
- [ ] 六个高强度subagent handoffs完整；
- [ ] Agent F最终结论为PASS；
- [ ] 所有commits已push；
- [ ] 未修改生产NewtonPipeline；
- [ ] 未开始Step3。

任一硬门槛未满足，最终报告必须写：

```text
Step 2: BLOCKED
```

并列出阻断项，不得通过措辞包装为“基本完成”。
