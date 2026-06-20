# Goal — Step 2 完整收敛：正确的离线运动学编译器与全量机器人验证

> **Codex 执行契约**
>
> 当前分支：`retargeting-v3-step2-kinematic-core`  
> 当前原型基线：`860a538ac25ce675b4b588bf4d2f5295ba80a463`  
> 干净代码祖先：`dev@aed9f9fbf09a1bf2816d8bfa5c8423ea2300cf2c`  
> Step 1 数学规范：
>
> - `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
> - `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
> - `docs/retargeting_v3/THREE_STEP_ROADMAP.md`
>
> 开始前必须完整阅读以上文件和本文件。当前代码只是 Step 2 原型，不能因为 `summary.json` 显示九台机器人 `compiled` 就宣称 Step 2 完成。
>
> **本轮只完成 Step 2。不得开始 Step 3，不得接入生产 `NewtonPipeline`，不得调 whole-body IK 权重。**

---

## 0. 当前审计结论：本轮必须逐项关闭

| 项目 | 当前判定 | 本轮要求 |
|---|---|---|
| 代码已推送 | 通过 | 保留 |
| runtime adapter 原型 | 基本通过 | 修复模型真值、site、fingerprint 与 cross-format |
| full-chain path | RPO 基本通过 | 扩展到全部机器人并验证 LCA/完整链 |
| numerical Jacobian | synthetic 基本通过 | 加强 Newton/MuJoCo cross-check 与完整模型 gate |
| multi-pose rank | synthetic 基本通过 | 全量机器人执行，禁止假 rank |
| rest calibration | 未通过 | 真正实现并独立测量误差 |
| distal hand/sole site | 未通过 | 自动生成并实际用于 target/projection |
| canonical complete-model projection | 未实现 | 对每台机器人、每个 canonical motion 真正运行 |
| G1 URDF/MJCF equivalence | 未通过 | 改成同源严格等价 + 异源变体兼容两层验证 |
| 完整机器人语义正确性 | 未通过 | verified semantic maps + 几何/拓扑证据 |
| artifacts 可复现性 | 未通过 | 最终干净 commit 重跑，两次确定性验证 |
| subagent/red-team 流程 | 无证据 | 必须使用 6 个专业 subagent 并保存 handoff |
| Step 2 总体 | 未完成 | 本轮全部硬门槛通过后停止 |

### 0.1 已确认的主要假阳性

当前实现中以下现象不得继续被视为“通过”：

1. `projection_reports` 只把 neutral FK 投影回 neutral FK，`desired == projected`、residual=0；
2. `calibrate_rest_frames()` 直接把 position/orientation errors 初始化为零；
3. RPO Hand 和 Foot semantic site 仍是 body 原点 `[0,0,0]`；
4. canonical target 虽生成，却没有逐 motion 输入真实 torso/endpoint chain projection；
5. 自动语义推断在 Newton 没有 world body 时，用 body 名称字符串长度近似“深度”；
6. 推断语义被包装成 `explicit_semantic_override`、confidence=1.0；
7. TALOS foot、Booster T1 Hips/Chest 等已出现明显错误映射；
8. G1 URDF/MJCF 差异被测试固化为 `documented_limitation`；
9. artifacts 在旧 HEAD、dirty worktree 下生成，命令包含本机绝对路径；
10. 没有正式 pytest/JUnit 结果、implementation report、六个 subagent handoff 和独立 red-team 结论。

不得通过改名、改 status 或降低 gate 来关闭这些问题。

---

## 1. 本轮唯一目标

把现有原型修正成一个可信的、离线的 `KinematicProfileV3` 编译器，能够对 **Robot Zoo manifest 中的全部机器人条目** 输出可复现、可审计的结果：

1. 使用实际运行时模型/FK 作为运动学真值；
2. 使用经过验证的 semantic body/site，而不是只靠名称猜测；
3. 正确生成 distal hand、sole/toe/heel 等局部 site；
4. 发现完整的相对运动学链和 active velocity coordinates；
5. 计算数值 relative Jacobian 与多姿态局部 rank；
6. 从真实 SOMA rest pose 到 robot neutral 做独立 rest calibration；
7. 使用正确的旋转共轭和逐段 target construction；
8. 对每一个 canonical motion 执行真实 torso/hand/foot chain projection；
9. 对 URDF/MJCF 做正确的同源严格等价和异源变体比较；
10. 对全量机器人产生结构化 pass/fail/unsupported 结果；
11. 在最终干净 commit 上重跑全部测试和 artifacts；
12. 由第六个 subagent 独立红队审查后才能宣布完成。

---

## 2. 硬性范围边界

### 2.1 本轮包含

- `soma_retargeter/robotics/v3/` 离线编译器；
- runtime model adapter；
- semantic site resolver 与 verified semantic maps；
- geometry/topology-based distal site inference；
- full-chain path；
- numerical relative Jacobian；
- multi-pose reachability；
- rest calibration；
- offset-free target builder；
- chain-only closest-reachable projection；
- canonical motion suite；
- Robot Zoo 全量模型加载与验证；
- cross-format equivalence；
- artifacts、CI、测试报告、subagent handoff、red-team report。

### 2.2 本轮禁止

- 修改或接入生产 `newton_pipeline.py`；
- production whole-body IK；
- contact runtime；
- collision objective；
- temporal smoothing；
- direction/pole-vector objective；
- hand/foot/chest weight sweep；
- viewer/UI/teacher refinement；
- 用机器人名字写数学特例；
- 为了让 summary 变绿而降低验收门槛；
- 开始 Step 3。

发现属于 Step 3 的问题，只写入 `known_limitations.md`，不得扩大本轮 scope。

---

## 3. 私有资产与公开模型规则

1. Robot Zoo 只使用 `assets/robot_zoo/robot_zoo_manifest.json` 中明确列出的公开来源模型和仓库内已有 RPO。
2. `franka_panda_mjcf` 指公开的 Franka Emika Panda/MuJoCo Menagerie 模型，仅作为单臂 negative control，可以保留。
3. 不得增加任何用户公司私有模型或内部代号模型，包括但不限于 `cxxx_190`。
4. 不得扫描、上传、复制用户机器上未列入 manifest 的模型目录。
5. 若本地 cache 中出现 manifest 外模型，必须忽略并在审计报告中记录“未使用”。
6. 所有上游模型必须固定 source/ref/hash；不得跟踪浮动 `main`。
7. 受许可证限制的模型只做固定版本 fetch-only 验证，不提交模型文件。

---

## 4. 六个专业 Subagent：必须全部启用

主 Codex 为 **Integrator**。启动后一次创建 6 个专业 subagent。六个 agent 必须全部工作，不能只把任务名列出来。

每个 agent 必须：

- 使用独立 worktree/branch；
- 先读 Step 1 文档、本 goal、当前代码审计；
- 提交小而可审查的 commits；
- 不修改其他 agent 拥有的文件，除非 Integrator 批准；
- 保存 handoff：`docs/retargeting_v3/subagents/step2_agent_<letter>_handoff.md`；
- handoff 必须列出 commit、文件、命令、测试、数值结果、假设、失败、风险、集成顺序；
- 不得自行降低 gate。

### Agent A — Runtime Model Truth & Cross-Format Adapter

**职责**

- Newton runtime adapter；
- MuJoCo reference adapter；
- URDF/MJCF loading/canonicalization；
- tangent integration；
- body/joint topology；
- q/qd coordinate metadata；
- model/site runtime truth；
- deterministic fingerprint；
- engine Jacobian cross-check；
- same-source URDF→canonical MJCF equivalence基础设施。

**文件所有权**

```text
soma_retargeter/robotics/v3/model_adapter.py
soma_retargeter/robotics/v3/kinematic_paths.py
soma_retargeter/robotics/v3/model_fingerprint.py        # 可新增
soma_retargeter/robotics/v3/model_conversion.py         # 可新增
soma_retargeter/robotics/v3/spatial.py                  # 仅底层变换部分
soma_retargeter/tools/convert_robot_model_v3.py         # 可新增
tests/v3/test_model_adapter_*.py
tests/v3/test_cross_format_adapter_*.py
```

**必须修复**

1. fingerprint 包含：主模型、include/xacro resolved 内容、实际引用 assets hash、loader version、conversion settings；
2. Newton model site 若运行时 API 不暴露，不得无条件把原始 XML site 当 compiled truth；需要用 MuJoCo compiled site cross-check或标记可信来源；
3. free/ball joint 使用 tangent integration；
4. fixed body 保留在 path，但不产生 DoF；
5. LCA 与 active-coordinate path 对 cross-branch 情况正确；
6. URDF mesh 绝对路径 patch 只作为临时 load copy，并记录 provenance；
7. 同源 URDF 转 MJCF 必须保留 topology、axis、limit、neutral 与 semantic FK。

**交付门槛**

- synthetic adapter tests 全通过；
- RPO、G1、H1、OP3、Booster、TALOS adapter report；
- same-source G1 URDF→canonical MJCF 严格等价报告；
- 无 static XML FK 替代 runtime FK。

### Agent B — Verified Semantics & Distal Geometry Sites

**职责**

- verified semantic maps；
- semantic confidence/evidence；
- topology-aware semantic resolver；
- distal hand site；
- sole/toe/heel/edge sites；
- geometry/site provenance；
- partial-humanoid 与 negative-control classification。

**文件所有权**

```text
soma_retargeter/robotics/v3/semantic_sites.py
soma_retargeter/robotics/v3/site_geometry.py             # 可新增
soma_retargeter/robotics/v3/semantic_validation.py       # 可新增
assets/robot_zoo/semantic_maps/
assets/robot_zoo/semantic_expectations/
tests/v3/test_semantic_sites_*.py
tests/v3/test_site_geometry_*.py
```

**必须修复**

1. 删除“用 body 名称字符串长度当树深度”的逻辑；
2. 推断结果不能标记为 `explicit_semantic_override` 或 confidence=1.0；
3. 每个 positive humanoid 都有 verified semantic map；
4. mapping 必须保存 evidence：name、topology、side、neutral position、chain、site/geom；
5. Hand position site 与 orientation capability 解耦；
6. RPO Hand 必须从 link geometry、explicit site、distal child anchor 或可靠 extrapolation 得到 distal local position；
7. RPO Foot 必须生成 sole center、toe、heel；不能只使用 ankle body 原点；
8. TALOS foot 必须映射到真实 distal foot/sole，而不是 `leg_left_1_link`；
9. Booster T1 Hips/Chest 必须反映真实 trunk/waist semantic split，不能简单同为 `Trunk` 而隐藏 waist joint；
10. partial humanoid 不得伪造 Hand；negative control 不得误判为 full humanoid。

**site 推断顺序**

1. explicit compiled site/frame；
2. verified semantic override；
3. distal child-joint anchor；
4. collision primitive bounds；
5. visual mesh bounds（只读、固定 hash）；
6. verified mirror；
7. kinematic extrapolation。

**site gate**

- positive humanoid 的 Hand/Foot 都必须有明确 local pose 和 provenance；
- body 原点只有在几何证明确为 distal/sole point 时才允许；
- `local_position=[0,0,0]` 不能自动视为通过；
- 左右镜像误差和 site confidence 必须落盘。

### Agent C — Rest Calibration & Offset-Free Target Geometry

**职责**

- SOMA source rest frames；
- robot neutral semantic frames；
- Wahba/Kabsch frame alignment；
- edge calibration；
- rotation conjugation；
- segment-local target builder；
- root scaling；
- independent neutral reconstruction validation；
- canonical source semantic motions。

**文件所有权**

```text
soma_retargeter/robotics/v3/source_rest.py
soma_retargeter/robotics/v3/rest_frames.py
soma_retargeter/robotics/v3/target_builder.py
soma_retargeter/robotics/v3/calibration_validation.py     # 可新增
tests/v3/test_rest_calibration_*.py
tests/v3/test_target_builder_*.py
```

**必须修复**

1. 禁止 `pos_errors={...:0}`、`rot_errors={...:0}` 这类循环验证；
2. neutral error 必须由独立 target generation 与 robot neutral FK 比较计算；
3. edge frame 不能只使用单方向 + 固定 world hint；应使用多语义方向、bend plane、bilateral axis 或 Wahba/Kabsch；
4. 对旋转 delta 使用正确共轭：

\[
\Delta R_r=A\,\Delta R_h\,A^T
\]

5. target builder 不读取任何 v1 joint scales/offsets/pose-pair结果；
6. source neutral 输入必须重建 robot neutral；
7. source 全局平移/旋转不改变局部 limb articulation；
8. root horizontal scale 使用 leg-length ratio；
9. vertical root 使用 pelvis-to-support-foot geometry；
10. zero-length、frame degeneracy、missing semantics 都有显式 fallback 和 confidence。

**独立 neutral gate**

- 先从 source rest 经 calibration 生成 robot targets；
- 再与独立 runtime robot neutral FK 比较；
- position max error `< 1 mm`；
- relative rotation max error `< 0.01 rad`；
- 不允许使用 robot neutral targets 直接复制成“重建结果”。

### Agent D — Reachability, Canonical Chain Projection & Mathematical Gates

**职责**

- numerical relative Jacobian；
- epsilon stability；
- deterministic multi-pose local rank；
- chain-only torso projection；
- endpoint projection；
- neutral/continuity priors；
- canonical motion projection；
- mathematical acceptance tests。

**文件所有权**

```text
soma_retargeter/robotics/v3/numerical_jacobian.py
soma_retargeter/robotics/v3/reachability.py
soma_retargeter/robotics/v3/chain_projection.py
soma_retargeter/robotics/v3/canonical_projection.py       # 可新增
tests/v3/test_numerical_jacobian_*.py
tests/v3/test_reachability_*.py
tests/v3/test_chain_projection_*.py
```

**必须修复**

1. projection 不再只做 neutral→neutral；
2. 对每个 canonical motion 的 torso、left/right hand、left/right foot 运行 projection；
3. torso desired 为真实 `Hips-relative Chest` calibrated delta；
4. endpoint desired 为 Agent C 生成的 morphology target；
5. 每个 projection 保存 desired、projected、residual、normalized residual、q、status、iterations；
6. redundant chain 优化加入 range-normalized neutral prior；
7. motion sequence 加 previous-q continuity prior；
8. 确定性初始化和必要时 deterministic multi-start；
9. rank 0 返回 `unreachable/rank_zero`，保留非零 source demand，不得将其计为零误差通过；
10. regular rank 仍基于 local ranks，不使用 concatenated Jacobian rank；
11. finite-difference epsilon-halving 失败时进入 hard warning/gate；
12. Jacobian engine cross-check 必须在完整模型代表样本运行。

**RPO torso gate**

对 complete RPO model：

- compatible axis canonical rotation被保留；
- pure incompatible pitch/roll 引起 torso hinge motion ≤ compatible response 的 5%；
- common world rotation不改变结果；
- joint limits生效；
- source不可达 demand单独报告。

### Agent E — Full Robot Zoo Validation & Cross-Model Matrix

**职责**

- manifest resolver；
- 固定上游 source/ref/hash；
- 全量模型运行；
- verified semantic-map调度；
- URDF/MJCF equivalence；
- positive/partial/negative 分类；
- per-robot reports；
- aggregate matrix；
- 无本机绝对路径的 reproduction commands。

**文件所有权**

```text
soma_retargeter/robotics/v3/validation.py
soma_retargeter/robotics/v3/robot_zoo.py                  # 可新增
soma_retargeter/tools/compile_kinematic_profile_v3.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
soma_retargeter/tools/sync_robot_zoo_v3.py                # 可新增
assets/robot_zoo/robot_zoo_manifest.json                  # 仅验证/补充固定ref
artifacts/retargeting_v3_step2/
tests/v3/test_full_robot_zoo_*.py
```

**全量定义**

`assets/robot_zoo/robot_zoo_manifest.json` 中每一个 entry 都必须产生结构化结果，不能静默跳过。

#### Positive humanoid

每个 positive humanoid 必须执行：

```text
resolve fixed source
→ license/provenance/hash
→ runtime load
→ verified semantics
→ distal sites
→ full chains
→ neutral FK
→ numerical Jacobians
→ 32-sample reachability
→ rest calibration
→ neutral exactness
→ 全部 canonical motions
→ torso/hand/foot chain projection
→ deterministic rerun
→ report
```

#### Partial humanoid / biped

- 运行所有可用链；
- 缺失 Hand 等语义时结构化降级；
- 不伪造 full humanoid readiness。

#### Negative control

包括公开 Franka Panda、Go2、Stretch、Cassie、Upkie、Bolt 等：

- 模型必须能加载或给出明确 source failure；
- morphology classification正确；
- 不生成伪造 humanoid profile；
- 结果为结构化 `negative_control_passed` 或明确 failure。

#### Fetch-only

- 使用固定 ref 下载到 cache，不提交受限资产；
- 有网络/cache时必须运行；
- 确实无法获取时输出 `source_unavailable`，包含固定 source、原因和复现命令；
- 不得把 unavailable 计入算法 pass；
- aggregate summary分别统计 algorithm pass 与 source unavailable。

**G1 cross-format 两层 gate**

1. **同源严格等价**：从同一个 G1 URDF 生成 deterministic canonical MJCF，再比较 topology、axis、limits、neutral semantic FK、chains、rank、canonical projection；必须通过。
2. **异源变体兼容**：vendor URDF 与 Menagerie MJCF 可能不是同一 DoF variant；必须识别 variant difference，比较共同语义能力，不得伪称严格等价，也不得把差异固化为“测试预期即完成”。

### Agent F — Reproducibility, CI & Independent Red Team

**职责**

- 首先写失败测试重现当前假阳性；
- acceptance ledger；
- pytest/JUnit；
- clean-worktree artifact rerun；
- deterministic rerun；
- CI workflows；
- 私有资产审计；
- robot-name special-case审计；
- legacy offset审计；
- false-pass审计；
- 最终独立 red-team report。

**文件所有权**

```text
.github/workflows/retargeting_v3_step2.yml
scripts/audit_retargeting_v3_step2.py                    # 可新增
tests/v3/test_acceptance_gates_*.py
docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md
docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md
docs/retargeting_v3/subagents/step2_agent_f_red_team.md
artifacts/retargeting_v3_step2/test_results/
```

**独立性规则**

- Agent F 不得参与 Agent A–E 的核心实现设计；
- 不得降低任何 gate；
- 必须寻找反例；
- 发现阻断问题时，Integrator 必须派回对应 agent 修复；
- 修复后 Agent F 重新审查；
- 最终 report 必须明确 `PASS` 或 `BLOCKED`，不得使用模糊“基本完成”。

---

## 5. 六个 Agent 的并行与集成顺序

Codex 启动后立即创建六个 agent，但按以下节奏避免冲突：

### Wave 1：并行审计与接口冻结

- A：runtime adapter/API审计；
- B：全量 semantic/site审计；
- C：calibration/target数学审计；
- D：projection/reachability审计；
- E：Robot Zoo source/manifest/validation矩阵准备；
- F：先写能暴露当前假阳性的 failing tests 和 acceptance ledger。

Wave 1 输出统一接口 ADR，Integrator 冻结：

```text
RuntimeModelAdapter
SemanticSiteV3
VerifiedSemanticMap
RestCalibrationV3
CanonicalSemanticMotion
ChainProjectionResult
KinematicProfileV3
RobotValidationResult
```

### Wave 2：并行实现

- A、B、C、D 实现各自模块；
- E 构建 resolver、全量调度、report schema；
- F 构建 CI、审计器与独立 gates。

### Wave 3：按依赖集成

严格顺序：

```text
A runtime truth
→ B verified sites
→ C calibration/targets
→ D canonical projection
→ E full-zoo validation
→ F red-team
```

### Wave 4：修复与最终重跑

- F 的每个 blocking finding分派回 A–E；
- 修复后重新运行所有 tests和全量 Robot Zoo；
- 最终 artifacts必须在 clean worktree、最终 commit 上生成；
- 最后只允许文档/metadata commit，不允许生成 artifacts 后再改核心代码。

---

## 6. Verified semantic maps 与证据格式

新增稳定格式，例如：

```json
{
  "schema_version": 2,
  "model_id": "...",
  "model_fingerprint": "...",
  "verification_status": "verified",
  "semantics": {
    "Hips": {
      "body": "...",
      "site": null,
      "local_position": [0, 0, 0],
      "local_rotation_xyzw": [0, 0, 0, 1],
      "confidence": 0.99,
      "evidence": ["topology", "neutral_position", "joint_chain"]
    }
  }
}
```

要求：

- inference 与 verified override明确区分；
- confidence有来源；
- verified map绑定model fingerprint；
- model变化使map失效；
- 左右语义必须通过side/topology/neutral geometry检查；
- complete positive humanoid不能只用临时自动推断结果作为最终 ground truth。

---

## 7. 真正的 Rest Calibration

### 7.1 独立数据流

```text
SOMA rest asset
→ source world semantic transforms
→ source semantic frames

runtime robot neutral q
→ runtime FK
→ robot semantic sites
→ robot semantic frames

source frames + robot frames
→ calibration rotations/lengths
→ independently reconstructed robot targets
→ compare with runtime neutral FK
```

禁止把 `robot_neutral_site_transforms` 原样复制成 neutral target 后再报告零误差。

### 7.2 Frame 构造

优先使用：

1. bilateral axis；
2. parent→child direction；
3. bend-plane normal；
4. torso vertical/forward directions；
5. explicit model frame；
6. multi-vector Wahba/Kabsch。

每个 frame 保存：

- basis；
- input vectors；
- conditioning；
- confidence；
- fallback；
- handedness check。

### 7.3 旋转映射

动态 relative rotation 必须通过 calibrated frame 共轭：

\[
\Delta R_r=A\Delta R_hA^T
\]

并在 robot neutral relative orientation上重建。

### 7.4 Calibration gates

- 不存在硬编码零误差；
- neutral position max error `<1e-3 m`；
- neutral relative rotation max error `<1e-2 rad`；
- frame determinant接近 +1；
- 左右镜像误差落盘；
- source global transform equivariance通过；
- 至少三个非中立 canonical姿态由独立几何预期验证。

---

## 8. Canonical Complete-Model Projection

### 8.1 Canonical motions

每台 positive humanoid必须运行：

1. neutral；
2. root translation；
3. global root yaw；
4. torso pitch；
5. torso roll；
6. torso yaw；
7. mixed torso rotation；
8. arms forward；
9. elbow bend；
10. overhead reach；
11. squat；
12. single step；
13. asymmetric arm reach；
14. crossed-body reach；
15. extreme-but-valid joint-limit stress。

Partial/negative模型只运行适用子集，并报告 skipped reason。

### 8.2 每个 motion 的真实流程

```text
canonical SOMA semantics
→ Agent C target builder
→ per-task desired robot target
→ Agent D chain projection
→ projected feasible target
→ residual/capability report
```

不得使用 neutral relative transform代替 desired target。

### 8.3 Projection cost

Endpoint：

\[
\min_q
\left\|\frac{p(q)-p_d}{L}\right\|^2
+\lambda_n\|D(q-q_0)\|^2
+\lambda_t\|D(q-q_{prev})\|^2
\]

Torso：

\[
\min_q
\|\log(R(q)^TR_d)\|^2
+\lambda_n\|D(q-q_0)\|^2
+\lambda_t\|D(q-q_{prev})\|^2
\]

要求：

- priors按joint range归一化；
- constants统一，不按机器人手调；
- deterministic；
- bounds生效；
- residual非零时诚实报告；
- rank0保留unreachable demand；
- sequence continuity可测。

---

## 9. 全量 Robot Zoo 验证

### 9.1 Manifest 是机器真值

测试必须动态读取 manifest，不能在代码里维护另一份容易漂移的机器人列表。

每个 entry 必须生成：

```text
artifacts/retargeting_v3_step2/per_robot/<id>.json
```

结果状态只能来自明确枚举：

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

`compiled` 不是最终通过状态。

### 9.2 Positive humanoid 硬门槛

- fixed source/hash可验证；
- runtime load成功；
- verified semantics；
- Hand/Foot distal sites；
- full chain正确；
- Jacobian有限且稳定；
- reachability有32个低差异样本；
- calibration独立通过；
- canonical motions全量执行；
- projection结果完整；
- two-run deterministic；
- 无 robot-name数学特例。

### 9.3 跨模型统计

aggregate summary至少包含：

- model count by class/status；
- semantic coverage；
- site provenance分布；
- rank分布；
- singularity fraction；
- calibration error；
- canonical projection residual p50/p90/max；
- solver convergence；
- deterministic mismatch；
- source unavailable；
- failure taxonomy；
- runtime compile时间。

### 9.4 不能只看 RPO

RPO 是关键回归模型，但任何算法默认必须同时通过不同形态：

- single-axis torso；
- zero-axis/fixed torso；
- multi-axis torso；
- no-wrist arm；
- wrist-rich arm；
- partial humanoid；
- compact humanoid；
- high-DoF humanoid；
- URDF/MJCF同源转换；
- negative controls。

---

## 10. G1 URDF/MJCF 正确验证设计

当前随意比较 vendor URDF 与 Menagerie MJCF 是不充分的，因为它们可能是不同 DoF variant。

### Gate A：同源严格等价

使用同一个固定 G1 URDF：

1. runtime直接加载URDF；
2. deterministic转换成canonical MJCF；
3. 两侧使用同一个 verified semantic map；
4. 比较：
   - body/joint topology；
   - joint type/axis/limit；
   - neutral q semantics；
   - semantic site FK；
   - active chains；
   - local ranks；
   - canonical projection residual。

必须在容差内通过。

### Gate B：异源变体兼容

vendor 29DoF URDF vs Menagerie MJCF：

- 自动识别实际 coordinate差异；
- 比较共同语义链；
- 记录额外/缺失DoF；
- 不要求ordered labels完全相同；
- 不得宣称strict equivalent；
- 两者都必须独立通过算法 gates。

### Gate C：23DoF真实模型

- 使用固定真实23DoF模型；
- 不得只通过锁定29DoF构造来替代；
- 可额外运行locked-29DoF作为controlled ablation；
- 结果应由实际model capability决定。

---

## 11. RPO 专项硬门槛

1. torso chain只包含真实torso coordinate；
2. torso regular rotational rank=1；
3. axis在pelvis frame表达，global root rotation不改变结果；
4. Hand chain包含完整shoulder/elbow链；
5. Foot chain包含hip/knee/ankle；
6. distal Hand site不是未经证明的 elbow body原点；
7. sole/toe/heel site不是ankle body原点；
8. source pure incompatible torso pitch/roll产生的hinge响应≤compatible响应5%；
9. compatible torso twist正确保留并受limit约束；
10. arms-forward、overhead、elbow-bend实际执行hand projection；
11. squat、single-step实际执行foot projection；
12. residual非零时不得被写成0或pass；
13. no-wrist orientation support与position site分离；
14. artifacts中保存site evidence和projection sequence。

---

## 12. 测试要求

### 12.1 首先添加失败回归测试

Agent F 应先为当前问题写测试，确认在修复前失败：

- calibration hardcoded zero；
- neutral→neutral假 projection；
- zero-offset RPO hand/sole；
- TALOS proximal foot误映射；
- Booster Hips=Chest隐藏waist；
- G1 arbitrary-source false equivalence；
- dirty artifact metadata；
- absolute cache path；
- inferred semantics confidence=1；
- rank0 false pass。

### 12.2 数学单元测试

- SO(3)/SE(3)；
- relative transform invariance；
- frame conjugation；
- Wahba/Kabsch；
- free/ball tangent integration；
- prismatic/revolute/oblique Jacobian；
- epsilon halving；
- local rank与curved manifold；
- multi-pose singularity；
- LCA/cross-branch path；
- redundant projection priors；
- continuity；
- bounds；
- rank0 demand保留；
- deterministic hash。

### 12.3 完整模型测试

所有 manifest entry 都要运行。测试不可写：

```python
if not path.exists():
    return
```

这种静默跳过。

允许：

- pytest skip，但必须带结构化原因；
- aggregate summary将source unavailable单独计数；
- required permissive models在CI缓存或sync后必须存在，否则CI失败。

### 12.4 两次确定性运行

对所有 required positive humanoid：

- 相同source/hash/config/seed运行两次；
- profile deterministic hash一致；
- rank一致；
- projection q/residual在容差内一致；
- semantic/site evidence一致。

---

## 13. Artifacts 与可复现性

最终目录：

```text
artifacts/retargeting_v3_step2/
  environment.json
  commands.txt
  source_inventory.json
  acceptance_ledger.json
  summary.json
  cross_format.json
  deterministic_rerun.json
  test_results/
    pytest.txt
    junit.xml
    coverage.json
  per_robot/
  failures/
```

### 13.1 clean generation protocol

1. 所有核心代码和测试先commit；
2. `git status --short`必须为空；
3. 记录当前commit；
4. 运行pytest；
5. 运行全量Robot Zoo；
6. 运行deterministic rerun；
7. 生成artifacts；
8. commit artifacts和reports；
9. 如果之后修改核心代码，必须重新执行步骤1–8。

### 13.2 environment.json

必须记录：

- artifact生成时的真实git HEAD；
- clean status；
- Python/Newton/Warp/MuJoCo版本；
- GPU/CPU；
- robot_descriptions版本；
- manifest hash；
- source cache root用变量表示；
- seed；
- commands；
- timestamp。

### 13.3 reproduction commands

禁止写死：

```text
/mnt/ssd1/song/...
```

改用：

```text
python -m ... --robot-id unitree_g1_mjcf --manifest assets/robot_zoo/robot_zoo_manifest.json
```

或 `${ROBOT_ZOO_CACHE}`。

---

## 14. 报告与 Handoff

必须新增：

```text
docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md
docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md
docs/retargeting_v3/STEP2_KNOWN_LIMITATIONS.md
docs/retargeting_v3/subagents/
  step2_agent_a_handoff.md
  step2_agent_b_handoff.md
  step2_agent_c_handoff.md
  step2_agent_d_handoff.md
  step2_agent_e_handoff.md
  step2_agent_f_red_team.md
```

Implementation report必须包含：

- commits；
- six-agent工作分配；
- 当前架构；
- 已关闭审计项；
- 未关闭项；
- 测试命令和真实结果；
- 全量机器人统计；
- G1 cross-format结果；
- RPO专项结果；
- 性能；
- source unavailable；
- red-team结论；
- 是否允许进入Step 3。

---

## 15. CI 设计

新增 workflow jobs：

```text
step2-unit-math
step2-runtime-adapter
step2-calibration-targets
step2-rpo-g1
step2-full-permissive-zoo
step2-negative-controls
step2-reproducibility
step2-red-team-audit
```

要求：

- unit jobs可CPU运行；
- full zoo使用cache；
- cache key包含manifest hash与source refs；
- required model source缺失时失败，不静默skip；
- fetch-only可单独manual job；
- JUnit和artifacts可下载；
- 当前commit没有CI status时不得宣称最终通过。

---

## 16. Acceptance Ledger：所有硬门槛

### Runtime truth

- [ ] Newton FK是主真值；
- [ ] MuJoCo仅作reference/cross-check；
- [ ] tangent integration正确；
- [ ] model fingerprint完整；
- [ ] include/assets/conversion provenance完整；
- [ ] full-chain/LCA正确。

### Semantics/sites

- [ ] 所有positive humanoid有verified map；
- [ ] inferred与verified明确区分；
- [ ] confidence非伪造；
- [ ] RPO distal hands正确；
- [ ] RPO sole/toe/heel正确；
- [ ] TALOS foot正确；
- [ ] Booster waist semantic正确；
- [ ] partial/negative不伪造语义。

### Calibration

- [ ] 无hardcoded zero error；
- [ ] source/robot frame独立；
- [ ] rotation conjugation正确；
- [ ] neutral position `<1mm`；
- [ ] neutral rotation `<0.01rad`；
- [ ] global equivariance；
- [ ] no legacy offsets。

### Projection

- [ ] 所有canonical motions真正投影；
- [ ] torso desired不是neutral placeholder；
- [ ] endpoint desired来自target builder；
- [ ] neutral/continuity prior；
- [ ] rank0 demand保留；
- [ ] residual真实；
- [ ] deterministic。

### Cross-format

- [ ] 同源G1 URDF→MJCF严格等价；
- [ ] 异源G1变体差异正确记录；
- [ ] 真实G1 23DoF验证；
- [ ] 不把limitation当pass。

### Full Robot Zoo

- [ ] manifest每个entry有结果；
- [ ] required positive全部算法通过；
- [ ] partial正确降级；
- [ ] negative controls正确；
- [ ] fetch-only明确状态；
- [ ] 没有私有资产；
- [ ] 无静默skip。

### Reproducibility

- [ ] pytest实际运行；
- [ ] JUnit保存；
- [ ] final HEAD artifacts；
- [ ] clean worktree；
- [ ] 无本机绝对路径；
- [ ] two-run deterministic；
- [ ] CI通过；
- [ ] 6个handoff完整；
- [ ] Agent F最终 `PASS`。

---

## 17. 明确禁止的捷径

以下任何做法都算失败：

- 把 `compiled_count == N` 当完成；
- desired直接取neutral FK；
- error字典直接填0；
- local_position为0就默认site正确；
- 用字符串长度判断body深度；
- 推断语义标confidence=1；
- 机器人名字数学分支；
- G1异源变体差异写成expected limitation后算通过；
- tests中`if not path.exists(): return`；
- rank0 residual置零并算pass；
- artifacts在dirty worktree生成；
- 命令写死用户本机路径；
- 只运行9台当前模型却称“全量”；
- 使用公司私有模型；
- 没有subagent证据；
- 没有red-team仍宣称完成；
- 开始生产IK或Step3。

---

## 18. 推荐提交顺序

1. `test: expose step2 false-positive calibration and projection gates`
2. `refactor: harden runtime model truth and fingerprints`
3. `feat: add verified semantic maps and distal geometry sites`
4. `feat: implement independent rest calibration`
5. `feat: fix calibrated target rotation and root scaling`
6. `feat: run canonical torso and endpoint chain projections`
7. `feat: add normalized neutral and continuity priors`
8. `feat: add same-source URDF MJCF equivalence`
9. `feat: validate all permissive positive humanoids`
10. `test: validate partial and negative-control robot zoo`
11. `test: add deterministic rerun and clean artifact audit`
12. `ci: add step2 full validation matrix`
13. `docs: add six subagent handoffs and implementation report`
14. `artifacts: publish final clean step2 validation results`

不要再次把所有工作压成一个无法审查的大提交。

---

## 19. Codex 启动步骤

1. `git status --short`；
2. `git rev-parse HEAD`；
3. 确认分支为 `retargeting-v3-step2-kinematic-core`；
4. 阅读三个Step1文档、本goal和现有known limitations；
5. 一次创建六个专业subagent；
6. 六个agent先完成Wave1审计；
7. Integrator冻结接口；
8. Agent F先提交失败回归测试；
9. A–E按ownership实现；
10. Integrator按依赖顺序集成；
11. 运行全量Robot Zoo；
12. Agent F红队；
13. 阻断项返回对应agent修复；
14. 最终clean rerun；
15. 保存完整artifacts和reports；
16. push全部commits；
17. 到此停止，不进入Step3。

---

## 20. Step 2 完成定义

只有同时满足以下条件才能写 `Step 2: PASS`：

- [ ] 当前审计表所有“未通过/未实现/无证据”项已关闭；
- [ ] six subagents全部有真实handoff；
- [ ] runtime adapter通过；
- [ ] verified semantics通过；
- [ ] distal hand/sole sites通过；
- [ ] independent rest calibration通过；
- [ ] calibrated target builder通过；
- [ ] canonical complete-model projection通过；
- [ ] RPO专项通过；
- [ ] G1同源cross-format严格等价通过；
- [ ] G1异源variant比较诚实；
- [ ] G1 23DoF真实模型通过；
- [ ] manifest全量条目有结构化结果；
- [ ] required positive humanoids全部算法通过；
- [ ] partial与negative controls通过；
- [ ] 无公司私有资产；
- [ ] tests/JUnit/CI通过；
- [ ] final clean artifacts可复现；
- [ ] Agent F最终红队结论为 `PASS`；
- [ ] 全部commits已push。

若任一硬门槛未满足，implementation report必须写 `Step 2: BLOCKED`，列出阻断原因，并停止，不得开始Step3。
