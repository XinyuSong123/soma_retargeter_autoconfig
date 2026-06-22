# Goal — Step 2.2：基于 44 个公开 Robot Zoo 资产完成全量开发

> 当前准备分支：`retargeting-v3-step2-assets-clean-sync`  
> 用户脚本输出分支：`retargeting-v3-step2-assets-vendored`  
> 数值基线：`retargeting-v3-step2-numerical-core-fix@b0d947b2367908295d1924e877ba863ec47d91b9`  
>
> 本轮必须创建并持续高强度使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator。
>
> **范围已经冻结：46 个公开 source 均已解析，其中 44 个进入后续项目范围，2 个 snapshot 生成失败项明确永久延期，不再要求修复。**
>
> 44 个 in-scope 资产的构成：
>
> ```text
> 38 个 permissive mesh-free kinematic snapshots
> 5 个 fetch-only、仅在外部 cache 使用的模型
> 1 个项目已有本地 RPO
> ```
>
> 两个延期项必须保留在 `robot_zoo_lock.json` 中，保存 ID、source、失败原因和 scope decision，但不得继续阻断本轮开发。

---

## 0. 用户首先执行

在正确 conda 环境中：

```bash
git fetch origin
git checkout retargeting-v3-step2-assets-clean-sync
git pull --ff-only

bash scripts/accept_two_deferred_robot_assets.sh
```

脚本只在以下精确条件下推送资产分支：

```text
manifest entries     = 46
source available     = 46
vendored snapshots   = 38
fetch-only cached    = 5
project-local RPO    = 1
snapshot deferred    = 2
其他 blocker         = 0
```

完成后：

```bash
git fetch origin
git checkout retargeting-v3-step2-assets-vendored
git pull --ff-only
```

然后让 Codex 读取本文件并开发。

---

## 1. 本轮目标

基于已经准备好的 44 个公开资产完成：

```text
clean committed assets / pinned external fetch-only cache
→ runtime load
→ verified semantic sites
→ full-chain paths
→ engine Jacobian and numerical validation
→ rest calibration
→ canonical target projection
→ deterministic rerun
→ per-robot result matrix
→ clean artifacts
→ CI and red-team
```

本轮不再定位或修复两个 deferred snapshots。

最终必须能够回答：

1. 44 个 in-scope 资产中有多少成功加载？
2. 每个 positive/partial humanoid 是否有 verified semantics？
3. 哪些机器人通过离线 retargeting compiler？
4. 哪些机器人失败于 projection residual、rank-zero demand 或 semantic geometry？
5. 当前剩余五个真实 algorithm failures 的具体 motion/task/metric 是什么？
6. 结果是否在 clean rerun 中完全复现？

---

## 2. 硬性边界

### 包含

- 38 个 committed snapshots；
- 5 个 fetch-only cached models；
- 本地 RPO；
- snapshot/runtime loader；
- public loader dependencies；
- verified semantic maps；
- partial morphology classification；
- full 44-source validation；
- deterministic rerun；
- remaining algorithm-failure analysis；
- clean artifacts；
- CI；
- six-agent handoffs；
- independent red-team。

### 禁止

- 修复或重新纳入两个 deferred snapshot IDs；
- 将 deferred 计为 pass；
- 将范围写成 46 个算法模型全部通过；
- 提交完整上游 cache；
- 提交 meshes；
- vendor GPL/LGPL/CC-SA/NASA fetch-only 模型；
- 使用私有资产；
- 使用 `cxxx_190`；
- 修改 numerical-core epsilon、rank 或 projection thresholds；
- 为机器人写数学特例；
- 修改生产 `NewtonPipeline`；
- 进入 Step 3。

---

## 3. 六个 xhigh Subagents

### Agent A — Asset Lock and Snapshot Integrity

负责：

- 审查 `robot_zoo_lock.json`；
- 验证 44/2 scope decision；
- snapshot hash、source hash、upstream commit、LICENSE；
- fetch-only 不进入 Git；
- deterministic snapshot rerun；
- mesh/private/absolute-path audit。

Owned files：

```text
assets/robot_zoo/robot_zoo_lock.json
assets/robot_zoo/source_inventory.json
assets/robot_zoo/snapshots/
scripts/build_robot_zoo_snapshots.py
scripts/audit_robot_zoo_assets_v3.py
tests/v3/test_robot_zoo_asset_*.py
docs/retargeting_v3/subagents/assets44_agent_a_handoff.md
```

### Agent B — Loader Closure

负责：

- URDF/MJCF snapshot load；
- Newton/MuJoCo backend load；
- `pycollada` 等 public dependencies；
- include/package/path resolution；
- loader taxonomy；
- 44 in-scope entries的 load matrix。

Owned files：

```text
pyproject.toml
environment*.yml
soma_retargeter/robotics/v3/model_adapter.py
soma_retargeter/robotics/v3/model_conversion.py
tests/v3/test_robot_zoo_loader_*.py
docs/retargeting_v3/subagents/assets44_agent_b_handoff.md
```

Gate：

```text
in-scope source/load failure = 0
```

fetch-only 也必须能从外部 cache 加载，但不能提交模型本体。

### Agent C — Verified Semantics: Core Humanoids

负责核心机器人：

- RPO；
- G1 URDF/MJCF；
- H1/H1_2；
- Booster T1；
- OP3；
- TALOS；
- Berkeley；
- Fourier N1。

必须建立：

- fingerprint-bound verified maps；
- Hips/Chest；
- distal Hand；
- sole/toe/heel where available；
- partial classification where morphology lacks arms；
- topology/FK/geometry evidence。

Owned files：

```text
assets/robot_zoo/semantic_maps/<core shard>
assets/robot_zoo/semantic_expectations/<core shard>
soma_retargeter/robotics/v3/semantic_validation.py
soma_retargeter/robotics/v3/site_geometry.py
tests/v3/test_assets44_core_semantics_*.py
docs/retargeting_v3/subagents/assets44_agent_c_handoff.md
```

### Agent D — Verified Semantics: Remaining Humanoids and Controls

负责其余 in-scope positive/partial humanoids，以及 negative controls。

要求：

- positive 不允许 inference→passed；
- biped-no-arms 诚实 partial/negative；
- Go2、Panda、Stretch 等不得伪装成 humanoid；
- semantic map 与 snapshot/source fingerprint 绑定；
- source variant 差异明确记录。

Owned files：

```text
assets/robot_zoo/semantic_maps/<remaining shard>
assets/robot_zoo/semantic_expectations/<remaining shard>
tests/v3/test_assets44_remaining_semantics_*.py
tests/v3/test_assets44_negative_controls_*.py
docs/retargeting_v3/subagents/assets44_agent_d_handoff.md
```

### Agent E — Full 44-Asset Validation and Failure Analysis

负责：

- 自动从 lock 的 scope decision 读取 44 个 in-scope IDs；
- 不在代码里另写清单；
- 对 44 个运行完整 validation；
- deterministic rerun；
- URDF/MJCF pair reports；
- RPO专项；
- 剩余 algorithm failures 的精确诊断。

Owned files：

```text
soma_retargeter/robotics/v3/validation.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
artifacts/retargeting_v3_step2_assets44/
tests/v3/test_assets44_full_validation_*.py
docs/retargeting_v3/STEP2_ASSETS44_VALIDATION_REPORT.md
docs/retargeting_v3/subagents/assets44_agent_e_handoff.md
```

每个 algorithm failure 必须保存：

```text
robot
motion
task
metric
actual value
threshold
rank/capability
projected target
unreachable demand
```

禁止只写：

```text
compiler recorded algorithm failures
```

### Agent F — Clean Provenance, CI and Red Team

负责：

- clean source commit；
- clean worktree generation；
- no absolute paths；
- no private assets；
- no meshes；
- no fetch-only vendoring；
- six handoffs；
- full tests；
- CI；
- final PASS/BLOCKED。

Owned files：

```text
scripts/audit_retargeting_v3_assets44.py
.github/workflows/retargeting_v3_assets44.yml
tests/v3/test_assets44_acceptance_*.py
docs/retargeting_v3/STEP2_ASSETS44_ACCEPTANCE.md
docs/retargeting_v3/subagents/assets44_agent_f_red_team.md
artifacts/retargeting_v3_step2_assets44/test_results/
```

---

## 4. Scope contract

`robot_zoo_lock.json` 必须包含：

```json
{
  "scope_decision": {
    "decision": "accept_exactly_two_deferred_snapshots",
    "deferred_snapshot_ids": ["...", "..."],
    "deferred_snapshot_count": 2,
    "in_scope_source_count": 44,
    "in_scope_breakdown": {
      "vendored_snapshots": 38,
      "fetch_only_cached": 5,
      "project_local_existing": 1
    },
    "blocking_count_for_current_scope": 0
  }
}
```

任何新的 source、snapshot、load 或 license failure 都是 blocker。

只有 lock 中列出的两个 ID 可 deferred。

---

## 5. 验证状态

对 44 个 in-scope 条目，只允许：

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

最终 asset/data 验收要求：

```text
source_unavailable = 0
model_load_failed  = 0
license_blocked    = 0 for in-scope IDs
semantic_failed    = 0
```

可以保留有真实数学证据的 `algorithm_failed`，但必须给出完整诊断。

两个 deferred IDs 不进入上述 44 条统计，并单独显示：

```text
deferred_snapshot = 2
```

---

## 6. 必须保留的 numerical-core 结果

不得修改以下全局 numerical contracts：

- engine relative Jacobian 是主能力依据；
- finite difference 是 uncertainty validator；
- task-specific translation/rotation block；
- backend-aware multi-scale FD；
- rank/subspace statistical gate；
- parent-frame rotation transfer；
- independent canonical capability motions；
- chain-length normalization；
- projection thresholds 使用当前已提交全局值。

当前 numerical baseline：

```text
positive profile pass       16
algorithm_failed             5
epsilon-only failures        0
```

资产轮结束后：

- 这 16 台不得回归；
- 新 source 可以增加 profile eligible/pass；
- 不得通过放宽阈值减少 5 个失败。

---

## 7. Artifacts

最终目录：

```text
artifacts/retargeting_v3_step2_assets44/
  environment.json
  commands.txt
  scope.json
  source_inventory.json
  load_matrix.json
  semantic_matrix.json
  summary.json
  deterministic_rerun.json
  cross_format.json
  deferred_snapshots.json
  per_robot/
  failures/
  test_results/
```

`summary.json` 至少包含：

```text
manifest_total                 46
in_scope_total                 44
deferred_snapshot_count         2
vendored_snapshot_count        38
fetch_only_cached_count          5
project_local_count              1
source_available_in_scope
load_passed_in_scope
semantic_passed_in_scope
profile_eligible
profile_passed
partial_passed
negative_control_passed
algorithm_failed
source_unavailable               0
model_load_failed                0
semantic_failed                  0
deterministic_compared
deterministic_matched
```

---

## 8. Clean generation protocol

1. 核心代码、maps、tests先提交；
2. 建 clean detached worktree；
3. 从 committed snapshot + pinned external cache运行；
4. full tests；
5. 44-asset validation；
6. deterministic rerun；
7. 生成临时 artifacts；
8. 扫描绝对路径、private token、mesh；
9. 提交 compact artifacts；
10. 核心代码若再改，全部重跑。

最终 `environment.json`：

```text
git_status_short = ""
source_code_commit 可解析
source_code_commit 是 artifact commit 祖先
```

---

## 9. Acceptance Gates

### Scope/assets

- [ ] 46 source available；
- [ ] exactly 2 deferred IDs；
- [ ] exactly 44 in scope；
- [ ] 38 snapshots committed；
- [ ] 5 fetch-only cache-only；
- [ ] 1 local RPO；
- [ ] 无新 asset blocker。

### Load/semantics

- [ ] 44 source unavailable=0；
- [ ] 44 model load failed=0；
- [ ] 44 semantic failed=0；
- [ ] partial morphology诚实；
- [ ] negative controls正确。

### Algorithms

- [ ] numerical-core 16 passes无回归；
- [ ] epsilon-only failures=0；
- [ ] remaining failures有具体证据；
- [ ] no threshold changes；
- [ ] RPO通过专项。

### Reproducibility

- [ ] clean worktree；
- [ ] no absolute paths；
- [ ] no meshes/private assets；
- [ ] deterministic rerun；
- [ ] six xhigh handoffs；
- [ ] full tests；
- [ ] CI run；
- [ ] Agent F PASS。

---

## 10. 完成定义

最终只能写：

```text
Step 2.2 Assets44: PASS
```

或：

```text
Step 2.2 Assets44: BLOCKED
```

PASS 不要求两个 deferred snapshots 被修复，也不允许未来重新把它们作为 blocker。

PASS 必须满足：

- [ ] asset branch 已干净推送；
- [ ] scope decision 可审计；
- [ ] 44 in-scope assets 全部可访问；
- [ ] 44 load/semantic scaffolding failures为0；
- [ ] full validation完成；
- [ ] deterministic rerun完成；
- [ ] numerical pass无回归；
- [ ] remaining algorithm failures有完整证据；
- [ ] no private/fetch-only/mesh leakage；
- [ ] six handoffs；
- [ ] tests/CI/red-team通过；
- [ ] 未进入Step3。
