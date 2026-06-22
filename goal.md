# Goal — Step 2.2：全量公开 Robot Zoo 拉取、轻量快照与干净回写

> **Codex 执行契约**
>
> 准备分支：`retargeting-v3-step2-assets-clean-sync`  
> 用户脚本输出分支：`retargeting-v3-step2-assets-vendored`  
> 数值基线：`retargeting-v3-step2-numerical-core-fix@b0d947b2367908295d1924e877ba863ec47d91b9`  
> 当前阶段：**Step 2.2 Asset Reproducibility and Full-Zoo Enablement**  
>
> 本轮必须使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator，负责接口、许可证边界、合并、全量验证和最终诚实结论。
>
> **不得进入 Step 3；不得修改生产 `NewtonPipeline`；不得调 Jacobian epsilon、rank threshold 或 projection residual threshold；不得为机器人写数学特例。**

---

## 0. 用户先执行的脚本

在正确 conda 环境中：

```bash
git fetch origin
git checkout -b retargeting-v3-step2-assets-clean-sync \
  origin/retargeting-v3-step2-assets-clean-sync

bash scripts/fetch_and_vendor_robot_zoo_assets.sh
```

脚本默认：

1. 不使用当前可能脏的工作目录作为生成环境；
2. 从远端准备分支创建独立 clean worktree；
3. 将完整上游仓库下载到 Git checkout 外部的 cache；
4. 拉取 manifest 中所有 `robot_descriptions` 模块；
5. 拉取固定 SHA 的 MuJoCo Menagerie；
6. 为 permissive `kinematic_snapshot` 条目生成无 meshes 的轻量快照；
7. 为每个快照保存 `SOURCE.json` 和上游许可证；
8. 对 GPL/LGPL/CC-SA/NASA 等 `fetch_only` 条目只保存 lock/provenance，不提交模型；
9. 生成 `source_inventory.json` 和 `robot_zoo_lock.json`；
10. 在 clean worktree 中提交；
11. 推送：

```text
origin/retargeting-v3-step2-assets-vendored
```

脚本环境变量：

```bash
ROBOT_ZOO_CACHE=/external/cache/path
INSTALL_MISSING_DEPS=1
ALLOW_PARTIAL_ASSETS=0
PUSH_ASSET_BRANCH=1
KEEP_WORKTREE=0
```

正常验收不允许 `ALLOW_PARTIAL_ASSETS=1`。它只能用于诊断。

---

## 1. 当前问题

数值核心修复后：

```text
baseline positive pass       7
corrected positive pass     16
algorithm_failed             5
semantic_failed              3
model_load_failed            2
source_unavailable          16
negative_control_passed      4
```

Step 2.1 已经证明：

- epsilon-only failures 从 14 降为 0；
- RPO 和 H1 等不再被旧 epsilon gate 误杀；
- 仍有 5 个真实 projection/compiler failure；
- 但当前最大覆盖率瓶颈已经变成 source、loader、semantic map 和可复现资产。

本轮不改数值公式。目标是让 manifest 中所有公开源都可被确定性获取，让许可证允许的轻量运动学模型进入干净 Git 历史，并使 Codex 能在同一套公开资产上完成全量验证。

---

## 2. 本轮唯一目标

建立以下可复现数据流：

```text
manifest entry
→ fixed upstream repository/ref
→ external full-source cache
→ source SHA/license verification
→ redistribution policy
→ mesh-free deterministic kinematic snapshot（仅 permissive）
→ SOURCE.json + LICENSE
→ robot_zoo_lock.json
→ clean Git asset commit
→ runtime load
→ verified semantics
→ full Robot Zoo validation
→ deterministic rerun
→ final red-team audit
```

本轮必须严格区分：

1. **完整上游 cache**：在 Git 仓库外，可以包含 meshes；
2. **Git kinematic snapshot**：只包含运动学所需文本模型、许可证和来源信息；
3. **fetch-only source**：可以拉取到 cache，但不得 vendor 到 Apache 项目；
4. **project-local RPO**：保留现有模型，不重复复制；
5. **private/unlisted assets**：完全不读取、不扫描、不提交。

---

## 3. 硬性范围

### 3.1 本轮包含

- 清理/隔离脏 worktree；
- external Robot Zoo cache；
- `robot_descriptions==2.0.0` 全量拉取；
- fixed-ref MuJoCo Menagerie 拉取；
- source inventory；
- lock file；
- permissive kinematic snapshots；
- LICENSE/SOURCE provenance；
- pycollada 等公开 loader dependency；
- source/load failure 修复；
- 新可用模型的 verified semantic maps；
- 全 46-entry 结构化验证；
- deterministic rerun；
- clean artifacts；
- CI 和 red-team。

### 3.2 本轮禁止

- 提交完整上游仓库；
- 提交 visual/collision meshes；
- Git LFS；
- 提交 GPL/LGPL/CC-BY-SA/NASA fetch-only 模型；
- 自动改许可证分类；
- 使用浮动 `main`；
- 执行上游未知脚本；
- 扫描 manifest 外目录；
- 使用公司私有模型；
- 使用 `cxxx_190`；
- 将公开 Franka Panda 错误当成私有资产；
- 修改 numerical-core thresholds；
- 为提高通过率降低 projection gate；
- 修改生产 pipeline；
- 进入 Step 3。

---

## 4. 六个 xhigh Subagents

每个 agent 必须：

- 独立 worktree/branch；
- 明确 owned files；
- 小 commits；
- handoff 记录 xhigh、commit、命令、测试、结果、风险；
- 不得降低许可证或算法 gate。

### Agent A — Source Fetch, Pins and Lock

**职责**

- 审计 manifest 46 entries；
- 运行和修复 `fetch_and_vendor_robot_zoo_assets.sh` 的 fetch 部分；
- `robot_descriptions` 2.0.0 拉取；
- Menagerie 固定 SHA；
- source repository/ref/path/hash；
- retry、resume、offline rerun；
- `source_inventory.json`；
- `robot_zoo_lock.json` schema。

**Owned files**

```text
scripts/fetch_and_vendor_robot_zoo_assets.sh
soma_retargeter/tools/sync_robot_zoo_v3.py
soma_retargeter/robotics/v3/robot_zoo.py
assets/robot_zoo/source_inventory.json
assets/robot_zoo/robot_zoo_lock.json
tests/v3/test_robot_zoo_fetch_*.py
docs/retargeting_v3/subagents/assets_agent_a_handoff.md
```

**硬门槛**

- manifest entry count 与 lock count 一致；
- required source unavailable = 0；
- optional fetch-only 也应尝试拉取到 cache；
- 所有 source 有 SHA；
- git upstream 有 commit SHA；
- Menagerie SHA 必须完全等于 manifest pin；
- 第二次 offline 执行不需要网络。

### Agent B — License Policy and Kinematic Snapshots

**职责**

- 审计 `scripts/build_robot_zoo_snapshots.py`；
- URDF visual/collision stripping；
- MJCF canonicalization 与 explicit inertial；
- snapshot runtime load；
- LICENSE/SOURCE；
- size/mesh/path/private-asset audit；
- deterministic snapshot hash。

**Owned files**

```text
scripts/build_robot_zoo_snapshots.py
assets/robot_zoo/snapshots/
docs/retargeting_v3/ASSET_POLICY.md
tests/v3/test_robot_zoo_snapshots_*.py
docs/retargeting_v3/subagents/assets_agent_b_handoff.md
```

**硬门槛**

- 每个 `kinematic_snapshot` entry：
  - snapshot 存在；
  - 可被相应 loader 加载；
  - 无 mesh/package URI；
  - 小于 2MB/robot；
  - 有 LICENSE；
  - 有 SOURCE.json；
  - source/ref/hash 完整；
- `fetch_only` 不产生 snapshot 目录；
- snapshot 二次生成 hash 一致；
- 不存在本机绝对路径；
- 不存在 private denylist token。

### Agent C — Loader and Environment Closure

**职责**

- `pycollada` 等公开依赖；
- URDF/MJCF snapshot load；
- include/package/path resolution；
- Newton/MuJoCo load parity；
- source failure taxonomy；
- CI environment lock。

**Owned files**

```text
pyproject.toml
environment*.yml                         # 若已有则更新
soma_retargeter/robotics/v3/model_adapter.py
soma_retargeter/robotics/v3/model_conversion.py
tests/v3/test_robot_zoo_snapshot_load_*.py
docs/retargeting_v3/subagents/assets_agent_c_handoff.md
```

**硬门槛**

- 当前 `jvrc_urdf` 和 `unitree_go2_urdf` 不再因缺 pycollada 失败；
- required snapshots 全部至少一个 runtime backend load；
- missing optional dependency 不再出现在 required reports；
- 不以删除几何 gate 来绕过 loader；
- loader patch 只能作用于临时 copy，不能改 upstream cache。

### Agent D — Verified Semantics for Newly Available Models

**职责**

- 只处理本轮新拉取且 source/load 成功的 positive/partial humanoids；
- verified Hips/Chest/Hand/Foot sites；
- distal geometry evidence；
- partial morphology 分类；
- map fingerprint/hash 绑定。

**Owned files**

```text
assets/robot_zoo/semantic_maps/
assets/robot_zoo/semantic_expectations/
soma_retargeter/robotics/v3/semantic_validation.py
soma_retargeter/robotics/v3/site_geometry.py
tests/v3/test_new_asset_semantics_*.py
docs/retargeting_v3/subagents/assets_agent_d_handoff.md
```

**硬门槛**

- positive humanoid 不允许 inference→passed；
- Berkeley 等实际 partial morphology 不伪造 hands；
- H1 URDF map 与真实 source fingerprint 绑定；
- Hand/Foot local site 有 topology/geometry evidence；
- body origin 不能无证据作为 distal site；
- map mismatch 必须 fail closed。

### Agent E — Full-Zoo Validation and Remaining Algorithm Evidence

**职责**

- 对 clean snapshots/cache 运行 46-entry validation；
- before/after source/load/semantic matrix；
- deterministic rerun；
- cross-format pairs；
- 记录剩余 5 个 algorithm failures 的真实 motion/task residual；
- 不改 numerical thresholds。

**Owned files**

```text
soma_retargeter/robotics/v3/validation.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
artifacts/retargeting_v3_step2_assets/
tests/v3/test_full_asset_zoo_validation_*.py
docs/retargeting_v3/STEP2_ASSET_VALIDATION_REPORT.md
docs/retargeting_v3/subagents/assets_agent_e_handoff.md
```

**硬门槛**

- 46 entries 都有 terminal structured status；
- required source unavailable = 0；
- required model_load_failed = 0；
- positive semantic_failed = 0，或有经验证的 partial classification；
- negative controls 正确拒绝；
- 当前 numerical pass 模型不得回归；
- 剩余 algorithm failure 不得再用 generic `compiler recorded algorithm failures`，必须列出 motion/task/metric/threshold；
- 不因资产轮修改数值公式。

### Agent F — Clean Provenance, CI and Red Team

**职责**

- clean worktree audit；
- source commit ancestry；
- license/mesh/size/private path audit；
- full tests；
- CI；
- six handoffs；
- 最终 PASS/BLOCKED。

**Owned files**

```text
scripts/audit_robot_zoo_assets_v3.py
.github/workflows/retargeting_v3_robot_zoo_assets.yml
tests/v3/test_robot_zoo_asset_acceptance_*.py
docs/retargeting_v3/STEP2_ASSET_ACCEPTANCE.md
docs/retargeting_v3/subagents/assets_agent_f_red_team.md
artifacts/retargeting_v3_step2_assets/test_results/
```

**红队必须检查**

- cache 是否在 repo 外；
- 生成时 worktree 是否 clean；
- 是否提交上游 `.git`；
- 是否提交 mesh；
- 是否提交 fetch-only；
- LICENSE 是否真实来自 upstream；
- SOURCE ref 是否为 commit SHA；
- 是否使用浮动 branch；
- 是否包含绝对路径；
- 是否包含私有资产 token；
- required unavailable/load/semantic failure 是否被伪装；
- numerical thresholds 是否被修改；
- artifacts 是否对应可解析的 source commit；
- CI 是否真实运行。

---

## 5. Clean Worktree 协议

最终资产提交必须来自独立 worktree：

```text
origin/retargeting-v3-step2-assets-clean-sync
→ clean worktree
→ external cache fetch
→ snapshot generation
→ tests
→ git add 仅允许目录
→ commit
→ git status empty
→ push retargeting-v3-step2-assets-vendored
```

允许提交的生成内容：

```text
assets/robot_zoo/source_inventory.json
assets/robot_zoo/robot_zoo_lock.json
assets/robot_zoo/snapshots/**
```

禁止提交：

```text
${ROBOT_ZOO_CACHE}/**
任何 upstream .git/**
任何 visual meshes
pull logs
临时 worktree
Python cache
本机路径
```

后续 Codex 代码和 semantic maps 应在 vendored branch 上提交。最终 artifacts 需要另一个 clean worktree 生成。

---

## 6. Snapshot 格式

每个 permissive snapshot：

```text
assets/robot_zoo/snapshots/<robot_id>/
  model.urdf | model.xml
  SOURCE.json
  LICENSES/
    LICENSE...
```

`SOURCE.json` 至少包含：

```json
{
  "robot_id": "...",
  "description_name": "...",
  "upstream_repository": "...",
  "upstream_ref": "40-char commit SHA",
  "source_file": "relative/path",
  "source_sha256": "...",
  "format": "urdf|mjcf",
  "license": "...",
  "license_files": [],
  "license_sha256": "...",
  "redistribution": "kinematic_snapshot",
  "generator_version": "...",
  "snapshot_file": "...",
  "snapshot_sha256": "..."
}
```

URDF snapshot：

- 保留 link/joint/inertial/origin/axis/limit；
- 移除 visual/collision/gazebo/transmission；
- 不保留 package URI 或 mesh filename。

MJCF snapshot：

- 从 compiled model 保存 canonical XML；
- 显式写入 inertial；
- 保留 body/joint/freejoint/site；
- 移除 asset/geom/visual/contact/actuator/sensor/keyframe；
- snapshot 必须重新被 MuJoCo 加载。

---

## 7. Lock 和 Inventory 语义

`source_inventory.json` 记录本次 cache resolution。

`robot_zoo_lock.json` 是长期可复现锁：

```text
manifest SHA
entry ID
source family
repository URL
commit SHA
source relative file
source SHA
license
redistribution policy
snapshot status/path/hash
```

状态至少包括：

```text
vendored
fetch_only
local_existing
snapshot_failed
source_unavailable
license_blocked
```

不得只记录 URL 而不记录 commit/hash。

---

## 8. 全量验证分层统计

不要再只报告 `pass/46`。必须输出：

```text
manifest_total
source_fetched
source_unavailable_required
source_unavailable_optional
snapshot_expected
snapshot_vendored
snapshot_failed
fetch_only_cached
load_attempted
load_passed
semantic_attempted
semantic_passed
profile_eligible
profile_passed
algorithm_failed
negative_control_passed
deterministic_compared
deterministic_matched
```

`fetch_only_cached` 不是 `snapshot_vendored`。

`source fetched` 也不是 `algorithm pass`。

---

## 9. 测试要求

### Script/asset tests

- manifest 46-entry lock coverage；
- retry/resume；
- offline second run；
- pinned Menagerie SHA；
- permissive snapshot generation；
- fetch-only non-vendoring；
- license missing fail closed；
- mesh/path/private denylist；
- deterministic snapshot hash；
- max size；
- clean worktree protocol。

### Loader tests

- all vendored URDFs parse；
- all vendored MJCFs parse；
- required snapshots load in Newton or MuJoCo；
- pycollada-dependent original sources load；
- no private asset dependency。

### Full validation

```bash
python -m pytest -q
python -m soma_retargeter.tools.validate_kinematic_profile_v3 \
  --manifest assets/robot_zoo/robot_zoo_manifest.json \
  --output-dir artifacts/retargeting_v3_step2_assets \
  --low-discrepancy-count 32 \
  --deterministic-rerun
```

---

## 10. Acceptance Gates

### Fetch/cache

- [ ] manifest 46 entries processed；
- [ ] all description modules attempted；
- [ ] Menagerie exact SHA；
- [ ] required unavailable=0；
- [ ] external cache；
- [ ] offline rerun works。

### Snapshot/license

- [ ] all kinematic_snapshot entries vendored；
- [ ] all snapshots load；
- [ ] no meshes；
- [ ] no package URI；
- [ ] under size limit；
- [ ] LICENSE/SOURCE complete；
- [ ] fetch-only not vendored；
- [ ] deterministic hash。

### Loader/semantics

- [ ] required load failures=0；
- [ ] pycollada closure；
- [ ] positive verified maps；
- [ ] partial morphology honest；
- [ ] negative controls correct。

### Reproducibility

- [ ] clean generation worktree；
- [ ] clean post-commit worktree；
- [ ] no absolute paths；
- [ ] no private assets；
- [ ] source commit resolvable；
- [ ] six xhigh handoffs；
- [ ] full tests；
- [ ] CI run；
- [ ] Agent F PASS。

---

## 11. 明确禁止的捷径

- 把 cache 目录直接 `git add`；
- vendor fetch-only/GPL/NASA/CC-SA；
- 删除 LICENSE；
- 用 manifest license 字符串代替真实 LICENSE 文件；
- 提交 meshes；
- 以 Git LFS 掩盖大文件；
- source 失败时换私有模型；
- 自动跟踪 upstream main；
- 修改 upstream cache；
- 未验证 snapshot load 就提交；
- 使用 `ALLOW_PARTIAL_ASSETS=1` 后宣布完成；
- 为通过全量验证放宽 numerical thresholds；
- 将 semantic/load/source failure 改名为 pass；
- 修改生产 pipeline；
- 进入 Step 3。

---

## 12. 推荐提交顺序

1. `test: add Robot Zoo fetch and lock gates`
2. `fix: make all manifest sources resumably fetchable`
3. `test: add license and snapshot rejection gates`
4. `feat: build deterministic mesh-free URDF snapshots`
5. `feat: build deterministic canonical MJCF snapshots`
6. `assets: vendor permissive Robot Zoo snapshots`
7. `deps: close public Robot Zoo loader dependencies`
8. `data: verify newly available humanoid semantic maps`
9. `test: run full 46-entry Robot Zoo validation`
10. `test: run deterministic asset and profile rerun`
11. `ci: add Robot Zoo asset workflow`
12. `docs: add six handoffs and asset reports`
13. `artifacts: publish clean Step 2.2 evidence`

---

## 13. 完成定义

最终报告只能写：

```text
Step 2.2 Robot Zoo Assets: PASS
```

或：

```text
Step 2.2 Robot Zoo Assets: BLOCKED
```

PASS 必须满足：

- [ ] clean asset branch 已由脚本推送；
- [ ] all required sources fetched；
- [ ] all permissive snapshots vendored；
- [ ] all fetch-only models只在cache/lock；
- [ ] all snapshots load；
- [ ] required model load failure=0；
- [ ] positive semantic failure=0或明确verified partial；
- [ ] full 46-entry structured validation；
- [ ] deterministic rerun；
- [ ] no private assets；
- [ ] no meshes/absolute paths；
- [ ] license/source/pin/hash完整；
- [ ] six xhigh handoffs；
- [ ] full tests通过；
- [ ] CI有通过记录；
- [ ] Agent F结论为PASS；
- [ ] 未修改 numerical thresholds；
- [ ] 未进入Step3。

本轮可以保留有真实 projection residual 证据的 `algorithm_failed`，但不能保留 source/load/semantic scaffolding failure。剩余算法问题必须为下一轮提供明确 robot/motion/task/metric/threshold，不得再只写 generic failure。
