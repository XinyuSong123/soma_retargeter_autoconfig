# Goal — Step 2 Asset-Ready Numerical Core：解除假失败，完成真实全量机器人验证

> **Codex 执行契约**
>
> 当前分支：`retargeting-v3-step2-assets-ready`  
> 起始代码：`retargeting-v3-step2-finalize@4a93386e41c31fa0ab55533b7f6d891042187bc7`  
> 当前阶段：**Step 2 数值数学修复 + 资产就绪后的全量复验**  
>
> 必须一次创建并持续高强度使用 **6 个 xhigh 专业 subagents**。主 Codex 是 Integrator，负责接口冻结、合并、全量测试和最终诚实结论。
>
> **不得开始 Step 3，不得修改生产 `NewtonPipeline`，不得调 whole-body IK 权重，不得加入 contact/collision/temporal/pole-vector 目标。**

---

## 0. 先读这些文件

1. `/goal.md`
2. `docs/retargeting_v3/STEP1_MATHEMATICAL_SPEC.md`
3. `docs/retargeting_v3/STEP1_COMPLETE_MODEL_CROSSCHECK.md`
4. `docs/retargeting_v3/ASSET_RESOLUTION.md`
5. `assets/robot_zoo/robot_zoo_manifest.json`
6. `assets/robot_zoo/source_lock.json`
7. `docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md`
8. `docs/retargeting_v3/STEP2_COMPLETION_AUDIT.md`
9. `artifacts/retargeting_v3_step2/summary.json`
10. RPO、G1、H1、Booster、OP3 的当前 per-robot reports

资产位置、上游版本和许可证已经锁定；不要重新搜索互联网或换成浮动 `main`。

---

## 1. 当前真实状态

当前旧 artifacts 的漏斗：

```text
manifest models              46
positive profile passed       7
negative controls passed      4
algorithm_failed             14
semantic_failed               3
model_load_failed             2
source_unavailable           16
```

本分支把 manifest 扩展到 **48** 个条目，新增官方真实 G1 23DoF：

```text
unitree_g1_23dof_urdf
unitree_g1_23dof_mjcf
```

资产侧已完成：

- `robot_descriptions==2.0.0` 固定到 commit `2431c405...`；
- MuJoCo Menagerie 固定到 `feadf76...`；
- Unitree ROS 固定到 `267182b...`；
- G1 23DoF URDF/MJCF 路径固定；
- `pycollada` 已进入 Robot Zoo 可选依赖；
- 有确定性 bootstrap 脚本；
- 禁止私有和 manifest 外资产。

### 1.1 不能误读 11/46

11 个 pass 中包括 4 个 negative controls。真正的 positive humanoid pass 是 7 个。

低通过率由四类原因构成：

1. 14 个 `algorithm_failed`：多数是 hand-chain epsilon stability 一票否决；
2. 3 个 `semantic_failed`：缺 verified map；
3. 2 个 `model_load_failed`：缺 `pycollada`；
4. 16 个 `source_unavailable`：之前没有完整拉取公开 cache。

只有第一类主要涉及数值数学。不要通过改公式试图修复缺源、缺依赖或缺 map。

---

## 2. 第一条命令：把公开资产真正拉齐

在正确环境中：

```bash
conda activate soma-retargeter-v2
python -m pip install -e '.[robot-zoo]'
python scripts/bootstrap_robot_zoo_assets.py
python scripts/bootstrap_robot_zoo_assets.py --verify-only
```

硬门槛：

```text
resolved_source_inventory.required_failure_count == 0
```

如果网络暂时失败，重试固定源；不得改 ref，不得用私有镜像或公司资产替代。

缓存目录：

```text
assets/robot_zoo/cache/
```

该目录不得提交。

---

## 3. 本轮唯一目标

把当前离线编译器从“有限差分每列必须零失败”改成：

```text
runtime engine Jacobian 是主真值
+ backend-aware finite difference 是独立验证器
+ 物理零列分类
+ task-specific stability
+ rank/subspace statistical gates
+ 正确 parent-frame rotation mapping
+ 独立 canonical capability tests
+ chain-length normalized residuals
```

然后对全部 48 个 manifest entries 重跑并生成可解释结果。

---

## 4. 六个 xhigh Subagents

所有 agent 使用独立 worktree/branch、小 commits、明确 ownership，并写：

```text
docs/retargeting_v3/subagents/assets_ready_agent_<a-f>_handoff.md
```

每个 handoff 必须包含：

- reasoning strength = xhigh；
- branch/commit；
- changed files；
- commands；
- tests；
- 数值结果；
- 未解决项；
- 风险；
- 集成顺序。

### Agent A — Public Asset Resolver & Source Integrity

**Owned files**

```text
scripts/bootstrap_robot_zoo_assets.py
soma_retargeter/robotics/v3/robot_zoo.py
soma_retargeter/tools/sync_robot_zoo_v3.py
soma_retargeter/tools/validate_kinematic_profile_v3.py
assets/robot_zoo/robot_zoo_manifest.json
assets/robot_zoo/source_lock.json
tests/v3/test_robot_zoo_asset_*.py
```

**任务**

1. 将 `pinned_git` 集成到正常 `resolve_robot_source()`；
2. 让 G1 23DoF 两个 entry 在 bootstrap 后由标准 validator 解析；
3. `allow_fetch=True` 时能够获取 direct Menagerie 固定 commit；
4. 验证 `robot_descriptions==2.0.0` 与 package-pinned upstream heads；
5. 安装/检查 `pycollada`，修复 JVRC URDF 和 Go2 URDF 加载；
6. required public entry 不得静默 `source_unavailable`；
7. fetch-only 可以测试，但绝不提交源资产；
8. inventory 保存 source hash、repo HEAD、license、reproduction command；
9. 不扫描 manifest 外目录；
10. 不包含任何私有资产或 `cxxx_190`。

### Agent B — Engine Relative Jacobian

**Owned files**

```text
soma_retargeter/robotics/v3/model_adapter.py
soma_retargeter/robotics/v3/engine_jacobian.py
soma_retargeter/robotics/v3/numerical_jacobian.py   # engine API integration only
tests/v3/test_engine_relative_jacobian_*.py
```

**主公式**

对 reference site \(A\)、target site \(B\)：

\[
p_{AB}=R_A^\top(p_B-p_A)
\]

\[
J_{p,AB}
=
R_A^\top(J_{p,B}-J_{p,A})
+
[p_{AB}]_\times R_A^\top J_{\omega,A}
\]

\[
J_{\omega,AB}
=
R_A^\top(J_{\omega,B}-J_{\omega,A})
\]

要求：

1. MuJoCo `mj_jac`/site Jacobian 为 MuJoCo 后端主结果；
2. Newton `eval_jacobian` 经 synthetic fixtures 确认 twist convention 后为 Newton 主结果；
3. site local offset 对线速度正确加入 \(\omega\times r\)；
4. cross-branch/LCA 公式完整，不能只处理 reference=LCA；
5. active coordinates 投影正确；
6. engine Jacobian finite；
7. analytical fixtures relative error `<1e-5`（float64）或按 backend 精度制定；
8. finite difference 不再是 runtime capability 的唯一真值。

### Agent C — Backend-Aware Finite-Difference Validator

**Owned files**

```text
soma_retargeter/robotics/v3/numerical_jacobian.py
soma_retargeter/robotics/v3/fd_validation.py
tests/v3/test_fd_validation_*.py
```

**任务**

1. adapter 暴露 scalar dtype 和 machine epsilon；
2. 步长：

\[
h_i=\operatorname{clip}
(c\,\epsilon_{\rm backend}^{1/3}s_i,h_{\min},h_{\max})
\]

3. 检查 `2h, h, h/2`，寻找稳定平台；
4. 使用中心差分和 Richardson：

\[
J^\star=(4J_{h/2}-J_h)/3
\]

5. 输出 truncation/cancellation 诊断；
6. 对近零 engine column 分类：
   - `structurally_zero`
   - `numerically_zero`
   - `unstable`
7. 不能因为近零平移列的相对误差大而失败；
8. task-specific：
   - hand/foot position 检查 \(J_p\)；
   - torso orientation 检查 \(J_\omega\)；
9. 禁止仅把 `stability_rtol` 放宽十倍；
10. engine-vs-FD error 必须按 characteristic scale 归一化。

### Agent D — Reachability Rank & Subspace Gates

**Owned files**

```text
soma_retargeter/robotics/v3/reachability.py
soma_retargeter/robotics/v3/rank_validation.py
tests/v3/test_reachability_rank_subspace_*.py
```

**任务**

当前错误：

```python
epsilon_stability_gate_passed = unstable == 0
```

这使数百列检查中的一次浮点噪声杀死整台机器人。

改成统计 gate：

1. no NaN/Inf；
2. relevant metric stable sample rate `>=95%`；
3. rank flip fraction `<=2%`；
4. retained singular subspace projector distance p95 `<=0.05`；
5. engine/FD normalized error有界；
6. physical-zero columns 不计 unstable；
7. rank threshold 包含估计误差：

\[
\tau_{\rm rank}
=
\max(\tau_{\rm rel}\sigma_1,k\|E_J\|_2,\tau_{\rm backend})
\]

8. structural rank 仍来自 per-sample local ranks，不使用 concatenated rank；
9. 输出每个 task 的：
   - relevant metric；
   - zero-column fraction；
   - stable-sample fraction；
   - rank histogram；
   - projector-distance percentiles；
   - engine/FD error percentiles。

### Agent E — Rotation Mapping, Target Normalization & Canonical Protocol

**Owned files**

```text
soma_retargeter/robotics/v3/rest_frames.py
soma_retargeter/robotics/v3/target_builder.py
soma_retargeter/robotics/v3/canonical_projection.py
soma_retargeter/robotics/v3/chain_projection.py
tests/v3/test_rotation_frame_mapping_*.py
tests/v3/test_canonical_protocol_*.py
```

**必须修正 parent-frame rotation delta**

定义：

\[
R_{h0}={} ^{P_h}R_{C_h,0},\qquad
R_h={} ^{P_h}R_{C_h}
\]

\[
\Delta R_h^{P_h}=R_hR_{h0}^\top
\]

\[
\Delta R_r^{P_r}=A_P\Delta R_h^{P_h}A_P^\top
\]

\[
R_r^*=\Delta R_r^{P_r}R_{r0}
\]

不得混用 body-frame delta 和 parent-frame alignment。

测试必须包含：

- source rest rotation 非 identity；
- robot rest rotation 非 identity；
- oblique joint axis；
- rotated parent frame；
- common world transform。

**Canonical protocol**

Capability motions 是独立样本，不是时间序列：

```text
每个 canonical motion:
q_seed = neutral_q
previous_q = None
```

continuity prior 只用于另外定义的 temporal sequences。

**Position normalization**

改为：

\[
e_p=(p(q)-p_d)/\max(L_{\rm chain},\epsilon)
\]

其中 \(L_{\rm chain}\) 是 neutral kinematic path length。禁止用“所有关节同时打到上下限后的最大位移”作为跨机器人 normalization。

### Agent F — Semantics, Full-Zoo Validation & Independent Red Team

**Owned files**

```text
assets/robot_zoo/semantic_maps/
assets/robot_zoo/semantic_expectations/
soma_retargeter/robotics/v3/semantic_validation.py
soma_retargeter/robotics/v3/validation.py
scripts/audit_retargeting_v3_step2.py
.github/workflows/retargeting_v3_step2.yml
artifacts/retargeting_v3_step2/
docs/retargeting_v3/STEP2_IMPLEMENTATION_REPORT.md
docs/retargeting_v3/STEP2_ACCEPTANCE_LEDGER.md
docs/retargeting_v3/subagents/assets_ready_agent_f_red_team.md
```

**任务**

1. 为 H1 URDF 和 Berkeley URDF/MJCF 补 verified/partial maps；
2. Berkeley 若无完整上肢，必须 `partial_passed`，不得伪造 Hand；
3. 为新 G1 23DoF URDF/MJCF 建 verified maps；
4.