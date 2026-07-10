# SOMA Retargeter vNext — `dev` 统一集成执行入口

> 本文件是 Codex 的最高优先级执行入口。完整技术规格见 [`docs/vnext/goal_spec.md`](docs/vnext/goal_spec.md)。
>
> **分支规则覆盖完整规格中所有旧的 `main` 表述：本次及后续 vNext 开发只能提交到 `dev`，不得直接提交、合并、force-push 或覆盖 `main`。** `main` 仅作为稳定发布分支，必须等人工验收后再单独决定是否从 `dev` 发起 PR。

## 1. 开始前必须完成分支整合审计

不要假设当前 `dev` 已包含仓库其他分支的全部有效成果，也不要把所有分支机械 merge。开始改代码前必须：

1. `git fetch --all --prune`，列出所有本地/远端分支、开放和已合并 PR、各自 head SHA、merge-base、提交范围和改动文件。
2. 至少审计当前已知的开发线：
   - `dev`；
   - 已合并到 `dev` 的 contact-aware foot IK PR #1、#2；
   - `retargeting-v3-step2-integrated-review`；
   - `retargeting-v3-step2-assets-clean-sync`；
   - `retargeting-v3-step2-assets-vendored` 与开放 PR #9；
   - `agent-a-runtime-model-truth`、`agent-b-verified-semantics`、`agent-c-rest-calibration`、`agent-d-reachability`、`retargeting-v3-step2-agent-e-code-only`、`retargeting-v3-step2-agent-f-audit`；
   - fetch 后发现的其他分支和未合并 PR。
3. 生成 `docs/vnext/branch_integration_inventory.md`，逐分支记录：head SHA、基线、独有提交、功能主题、测试状态、是否已被其他集成分支包含、冲突风险、处理决定和吸收后的目标提交。
4. 以功能和提交 patch-id 去重：已包含的提交不得重复 cherry-pick；相互替代的实验实现必须择优整合，不能同时保留两套互相冲突的默认路径。
5. 优先吸收已验证的源码、测试、schema、语义映射、runtime adapter、rest calibration、reachability、validation 和 CI；生成物、临时报告、缓存、失败快照、大型运行输出及来源/许可证不清楚的 vendored 资产不得因“整合所有分支”而盲目进入 `dev`。
6. 开放 PR #9 不能直接视为已验收。先检查其 38 snapshot / 5 fetch-only / 1 local Assets44 范围、锁文件、许可证、LFS/仓库体积、可复现性和测试证据，再决定完整吸收、部分吸收或拒绝，并在 inventory 中写明理由。
7. 完成盘点后，在 `dev` 上建立一个明确的 integration checkpoint；后续 vNext 重构都基于该 checkpoint。不要删除原分支，便于追溯。

## 2. 完整版本实施要求

完整读取并实施 [`docs/vnext/goal_spec.md`](docs/vnext/goal_spec.md) 的全部 P0、P1、P2、公式、架构、测试、CI、benchmark、迁移和文档要求，但应用以下覆盖规则：

- 所有“当前 `main`”“覆盖 `main`”“从 `main` 直接交付”等措辞统一解释为“当前完成分支整合后的 `dev`”。
- 不允许把 `dev` reset 成旧 `main`；必须保留并整合 `dev` 已有 contact-aware foot IK、grounding、diagnostics 等有效成果。
- 不允许继续按 scale / translation / rotation 分阶段冻结优化；所有可训练 scaler 参数必须在同一联合优化循环中更新。
- runtime、optimizer、preview、metrics 必须共享唯一 target generator。
- 所有完成定义仍然有效；未实际运行的 GPU/LFS/真实资产测试必须如实标记为未验证。
- 开发完成后只 push `dev`。不要自动创建或合并 `dev -> main` PR，除非仓库所有者之后明确要求。

## 3. 最终交付附加项

除完整规格要求的交付外，还必须：

- 在 `docs/vnext/implementation_report.md` 中增加“分支整合来源”章节，列出实际吸收的分支、PR、commit SHA 和被拒绝内容；
- 保证 `git log --first-parent dev` 能清楚识别 integration checkpoint 与 vNext 实施提交；
- 报告 `dev` 相对 `main` 的 ahead/behind、未合并 PR 状态和发布前剩余风险；
- 最终工作树干净，并确认没有对 `main` 做任何写操作。

开始执行时不要只回复计划；先完成分支 inventory，再直接实施整个版本。