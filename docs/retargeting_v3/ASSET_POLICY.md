# Robot Zoo Asset and License Policy

## 原则

1. 固定版本，不使用浮动 main。
2. 不执行未知上游脚本。
3. 不提交不明许可证资产。
4. Apache/MIT/BSD 类模型可生成并提交轻量 kinematic snapshot，同时保留 LICENSE/SOURCE。
5. GPL/LGPL/CC-BY-SA/NASA 等进入 fetch-only，除非法律/项目负责人明确批准再分发。
6. 不提交非商业/禁止衍生条款资产。
7. 大型 meshes 默认不提交。
8. 每个 snapshot 必须可由工具确定性重建。
9. 接受仓库现有 `.gitattributes` 对 XML 等文本资产使用 Git LFS，但 LFS 只作为传输机制，不改变许可证、体积、内容和可复现性约束。

## Git LFS 规则

允许：

- 轻量、mesh-free 的 MJCF/XML kinematic snapshot；
- 已经在 `.gitattributes` 中声明的其他项目文本/二进制格式；
- GitHub Actions 使用 `actions/checkout` 的 `lfs: true`；
- 本地通过 `git lfs pull` 获取对象。

必须：

- `git lfs fsck` 通过；
- CI 和完整验证不得在 LFS pointer 文件上运行；
- snapshot 仍受每机器人 2 MB 的 policy limit；
- `SOURCE.json` 必须记录源文件和 snapshot SHA；
- clone 后的复现说明必须包含 Git LFS 安装和拉取步骤；
- audit 检查对象已 materialize，而不是只检查 pointer 存在。

仍然禁止：

- 用 Git LFS 提交完整上游仓库；
- 用 Git LFS 提交 visual/collision meshes；
- 用 Git LFS 绕过许可证或 fetch-only 规则；
- 用 Git LFS 隐藏数百 MB、不可审计或未固定版本的资产。

## SOURCE.json 最低字段

```json
{
  "robot_id": "...",
  "description_name": "...",
  "upstream_repository": "...",
  "upstream_ref": "...",
  "source_file": "...",
  "source_sha256": "...",
  "format": "urdf|mjcf",
  "license": "...",
  "license_sha256": "...",
  "redistribution": "kinematic_snapshot|fetch_only|excluded",
  "generator_version": "...",
  "snapshot_sha256": "..."
}
```

## 禁止

- 删除许可证；
- 只写 URL 不锁 commit；
- 将 fetch 失败当作 pass；
- 自动更新 pins；
- 许可证 unknown 时继续 vendor；
- 在未执行 `git lfs pull` 的 pointer-only checkout 上声称模型加载或算法验证通过。
