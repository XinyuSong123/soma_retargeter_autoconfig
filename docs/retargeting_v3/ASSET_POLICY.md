# Robot Zoo Asset and License Policy

## 原则

1. 固定版本，不使用浮动 main。
2. 不执行未知上游脚本。
3. 不提交不明许可证资产。
4. Apache/MIT/BSD类模型可生成并提交轻量kinematic snapshot，同时保留LICENSE/SOURCE。
5. GPL/LGPL/CC-BY-SA/NASA等进入fetch-only，除非法律/项目负责人明确批准再分发。
6. 不提交非商业/禁止衍生条款资产。
7. 大型meshes默认不提交。
8. 每个snapshot必须可由工具确定性重建。

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
- 只写URL不锁commit；
- 用Git LFS掩盖数百MB资产；
- 将fetch失败当作pass；
- 自动更新pins；
- 许可证unknown时继续vendor。
