# Dify 日志样例

在仓库根目录运行，所有命令使用 `--adapter dify`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [basic.jsonl](basic.jsonl) | 基础调用、差额、缺字段与重复记录 | [默认绑定](../../config/modulo.yaml) | [登记](../../tests/fixtures/expected/basic.json) |
| [platforms.jsonl](platforms.jsonl) | 同一模型的不同平台标签 | [platforms.bindings.yaml](platforms.bindings.yaml) | [登记](../../tests/fixtures/expected/platforms.json) |
| [model-aliases.jsonl](model-aliases.jsonl) | `volc-deepseek` 日志名称映射到 DeepSeek tokenizer | [model-aliases.bindings.yaml](model-aliases.bindings.yaml) | [登记](../../tests/fixtures/expected/model-aliases.json) |
| [reasoning-tools.jsonl](reasoning-tools.jsonl) | 11 个型号的思考、多轮历史、工具和模式参数 | [reasoning-tools.bindings.yaml](reasoning-tools.bindings.yaml) | [登记](../../tests/fixtures/provenance/reasoning-tools.json) |

```bash
.venv/bin/modulo import --source log --adapter dify \
  --path examples/dify/basic.jsonl

.venv/bin/modulo import --source log --adapter dify \
  --path examples/dify/platforms.jsonl --payload-field payload \
  --bindings examples/dify/platforms.bindings.yaml \
  --platform-config examples/dify/platforms-metadata.yaml
```

其他场景按表替换 `--path` 和 `--bindings`。[platforms-metadata.yaml](platforms-metadata.yaml) 从 metadata 取平台；[platforms-endpoint.yaml](platforms-endpoint.yaml) 按 endpoint 映射；[platforms-endpoints.yaml](platforms-endpoints.yaml) 是单独传给 `--endpoint-map` 的映射表。平台样例需要 `--payload-field payload`。详细参数见 [平台说明](../../README.md#configuration)。

这些是合成的格式与计算样例；来源和独立预期见上表对应的登记文件。
