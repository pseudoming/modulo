# Dify 日志样例

在仓库根目录运行，所有命令使用 `--adapter dify`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [basic.jsonl](basic.jsonl) | 基础调用、差额、缺字段与重复记录 | [默认绑定](../../config/modulo.yaml) | [登记](../../tests/fixtures/expected/basic.json) |
| [platforms.jsonl](platforms.jsonl) | 同一模型的不同平台标签 | [platforms.bindings.yaml](platforms.bindings.yaml) | [登记](../../tests/fixtures/expected/platforms.json) |
| [model-aliases.jsonl](model-aliases.jsonl) | `volc-deepseek` 日志名称映射到 DeepSeek tokenizer | [model-aliases.bindings.yaml](model-aliases.bindings.yaml) | [登记](../../tests/fixtures/expected/model-aliases.json) |
| [reasoning-tools.jsonl](reasoning-tools.jsonl) | 19 个型号的思考、多轮历史、工具和模式参数 | [reasoning-tools.bindings.yaml](reasoning-tools.bindings.yaml) | [登记](../../tests/fixtures/provenance/reasoning-tools.json) |

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

## 结构化 thought 导出

```bash
uv run modulo import --source log --adapter dify \
  --path examples/dify/thought-exports.jsonl
```

thought 样例是人工合成的结构化导出：`message` 为消息数组、该数组的 JSON 字符串，或带 `messages`／`tools`／参数的请求对象时均可读取。样例额外保存完整 `response`；不声称 Dify 默认 thought 表保存这些正文。这里只有一个输入 53／输出 9 token 的调用，三项结论均一致。空 `message` 继续保留 usage 与已有输出；不会从 `thought` 或 `tool_input` 猜补请求。有工具操作却未导出工具定义时，输入保留对应缺口。

数据由人工构造；53／9 的独立固定模板依据及文件／ES、报告重读验证见 [适配器贯通测试](../../tests/integration/test_adapter_imports.py)。
