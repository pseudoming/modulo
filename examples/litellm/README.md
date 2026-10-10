# LiteLLM 日志样例

在仓库根目录运行，所有命令使用 `--adapter litellm`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [basic.jsonl](basic.jsonl) | 基础调用、模型别名、差额、缺字段和工具上下文 | [basic.bindings.yaml](basic.bindings.yaml) | [登记](../../tests/fixtures/expected/gateway-logs.json) |
| [reasoning-tools.jsonl](reasoning-tools.jsonl) | 11 个型号的思考、多轮历史、工具和模式参数 | [reasoning-tools.bindings.yaml](reasoning-tools.bindings.yaml) | [登记](../../tests/fixtures/provenance/reasoning-tools.json) |
| [model-families.jsonl](model-families.jsonl) | 各型号的文本复算与缺口对照 | [model-families.bindings.yaml](model-families.bindings.yaml) | [登记](../../tests/fixtures/expected/model-families.json) |

```bash
.venv/bin/modulo import --source log --adapter litellm \
  --path examples/litellm/reasoning-tools.jsonl \
  --bindings examples/litellm/reasoning-tools.bindings.yaml

.venv/bin/modulo import --source log --adapter litellm \
  --path examples/litellm/model-families.jsonl \
  --bindings examples/litellm/model-families.bindings.yaml \
  --platform-tag-source metadata:platform --platforms synthetic-reference
```

运行基础样例时，使用 `basic.jsonl` 和 `basic.bindings.yaml`。`synthetic-reference` 是样例中的平台标签；计量范围见 [型号说明](../../README.md#tokenizers) 和 [思考说明](../../README.md#faq)。

[model-families.bindings.yaml](model-families.bindings.yaml) 的平台条目已自带 `platform-tag-source: metadata:platform`；全局条目与平台条目都在同一个 `models` 数组。上面的批次来源参数也给未绑定型号提取平台，因此按平台筛选时仍保留这些型号以展示未绑定原因。平台条件、全局兜底及冲突处理见 [配置说明](../../README.md#configuration)。
