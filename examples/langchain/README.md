# LangChain 日志样例

在仓库根目录运行，所有命令使用 `--adapter langchain`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [basic.jsonl](basic.jsonl) | FakeModel 生成的基础轨迹，含差额、缺字段与重复记录 | [默认绑定](../../config/modulo.yaml) | [登记](../../tests/fixtures/expected/basic.json) |
| [platforms.jsonl](platforms.jsonl) | 同一模型的不同平台标签 | [platforms.bindings.yaml](platforms.bindings.yaml) | [登记](../../tests/fixtures/expected/platforms.json) |
| [model-aliases.jsonl](model-aliases.jsonl) | `volc-deepseek` 日志名称映射到 DeepSeek tokenizer | [model-aliases.bindings.yaml](model-aliases.bindings.yaml) | [登记](../../tests/fixtures/expected/model-aliases.json) |
| [reasoning-tools.jsonl](reasoning-tools.jsonl) | 19 个型号的思考、多轮历史、工具和模式参数 | [reasoning-tools.bindings.yaml](reasoning-tools.bindings.yaml) | [登记](../../tests/fixtures/provenance/reasoning-tools.json) |

```bash
.venv/bin/modulo import --source log --adapter langchain \
  --path examples/langchain/basic.jsonl

.venv/bin/modulo import --source log --adapter langchain \
  --path examples/langchain/platforms.jsonl --payload-field payload \
  --bindings examples/langchain/platforms.bindings.yaml \
  --platform-config examples/langchain/platforms-metadata.yaml
```

其他场景按表替换 `--path` 和 `--bindings`。[platforms-metadata.yaml](platforms-metadata.yaml) 从 metadata 取平台；[platforms-endpoint.yaml](platforms-endpoint.yaml) 按 endpoint 映射；[platforms-endpoints.yaml](platforms-endpoints.yaml) 是单独传给 `--endpoint-map` 的映射表。平台样例需要 `--payload-field payload`。详细参数见 [平台说明](../../README.md#configuration)。

## 多组输入与整批 usage

```bash
uv run modulo import --source log --adapter langchain \
  --path examples/langchain/input-batches.jsonl
```

两个合成 Run 各包含两组输入及对应 generations；一条只报整批 usage，另一条含每组 AIMessage usage。每组输入 53／输出 9 token。共生成 4 条明细：两条可逐组比较，两条保留本地计数、逐组结论不可核验；两个 Run 均在汇总 `input_batches` 中比较 106／18，结果一致。`call.input_batch` 保留原 Run ID、组位置、组数和整批报告值。整批 usage 不重复加入 `by_model` 的逐条报告总和。输入／输出组数不符、来源冲突或有组未完成时，不能把局部和当作完整整批结论。

数据由人工构造；53／9 的独立固定模板依据及文件／ES、报告重读验证见 [适配器贯通测试](../../tests/integration/test_adapter_imports.py)。
