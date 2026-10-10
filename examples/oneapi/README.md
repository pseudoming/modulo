# One API 日志样例

在仓库根目录运行，所有命令使用 `--adapter oneapi`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [basic.jsonl](basic.jsonl) | 消费表日志；有 usage，缺请求／响应正文 | [basic.bindings.yaml](basic.bindings.yaml) | [登记](../../tests/fixtures/expected/gateway-logs.json) |
| [request-response.log](request-response.log) | 多行请求／响应文本，含思考、工具和 SSE | [request-response.bindings.yaml](request-response.bindings.yaml) | [登记](../../tests/fixtures/provenance/request-response.json) |
| [request-response-events.jsonl](request-response-events.jsonl) | 相同调用的 ES 事件包装，含 `message` 与 `event.sequence`，故意乱序 | 同上 | 同上 |
| [request-response-exchanges.jsonl](request-response-exchanges.jsonl) | 相同调用的已配对 HTTP 记录 | 同上 | 同上 |

```bash
.venv/bin/modulo import --source log --adapter oneapi \
  --path examples/oneapi/request-response.log \
  --bindings examples/oneapi/request-response.bindings.yaml
```

另外两种正文样例可替换 `--path`，沿用相同绑定。运行消费表样例时，改用 `basic.jsonl` 和 `basic.bindings.yaml`。

正文样例中的响应日志标签沿用历史扩展格式，不代表原版 One API 默认输出；来源登记保留了这一差别。缺字段时按实际输入诊断，格式与条件见 [正文接入](../../README.md#log-formats)。
