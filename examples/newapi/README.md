# New API 日志样例

在仓库根目录运行，所有命令使用 `--adapter newapi`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [basic.jsonl](basic.jsonl) | 消费表日志；有 usage，缺请求／响应正文 | [basic.bindings.yaml](basic.bindings.yaml) | [登记](../../tests/fixtures/expected/gateway-logs.json) |
| [request-response.log](request-response.log) | 请求／响应 DEBUG 文本，含思考、工具和 SSE | [request-response.bindings.yaml](request-response.bindings.yaml) | [登记](../../tests/fixtures/provenance/request-response.json) |
| [request-response-events.jsonl](request-response-events.jsonl) | 相同调用的 ES 事件包装，含 `message` 与 `event.sequence`，故意乱序 | 同上 | 同上 |
| [request-response-exchanges.jsonl](request-response-exchanges.jsonl) | 相同调用的已配对 HTTP 记录 | 同上 | 同上 |

```bash
.venv/bin/modulo import --source log --adapter newapi \
  --path examples/newapi/request-response.log \
  --bindings examples/newapi/request-response.bindings.yaml
```

另外两种正文样例可替换 `--path`，沿用相同绑定。运行消费表样例时，改用 `basic.jsonl` 和 `basic.bindings.yaml`；缺正文的分项保留不可复算状态。记录格式、ES 顺序要求见 [正文接入](../../README.md#log-formats)。
