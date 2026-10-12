# One API 日志样例

在仓库根目录运行，所有命令使用 `--adapter oneapi`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [api-bodies.jsonl](api-bodies.jsonl) | Responses／Anthropic 原始正文，含思考与工具 | [request-response.bindings.yaml](request-response.bindings.yaml) | 合成数据，预期见下文 |
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

## API 正文

```bash
uv run modulo import --source log --adapter oneapi \
  --path examples/oneapi/api-bodies.jsonl \
  --bindings examples/oneapi/request-response.bindings.yaml
```

4 条人工合成的 Qwen 调用：Responses、Anthropic 各有普通回答和流式工具输出。输入参照均为 53 token；普通输出为 9，含工具的输出为 45，输入／输出均为 4 条一致。正文放在已有的配对记录中；流式响应保留 SSE 文本。

字段依据为 [Responses 官方说明](https://developers.openai.com/api/docs/guides/migrate-to-responses) 与 [Anthropic Messages 官方说明](https://platform.claude.com/docs/en/api/messages/create)；样例代表导入格式。计数依据及两种来源的回归见 [完整流程测试](../../tests/integration/test_api_bodies.py)。

报告中的 `warnings` 会按本次导入发现的缺口列出缺失字段、影响及操作建议：正文、共同 request_id、重试 attempt_id、流事件顺序或结束事件。补齐已有日志后重新导入即可产生新的结果包；缺口不自动视为通过。警告覆盖本次读到的记录，原始错误仍保存在诊断中。
