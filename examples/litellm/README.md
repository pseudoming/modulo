# LiteLLM 日志样例

在仓库根目录运行，所有命令使用 `--adapter litellm`。

| 日志文件 | 内容 | 配套绑定 | 来源与预期 |
| --- | --- | --- | --- |
| [api-bodies.jsonl](api-bodies.jsonl) | Responses／Anthropic 原始正文，含思考与工具 | [basic.bindings.yaml](basic.bindings.yaml) | 合成数据，预期见下文 |
| [images.jsonl](images.jsonl) | 十个视觉型号的内嵌图片、输入差额和图片缺口 | [model-families.bindings.yaml](model-families.bindings.yaml) | 合成数据，预期见下文 |
| [content-blocks.jsonl](content-blocks.jsonl) | 结构化思考、两个候选、缺候选与摘要缺口 | [basic.bindings.yaml](basic.bindings.yaml) | 合成数据，预期见下文 |
| [output-audit.jsonl](output-audit.jsonl) | 完整工具输出参照、正负差额与缺失／截断 | [basic.bindings.yaml](basic.bindings.yaml) | 合成数据，预期见下文 |
| [basic.jsonl](basic.jsonl) | 基础调用、模型别名、差额、缺字段和工具上下文 | [basic.bindings.yaml](basic.bindings.yaml) | [登记](../../tests/fixtures/expected/gateway-logs.json) |
| [reasoning-tools.jsonl](reasoning-tools.jsonl) | 19 个型号的思考、多轮历史、工具和模式参数 | [reasoning-tools.bindings.yaml](reasoning-tools.bindings.yaml) | [登记](../../tests/fixtures/provenance/reasoning-tools.json) |
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

输出样例：

```bash
uv run modulo import --source log --adapter litellm \
  --path examples/litellm/output-audit.jsonl \
  --bindings examples/litellm/basic.bindings.yaml
```

六条均是人工合成的 Qwen 调用，包含 `THOUGHT`、`ANSWER` 及一次 `lookup(z=2,a="中")` 工具调用。固定完整输出参照为 45 token；工具片段单独为 36，未含结束部分的组合正文为 44。正常记录报告 45／48／42，差额分别为 0／+3／−3；`length` 记录表示达到 token 上限，按 44 计且不补 EOS，报告 45，差额 +1。缺 usage、缺 finish_reason 两条不可核验。预期 `output_verdicts` 为 1 一致、3 有差异、2 不可核验，可比较 4 条，有符号差额合计 +1、绝对差额合计 7。输入未提供 usage，`input_verdicts` 为 6 不可核验；整体 `verdicts` 为 3 有差异、3 不可核验。

来源与预期构造见 [流程测试](../../tests/integration/test_output.py)，45 token 的完整拼接及 token ID 由 [独立参照测试](../../tests/adapters/tokenizers/test_output.py) 对照固定官方模板和原生词表检查；样例不是网关生产实录。不会因复算结果变化重写预期。

## 结构化正文与多候选

先准备 Qwen 资产：`uv run modulo prepare-assets --config config/modulo.yaml --only qwen-3.8-2.4t`，然后运行：

```bash
uv run modulo import --source log --adapter litellm \
  --path examples/litellm/content-blocks.jsonl \
  --bindings examples/litellm/basic.bindings.yaml --output runs/content-blocks
# 将 RUN_DIR 换成上一步返回的 run 路径
uv run modulo report RUN_DIR --details
```

5 条人工合成的 Qwen 请求，每条保存两个候选。正文由 `thinking` 和 `text` 块组成，签名保留但不计为模型正文。固定参照输入为 53 token，每个完整候选为 9 token，总输出为 18：

| 调用 ID | 场景 | 报告输出 | 完整输出参照 | 差额／结论 |
| --- | --- | --- | --- | --- |
| `blocks-complete` | 两个完整候选 | 18 | 18 | 0／一致 |
| `blocks-more` | 报告多 3 | 21 | 18 | +3／有差异 |
| `blocks-truncated` | 第二个达到 token 上限 | 18 | 17（9＋8，无末尾 EOS） | +1／有差异 |
| `blocks-missing-candidate` | 请求 n=3，只记了两个 | 18 | 缺失 | 不可核验 |
| `blocks-summary` | 第二个只有思考摘要 | 18 | 缺失 | 不可核验 |

输入结论 5 条一致；输出 1 条一致、2 条有差异、2 条不可核验。输入只计 5 次，不能按候选重复累计。缺候选和只有摘要的最后两条保留可算分项，不给完整输出结论。

`report --details` 中的 `call.output_candidates` 保存各候选原文与结束原因；`measurement.output_candidates` 保存各自的计数、执行状态、参数和摘要。DuckDB 的 `output_candidates` 视图可按 `call_id`、`candidate_index` 查看这 10 条候选，`calls` 仍只有 5 条请求。

53／9 的依据是手写展开的固定 Qwen 格式与原生词表编码，见 [流程测试](../../tests/integration/test_candidates.py)。这些是解析与计量样例，不是 LiteLLM 生产日志；同一结构链路也验证了 LangChain、New API、One API 多候选，五种适配器的单候选结构化正文，以及文件／模拟 ES 两种来源。

## 图片输入

```bash
uv run modulo prepare-assets --config config/modulo.yaml --only deepseek-v4.1 kimi-k3 minimax-m3 mimo-v2.6 step-3.7 \
  qwen3.5-122b qwen3.5-397b glm-5.3-flash kimi-k2.6 kimi-k2.7-code
uv run modulo import --source log --adapter litellm \
  --path examples/litellm/images.jsonl \
  --bindings examples/litellm/model-families.bindings.yaml --output runs/images
# 将 RUN_DIR 换成上一步返回的 run 路径
uv run modulo report RUN_DIR --details
```

前十条日志各包含一张合成的 640×480 白色 PNG（data URL），前后有文字。usage 来自[冻结的官方参照](../../tests/fixtures/tokenizers/image_reference_vectors.json)，不是服务返回的实际账单：

| 固定型号 | 视觉位置数 | 完整图片段 token | 完整输入 token |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | 192 | 206 | 248 |
| Kimi-K3 | 414 | 422 | 521 |
| MiniMax-M3 | 391 | 393 | 578 |
| MiMo-V2.6-Pro-RL | 300 | 302 | 318 |
| Step-3.7-Flash | 169 | 171 | 194 |
| Qwen3.5-122B-A10B | 300 | 302 | 320 |
| Qwen3.5-397B-A17B | 300 | 302 | 320 |
| GLM-5.3-Flash | 414 | 416 | 437 |
| Kimi-K2.6 | 414 | 418 | 438 |
| Kimi-K2.7-Code | 414 | 418 | 438 |

完整图片段包括图片边界、必要的行分隔或分辨率文字；完整输入另含消息文本和模板。Kimi 的 tokenizer 先生成一个媒体占位，模型再展开视觉位置；结果中的 `input_ids_scope=tokenizer_ids_before_image_expansion` 标明 ID 摘要覆盖展开前的序列，`input_tokens` 已计入展开后的长度。

另三条 DeepSeek 样例分别为报告多 7、只有图片 URL、显式 `detail=high`。后两条缺少可复现的图片依据或参数映射，输入不可核验。预期输入结果：**10 一致、1 有差异、2 不可核验**；差额合计 +7。13 条输出均有本地计数，但未提供输出 usage，不能判定输出一致。

`measurement.input_images` 保存每张图的消息位置、尺寸来源、图片摘要、布局、参数来源、计数及缺口；`image_input_tokens` 是本次完整图片段总数，已包含在 `input_tokens` 中，不能再加一次。汇总中的 `image_input_recalculated` 是完成计数的调用数，`image_input_token_sum` 是这些调用的图片段合计。DuckDB `input_images` 视图可查询逐图明细。

接受有序正文块中的 `image`、`image_url`、`input_image`。图片数据可以是 `image_url.url` 的 data URL，或 `source.data`／`base64`；也可直接记录正整数 `width`、`height`。Qwen3.5／GLM-5.3-Flash／MiniMax-M3／MiMo 另接受图片块上的 `image_grid_thw: [1, H, W]`；同时有尺寸时会核对两者是否一致。宽高和网格是日志证据，并不证明图片真实性。默认不访问 URL；缺少尺寸的公网图片可通过主配置 `image-download` 或 `--download-images` 显式启用下载，配置见 [README](../../README.md)。本地文件路径不读取。

本例的 `example.invalid` 是故意不可访问的合成地址，启用下载后仍应不可核验。换成实际可访问的图片 URL 后，可从 `measurement.input_images[].download` 查看下载时间、文件摘要、字节数和缓存相对路径。[URL 贯通测试](../../tests/integration/test_image_download.py)使用合成 HTTP 响应验证五种协议与十个图片参照，覆盖缓存复用和失败隔离。

处理参数可从请求的 `mm_processor_kwargs`／`images_kwargs` 读取；图片块支持 `min_pixels`／`max_pixels`。支持的键对应[tokenizer 资源清单](../../static/tokenizer-assets.json)中的 `image_processing.parameters`，冲突和未知键报缺口；缺省值标为 `asset_profile`。`detail` 仅接受省略或 `auto`，不猜测托管服务的 high／low 映射；自定义裁剪、动画图片和音视频不在此参照内。内嵌数据按固定 processor 的 EXIF 规则处理；仅给尺寸时将其视为该 processor 读取到的有效尺寸。

五种协议的完整正文块都走同一条链路；Dify 默认保存记录若把图片拆进 `files` 并截断数据，会保留文字并标明图片顺序／数据缺失，不补造正文。参照由[独立工具](../../tests/tools/references/build_image_reference.py)执行摘要核对后的官方布局函数及模板建立；[贯通测试](../../tests/integration/test_images.py)覆盖五适配器及文件／模拟 ES。样例生成器见 [generate_image_fixtures.py](../../tests/tools/fixtures/generate_image_fixtures.py)。

## API 正文

```bash
uv run modulo import --source log --adapter litellm \
  --path examples/litellm/api-bodies.jsonl \
  --bindings examples/litellm/basic.bindings.yaml
```

4 条人工合成的 Qwen 调用：Responses、Anthropic 各有普通回答和流式工具输出。输入参照均为 53 token；普通输出为 9，含工具的输出为 45，输入／输出均为 4 条一致。本文件按 LiteLLM 的终态日志保存已组装响应。

字段依据为 [Responses 官方说明](https://developers.openai.com/api/docs/guides/migrate-to-responses) 与 [Anthropic Messages 官方说明](https://platform.claude.com/docs/en/api/messages/create)；样例代表导入格式。计数依据及两种来源的回归见 [完整流程测试](../../tests/integration/test_api_bodies.py)。

## 请求参数与自定义模板

五种日志适配器读取已有的 `response_format`、`response_schema`、`tool_choice`、`parallel_tool_calls` 和 `add_generation_prompt`。`tool_choice` 支持 auto／none／required／指定函数。约束按所选模板处理，不因 JSON Schema 凭空增加系统提示词；参见 [vLLM 结构化输出](https://docs.vllm.ai/en/stable/features/structured_outputs/)。

模型绑定可以声明参照默认参数：

```yaml
models:
  - name: qwen-3.8-2.4t
    aliases: [internal-qwen]
    template-kwargs:
      preserve_thinking: false
```

日志中的显式值优先于这里的默认值；false／0 保留原义。需要其他模板时，在同一条目加 `chat-template: |`，多行值填写经核对的完整 Jinja 模板，特殊 token 沿用对应型号的词表。未配置时使用固定资产模板；不会执行日志里的模板。绑定全文与参数随结果冻结，明细保留 `template_hash` 及参数来源。Jinja 基础见 [Hugging Face 模板说明](https://huggingface.co/docs/transformers/chat_templating_writing)。

模板可读取 message 的 role／content／reasoning_content／tool_calls。复杂 assistant 正文另提供 `content_blocks` 有序数组：text／reasoning 块使用 text 字段，tool_calls 块使用 tool_calls 数组。读取此数组的模板可保留“文本→思考→工具→文本”等交错顺序；内置模板不能表达且未配置相应模板时，明细记录 `content_order_not_supported_by_template`。Responses 多条 assistant 消息保留消息边界，与多个候选区分。

`stop`／`stop_sequences`、`stop_token_ids`、token 上限与来源单独存入 `generation_options`／`generation_option_sources`。`length`／`max_tokens` 等按实际生成正文计数，不补 EOS；已命中的停止串／token ID 只计一次，已包含与已剔除的停止串分别处理。若只知道 finish_reason=stop、却不知道究竟命中了哪个自定义停止项，输入仍可算，输出保留 `output_stop_reason_missing`。结果的 `output_termination` 说明采用的规则，多候选则分别看 `measurement.output_candidates[].termination`。参考 [Anthropic 停止原因](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)。

## 流片段与缓存命中

```bash
uv run modulo import --source log --adapter litellm \
  --path examples/litellm/stream-and-cache.jsonl
```

合成样例包含同一次请求的 4 个流片段，以及 1 条缓存命中终态。两条明细均复算输入 53／输出 9 token；流调用三项结论一致，缓存调用保留计数和原始 usage，但两项差额为空、结论不可核验，原因是缓存记录不能证明本次新推理的用量。`cache_hit_calls` 单独统计缓存调用。

接受日志 `response` 中单个 SDK chunk、有序 chunk 数组或 SSE 字符串，也接受已序列化的 `kwargs`＋`response_obj` 回调参数。需要 `call_type`、型号、请求正文与 `id`／`request_id`，回调导出使用 `kwargs.litellm_call_id`。跨行优先按 `litellm_call_id` 关联，再使用日志 ID；有重试时保留 `attempt_id`。这描述可读取的日志结构，不安装采集器，也不表示 LiteLLM 默认输出全部片段。

ES 流片段保留 `sequence`／`event.sequence`，或同一文件的 `log.file.path`＋`log.offset`；原生 Responses 的 `sequence_number` 也可排序。直接文件可使用原有行顺序。只有明确事件 ID 或顺序位置相同的重复记录才去重，所有原始引用保留。终态与完整流同时存在时校验正文一致，计数一次。缺序、缺结束事件、身份冲突及请求冲突会保留缺口。SDK JSON chunk 不提供 HTTP `[DONE]`，因此逐候选检查 `finish_reason`；SSE 文本仍要求真实结束事件。

数据由人工构造；53／9 的独立固定模板依据及文件／ES、报告重读验证见 [适配器贯通测试](../../tests/integration/test_adapter_imports.py)。
