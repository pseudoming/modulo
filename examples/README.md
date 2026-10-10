# 日志样例

先按 `--adapter` 选择目录，再按文件名选择场景。全部是合成数据；每个目录的 README 列出日志用途、配套配置、来源与运行命令。

| 适配器目录 | 样例内容 |
| --- | --- |
| [dify](dify/README.md) | 基础调用、平台标签、模型别名、思考与工具 |
| [langchain](langchain/README.md) | 基础调用、平台标签、模型别名、思考与工具 |
| [newapi](newapi/README.md) | 消费日志、请求／响应正文、ES 事件、配对 HTTP 记录 |
| [oneapi](oneapi/README.md) | 消费日志、请求／响应正文、ES 事件、配对 HTTP 记录 |
| [litellm](litellm/README.md) | 基础日志、思考与工具、各型号复算 |

`*.bindings.yaml` 是同名场景的模型到 tokenizer 绑定，每个数组条目用 `modelnames` 收拢别名。跨适配器相同的场景也就近放置配置，单个样例不依赖其他适配器目录。正式默认绑定在 [config/modulo.yaml](../config/modulo.yaml)，包含全部 11 个固定型号；不传 `--bindings` 时使用它。ES 查询条件的写法见 [运行说明](../README.md#es-source)。

命令均从仓库根目录执行，环境准备见 [项目 README](../README.md)。独立标准答案与来源登记保存在 [tests/fixtures](../tests/fixtures)，各目录直接链接到对应文件；样例日志和标准答案分开保存。
