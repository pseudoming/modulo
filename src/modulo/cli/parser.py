"""Command syntax and help text, without audit decisions."""

import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="modulo", description="日志批量导入与本地参照 token 复算")
    root.add_argument("--plugin", action="append", default=[], help="显式加载受信 Python 模块")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("plugins", help="列出三类已注册适配器")
    assets = commands.add_parser(
        "prepare-assets", help="显式联网准备固定 tokenizer 资产", argument_default=argparse.SUPPRESS
    )
    assets.add_argument("--config", type=Path, help="读取主配置中的 assets 与 asset-source")
    assets.add_argument("--assets", type=Path, help="本地 tokenizer 目录，默认 assets/tokenizers")
    assets.add_argument(
        "--asset-source",
        type=str.lower,
        choices=["global", "cn"],
        help="global 使用 Hugging Face（默认），cn 使用魔搭；覆盖主配置",
    )
    assets.add_argument("--only", nargs="+", metavar="ASSET", help="仅准备指定资产；省略时准备全部")
    run = commands.add_parser(
        "import",
        help="统一读取日志、提取平台、筛选平台／模型、复算并保存",
        argument_default=argparse.SUPPRESS,
    )
    run.add_argument(
        "--config", type=Path, help="主配置 YAML：来源选择、日志格式、模型映射和平台筛选"
    )
    run.add_argument("--source", help="log 或 es；也支持已注册的物理源插件")
    run.add_argument(
        "--adapter",
        "--protocol",
        dest="protocol",
        help="dify / langchain / newapi / oneapi / litellm 或协议插件名",
    )
    location = run.add_mutually_exclusive_group()
    location.add_argument("--path", help="日志文件／目录路径，或 ES 服务地址")
    location.add_argument("--file", type=Path, help="兼容旧版 log/file 路径参数")
    location.add_argument("--es-url", help="兼容旧版 ES 地址参数")
    run.add_argument("--pattern", help="日志目录内的文件模式，如 dify-2026-10-*.jsonl")
    run.add_argument(
        "--index", "--es-index", dest="es_index", help="ES 索引名或模式；多个索引用逗号分隔"
    )
    run.add_argument(
        "--models",
        nargs="+",
        action="extend",
        metavar="MODEL",
        help="模型名称列表，精确匹配；省略时处理全部模型",
    )
    run.add_argument(
        "--platform-config", type=Path, help="平台提取及平台／模型筛选的 YAML 或 JSON 配置"
    )
    run.add_argument(
        "--platform-tag-source",
        help="field:<路径> / metadata:<路径> / endpoint / literal:<名称> / none（默认）",
    )
    run.add_argument(
        "--platforms",
        nargs="+",
        action="extend",
        metavar="PLATFORM",
        help="平台名称列表；与模型条件联合筛选",
    )
    run.add_argument(
        "--endpoint-source",
        help="endpoint 模式取值位置，默认 field:endpoint；也支持 metadata:<路径>",
    )
    run.add_argument(
        "--endpoint-map", type=Path, help="endpoint 到平台名称的精确映射，YAML 或 JSON 对象"
    )
    run.add_argument("--namespace", help="账号／应用来源命名空间，默认 local")
    run.add_argument("--es-query", type=Path, help="包含 ES query 子句的 JSON 文件")
    run.add_argument("--es-username", help="ES 用户名；与 --es-password-env 配套")
    run.add_argument("--es-password-env", help="保存 ES 密码的环境变量名")
    run.add_argument("--es-api-key-env", help="保存 ES API key 的环境变量名；与用户名密码二选一")
    run.add_argument("--page-size", type=int, help="ES 每页记录数，默认 500")
    run.add_argument("--payload-field", help="外层日志中的协议字段路径，例如 payload 或 message")
    run.add_argument(
        "--source-config", type=Path, help="物理源 YAML：文件路径或 ES 连接；也用于源插件"
    )
    run.add_argument(
        "--bindings",
        type=Path,
        help="单独指定模型绑定 YAML；通常直接编辑 config/modulo.yaml 的 models 即可",
    )
    run.add_argument("--assets", type=Path, help="本地 tokenizer 目录，默认 assets/tokenizers")
    run.add_argument("--output", type=Path, help="结果父目录，默认 runs")
    run.add_argument("--max-records", type=int, help="本次新增读取上限；默认 0 不限条数")
    run.add_argument("--memory-mib", type=int, help="磁盘工作区缓存预算 MiB，默认 64")
    run.add_argument("--max-record-mib", type=int, help="单条记录上限 MiB，默认 8")
    run.add_argument("--state", type=Path, help="保存读取进度与已有日志；复用目录继续导入和关联")
    run.add_argument(
        "--follow", action=argparse.BooleanOptionalAction, help="持续轮询新增日志，需要 --state"
    )
    run.add_argument("--poll-seconds", type=float, help="持续导入轮询间隔，默认 5 秒")
    run.add_argument(
        "--sample-mode", choices=["all", "ratio", "count", "window"], help="采样模式，默认全量"
    )
    run.add_argument(
        "--sample-ratio", help="ratio 模式的纳入概率，如 0.1；按调用／共享 usage 组抽样"
    )
    run.add_argument(
        "--sample-count", type=int, help="count 每个平台／模型组的额度；window 每个时间窗口的额度"
    )
    run.add_argument("--sample-seed", help="可复现随机种子，仅保存于买方本地")
    run.add_argument(
        "--sample-window-minutes", type=int, help="时间窗口长度；默认取最近已结束的 UTC 窗口"
    )
    run.add_argument("--sample-before", help="历史窗口结束时间，带时区的 ISO 格式；结束边界不包含")
    run.add_argument(
        "--download-images",
        action=argparse.BooleanOptionalAction,
        help="下载缺尺寸的 URL 图片，默认关闭；缓存目录与限额见主配置 image-download",
    )
    report = commands.add_parser("report", help="校验并读取封存摘要或调用明细")
    report.add_argument("run", type=Path)
    report.add_argument("--details", action="store_true")
    report.add_argument("--call-id")
    export = commands.add_parser(
        "export-prometheus",
        help="将已有结果的聚合指标直接写入 Prometheus",
        argument_default=argparse.SUPPRESS,
    )
    export.add_argument("run", type=Path, help="已有结果包目录")
    export.add_argument("--config", type=Path, help="Prometheus 导出 YAML 配置")
    export.add_argument(
        "--url", help="完整 Remote Write 接口地址，如 http://localhost:9090/api/v1/write"
    )
    export.add_argument("--scope", help="稳定的统计范围名称；同一范围展示最近导出的结果")
    export.add_argument("--username", help="HTTP Basic 用户名")
    export.add_argument("--password-env", help="保存 Basic 密码的环境变量名")
    export.add_argument("--bearer-token-env", help="保存 Bearer token 的环境变量名；与 Basic 互斥")
    export.add_argument("--ca-file", type=Path, help="自定义 TLS CA 证书文件")
    export.add_argument("--timeout-seconds", type=float, help="HTTP 超时秒数，默认 30")
    export.add_argument("--max-series", type=int, help="单次指标序列上限，默认 10000")
    export.add_argument(
        "--visibility",
        choices=["auto", "full", "sample-only"],
        help="auto：时间窗口只导出样本，其余完整导出；full 仅用于买方内部",
    )
    verify = commands.add_parser("verify", help="校验结果包文件摘要")
    verify.add_argument("run", type=Path)
    reindex = commands.add_parser("reindex", help="由权威 JSONL 重建 DuckDB")
    reindex.add_argument("run", type=Path)
    return root
