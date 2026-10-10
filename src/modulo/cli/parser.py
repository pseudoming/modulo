"""Command syntax and help text, without audit decisions."""

import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="modulo", description="日志批量导入与本地参照 token 复算")
    root.add_argument("--plugin", action="append", default=[], help="显式加载受信 Python 模块")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("plugins", help="列出三类已注册适配器")
    assets = commands.add_parser("prepare-assets", help="显式联网准备固定 tokenizer 资产")
    assets.add_argument("--assets", type=Path, default=Path("assets/tokenizers"))
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
    run.add_argument("--max-records", type=int, help="整批读取上限，默认 10000")
    report = commands.add_parser("report", help="校验并读取封存摘要或调用明细")
    report.add_argument("run", type=Path)
    report.add_argument("--details", action="store_true")
    report.add_argument("--call-id")
    verify = commands.add_parser("verify", help="校验结果包文件摘要")
    verify.add_argument("run", type=Path)
    reindex = commands.add_parser("reindex", help="由权威 JSONL 重建 DuckDB")
    reindex.add_argument("run", type=Path)
    return root
