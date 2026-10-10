"""Execute CLI commands using the log importer and persistence services."""

import json
import sys

from modulo.application.factory import build_registry, make_calculator
from modulo.application.importer import run_batch
from modulo.application.tokenizer_assets import asset_lock, prepare_assets
from modulo.shared.serialization import dumps, loads
from modulo.storage.query_index import details, rebuild_index
from modulo.storage.result_bundle import verify_bundle

from .configuration import import_options, import_source, platform_options
from .parser import parser


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        registry = build_registry(arguments.plugin)
        if arguments.command == "plugins":
            print(dumps(registry.describe()))
        elif arguments.command == "prepare-assets":
            print(dumps({"verified_assets": prepare_assets(arguments.assets, arguments.only)}))
        elif arguments.command == "import":
            arguments = import_options(arguments)
            platform_config = platform_options(arguments)
            protocol = registry.get("protocols", arguments.protocol)
            source = import_source(arguments, registry)
            calculator = make_calculator(
                registry, arguments.assets, arguments.bindings, asset_lock()
            )
            # 配置与适配器就绪；外层准备存储后进入 core.main.run。
            path, reused = run_batch(
                source,
                protocol,
                calculator,
                namespace=arguments.namespace,
                output_root=arguments.output,
                payload_field=arguments.payload_field,
                max_records=arguments.max_records,
                plugin_modules=arguments.plugin,
                platform_config=platform_config,
            )
            summary = loads((path / "summary.json").read_bytes())
            print(dumps({"run": str(path.resolve()), "reused": reused, "summary": summary}))
            return 0 if summary["run_status"] == "COMPLETE" else 2
        elif arguments.command == "report":
            verify_bundle(arguments.run)
            value = (
                details(arguments.run, arguments.call_id)
                if arguments.details or arguments.call_id
                else loads((arguments.run / "summary.json").read_bytes())
            )
            print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        elif arguments.command == "verify":
            manifest = verify_bundle(arguments.run)
            print(
                dumps(
                    {
                        "run_id": manifest["run_id"],
                        "integrity": "verified",
                        "status": manifest["status"],
                    }
                )
            )
        elif arguments.command == "reindex":
            print(dumps({"index": str(rebuild_index(arguments.run).resolve())}))
        return 0
    except (ValueError, OSError) as exc:
        # Data-level errors are captured in bundles; configuration errors stop the command.
        print(f"modulo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
