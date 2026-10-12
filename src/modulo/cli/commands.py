"""Execute CLI commands using the log importer and persistence services."""

import shutil
import sys
import time

from modulo.application.factory import build_registry, make_calculator
from modulo.application.importer import run_batch
from modulo.application.tokenizer_assets import load_tokenizer_assets, prepare_assets
from modulo.shared.serialization import dumps
from modulo.storage.query_index import iter_details, rebuild_index
from modulo.storage.result_bundle import verify_bundle

from .configuration import asset_options, import_options, import_source, platform_options
from .parser import parser


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        registry = build_registry(arguments.plugin)
        if arguments.command == "plugins":
            print(dumps(registry.describe()))
        elif arguments.command == "prepare-assets":
            options = asset_options(arguments)
            print(
                dumps(
                    {
                        "verified_assets": prepare_assets(
                            options.assets,
                            getattr(arguments, "only", None),
                            source=options.asset_source,
                        )
                    }
                )
            )
        elif arguments.command == "import":
            arguments = import_options(arguments)
            platform_config = platform_options(arguments)
            protocol = registry.get("protocols", arguments.protocol)
            calculator = make_calculator(
                registry, arguments.assets, arguments.bindings, load_tokenizer_assets()
            )
            previous = None
            while True:
                # 每轮重新固定文件清单／ES 快照，状态目录保留读取进度。
                path, reused = run_batch(
                    import_source(arguments, registry),
                    protocol,
                    calculator,
                    namespace=arguments.namespace,
                    output_root=arguments.output,
                    payload_field=arguments.payload_field,
                    max_records=arguments.max_records,
                    plugin_modules=arguments.plugin,
                    platform_config=platform_config,
                    image_download=arguments.image_download,
                    state=arguments.state,
                    memory_mib=arguments.memory_mib,
                    max_record_bytes=arguments.max_record_mib * 1024 * 1024,
                )
                manifest = verify_bundle(path)
                if path != previous:
                    sys.stdout.write(
                        '{"run":'
                        + dumps(str(path.resolve()))
                        + ',"reused":'
                        + dumps(reused)
                        + ',"summary":'
                    )
                    with (path / "summary.json").open(encoding="utf-8") as stream:
                        shutil.copyfileobj(stream, sys.stdout, length=64 * 1024)
                    print("}", flush=True)
                    previous = path
                if not arguments.follow:
                    return 0 if manifest["status"] == "COMPLETE" else 2
                time.sleep(arguments.poll_seconds)
        elif arguments.command == "report":
            verify_bundle(arguments.run)
            if arguments.details or arguments.call_id:
                sys.stdout.write("[")
                for index, row in enumerate(iter_details(arguments.run, arguments.call_id)):
                    if index:
                        sys.stdout.write(",")
                    sys.stdout.write(dumps(row))
                print("]")
            else:
                with (arguments.run / "summary.json").open(encoding="utf-8") as stream:
                    shutil.copyfileobj(stream, sys.stdout, length=64 * 1024)
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
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as exc:
        # Data-level errors are captured in bundles; configuration errors stop the command.
        print(f"modulo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
