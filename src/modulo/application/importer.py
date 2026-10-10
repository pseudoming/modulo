"""Run one synchronous log import with a frozen plan and result storage."""

from functools import partial
from pathlib import Path

from modulo.core.domain.interfaces import ProtocolAdapter, SourceAdapter
from modulo.core.domain.platforms import PlatformConfig
from modulo.core.main import run
from modulo.core.processing.recalculation import Recalculator
from modulo.shared.serialization import digest
from modulo.storage.query_index import rebuild_index
from modulo.storage.result_bundle import BundleWriter

from .runtime_info import collect_runtime_info


def run_batch(
    source: SourceAdapter,
    protocol: ProtocolAdapter,
    calculator: Recalculator,
    *,
    namespace: str,
    output_root: Path,
    payload_field: str | None = None,
    max_records: int = 10_000,
    plugin_modules: list[str] | None = None,
    models: list[str] | None = None,
    platform_config: PlatformConfig | None = None,
) -> tuple[Path, bool]:
    if not namespace or max_records < 1:
        raise ValueError("namespace and a positive max_records are required")
    config = (platform_config or PlatformConfig()).describe()
    if models is not None:
        config["models"] = models
    platform_config = PlatformConfig.model_validate(config)
    plugin_modules = plugin_modules or []
    execution = collect_runtime_info(plugin_modules)
    plan = {
        "source": source.describe(),
        "namespace": namespace,
        "protocol": protocol.name,
        "protocol_version": protocol.version,
        "payload_field": payload_field,
        "max_records": max_records,
        "selection": {
            "schema_version": "selection.v2",
            "models": platform_config.models,
            "platforms": platform_config.platforms,
        },
        "platform_config": platform_config.describe(),
        "plugins": plugin_modules,
        "bindings": calculator.bindings.model_dump(by_alias=True),
        "assets": calculator.lock,
        "execution_hash": digest(execution),
    }

    with BundleWriter(output_root) as bundle:
        save = partial(
            bundle.publish,
            plan,
            execution=execution,
            bindings=calculator.bindings,
            asset_lock=calculator.lock,
            rebuild_index=rebuild_index,
        )
        return run(
            source,
            protocol,
            calculator,
            archive=bundle.record,
            save=save,
            namespace=namespace,
            payload_field=payload_field,
            max_records=max_records,
            platform_config=platform_config,
        )
