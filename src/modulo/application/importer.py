"""Run one synchronous log import with a frozen plan and result storage."""

from functools import partial
from pathlib import Path

from modulo.core.domain.interfaces import ProtocolAdapter, SourceAdapter
from modulo.core.domain.platforms import PlatformConfig
from modulo.core.main import run
from modulo.core.processing.recalculation import Recalculator
from modulo.shared.serialization import digest
from modulo.storage.query_index import rebuild_index
from modulo.storage.result_bundle import BundleWriter, verify_bundle
from modulo.storage.working_set import WorkingSet

from .continuation import JournalSource
from .image_cache import ImageCache, ImageDownloadOptions
from .runtime_info import collect_runtime_info


def run_batch(
    source: SourceAdapter,
    protocol: ProtocolAdapter,
    calculator: Recalculator,
    *,
    namespace: str,
    output_root: Path,
    payload_field: str | None = None,
    max_records: int = 0,
    plugin_modules: list[str] | None = None,
    models: list[str] | None = None,
    platform_config: PlatformConfig | None = None,
    image_download: ImageDownloadOptions | None = None,
    memory_mib: int = 64,
    max_record_bytes: int = 8 * 1024 * 1024,
    state: Path | None = None,
) -> tuple[Path, bool]:
    if not namespace or max_records < 0:
        raise ValueError("namespace and a nonnegative max_records are required")
    config = (platform_config or PlatformConfig()).describe()
    if models is not None:
        config["models"] = models
    platform_config = PlatformConfig.model_validate(config)
    plugin_modules = plugin_modules or []
    execution = collect_runtime_info(plugin_modules)
    image_download = image_download or ImageDownloadOptions()
    plan = {
        "source": source.describe(),
        "namespace": namespace,
        "protocol": protocol.name,
        "protocol_version": protocol.version,
        "payload_field": payload_field,
        "max_records": max_records,
        "max_record_bytes": max_record_bytes,
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
        "image_download": image_download.model_dump(mode="json", by_alias=True),
    }

    with (
        WorkingSet(state, memory_mib=memory_mib) as working,
        ImageCache(image_download) as images,
        BundleWriter(output_root) as bundle,
    ):
        if source.name == "es":
            source.seen = working.mapping(
                ("saved:" if state else "")
                + "es_seen:"
                + digest([source.url, source.index, source.query])
            )
            source.max_page_bytes = memory_mib * 1024 * 1024
        source.max_record_bytes = max_record_bytes
        if state is not None:
            working.bind(
                {
                    **{
                        key: value
                        for key, value in plan.items()
                        if key not in {"source", "max_records"}
                    },
                    "source_kind": source.name,
                }
            )
            source = JournalSource(source, working, max_records)
            source.capture()
            plan["source"] = source.describe()
            previous = working.mapping("saved:settings").get("last_result")
            if previous and previous["plan"] == digest(plan):
                path = Path(previous["path"])
                if path.parent == output_root.resolve() and path.exists():
                    if verify_bundle(path)["status"] == "COMPLETE":
                        return path, True

        def prepare_call(call, matches):
            binding = calculator.bindings.select(matches)
            if binding and calculator.lock["assets"][binding.name].get("image_processing"):
                return images.prepare(call)
            return call

        save = partial(
            bundle.publish,
            plan,
            execution=execution,
            bindings=calculator.bindings,
            asset_lock=calculator.lock,
            rebuild_index=partial(rebuild_index, memory_mib=memory_mib),
        )
        result = run(
            source,
            protocol,
            calculator,
            archive=bundle.record,
            save=save,
            namespace=namespace,
            payload_field=payload_field,
            max_records=0 if state else max_records,
            platform_config=platform_config,
            prepare_call=prepare_call if image_download.enabled else None,
            working=working,
            max_record_bytes=max_record_bytes,
            measurement_cache=working.mapping("saved:measurements") if state else None,
        )
        if state is not None:
            working.mapping("saved:settings")["last_result"] = {
                "plan": digest(plan),
                "path": str(result[0].resolve()),
            }
        return result
