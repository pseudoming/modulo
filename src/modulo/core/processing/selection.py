"""Select complete platform/model variants only after identities have been resolved."""

from collections.abc import Iterator

from modulo.shared.serialization import dumps

from ..domain.platforms import PlatformConfig
from .ingestion import CollectedBatch


def select_variant(platform: str | None, model: str | None, config: PlatformConfig) -> str:
    """Three-valued AND; both filters must match the same observed variant."""
    unknown = False
    for value, requested in ((platform, config.platforms), (model, config.models)):
        if requested is None:
            continue
        if value is None:
            unknown = True
        elif value not in requested:
            return "excluded"
    return "unresolved" if unknown else "selected"


def selection_summary(batch: CollectedBatch, platform_config: PlatformConfig) -> dict:
    calls, observed_dimensions = batch.calls, batch.dimensions
    seen = {pair for pairs in observed_dimensions.values() for pair in pairs}
    seen_models = {model for _, model in seen if model is not None}
    seen_platforms = {platform for platform, _ in seen if platform is not None}
    requested = set(platform_config.models) if platform_config.models is not None else None
    requested_platforms = (
        set(platform_config.platforms) if platform_config.platforms is not None else None
    )
    selection = {
        "schema_version": "selection.v2",
        "models": platform_config.models,
        "platforms": platform_config.platforms,
        "parsed_calls": len(calls),
        "selected_calls": 0,
        "excluded_calls": 0,
        "unresolved_calls": 0,
        "matched_models": sorted(seen_models if requested is None else seen_models & requested),
        "unmatched_models": sorted(requested - seen_models) if requested is not None else [],
        "matched_platforms": sorted(
            seen_platforms if requested_platforms is None else seen_platforms & requested_platforms
        ),
        "unmatched_platforms": sorted(requested_platforms - seen_platforms)
        if requested_platforms is not None
        else [],
    }
    return selection


def select_calls(
    batch: CollectedBatch, platform_config: PlatformConfig, selection: dict
) -> Iterator[dict]:
    calls, observed_dimensions, diagnostics = batch.calls, batch.dimensions, batch.diagnostics
    for key in sorted(calls):
        row = calls[key]
        pairs = observed_dimensions[key]
        row["observed_dimensions"] = [
            {"platform": platform, "model": model} for platform, model in sorted(pairs, key=dumps)
        ]
        decisions = {select_variant(platform, model, platform_config) for platform, model in pairs}
        if "selected" not in decisions:
            unknown = "unresolved" in decisions
            selection["unresolved_calls" if unknown else "excluded_calls"] += 1
            prefix = "model" if platform_config.platforms is None else "platform_model"
            diagnostics.append(
                {
                    "level": "warning" if unknown else "info",
                    "reason": f"{prefix}_selection_unresolved"
                    if unknown
                    else f"{prefix}_not_selected",
                    "call_key": key,
                    "observed_dimensions": row["observed_dimensions"],
                    "source_refs": row["source_refs"],
                }
            )
            continue
        yield row
