"""Platform configuration semantics and provenance-preserving tag extraction."""

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from modulo.shared.serialization import digest

from .data_models import Contract, ProtocolContext


def field_path(value: str) -> str:
    if not value or any(not part.strip() for part in value.split(".")):
        raise ValueError("field paths must contain nonempty dot-separated keys")
    return value


def field_reference(value: str) -> str:
    kind, separator, path = value.partition(":")
    if separator and kind in {"field", "metadata"}:
        field_path(path)
        return value
    raise ValueError("expected field:<path> or metadata:<path>")


def name_list(values: list[str] | None) -> list[str] | None:
    if values is not None:
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError("expected a nonempty list of names, or None for all")
        return sorted(set(values))
    return None


class PlatformConfig(Contract):
    schema_version: Literal["platform-config.v1"] = "platform-config.v1"
    source: str = Field(default="none", alias="platform-tag-source")
    endpoint_source: str = Field(default="field:endpoint", alias="endpoint-source")
    endpoint_map: dict[str, str] = Field(default_factory=dict, alias="endpoint-map")
    platforms: list[str] | None = None
    models: list[str] | None = None

    @field_validator("source")
    @classmethod
    def source_form(cls, value):
        if value in {"none", "endpoint"}:
            return value
        if value.startswith("literal:") and value.removeprefix("literal:").strip():
            return value
        return field_reference(value)

    @field_validator("endpoint_source")
    @classmethod
    def endpoint_form(cls, value):
        return field_reference(value)

    @field_validator("platforms", "models", mode="before")
    @classmethod
    def names(cls, value):
        return name_list(value)

    @field_validator("endpoint_map")
    @classmethod
    def endpoint_names(cls, value):
        if any(
            not endpoint.strip() or not platform.strip() for endpoint, platform in value.items()
        ):
            raise ValueError("endpoint mapping keys and platform names must be nonempty")
        return value

    @model_validator(mode="after")
    def mapping_scope(self):
        if self.source == "endpoint" and not self.endpoint_map:
            raise ValueError("endpoint source requires an explicit endpoint map")
        if self.source != "endpoint" and (
            self.endpoint_map or self.endpoint_source != "field:endpoint"
        ):
            raise ValueError("endpoint mapping options require platform-tag-source=endpoint")
        return self

    def describe(self) -> dict:
        return self.model_dump(by_alias=True)


class PlatformTag(Contract):
    schema_version: Literal["platform-tag.v1"] = "platform-tag.v1"
    value: str | None = None
    source: str
    provenance: Literal["not_requested", "source_reported", "configuration", "mapped"]
    field_path: str | None = None
    raw_value: str | None = None
    mapping_hash: str | None = None
    reason: str | None = None
    source_ref: dict[str, str] | None = None


def lookup(value: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def read_tag_value(reference: str, record: dict, context: ProtocolContext) -> tuple[Any, str]:
    kind, path = reference.split(":", 1)
    if kind == "field":
        return lookup(record, path), f"record:{path}"
    location = context.metadata_path or "metadata_unavailable"
    return lookup(context.metadata, path), f"{context.metadata_root}:{location}.{path}"


def resolve_platform(config: PlatformConfig, record: dict, context: ProtocolContext) -> PlatformTag:
    source = config.source
    if source == "none":
        return PlatformTag(source=source, provenance="not_requested", reason="not_requested")
    if source.startswith("literal:"):
        value = source.removeprefix("literal:")
        return PlatformTag(source=source, value=value, raw_value=value, provenance="configuration")
    reference = config.endpoint_source if source == "endpoint" else source
    value, path = read_tag_value(reference, record, context)
    base = {
        "source": source,
        "field_path": path,
        "provenance": "mapped" if source == "endpoint" else "source_reported",
        "mapping_hash": digest(config.endpoint_map) if source == "endpoint" else None,
    }
    if value is None:
        return PlatformTag(**base, reason="platform_field_missing")
    if not isinstance(value, str) or not value.strip():
        return PlatformTag(**base, reason="platform_field_invalid")
    if source == "endpoint":
        platform = config.endpoint_map.get(value)
        return PlatformTag(
            **base,
            raw_value=value,
            value=platform,
            reason="endpoint_unmapped" if platform is None else None,
        )
    return PlatformTag(**base, raw_value=value, value=value)
