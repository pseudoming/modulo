"""Read configuration files and resolve command-line source and platform options."""

import argparse
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from modulo.application.image_cache import ImageDownloadOptions
from modulo.application.registry import Registry
from modulo.core.domain.bindings import Bindings
from modulo.core.domain.interfaces import SourceAdapter
from modulo.core.domain.platforms import PlatformConfig
from modulo.shared.serialization import loads


class AssetOptions(BaseModel):
    """Local asset directory and explicit download region, shared by both commands."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, alias_generator=lambda name: name.replace("_", "-")
    )

    assets: Path = Path("assets/tokenizers")
    asset_source: Literal["global", "cn"] = "global"

    @field_validator("asset_source", mode="before")
    @classmethod
    def normalize_source(cls, value):
        return value.lower() if isinstance(value, str) else value


class ImportOptions(AssetOptions):
    """Resolved options passed from configuration and CLI into one import."""

    source: str
    protocol: str = Field(alias="adapter")
    path: str | None = None
    pattern: str | None = Field(default=None, min_length=1)
    file: Path | None = None
    es_url: str | None = None
    es_index: str | None = Field(default=None, alias="index")
    models: list[str] | None = None
    platform_config: Path | None = None
    platform_tag_source: str | None = None
    platforms: list[str] | None = None
    endpoint_source: str | None = None
    endpoint_map: Path | None = None
    namespace: str = Field(default="local", min_length=1)
    es_query: Path | None = None
    es_username: str | None = Field(default=None, min_length=1)
    es_password_env: str | None = Field(default=None, min_length=1)
    es_api_key_env: str | None = Field(default=None, min_length=1)
    page_size: int = Field(default=500, ge=1, le=10_000)
    payload_field: str | None = None
    source_config: Path | None = None
    bindings: Bindings
    output: Path = Path("runs")
    max_records: int = Field(default=0, ge=0)
    memory_mib: int = Field(default=64, ge=8)
    max_record_mib: int = Field(default=8, ge=1)
    state: Path | None = None
    follow: bool = False
    poll_seconds: float = Field(default=5, gt=0, allow_inf_nan=False)
    image_download: ImageDownloadOptions = Field(default_factory=ImageDownloadOptions)

    @field_validator("page_size", "max_records", "memory_mib", "max_record_mib", mode="before")
    @classmethod
    def integer_option(cls, value):
        if isinstance(value, str):
            return int(value)
        if type(value) is not int:
            raise ValueError("expected an integer")
        return value


class ConfigLoader(yaml.BaseLoader):
    """Keep names as strings; template parameters alone allow JSON scalar types."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported")
        return super().compose_node(parent, index)

    def construct_object(self, node, deep=False):
        if node.tag not in {f"tag:yaml.org,2002:{kind}" for kind in ("str", "seq", "map")}:
            raise ValueError("YAML explicit types and custom tags are not supported")
        return super().construct_object(node, deep)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, str) or key == "<<":
                raise ValueError("configuration keys must be strings; YAML merges are unsupported")
            if key in result:
                raise ValueError("duplicate configuration key")
            value = self.construct_object(value_node, deep=True)
            result[key] = (
                self.template_parameters(value_node, value) if key == "template-kwargs" else value
            )
        return result

    def template_parameters(self, node, value):
        if isinstance(node, yaml.ScalarNode) and node.style is None:
            try:
                return loads(node.value)
            except ValueError:
                return value
        if isinstance(node, yaml.SequenceNode):
            return [
                self.template_parameters(child, item)
                for child, item in zip(node.value, value, strict=True)
            ]
        if isinstance(node, yaml.MappingNode):
            return {
                key: self.template_parameters(child, value[key])
                for (_, child), key in zip(node.value, value, strict=True)
            }
        return value


def read_config(path: Path) -> dict:
    with path.open("rb") as stream:
        content = stream.read(128 * 1024 + 1)
    if len(content) > 128 * 1024:
        raise ValueError("configuration exceeds 128 KiB")
    try:
        value = (
            loads(content)
            if path.suffix.lower() == ".json"
            else yaml.load(content.decode("utf-8"), Loader=ConfigLoader)
        )
    except (yaml.YAMLError, RecursionError) as exc:
        # Parser errors can contain the input; expose no configuration contents.
        raise ValueError("invalid YAML configuration") from exc
    if not isinstance(value, dict):
        raise ValueError("configuration must contain an object")
    return value


SOURCE_FIELDS = {
    "file": {"path": "path", "pattern": "pattern"},
    "es": {
        "path": "path",
        "index": "es_index",
        "username": "es_username",
        "password-env": "es_password_env",
        "api-key-env": "es_api_key_env",
        "page-size": "page_size",
        "query-file": "es_query",
    },
}
COMMON_FIELDS = {
    "source",
    "source_config",
    "protocol",
    "platform_config",
    "platform_tag_source",
    "platforms",
    "endpoint_source",
    "endpoint_map",
    "namespace",
    "payload_field",
    "assets",
    "asset_source",
    "output",
    "max_records",
    "image_download",
    "memory_mib",
    "max_record_mib",
    "state",
    "follow",
    "poll_seconds",
}


def named_options(config: dict, aliases: dict[str, str], label: str) -> dict:
    if unknown := config.keys() - aliases.keys():
        raise ValueError(f"unknown {label} configuration keys: {', '.join(sorted(unknown))}")
    return {aliases[key]: value for key, value in config.items()}


def relative_paths(options: dict, config_path: Path, names: set[str]) -> None:
    for name in names & options.keys():
        if isinstance(options[name], str):
            options[name] = str((config_path.resolve().parent / options[name]).resolve())


def asset_options(arguments: argparse.Namespace) -> AssetOptions:
    config_path = getattr(arguments, "config", None)
    config = read_config(config_path) if config_path else {}
    config.pop("models", None)
    aliases = {ImportOptions.model_fields[name].alias: name for name in COMMON_FIELDS}
    options = named_options(config, aliases, "main")
    options = {key: value for key, value in options.items() if key in AssetOptions.model_fields}
    if config_path:
        relative_paths(options, config_path, {"assets"})
    options.update(
        {key: value for key, value in vars(arguments).items() if key in AssetOptions.model_fields}
    )
    return AssetOptions.model_validate(options)


def import_options(arguments: argparse.Namespace) -> argparse.Namespace:
    config_path = getattr(arguments, "config", None)
    config = read_config(config_path) if config_path else {}
    has_models = "models" in config
    model_bindings = config.pop("models", None)
    aliases = {ImportOptions.model_fields[name].alias: name for name in COMMON_FIELDS}
    options = named_options(config, aliases, "main")
    if config_path:
        relative_paths(
            options,
            config_path,
            {
                "platform_config",
                "endpoint_map",
                "source_config",
                "assets",
                "output",
                "state",
            },
        )
        if isinstance(options.get("image_download"), dict):
            relative_paths(options["image_download"], config_path, {"cache-dir", "cache_dir"})
    explicit = {
        key: value
        for key, value in vars(arguments).items()
        if key in ImportOptions.model_fields and key != "bindings"
    }
    # A different source must not inherit the old source's connection settings.
    if "source" in explicit and explicit["source"] != options.get("source"):
        options.pop("source_config", None)
    options.update(explicit)
    if hasattr(arguments, "download_images"):
        if not isinstance(options.get("image_download", {}), dict):
            raise ValueError("image-download configuration must contain an object")
        options["image_download"] = {
            **options.get("image_download", {}),
            "enabled": arguments.download_images,
        }
    if getattr(arguments, "bindings", None):
        bindings = read_config(arguments.bindings)
    elif has_models:
        bindings = {"models": model_bindings}
    elif config_path:
        raise ValueError(
            "main configuration requires a models array or an explicit --bindings file"
        )
    else:
        # Direct CLI use shares the shipped model mappings, not its example source.
        bindings = {"models": read_config(Path("config/modulo.yaml")).get("models")}
    options["bindings"] = Bindings.model_validate(bindings)
    validated = ImportOptions.model_validate(options)
    source_path = validated.source_config
    kind = "file" if validated.source == "log" else validated.source
    if source_path and kind in SOURCE_FIELDS:
        source = named_options(read_config(source_path), SOURCE_FIELDS[kind], kind)
        relative_paths(source, source_path, {"path"} if kind == "file" else {"es_query"})
        if {"path", "file", "es_url"} & explicit.keys():
            source.pop("path", None)
        validated = ImportOptions.model_validate(source | options)
    if sum(getattr(validated, name) is not None for name in ("path", "file", "es_url")) > 1:
        raise ValueError("use only one of --path, --file and --es-url")
    if validated.follow and validated.state is None:
        raise ValueError("--follow requires --state to save reading progress")
    return argparse.Namespace(
        **(
            vars(arguments)
            | {name: getattr(validated, name) for name in ImportOptions.model_fields}
        )
    )


def environment_secret(name: str | None) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required credential environment variable is missing or empty: {name}")
    return value


def import_source(arguments: argparse.Namespace, registry: Registry) -> SourceAdapter:
    """Resolve CLI aliases and reject source options that would otherwise be ignored."""
    kind = "file" if arguments.source == "log" else arguments.source
    factory = registry.get("sources", kind)
    if kind == "file":
        if any(
            (
                arguments.es_url,
                arguments.es_index,
                arguments.es_query,
                arguments.es_username,
                arguments.es_password_env,
                arguments.es_api_key_env,
            )
        ):
            raise ValueError("log source does not accept ES options")
        path = arguments.path or arguments.file
        if not path:
            raise ValueError("--path is required for log source")
        return (
            factory(Path(path), pattern=arguments.pattern, allow_empty=arguments.follow)
            if arguments.pattern is not None
            else factory(Path(path))
        )
    if kind == "es":
        if arguments.file or arguments.pattern is not None:
            raise ValueError("ES source does not accept --file or --pattern")
        query = json.loads(arguments.es_query.read_bytes()) if arguments.es_query else None
        if query is not None and not isinstance(query, dict):
            raise ValueError("--es-query must contain a query object")
        url = arguments.path or arguments.es_url
        if not url or not arguments.es_index:
            raise ValueError("--path and --index are required for ES source")
        return factory(
            url,
            arguments.es_index,
            query=query,
            page_size=arguments.page_size,
            username=arguments.es_username,
            password=environment_secret(arguments.es_password_env),
            api_key=environment_secret(arguments.es_api_key_env),
        )
    if any(
        (
            arguments.path,
            arguments.file,
            arguments.es_url,
            arguments.es_index,
            arguments.es_query,
            arguments.pattern,
            arguments.es_username,
            arguments.es_password_env,
            arguments.es_api_key_env,
        )
    ):
        raise ValueError("custom sources use --source-config for location and source options")
    config = read_config(arguments.source_config) if arguments.source_config else {}
    return factory(**config)


def platform_options(arguments: argparse.Namespace) -> PlatformConfig:
    config = read_config(arguments.platform_config) if arguments.platform_config else {}
    PlatformConfig.model_validate(config)
    for field in ("platform_tag_source", "endpoint_source", "platforms", "models"):
        value = getattr(arguments, field)
        if value is not None:
            config[field.replace("_", "-")] = value
    if arguments.endpoint_map:
        config["endpoint-map"] = read_config(arguments.endpoint_map)
    return PlatformConfig.model_validate(config)
