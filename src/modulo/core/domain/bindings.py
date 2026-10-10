"""Explicit platform/model aliases bound to fixed reference assets."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from .data_models import Call, Contract, ProtocolContext
from .platforms import PlatformConfig, PlatformTag, resolve_platform


class Binding(Contract):
    modelnames: list[str] = Field(min_length=1)
    asset: str
    platform: str | None = None
    platform_tag_source: str | None = Field(default=None, alias="platform-tag-source")
    endpoint_source: str = Field(default="field:endpoint", alias="endpoint-source")
    endpoint_map: dict[str, str] = Field(default_factory=dict, alias="endpoint-map")
    mode: Literal["text_chat"] = "text_chat"
    add_generation_prompt: Literal[True] = True

    @model_validator(mode="after")
    def validate_platform(self):
        if self.platform is not None and not self.platform.strip():
            raise ValueError("binding platform must be nonempty")
        if self.platform_tag_source is not None:
            if self.platform is None:
                raise ValueError("platform-tag-source requires a platform condition")
            if self.platform_tag_source == "none":
                raise ValueError("platform condition cannot use platform-tag-source=none")
            self.tag_config(PlatformConfig())
        elif self.endpoint_map or self.endpoint_source != "field:endpoint":
            raise ValueError("binding endpoint options require platform-tag-source=endpoint")
        return self

    def tag_config(self, default: PlatformConfig) -> PlatformConfig:
        if self.platform_tag_source is None:
            return default
        return PlatformConfig.model_validate(
            {
                "platform-tag-source": self.platform_tag_source,
                "endpoint-source": self.endpoint_source,
                "endpoint-map": self.endpoint_map,
            }
        )


class BindingConflict(ValueError):
    pass


@dataclass
class BindingResolution:
    tag: PlatformTag
    tags: list[dict]
    matches: list[int]
    error: str | None = None


class Bindings(Contract):
    models: list[Binding] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_names(self):
        seen = set()
        for binding in self.models:
            scope = (
                binding.platform,
                binding.platform_tag_source,
                binding.endpoint_source,
                tuple(sorted(binding.endpoint_map.items())),
            )
            for name in binding.modelnames:
                if not name.strip():
                    raise ValueError("binding modelnames must be nonempty")
                key = (*scope, name)
                if key in seen:
                    raise ValueError(f"duplicate modelname {name!r} in the same binding condition")
                seen.add(key)
        return self

    def select(self, matches: list[int]) -> Binding | None:
        candidates = [self.models[index] for index in matches]
        specific = [item for item in candidates if item.platform is not None]
        candidates = specific or candidates
        targets = {
            (item.platform, item.asset, item.mode, item.add_generation_prompt)
            for item in candidates
        }
        if len(targets) > 1:
            raise BindingConflict("binding_rule_conflict")
        return candidates[0] if candidates else None

    def match_call(self, call: Call) -> list[int]:
        """Match an already normalized call when no raw extraction context is available."""
        return [
            index
            for index, item in enumerate(self.models)
            if call.model in item.modelnames
            and (
                item.platform is None
                or (
                    item.platform == call.platform
                    and item.platform_tag_source in {None, call.platform_source}
                )
            )
        ]

    def resolve(
        self, call: Call, config: PlatformConfig, record: dict, context: ProtocolContext
    ) -> BindingResolution:
        matches, tags, matched_tags = [], [], []
        evidence = []
        for index, item in enumerate(self.models):
            if call.model not in item.modelnames:
                continue
            if item.platform is None:
                matches.append(index)
                continue
            tag = resolve_platform(item.tag_config(config), record, context)
            matched = tag.value == item.platform
            tags.append(tag)
            evidence.append({**tag.model_dump(), "binding_index": index, "matched": matched})
            if matched:
                matches.append(index)
                matched_tags.append(tag)
        if not tags:
            tags = [resolve_platform(config, record, context)]
            evidence = [tags[0].model_dump()]
        # Matching rules establish the platform; otherwise retain the observed tag
        # for global fallback and filtering. Never choose between contradictory values.
        candidates = matched_tags or tags
        known = [tag for tag in candidates if tag.value is not None]
        error = None
        if len({tag.value for tag in known}) > 1:
            error = "binding_rule_conflict" if matched_tags else "platform_tag_conflict"
            tag = PlatformTag(source="conflict", provenance="source_reported", reason=error)
        else:
            tag = min(
                known or candidates,
                key=lambda item: (item.source, item.field_path or "", item.mapping_hash or ""),
            )
        try:
            self.select(matches)
        except BindingConflict as exc:
            error = str(exc)
        return BindingResolution(tag, evidence, matches, error)
