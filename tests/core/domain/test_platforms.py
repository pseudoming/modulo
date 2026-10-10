import pytest

from modulo.core.domain.data_models import ProtocolContext
from modulo.core.domain.platforms import PlatformConfig, resolve_platform


def configuration(**values):
    return PlatformConfig.model_validate(
        {key.replace("_", "-"): value for key, value in values.items()}
    )


@pytest.mark.parametrize(
    "source, expected, provenance",
    [
        ("field:labels.platform", "volcengine", "source_reported"),
        ("metadata:platform", "huaweicloud", "source_reported"),
        ("literal:deepseek-official", "deepseek-official", "configuration"),
        ("none", None, "not_requested"),
    ],
)
def test_selected_source_is_used_without_implicit_fallback(source, expected, provenance):
    tag = resolve_platform(
        configuration(platform_tag_source=source),
        {"labels": {"platform": "volcengine"}},
        ProtocolContext(metadata={"platform": "huaweicloud"}, metadata_path="extra.metadata"),
    )
    assert tag.value == expected and tag.provenance == provenance


@pytest.mark.parametrize(
    "value, reason",
    [
        (None, "platform_field_missing"),
        ("", "platform_field_invalid"),
        (12, "platform_field_invalid"),
        (["volcengine"], "platform_field_invalid"),
    ],
)
def test_missing_or_invalid_platform_is_not_inferred_from_provider(value, reason):
    tag = resolve_platform(
        configuration(platform_tag_source="metadata:platform"),
        {"platform": "other", "provider": "deepseek", "model": "deepseek-v4.1"},
        ProtocolContext(metadata={"platform": value}),
    )
    assert tag.value is None and tag.reason == reason


def test_endpoint_mapping_is_exact_and_keeps_frozen_evidence():
    endpoint = "https://gateway.example.test/v1"
    config = configuration(
        platform_tag_source="endpoint",
        endpoint_source="metadata:base_url",
        endpoint_map={endpoint: "volcengine"},
    )
    context = ProtocolContext(metadata={"base_url": endpoint}, metadata_path="extra.metadata")
    tag = resolve_platform(config, {}, context)
    assert tag.value == "volcengine" and tag.provenance == "mapped"
    assert tag.mapping_hash and tag.raw_value == endpoint
    assert tag.field_path == "payload:extra.metadata.base_url"
    for unmatched in [endpoint + "/", "https://gateway.example.test.attacker.invalid/v1"]:
        result = resolve_platform(config, {}, ProtocolContext(metadata={"base_url": unmatched}))
        assert result.value is None and result.reason == "endpoint_unmapped"
