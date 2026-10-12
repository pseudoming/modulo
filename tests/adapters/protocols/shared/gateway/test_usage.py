from decimal import Decimal

import pytest

from modulo.adapters.protocols import NewAPIProtocol, OneAPIProtocol
from modulo.core.domain.data_models import DataError, ProtocolContext
from modulo.core.domain.platforms import PlatformConfig, resolve_platform


@pytest.fixture
def usage_row():
    return {
        "id": 17,
        "type": 2,
        "model_name": "volc-deepseek",
        "channel": 3,
        "request_id": "gateway-request",
        "created_at": 1791417600,
        "prompt_tokens": 30,
        "completion_tokens": 2,
        "content": '{"messages":[{"role":"user","content":"NOT a request body"}]}',
    }


@pytest.mark.parametrize("protocol", [NewAPIProtocol(), OneAPIProtocol()])
def test_usage_logs_keep_counters_without_inventing_messages_or_platform(protocol, usage_row):
    call = protocol.parse(usage_row).calls[0]
    assert call.call_id == "17"
    assert call.model == "volc-deepseek"
    assert call.usage.input_tokens == 30 and call.usage.output_tokens == 2
    assert call.usage.total_tokens is None
    assert call.messages is None and call.output_text is None
    assert call.gaps == ["missing_messages"]
    assert call.provider is None and call.platform is None
    assert call.started_at is None  # Log creation cannot be relabelled as request start.
    assert call.status == "unknown"  # Consumption is not evidence of a final HTTP outcome.


def test_native_duration_units_and_type_five_are_different(usage_row):
    usage_row.update(use_time=2, elapsed_time=250)
    assert NewAPIProtocol().parse(usage_row).calls[0].duration_ms == "2000"
    assert OneAPIProtocol().parse(usage_row).calls[0].duration_ms == "250"
    usage_row["type"] = 5
    assert NewAPIProtocol().parse(usage_row).calls[0].status == "failed"
    assert OneAPIProtocol().parse(usage_row).ignored == "non_consumption_log"
    usage_row["type"] = 6
    assert NewAPIProtocol().parse(usage_row).ignored == "non_consumption_log"
    with pytest.raises(DataError):
        OneAPIProtocol().parse(usage_row)


@pytest.mark.parametrize("value", [None, 0, 2**70])
@pytest.mark.parametrize("protocol", [NewAPIProtocol(), OneAPIProtocol()])
def test_usage_log_missing_zero_and_large_values_remain_distinct(protocol, usage_row, value):
    usage_row["prompt_tokens"] = value
    assert protocol.parse(usage_row).calls[0].usage.input_tokens == value


@pytest.mark.parametrize("value", [True, "30", -1, Decimal("30.0")])
@pytest.mark.parametrize("protocol", [NewAPIProtocol(), OneAPIProtocol()])
def test_usage_log_rejects_coercible_counters(protocol, usage_row, value):
    usage_row["prompt_tokens"] = value
    with pytest.raises(DataError):
        protocol.parse(usage_row)


@pytest.mark.parametrize("protocol", [NewAPIProtocol(), OneAPIProtocol()])
@pytest.mark.parametrize("value", [None, 0, -1, True, "17"])
def test_usage_log_requires_stable_integer_row_id(protocol, usage_row, value):
    usage_row["id"] = value
    with pytest.raises(DataError, match="stable log row id"):
        protocol.parse(usage_row)


@pytest.mark.parametrize("protocol", [NewAPIProtocol(), OneAPIProtocol()])
def test_usage_log_unknown_shapes_and_types_fail(protocol, usage_row):
    for value in (None, True, "2", 0, 99):
        usage_row["type"] = value
        with pytest.raises(DataError):
            protocol.parse(usage_row)
    with pytest.raises(DataError):
        protocol.parse({"data": [usage_row]})


def test_newapi_metadata_decodes_strict_json_without_setting_platform(usage_row):
    usage_row["other"] = '{"platform":"volcengine","model_ratio":1.10}'
    parsed = NewAPIProtocol().parse(usage_row)
    assert parsed.contexts[0].metadata["model_ratio"] == Decimal("1.10")
    config = PlatformConfig.model_validate({"platform-tag-source": "metadata:platform"})
    tag = resolve_platform(config, usage_row, parsed.contexts[0])
    assert tag.value == "volcengine" and tag.field_path == "payload:other.platform"
    assert parsed.calls[0].platform is None
    assert OneAPIProtocol().parse(usage_row).contexts == [ProtocolContext()]
    for invalid in ('{"platform":"a","platform":"b"}', "[]", "not-json"):
        usage_row["other"] = invalid
        with pytest.raises(DataError):
            NewAPIProtocol().parse(usage_row)
