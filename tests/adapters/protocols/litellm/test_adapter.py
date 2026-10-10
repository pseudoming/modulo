import copy
import json
from decimal import Decimal

import pytest

from modulo.adapters.protocols import (
    LiteLLMProtocol,
)
from modulo.core.domain.data_models import DataError
from modulo.core.domain.platforms import PlatformConfig, resolve_platform
from modulo.shared.serialization import loads


@pytest.fixture
def spend_log():
    # Fixed LiteLLM spend_tracking_utils stores the body directly and leaves
    # top-level messages empty for ordinary chat completions.
    return {
        "request_id": "native-spend-1",
        "call_type": "acompletion",
        "model": "test-model",
        "messages": {},
        "proxy_server_request": json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "  正文\r\n"}],
                "extra_body": {"enable_thinking": False, "thinking_budget": 0},
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
            ensure_ascii=False,
        ),
        "response": json.dumps(
            {"choices": [{"message": {"content": "回答", "reasoning_content": "思考"}}]}
        ),
        "prompt_tokens": 13,
        "completion_tokens": 2,
    }


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("empty", [None, {}, [], "{}", "[]"])
def test_spendlogs_read_native_or_wrapped_body_and_preserve_options(spend_log, wrapped, empty):
    spend_log["messages"] = empty
    if wrapped:
        spend_log["proxy_server_request"] = {"body": spend_log["proxy_server_request"]}
    original = copy.deepcopy(spend_log)
    call = LiteLLMProtocol().parse(spend_log).calls[0]
    assert not call.gaps and not call.output_gaps
    assert call.messages[0].content == "  正文\r\n"
    assert call.output_reasoning == "思考"
    assert call.render_options == {"enable_thinking": False, "thinking_budget": 0}
    prefix = "proxy_server_request.body" if wrapped else "proxy_server_request"
    assert call.render_option_sources["enable_thinking"] == [f"{prefix}.extra_body.enable_thinking"]
    assert json.loads(call.tools_json)[0]["function"]["name"] == "lookup"
    assert spend_log == original


def test_explicit_redaction_is_not_treated_as_missing(spend_log):
    spend_log["messages"] = [{"role": "user", "content": "redacted-by-litellm"}]
    call = LiteLLMProtocol().parse(spend_log).calls[0]
    assert call.messages is None and "redacted_messages" in call.gaps


def test_spendlogs_serialized_none_cache_status_is_unknown(spend_log):
    spend_log["cache_hit"] = "None"
    assert not LiteLLMProtocol().parse(spend_log).calls[0].gaps


@pytest.mark.parametrize("provider", [{}, {"refusal": None}, {"refusal": "synthetic refusal"}])
def test_litellm_empty_provider_envelope_is_not_output_content(spend_log, provider):
    response = json.loads(spend_log["response"])
    response["choices"][0]["message"]["provider_specific_fields"] = provider
    spend_log["response"] = response
    call = LiteLLMProtocol().parse(spend_log).calls[0]
    assert bool(call.output_gaps) == bool(provider.get("refusal"))


@pytest.mark.parametrize("component", ["request", "response"])
def test_native_litellm_truncation_marker_blocks_affected_component(spend_log, component):
    marker = "... (litellm_truncated skipped 4000 chars. Truncation is a DB storage safeguard."
    if component == "request":
        request = json.loads(spend_log["proxy_server_request"])
        request["messages"][0]["content"] = "prefix " + marker + " suffix"
        spend_log["proxy_server_request"] = json.dumps(request)
    else:
        response = json.loads(spend_log["response"])
        response["choices"][0]["message"]["reasoning_content"] = marker
        spend_log["response"] = json.dumps(response)
    call = LiteLLMProtocol().parse(spend_log).calls[0]
    assert call.gaps == (["truncated_request"] if component == "request" else [])
    assert call.output_gaps == (["truncated_output"] if component == "response" else [])


def test_conflicting_body_options_and_ambiguous_body_are_rejected(spend_log):
    spend_log["model_parameters"] = {"enable_thinking": True}
    call = LiteLLMProtocol().parse(spend_log).calls[0]
    assert "conflicting_request_option:enable_thinking" in call.gaps
    spend_log["proxy_server_request"] = {"body": {}, "messages": []}
    with pytest.raises(DataError, match="ambiguous_litellm_request_body"):
        LiteLLMProtocol().parse(spend_log)


@pytest.fixture
def litellm_record():
    return {
        "id": "completion-1",
        "litellm_call_id": "separate-correlation-id",
        "trace_id": "shared-trace",
        "call_type": "acompletion",
        "status": "success",
        "model": "volc-deepseek",
        "model_group": "public-group",
        "custom_llm_provider": "openai",
        "cache_hit": False,
        "startTime": Decimal("1791417600.125"),
        "endTime": Decimal("1791417600.375"),
        "completionStartTime": Decimal("1791417600.225"),
        "response_time": Decimal("0.1"),
        "prompt_tokens": 30,
        "completion_tokens": 2,
        "total_tokens": 32,
        "metadata": {"spend_logs_metadata": {"platform": "volcengine"}},
        "messages": [{"role": "user", "content": "  中e\u0301\r\n🙂\t "}],
        "model_parameters": {},
        "response": {"choices": [{"message": {"role": "assistant", "content": "收到。"}}]},
    }


def test_litellm_preserves_logged_model_text_and_uses_full_duration(litellm_record):
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert call.call_id == "completion-1"
    assert call.model == "volc-deepseek" and call.provider == "openai"
    assert call.platform is None
    assert call.messages[0].content == "  中e\u0301\r\n🙂\t "
    assert call.started_at == "2026-10-08T00:00:00.125000+00:00"
    assert Decimal(call.duration_ms) == 250  # response_time can be TTFT, so do not use it.
    assert call.output_text == "收到。" and call.usage.total_tokens == 32
    assert call.gaps == []


def test_litellm_spend_row_with_iso_dates_and_json_fields(litellm_record):
    litellm_record["request_id"] = litellm_record.pop("id")
    litellm_record.update(
        startTime="2026-10-08T08:00:00.125+08:00", endTime="2026-10-08T00:00:00.375Z"
    )
    litellm_record["messages"] = json.dumps(litellm_record["messages"])
    litellm_record["response"] = json.dumps(litellm_record["response"])
    litellm_record["metadata"] = json.dumps(litellm_record["metadata"])
    litellm_record["cache_hit"] = "False"
    del litellm_record["status"]
    parsed = LiteLLMProtocol().parse(litellm_record)
    call = parsed.calls[0]
    assert call.call_id == "completion-1" and call.status == "unknown"
    assert Decimal(call.duration_ms) == 250 and call.output_text == "收到。"
    config = PlatformConfig.model_validate(
        {"platform-tag-source": "metadata:spend_logs_metadata.platform"}
    )
    tag = resolve_platform(config, {}, parsed.contexts[0])
    assert tag.value == "volcengine"
    assert tag.field_path == "payload:metadata.spend_logs_metadata.platform"


@pytest.mark.parametrize(
    "messages", [None, {}, [], "redacted-by-litellm", "Human: Hello", {"user": "Hello"}]
)
def test_litellm_missing_or_display_messages_do_not_get_reconstructed(litellm_record, messages):
    litellm_record["messages"] = messages
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert call.messages is None and call.gaps
    assert call.usage.input_tokens == 30


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "tool", "content": "Hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
    ],
)
def test_litellm_tool_and_multimodal_messages_have_capability_gaps(litellm_record, message):
    litellm_record["messages"] = [message]
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert call.messages is None and call.gaps


@pytest.mark.parametrize("field", ["model_parameters", "proxy_server_request"])
def test_litellm_request_tool_definitions_are_retained(litellm_record, field):
    parameters = {"tools": [{"type": "function", "function": {"name": "lookup"}}]}
    litellm_record[field] = parameters if field == "model_parameters" else {"body": parameters}
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert not call.gaps
    assert json.loads(call.tools_json) == parameters["tools"]


@pytest.mark.parametrize(
    "fields, reason",
    [
        ({"call_type": "embedding"}, "unsupported_call_type"),
        ({"status": "failure"}, "failed_call"),
        ({"cache_hit": True}, "cache_hit"),
        ({"cache_hit": "True"}, "cache_hit"),
    ],
)
def test_litellm_nontext_failed_and_cached_calls_are_retained_with_gaps(
    litellm_record, fields, reason
):
    litellm_record.update(fields)
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert reason in call.gaps and call.usage.input_tokens == 30


def test_litellm_does_not_derive_missing_usage_or_replace_top_level_counts(litellm_record):
    litellm_record["response"]["usage"] = {"prompt_tokens": 999, "completion_tokens": 888}
    litellm_record.pop("total_tokens")
    litellm_record.pop("prompt_tokens")
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert call.usage.input_tokens is None and call.usage.total_tokens is None
    assert call.usage.output_tokens == 2
    litellm_record["completion_tokens"] = True
    with pytest.raises(DataError):
        LiteLLMProtocol().parse(litellm_record)


@pytest.mark.parametrize(
    "marker",
    [
        "redacted-by-litellm",
        "redacted by litellm. 'litellm.turn_off_message_logging=True'",
    ],
)
def test_structured_redaction_placeholders_are_not_counted_as_request_text(litellm_record, marker):
    litellm_record["messages"] = [{"role": "user", "content": marker}]
    litellm_record["response"]["choices"][0]["message"]["content"] = marker
    call = LiteLLMProtocol().parse(litellm_record).calls[0]
    assert call.messages is None and call.output_text is None
    assert "redacted_messages" in call.gaps


def test_litellm_trace_id_cannot_replace_call_identity(litellm_record):
    other = copy.deepcopy(litellm_record)
    other["id"] = "completion-2"
    assert (
        LiteLLMProtocol().parse(other).calls[0].call_id
        != LiteLLMProtocol().parse(litellm_record).calls[0].call_id
    )
    del other["id"]
    with pytest.raises(DataError):
        LiteLLMProtocol().parse(other)
    litellm_record["request_id"] = "ambiguous"
    with pytest.raises(DataError):
        LiteLLMProtocol().parse(litellm_record)


def test_litellm_stream_chunk_is_not_a_terminal_log(litellm_record):
    litellm_record["response"] = {"choices": [{"delta": {"content": "partial"}}]}
    assert LiteLLMProtocol().parse(litellm_record).calls[0].output_text is None
    with pytest.raises(DataError):
        LiteLLMProtocol().parse({"id": "chunk", "choices": [], "object": "chat.completion.chunk"})


def test_litellm_epoch_decimal_survives_file_and_es_representation(litellm_record):
    litellm_record.update(
        startTime=Decimal("1791417600.000001"), endTime=Decimal("1791417600.000003")
    )
    assert LiteLLMProtocol().parse(litellm_record).calls[0].duration_ms == "0.002000"
    litellm_record.update(startTime="1791417600.000001", endTime="1791417600.000003")
    assert LiteLLMProtocol().parse(litellm_record).calls[0].duration_ms == "0.002000"
    litellm_record["startTime"] = "2026-10-08T00:00:00"
    with pytest.raises(DataError, match="timezone"):
        LiteLLMProtocol().parse(litellm_record)
    for start, end in [(3, 2), (True, 2), (Decimal("NaN"), 2), (10**30, None)]:
        litellm_record.update(startTime=start, endTime=end)
        with pytest.raises(DataError):
            LiteLLMProtocol().parse(litellm_record)


def test_nested_json_duplicate_keys_are_rejected(litellm_record):
    litellm_record["metadata"] = '{"platform":"a","platform":"b"}'
    with pytest.raises(DataError):
        LiteLLMProtocol().parse(litellm_record)
    with pytest.raises(ValueError):
        loads('{"prompt_tokens":0,"prompt_tokens":30}')
