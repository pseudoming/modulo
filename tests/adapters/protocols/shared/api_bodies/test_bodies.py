"""Wire semantics: content, source paths, tools, usage and independent gaps."""

import json
from copy import deepcopy

import pytest

from modulo.adapters.protocols import LiteLLMProtocol, NewAPIProtocol, OneAPIProtocol
from modulo.adapters.protocols.shared.api_bodies.parsing import request_fields, response_fields
from modulo.core.domain.data_models import DataError
from tests.support.api_bodies import request_body, response_body

TEXT = "  中e\u0301🙂\r\n "
CALLS = [
    {
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{ "z":2,"a":"中" }'},
    }
]
MESSAGES = [
    {"role": "system", "content": "系统"},
    {"role": "user", "content": TEXT},
    {"role": "assistant", "content": "", "reasoning_content": "先查", "tool_calls": CALLS},
    {"role": "tool", "content": "结果", "tool_call_id": "c1"},
    {"role": "user", "content": "继续"},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "查询",
            "parameters": {"type": "object", "properties": {"z": {"type": "integer"}}},
        },
    }
]


def test_responses_schema_and_anthropic_stop_rules_are_separate_from_prompt_options():
    schema = {"type": "json_schema", "name": "answer", "strict": True, "schema": {"type": "object"}}
    fields = request_fields({"input": "x", "text": {"format": schema}, "max_output_tokens": 20})
    assert fields["render_options"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {k: v for k, v in schema.items() if k != "type"},
    }
    assert fields["render_option_sources"]["response_format"] == ["request_body.text.format"]
    assert fields["generation_options"] == {"max_output_tokens": 20}
    request = request_body(
        "anthropic",
        [{"role": "user", "content": "x"}],
        tools=TOOLS,
        options={
            "stop_sequences": ["END"],
            "max_tokens": 20,
            "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
        },
    )
    fields = request_fields(request)
    assert not fields["gaps"]
    assert fields["generation_options"] == {"stop": ["END"], "max_tokens": 20}
    assert fields["generation_option_sources"]["stop"] == ["request_body.stop_sequences"]
    assert fields["render_options"]["parallel_tool_calls"] is False
    assert fields["render_options"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }


def test_responses_keep_multiple_messages_and_their_reasoning_in_both_directions():
    first = response_body("responses", "FIRST", "THOUGHT-1")
    second = response_body("responses", "SECOND", "THOUGHT-2")
    body = {**first, "output": [*first["output"], *second["output"]]}
    fields, _, _ = response_fields(body)
    assert not fields.get("output_gaps")
    assert fields["output_text"] == "FIRSTSECOND"
    assert [(m.content, m.reasoning_content) for m in fields["output_messages"]] == [
        ("FIRST", "THOUGHT-1"),
        ("SECOND", "THOUGHT-2"),
    ]
    assert fields["output_messages"][1].reasoning_source == "response_body.output.2.content.0.text"
    request = request_fields({"input": body["output"]})
    assert not request["gaps"]
    assert [(m.content, m.reasoning_content) for m in request["messages"]] == [
        ("FIRST", "THOUGHT-1"),
        ("SECOND", "THOUGHT-2"),
    ]


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
@pytest.mark.parametrize("only_reasoning", [False, True])
def test_token_limit_is_not_treated_as_incomplete_log(dialect, only_reasoning):
    body = response_body(dialect, "ANSWER", "THOUGHT")
    if dialect == "responses":
        body.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        body["output"][-1]["status"] = "incomplete"
        if only_reasoning:
            body["output"].pop()
    else:
        body["stop_reason"] = "max_tokens"
        if only_reasoning:
            body["content"].pop()
    fields, _, _ = response_fields(body)
    assert not fields.get("output_gaps")
    assert fields["output_finish_reason"] in {"max_tokens", "max_output_tokens"}
    assert fields["output_text"] == (None if only_reasoning else "ANSWER")
    assert fields["output_reasoning"] == "THOUGHT"


def test_anthropic_matched_stop_sequence_has_original_source():
    body = response_body("anthropic", "ANSWER", "THOUGHT")
    body.update(stop_reason="stop_sequence", stop_sequence="END")
    fields, _, _ = response_fields(body)
    assert not fields.get("output_gaps")
    assert fields["output_stop_reason"] == "END"
    assert fields["output_stop_reason_source"] == "response_body.stop_sequence"


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_system_history_tools_and_source_text_are_preserved(dialect):
    body = request_body(dialect, MESSAGES, tools=TOOLS)
    original = deepcopy(body)
    fields = request_fields(body)
    messages = fields["messages"]
    assert not fields["gaps"]
    assert [m.role for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert [m.content for m in messages] == ["系统", TEXT, "", "结果", "继续"]
    assert messages[2].reasoning_content == "先查"
    assert messages[3].tool_call_id == "c1"
    assert json.loads(fields["tools_json"]) == TOOLS
    if dialect == "responses":
        assert json.loads(messages[2].tool_calls_json) == CALLS
        assert messages[2].reasoning_source == "request_body.input.1.content.0.text"
    else:
        assert messages[2].reasoning_source == "request_body.messages.1.content.0.thinking"
    assert body == original


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_output_body_is_one_turn_with_reasoning_text_and_tool_calls(dialect):
    body = response_body(dialect, TEXT, "思考\n", calls=CALLS, input_tokens=0, output_tokens=15)
    fields, usage, model = response_fields(body)
    assert not fields.get("output_gaps")
    assert fields["output_text"] == TEXT and fields["output_reasoning"] == "思考\n"
    assert not fields.get("output_candidates")
    assert usage.input_tokens == 0 and usage.output_tokens == 15
    assert model == "synthetic-model"
    assert json.loads(fields["output_tool_calls_json"])[0]["id"] == "c1"
    assert json.loads(fields["output_structured_content_json"]) == body


def test_responses_string_input_and_parameters_have_original_sources():
    fields = request_fields(
        {"input": TEXT, "instructions": "系统", "reasoning": {"effort": "high", "summary": "auto"}}
    )
    assert [m.content for m in fields["messages"]] == ["系统", TEXT]
    assert fields["render_options"] == {"reasoning_effort": "high"}
    assert fields["render_option_sources"] == {
        "reasoning_effort": ["request_body.reasoning.effort"]
    }


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("previous_response_id", "previous-1", "unresolved_responses_context"),
        ("conversation", "conversation-1", "unresolved_responses_context"),
        ("truncation", "auto", "responses_truncation_unknown"),
        (
            "input",
            [{"type": "item_reference", "id": "item-1"}],
            "unresolved_response_item_reference",
        ),
    ],
)
def test_references_and_server_truncation_are_not_invented(key, value, reason):
    fields = request_fields({"input": "test", key: value})
    assert reason in fields["gaps"]


def test_anthropic_tool_results_and_following_images_keep_original_order_and_paths():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "结果"},
                    {"type": "text", "text": "看图"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.invalid/a.png"},
                    },
                ],
            }
        ]
    }
    fields = request_fields(body)
    assert [m.role for m in fields["messages"]] == ["tool", "user"]
    image = fields["messages"][1].content_parts[1]
    assert image.source == "request_body.messages.0.content.2"
    assert image.url_source == "request_body.messages.0.content.2.source.url"
    assert fields["messages"][1].content == "看图"


def test_anthropic_cache_counters_add_to_full_input_and_missing_stays_missing():
    body = response_body("anthropic", "text", input_tokens=2, output_tokens=3)
    body["usage"].update(cache_creation_input_tokens=5, cache_read_input_tokens=7)
    assert response_fields(body)[1].input_tokens == 14
    body["usage"]["cache_creation_input_tokens"] = None
    assert response_fields(body)[1].input_tokens is None
    body["usage"]["cache_creation_input_tokens"] = False
    with pytest.raises(DataError, match="cache_usage"):
        response_fields(body)


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
@pytest.mark.parametrize("adapter", [NewAPIProtocol, OneAPIProtocol, LiteLLMProtocol])
def test_adapters_read_native_body_and_keep_output_when_input_is_missing(adapter, dialect):
    request = request_body(dialect, MESSAGES, tools=TOOLS)
    response = response_body(dialect, TEXT, "想", input_tokens=80, output_tokens=12)
    if adapter is LiteLLMProtocol:
        row = {
            "request_id": "r1",
            "model": "synthetic-model",
            "call_type": "aresponses" if dialect == "responses" else "anthropic_messages",
            "proxy_server_request": json.dumps(request),
            "response": json.dumps(response),
        }
    else:
        row = {
            "schema_version": "gateway.exchange.v1",
            "request_id": "r1",
            "request_body": request,
            "response_body": response,
        }
    call = adapter().parse(row).calls[0]
    assert not call.gaps and not call.output_gaps
    assert call.output_text == TEXT
    assert call.usage.input_tokens == 80 and call.usage.output_tokens == 12
    row["proxy_server_request" if adapter is LiteLLMProtocol else "request_body"] = {}
    call = adapter().parse(row).calls[0]
    assert "missing_messages" in call.gaps
    assert call.output_text == TEXT and not call.output_gaps


def test_litellm_usage_conflicts_do_not_choose_a_value():
    row = {
        "id": "r1",
        "call_type": "aresponses",
        "messages": "hello",
        "model": "model",
        "prompt_tokens": 99,
        "response": response_body("responses", "answer", input_tokens=2, output_tokens=3),
    }
    call = LiteLLMProtocol().parse(row).calls[0]
    assert call.usage.input_tokens is None and call.usage.output_tokens == 3
    assert "conflicting_litellm_input_tokens" in call.gaps
    assert not call.output_gaps


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_unknown_blocks_and_redaction_cannot_pass_as_complete(dialect):
    body = response_body(dialect, "answer")
    if dialect == "responses":
        body["output"].insert(
            0,
            {
                "type": "reasoning",
                "content": [],
                "summary": [{"type": "summary_text", "text": "summary"}],
                "encrypted_content": "opaque",
            },
        )
        expected = "reasoning_summary_only"
    else:
        body["content"].insert(0, {"type": "redacted_thinking", "data": "opaque"})
        expected = "redacted_reasoning"
    fields, _, _ = response_fields(body)
    assert fields["output_text"] == "answer" and expected in fields["output_gaps"]


def test_conflicting_normalized_and_native_options_remain_errors():
    fields = request_fields(
        {"input": "test", "reasoning": {"effort": "high"}},
        extra_roots=(("model_parameters", {"reasoning_effort": "low"}),),
    )
    assert "conflicting_request_option:reasoning_effort" in fields["gaps"]


def test_native_litellm_standard_parameters_preserve_instructions():
    # Checked with LiteLLM's real StandardLoggingPayload serializer: input
    # messages and optional instructions are stored in separate locations.
    row = {
        "id": "r1",
        "model": "model",
        "call_type": "aresponses",
        "messages": [{"type": "message", "role": "user", "content": TEXT}],
        "model_parameters": {"instructions": "系统"},
        "response": response_body("responses", "answer", "thought"),
    }
    call = LiteLLMProtocol().parse(row).calls[0]
    assert [m.content for m in call.messages] == ["系统", TEXT]
    assert not call.gaps
    row["model_parameters"]["previous_response_id"] = "previous"
    assert "unresolved_responses_context" in LiteLLMProtocol().parse(row).calls[0].gaps


def test_litellm_role_only_projection_does_not_invent_tool_history():
    row = {
        "id": "r1",
        "model": "model",
        "call_type": "aresponses",
        "messages": [{"role": "user", "content": TEXT}],
        "model_parameters": {"tools": [{"type": "function", "name": "lookup", "parameters": {}}]},
        "response": response_body("responses", "answer", "thought"),
    }
    call = LiteLLMProtocol().parse(row).calls[0]
    assert "missing_responses_input_items" in call.gaps and not call.output_gaps


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_api_image_blocks_feed_existing_image_contract(dialect):
    content = [
        {"type": "input_text" if dialect == "responses" else "text", "text": "看图"},
        {
            "type": "input_image",
            "image_url": "https://example.invalid/a.png",
            "width": 32,
            "height": 64,
        }
        if dialect == "responses"
        else {
            "type": "image",
            "source": {"type": "url", "url": "https://example.invalid/a.png"},
            "width": 32,
            "height": 64,
        },
    ]
    body = {
        "input" if dialect == "responses" else "messages": [{"role": "user", "content": content}]
    }
    fields = request_fields(body, format=dialect)
    image = fields["messages"][0].content_parts[1]
    assert image.width == 32 and image.height == 64
    assert image.url == "https://example.invalid/a.png" and not image.gaps


@pytest.mark.parametrize("system", [True, False])
def test_anthropic_thinking_budget_is_not_included_in_prompt_text(system):
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    if system:
        body["system"] = "系统"
    fields = request_fields(body)
    assert fields["render_options"] == {"thinking": {"type": "enabled"}}
    assert fields["render_option_sources"] == {"thinking": ["request_body.thinking.type"]}
