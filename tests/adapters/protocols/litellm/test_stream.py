"""Saved stream events: preserve identity, order, body bytes and terminal evidence."""

from copy import deepcopy

import pytest

from modulo.adapters.protocols.litellm import LiteLLMProtocol
from modulo.core.domain.data_models import ProtocolRecord
from tests.support.api_bodies import response_body, stream_events


def chunk(delta=None, *, finish=None, usage=None):
    return {
        "id": "upstream-1",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **({"usage": usage} if usage is not None else {}),
    }


def records(events):
    return [
        {
            "id": "log-1",
            "litellm_call_id": "attempt-1",
            "call_type": "acompletion",
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "response": event,
            "event": {"sequence": i},
        }
        for i, event in enumerate(events)
    ]


def assemble(rows, *, ordered=False, complete=True):
    session = LiteLLMProtocol().new_session()
    emitted = []
    for index, row in enumerate(rows):
        emitted.extend(session.feed(ProtocolRecord(row, row, {"line": index + 1}, index, ordered)))
    emitted.extend(session.finish(complete=complete))
    assert len(emitted) == 1
    assert emitted[0].error is None, emitted[0].error
    return emitted[0]


def test_shuffled_events_and_duplicate_identity_preserve_text_and_sources():
    rows = records(
        [
            chunk({"role": "assistant", "reasoning_content": "想\n"}),
            chunk({"content": "答"}),
            chunk({"content": "答"}),
            chunk(finish="stop", usage={"prompt_tokens": 13, "completion_tokens": 7}),
        ]
    )
    original = deepcopy(rows)
    result = assemble([*reversed(rows), rows[1]])
    call = result.parsed.calls[0]
    assert call.output_text == "答答" and call.output_reasoning == "想\n"
    assert call.usage.input_tokens == 13 and call.usage.output_tokens == 7
    assert not call.gaps and not call.output_gaps
    assert result.assembly["sequence_basis"] == "event.sequence"
    assert result.assembly["duplicate_events"] == 1 and len(result.records) == 5
    assert rows == original


def test_tool_argument_fragments_are_not_reserialized():
    rows = records(
        [
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{ "x": '},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1 }"}}]}),
            chunk(finish="tool_calls"),
        ]
    )
    call = assemble(rows).parsed.calls[0]
    assert '{ \\"x\\": 1 }' in call.output_tool_calls_json
    assert not call.output_gaps


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_native_api_events_use_the_shared_parsers(dialect):
    body = response_body(dialect, "answer", "thought", input_tokens=13, output_tokens=7)
    rows = records(stream_events(dialect, body))
    call = assemble(list(reversed(rows))).parsed.calls[0]
    assert call.output_text == "answer" and call.output_reasoning == "thought"
    assert call.usage.input_tokens == 13 and call.usage.output_tokens == 7
    assert not call.output_gaps


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("missing_end", "stream_done_missing"),
        ("missing_order", "stream_order_unknown"),
        ("conflicting_event", "conflicting_litellm_stream_event"),
        ("request_conflict", "conflicting_litellm_stream_request"),
        ("partial_source", "source_incomplete"),
        ("response_id_conflict", "conflicting_stream_id"),
    ],
)
def test_bad_stream_cannot_pass_as_complete(problem, reason):
    rows = records([chunk({"content": "answer"}), chunk(finish="stop")])
    if problem == "missing_end":
        rows.pop()
    elif problem == "missing_order":
        for row in rows:
            row.pop("event")
    elif problem == "conflicting_event":
        rows.append(deepcopy(rows[0]))
        rows[-1]["response"]["choices"][0]["delta"]["content"] = "other"
    elif problem == "request_conflict":
        rows[-1]["messages"] = [{"role": "user", "content": "different"}]
    elif problem == "response_id_conflict":
        rows[-1]["response"]["id"] = "other-attempt"
    call = assemble(rows, complete=problem != "partial_source").parsed.calls[0]
    assert reason in call.output_gaps
    assert call.messages[0].content == "Hi"


def test_ordered_file_and_batched_response_array():
    rows = records([chunk({"content": "answer"}), chunk(finish="stop")])
    for row in rows:
        row.pop("event")
    assert not assemble(rows, ordered=True).parsed.calls[0].output_gaps
    single = {**rows[0], "response": [r["response"] for r in rows]}
    assert not assemble([single]).parsed.calls[0].output_gaps
    sse = {
        **single,
        "response": (
            'data: {"id":"s","choices":[{"index":0,"delta":{"content":"a"},'
            '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        ),
    }
    assert assemble([sse]).parsed.calls[0].output_text == "a"


def test_serialized_callback_arguments_keep_request_context():
    kwargs = {
        "litellm_call_id": "callback-call",
        "call_type": "completion",
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "optional_params": {"enable_thinking": False},
        "litellm_params": {"metadata": {"platform": "private"}},
    }
    rows = [
        {"kwargs": kwargs, "response_obj": event, "sequence": i}
        for i, event in enumerate([chunk({"content": "a"}), chunk(finish="stop")])
    ]
    result = assemble(rows)
    call = result.parsed.calls[0]
    assert call.call_id == "callback-call" and call.output_text == "a"
    assert call.render_options["enable_thinking"] is False
    assert call.render_option_sources["enable_thinking"] == [
        "kwargs.optional_params.enable_thinking"
    ]
    assert result.parsed.contexts[0].metadata_path == "kwargs.litellm_params.metadata"
    assert result.parsed.contexts[0].metadata == {"platform": "private"}


def test_complete_terminal_and_chunks_are_counted_once_and_conflicts_block_output():
    rows = records([chunk({"content": "answer"}), chunk(finish="stop")])
    terminal = {
        **rows[0],
        "response": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ]
        },
    }
    result = assemble([terminal, *rows])
    assert result.parsed.calls[0].output_text == "answer" and not result.parsed.calls[0].output_gaps
    terminal["response"]["choices"][0]["message"]["content"] = "different"
    call = assemble([terminal, *rows]).parsed.calls[0]
    assert "conflicting_litellm_terminal_response" in call.output_gaps


def test_actual_litellm_sdk_stream_serialization():
    import json

    from tests.support.paths import ROOT

    path = ROOT / "tests/fixtures/native-logs/litellm-stream-chunks.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = assemble(list(reversed(rows)))
    call = result.parsed.calls[0]
    assert call.output_text == "synthetic answer"
    assert call.output_finish_reason == "stop"
    assert not call.gaps and not call.output_gaps
    assert call.messages[0].content == "synthetic question"
    assert call.usage.input_tokens is None


def test_native_responses_sequence_numbers_are_usable_without_es_envelope_sequence():
    rows = records(stream_events("responses", response_body("responses", "answer")))
    for row in rows:
        row.pop("event")
    result = assemble(list(reversed(rows)))
    assert result.assembly["sequence_basis"] == "response.sequence_number"
    assert not result.parsed.calls[0].output_gaps


def test_intermediate_usage_snapshots_are_not_added_or_compared_as_totals():
    rows = records(
        [
            chunk({"content": "answer"}),
            chunk(finish="stop", usage={"prompt_tokens": 13, "completion_tokens": 7}),
        ]
    )
    rows[0].update(prompt_tokens=0, completion_tokens=0)
    rows[1].update(prompt_tokens=13, completion_tokens=7)
    call = assemble(rows).parsed.calls[0]
    assert not call.gaps and not call.output_gaps
    assert call.usage.input_tokens == 13 and call.usage.output_tokens == 7


def test_matching_tool_terminal_does_not_depend_on_transport_key_order():
    tool = {
        "id": "t",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{ "x": 1 }'},
    }
    rows = records([chunk({"tool_calls": [{"index": 0, **tool}]}, finish="tool_calls")])
    final = {
        **rows[0],
        "response": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [tool]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    }
    final = deepcopy(final)
    call = assemble([*rows, final]).parsed.calls[0]
    assert not call.output_gaps
    final["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"x":1}'
    call = assemble([*rows, final]).parsed.calls[0]
    assert "conflicting_litellm_terminal_response" in call.output_gaps


def test_saved_sse_lines_include_delimiters_and_heartbeats():
    import json

    lines = [
        ": heartbeat",
        "data: " + json.dumps(chunk({"content": "a"})),
        "",
        "data: " + json.dumps(chunk(finish="stop")),
        "",
        "data: [DONE]",
        "",
    ]
    call = assemble(records(lines)).parsed.calls[0]
    assert call.output_text == "a" and not call.output_gaps
