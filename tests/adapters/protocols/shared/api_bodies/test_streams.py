import json
from copy import deepcopy

import pytest

from modulo.adapters.protocols.shared.api_bodies.parsing import assemble_sse
from tests.support.api_bodies import response_body, sse, stream_events

CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"z":2,"a":"中"}'},
}


@pytest.mark.parametrize("dialect", ["chat", "anthropic", "responses"])
def test_stream_termination_metadata_is_preserved(dialect):
    if dialect == "chat":
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "ANSWER", "reasoning_content": "THOUGHT"},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": "END"}]},
        ]
        lines = [line for event in events for line in ("data: " + json.dumps(event), "")] + [
            "data: [DONE]",
            "",
        ]
    else:
        body = response_body(dialect, "ANSWER", "THOUGHT", input_tokens=53, output_tokens=8)
        if dialect == "anthropic":
            body.update(stop_reason="stop_sequence", stop_sequence="END")
        else:
            body.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        lines = sse(stream_events(dialect, body)).splitlines()
    result = assemble_sse(lines)
    assert result.done and not result.output.get("output_gaps")
    if dialect == "responses":
        assert result.output["output_finish_reason"] == "max_output_tokens"
    else:
        assert result.output["output_stop_reason"] == "END"
        assert result.output["output_stop_reason_source"]


def test_chat_stream_retains_interleaving_but_joins_adjacent_scalar_fragments():
    deltas = [
        {"content": "before"},
        {"reasoning_content": "TH"},
        {"reasoning_content": "OUGHT"},
        {"content": "after"},
    ]
    events = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas
    ]
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    output = assemble_sse(
        [line for event in events for line in ("data: " + json.dumps(event), "")]
        + ["data: [DONE]", ""]
    ).output
    assert not output.get("output_gaps")
    assert json.loads(output["output_parts_json"]) == [
        {"type": "text", "text": "before"},
        {"type": "reasoning", "text": "THOUGHT"},
        {"type": "text", "text": "after"},
    ]


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
@pytest.mark.parametrize("separators", [True, False])
def test_stream_merges_fragments_once_and_uses_final_usage(dialect, separators):
    body = response_body(
        dialect, "回答\n", " 思考🙂 ", calls=[CALL], input_tokens=23, output_tokens=45
    )
    lines = sse(stream_events(dialect, body)).splitlines()
    if not separators:
        lines = [line for line in lines if line.startswith("data:")]
    result = assemble_sse(lines)
    assert result.done and not result.output.get("output_gaps")
    assert result.output["output_text"] == "回答\n"
    assert result.output["output_reasoning"] == " 思考🙂 "
    assert result.usage.input_tokens == 23 and result.usage.output_tokens == 45
    call = json.loads(result.output["output_tool_calls_json"])[0]
    arguments = call["function"]["arguments"]
    assert (json.loads(arguments) if isinstance(arguments, str) else arguments) == {
        "z": 2,
        "a": "中",
    }


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_missing_terminator_retains_partial_text(dialect):
    body = response_body(dialect, "回答", "思考", input_tokens=2, output_tokens=3)
    events = stream_events(dialect, body)[:-1]
    result = assemble_sse(sse(events).splitlines())
    assert not result.done and result.output["output_text"] == "回答"
    assert "stream_done_missing" in result.output["output_gaps"]


def test_responses_final_snapshot_conflict_is_not_hidden():
    body = response_body("responses", "answer", "thought", input_tokens=2, output_tokens=3)
    events = deepcopy(stream_events("responses", body))
    delta = next(e for e in events if e["type"] == "response.output_text.delta")
    delta["delta"] = "other"
    result = assemble_sse(sse(events).splitlines())
    assert "conflicting_stream_text" in result.output["output_gaps"]


def test_responses_final_annotations_do_not_change_text_countability():
    body = response_body("responses", "answer", input_tokens=2, output_tokens=3)
    annotation = {"type": "url_citation", "url": "https://example.invalid", "title": "source"}
    body["output"][0]["content"][0]["annotations"] = [annotation]
    events = deepcopy(stream_events("responses", body))
    part = next(e for e in events if e["type"] == "response.content_part.added")
    part["part"]["annotations"] = []
    result = assemble_sse(sse(events).splitlines())
    assert result.output["output_text"] == "answer" and not result.output.get("output_gaps")
    assert json.loads(result.output["output_structured_content_json"]) == body


def test_responses_done_text_is_checked_even_without_deltas_or_item_snapshot():
    body = response_body("responses", "answer", input_tokens=2, output_tokens=3)
    events = [
        e
        for e in stream_events("responses", body)
        if not e["type"].endswith(".delta") and e["type"] != "response.output_item.done"
    ]
    events[-1] = deepcopy(events[-1])
    events[-1]["response"]["output"][0]["content"][0]["text"] = "different answer"
    result = assemble_sse(sse(events).splitlines())
    assert "conflicting_stream_text" in result.output["output_gaps"]


def test_anthropic_tool_json_remains_raw_and_missing_block_end_is_detected():
    body = response_body("anthropic", None, calls=[CALL], input_tokens=2, output_tokens=3)
    events = stream_events("anthropic", body)
    raw = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e.get("delta", {}).get("type") == "input_json_delta"
    )
    result = assemble_sse(sse(events).splitlines())
    assert json.loads(result.output["output_tool_calls_json"])[0]["function"][
        "arguments"
    ] == json.loads(raw)
    assert (
        json.loads(result.output["output_structured_content_json"])["tool_input_json"]["0"] == raw
    )
    events = [e for e in events if e["type"] != "content_block_stop"]
    result = assemble_sse(sse(events).splitlines())
    assert "stream_block_stop_missing" in result.output["output_gaps"]


@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_unknown_and_malformed_stream_events_are_visible(dialect):
    events = stream_events(
        dialect, response_body(dialect, "answer", input_tokens=2, output_tokens=3)
    )
    events.insert(1, {"type": "unexpected", "output_index": 0})
    result = assemble_sse(sse(events).splitlines())
    assert result.output["output_gaps"]
    lines = sse(events).splitlines()
    lines.insert(0, "data: {invalid}")
    result = assemble_sse(lines)
    assert "invalid_stream_json" in result.output["output_gaps"]
