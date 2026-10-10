"""Regression cases observed while running an actual Dify Agent application."""

import copy
import json

import pytest

from modulo.adapters.protocols.dify import DifyProtocol
from modulo.core.domain.data_models import DataError, ProtocolRecord
from tests.support.paths import ROOT


def captures():
    return [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/dify-agent/agent-llm.jsonl").read_text().splitlines()
    ]


def terminal(stream=True, position=1):
    return next(
        r
        for r in captures()
        if r["phase"] == "finished"
        and r["request"]["stream"] == stream
        and r["metadata"]["position"] == position
    )


def assembled(rows, complete=True):
    session = DifyProtocol().new_session()
    for i, row in enumerate(rows):
        assert session.feed(ProtocolRecord(row, row, {"locator": str(i)}, i)) == []
    return session.finish(complete=complete)


def test_real_stream_terminal_omits_tools_but_actual_chunks_preserve_them():
    row = terminal()
    assert row["response"]["message"]["tool_calls"] == []
    call = DifyProtocol().parse(row).calls[0]
    assert not call.output_gaps
    assert (
        json.loads(call.output_tool_calls_json)
        == row["chunks"][0]["delta"]["message"]["tool_calls"]
    )
    assert call.output_text == ""


@pytest.mark.parametrize("stream", [False, True])
def test_actual_tool_result_and_reasoning_remain_verbatim(stream):
    row = terminal(stream, 2)
    original = copy.deepcopy(row)
    call = DifyProtocol().parse(row).calls[0]
    assert not call.gaps and not call.output_gaps
    assert call.messages[-1].content == '{"result": 42}'
    assert call.messages[-1].tool_call_id == json.loads(call.messages[-2].tool_calls_json)[0]["id"]
    assert call.output_text == "<think>\n工具已返回42，按工具结果作答。\n</think>工具计算结果是42。"
    assert row == original


@pytest.mark.parametrize("missing", ["chunks", "finish"])
def test_missing_stream_evidence_is_not_silently_completed(missing):
    row = terminal()
    if missing == "chunks":
        del row["chunks"]
        expected = "missing_dify_stream_chunks"
    else:
        row["chunks"].pop()
        expected = "missing_dify_stream_finish_reason"
    call = DifyProtocol().parse(row).calls[0]
    assert call.output_gaps == [expected]
    assert not call.gaps  # Input evidence remains available independently.


def test_unordered_events_and_exact_duplicates_keep_all_source_references():
    rows = captures()
    result = assembled([*reversed(rows), rows[0]])
    assert len(result) == 4
    assert sum(len(r.records) for r in result) == 9
    assert all(r.parsed and len(r.parsed.calls) == 1 for r in result)
    assert all(r.assembly["method"] == "explicit_invocation_id" for r in result)


@pytest.mark.parametrize("field", ["request", "usage"])
def test_conflicting_identity_is_reported_even_when_numeric_values_match(field):
    end = terminal()
    start = next(
        r
        for r in captures()
        if r["invocation_id"] == end["invocation_id"] and r["phase"] == "started"
    )
    if field == "request":
        end["request"]["messages"][0]["content"] += " changed"
        rows = [start, end]
    else:
        other = copy.deepcopy(end)
        other["response"]["usage"]["prompt_tokens"] += 1
        rows = [start, end, other]
    result = assembled(rows)
    assert len(result) == 1 and result[0].error.startswith("conflicting_dify_invocation_")
    assert result[0].parsed is None


def test_partial_batch_keeps_started_call_and_explicit_missing_response():
    row = captures()[0]
    assert row["phase"] == "started"
    result = assembled([row], complete=False)[0]
    call = result.parsed.calls[0]
    assert call.status == "unknown" and call.usage.input_tokens is None
    assert call.output_gaps == ["missing_dify_invocation_finish"]
    assert result.assembly["source_read_complete"] is False


def test_unknown_multimodal_part_does_not_turn_into_text():
    row = terminal(False, 2)
    row["response"]["message"]["content"] = [{"type": "image", "data": "synthetic"}]
    assert DifyProtocol().parse(row).calls[0].output_gaps == ["unsupported_dify_content_parts"]


def test_explicit_false_and_zero_options_are_not_dropped():
    row = terminal(False, 2)
    row["request"]["model_parameters"] = {"enable_thinking": False, "thinking_budget": 0}
    assert DifyProtocol().parse(row).calls[0].render_options == {
        "enable_thinking": False,
        "thinking_budget": 0,
    }


@pytest.mark.parametrize("duration", ["NaN", "-1", True])
def test_invalid_duration_is_rejected(duration):
    row = terminal(False)
    row["duration_ms"] = duration
    with pytest.raises(DataError, match="duration"):
        DifyProtocol().parse(row)


def test_native_agent_database_row_preserves_usage_without_guessing_request():
    row = json.loads(
        (ROOT / "tests/fixtures/dify-agent/agent-thoughts.jsonl").read_text().splitlines()[0]
    )
    call = DifyProtocol().parse(row).calls[0]
    assert call.usage.input_tokens == row["message_token"]
    assert call.usage.output_tokens == row["answer_token"]
    assert call.messages is None
    assert call.gaps == ["missing_agent_request_body"]
    assert call.output_gaps == ["missing_agent_tool_call_ids"]
