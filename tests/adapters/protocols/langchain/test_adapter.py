import copy
import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from modulo.adapters.protocols import LangChainProtocol


def test_real_langchain_export_and_nested_children():
    from tests.support.paths import ROOT

    record = json.loads((ROOT / "examples/langchain/basic.jsonl").read_text().splitlines()[0])
    call = LangChainProtocol().parse(record).calls[0]
    assert call.model == "Qwen/Qwen3.8-2.4T-A95B"
    assert call.messages[0].content == "你是一个简洁的助手。"
    assert call.output_text == "5。"
    assert call.duration_ms == "250"
    child = copy.deepcopy(record)
    child["id"] = "second-attempt"
    nested = {"run_type": "chain", "child_runs": [record, child]}
    calls = LangChainProtocol().parse(nested).calls
    assert [call.call_id for call in calls] == [record["id"], "second-attempt"]


def test_flattened_langchain_prompts_are_not_guessed_as_messages():
    run = {"run_type": "llm", "id": "flat", "inputs": {"prompts": ["Human: Hi"]}}
    call = LangChainProtocol().parse(run).calls[0]
    assert call.messages is None
    assert "missing_messages" in call.gaps


def native_tool_run(status="success", artifact=None):
    collector = RunCollectorCallbackHandler(_schema_format="original+chat")
    response = AIMessage(
        content="合成回答\n",
        additional_kwargs={"reasoning_content": "合成思考\n"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    FakeMessagesListChatModel(responses=[response]).invoke(
        [
            HumanMessage(content="合成问题"),
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "历史思考"},
                tool_calls=[{"name": "lookup", "args": {"q": "合成"}, "id": "call-a"}],
            ),
            ToolMessage(
                content="工具结果\r\n ", tool_call_id="call-a", status=status, artifact=artifact
            ),
        ],
        config={"callbacks": [collector], "metadata": {"ls_model_name": "test-model"}},
    )
    return json.loads(collector.traced_runs[0].model_dump_json())


@pytest.mark.parametrize("status", ["success", "error"])
def test_real_langchain_tool_messages_do_not_drop_the_request(status):
    run = native_tool_run(status, artifact={"not_sent_to_model": [1, 2, 3]})
    original = copy.deepcopy(run)
    tool = run["inputs"]["messages"][0][-1]["kwargs"]
    assert tool["status"] == status
    call = LangChainProtocol().parse(run).calls[0]
    assert not call.gaps and not call.output_gaps
    assert call.messages[-1].content == "工具结果\r\n "
    assert call.messages[-1].tool_call_id == "call-a"
    assert call.messages[-2].reasoning_content == "历史思考"
    assert call.output_reasoning == "合成思考\n"
    assert call.usage.input_tokens == 10
    assert run == original


def test_unknown_tool_status_remains_a_gap():
    run = native_tool_run()
    run["inputs"]["messages"][0][-1]["kwargs"]["status"] = "not-a-framework-status"
    call = LangChainProtocol().parse(run).calls[0]
    assert call.gaps == ["invalid_langchain_tool_status"]
    assert not call.output_gaps


def test_native_aggregated_tool_chunks_preserve_argument_bytes():
    collector = RunCollectorCallbackHandler(_schema_format="original+chat")
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "lookup", "args": '{ "q": ', "id": "native-tool", "index": 0}],
    )
    last = AIMessageChunk(
        content="",
        chunk_position="last",
        tool_call_chunks=[{"name": None, "args": '"合成" }', "id": None, "index": 0}],
    )
    FakeMessagesListChatModel(responses=[first + last]).invoke(
        [HumanMessage(content="工具问题")], config={"callbacks": [collector]}
    )
    run = json.loads(collector.traced_runs[0].model_dump_json())
    call = LangChainProtocol().parse(run).calls[0]
    assert not call.output_gaps
    assert json.loads(call.output_tool_calls_json)[0]["function"]["arguments"] == '{ "q": "合成" }'


def test_real_langchain_serializes_reasoning_blocks_without_losing_text():
    collector = RunCollectorCallbackHandler(_schema_format="original+chat")
    response = AIMessage(
        content=[
            {"type": "thinking", "thinking": "完整思考\n", "signature": "sig"},
            {"type": "text", "text": "答案"},
        ],
        response_metadata={"finish_reason": "stop"},
    )
    FakeMessagesListChatModel(responses=[response]).invoke(
        [HumanMessage(content=[{"type": "text", "text": "问题"}])],
        config={"callbacks": [collector]},
    )
    run = json.loads(collector.traced_runs[0].model_dump_json())
    call = LangChainProtocol().parse(run).calls[0]
    assert call.messages[0].content == "问题"
    assert call.output_reasoning == "完整思考\n" and call.output_text == "答案"
    assert call.output_finish_reason == "stop" and not call.output_gaps
    assert "sig" in call.output_structured_content_json


def test_conflicting_usage_copies_are_not_added_or_arbitrarily_selected():
    run = native_tool_run()
    generations = run["outputs"]["generations"][0]
    second = copy.deepcopy(generations[0])
    second["message"]["kwargs"]["usage_metadata"]["output_tokens"] = 7
    generations.append(second)
    call = LangChainProtocol().parse(run).calls[0]
    assert len(call.output_candidates) == 2
    assert call.usage.input_tokens is None and call.usage.output_tokens is None
    assert "conflicting_langchain_usage" in call.gaps
    assert "conflicting_langchain_usage" in call.output_gaps


def test_invalid_candidate_usage_keeps_available_response_content():
    run = native_tool_run()
    generations = run["outputs"]["generations"][0]
    second = copy.deepcopy(generations[0])
    second["message"]["kwargs"]["usage_metadata"]["output_tokens"] = -1
    generations.append(second)
    call = LangChainProtocol().parse(run).calls[0]
    assert len(call.output_candidates) == 2
    assert all(item.output_text == "合成回答\n" for item in call.output_candidates)
    assert call.usage.output_tokens is None
    assert "invalid_langchain_usage" in call.output_gaps


def test_multiple_input_groups_keep_pairing_candidates_and_usage_scope():
    run = native_tool_run()
    second = copy.deepcopy(run["inputs"]["messages"][0])
    second[0]["kwargs"]["content"] = "第二个问题"
    run["inputs"]["messages"].append(second)
    output = copy.deepcopy(run["outputs"]["generations"][0])
    output[0]["message"]["kwargs"]["content"] = "第二个答案"
    output[0]["message"]["kwargs"]["usage_metadata"] = {
        "input_tokens": 20,
        "output_tokens": 8,
        "total_tokens": 28,
    }
    run["outputs"]["generations"].append([*output, copy.deepcopy(output[0])])
    run["outputs"]["llm_output"] = {
        "token_usage": {"prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43}
    }
    calls = LangChainProtocol().parse(run).calls
    assert len(calls) == 2 and all(not call.gaps for call in calls)
    assert calls[1].messages[0].content == "第二个问题"
    assert len(calls[1].output_candidates) == 2
    assert [c.usage.input_tokens for c in calls] == [10, 20]
    assert calls[1].usage.output_tokens == 8  # Repeated candidate metadata counted once.
    assert [c.input_batch.index for c in calls] == [0, 1]
    assert all(c.input_batch.usage.input_tokens == 30 for c in calls)
    assert all(c.input_batch.run_id == run["id"] for c in calls)
    for group in run["outputs"]["generations"]:
        for generation in group:
            generation["message"]["kwargs"].pop("usage_metadata")
    calls = LangChainProtocol().parse(run).calls
    assert all(c.usage.input_tokens is None and c.usage.output_tokens is None for c in calls)
    assert all(not c.gaps and not c.output_gaps for c in calls)


def test_unmatched_input_output_groups_remain_visible():
    run = native_tool_run()
    run["outputs"]["generations"].append(copy.deepcopy(run["outputs"]["generations"][0]))
    calls = LangChainProtocol().parse(run).calls
    assert len(calls) == 2 and calls[1].messages is None
    assert all("langchain_input_output_group_mismatch" in c.output_gaps for c in calls)
    assert calls[1].output_text == "合成回答\n"
