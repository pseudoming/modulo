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
