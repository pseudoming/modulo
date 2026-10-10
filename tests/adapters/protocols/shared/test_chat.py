import copy

import pytest

from modulo.adapters.protocols.shared.chat import message
from modulo.core.domain.data_models import DataError
from modulo.shared.serialization import loads


def test_langchain_parsed_tool_args_never_rewrite_raw_arguments_and_conflicts_reject():
    raw = '{"z":1.2300,"a":"中"}'
    body = {
        "content": "",
        "additional_kwargs": {
            "reasoning_content": "先查询",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": raw}}
            ],
        },
        "tool_calls": [{"id": "c1", "type": "tool_call", "name": "lookup", "args": loads(raw)}],
    }
    parsed = message(body, role="assistant")
    assert loads(parsed.tool_calls_json)[0]["function"]["arguments"] == raw
    conflict = copy.deepcopy(body)
    conflict["tool_calls"][0]["args"]["a"] = "different"
    with pytest.raises(DataError, match="conflicting_tool_calls"):
        message(conflict, role="assistant")


@pytest.mark.parametrize("reasoning", ["", " \n "])
def test_reasoning_only_assistant_message_is_preserved(reasoning):
    parsed = message({"role": "assistant", "reasoning_content": reasoning})
    assert parsed.content == "" and parsed.reasoning_content == reasoning
