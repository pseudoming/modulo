import copy

import pytest

from modulo.adapters.protocols.shared.messages.chat import message
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


@pytest.mark.parametrize("finish", ["stop", "tool_calls", "length", "content_filter"])
def test_response_retains_choice_and_message_finish_reasons_without_mutating_source(finish):
    from modulo.adapters.protocols.shared.messages.chat import decode_response, output_fields

    response = {"choices": [{"message": {"content": "答案"}, "finish_reason": finish}]}
    before = copy.deepcopy(response)
    fields = output_fields(decode_response(response))
    assert fields["output_finish_reason"] == finish
    assert response == before
    response["choices"][0]["message"]["finish_reason"] = "different"
    fields = output_fields(decode_response(response))
    assert fields["output_gaps"] == ["conflicting_finish_reason"]
    assert fields["output_text"] == "答案"


@pytest.mark.parametrize("value", [0, False, [], ""])
def test_invalid_finish_keeps_the_observed_body(value):
    from modulo.adapters.protocols.shared.messages.chat import output_fields

    fields = output_fields({"content": "答案"}, finish_reason=value)
    assert fields["output_gaps"] == ["invalid_finish_reason"]
    assert fields["output_text"] == "答案"
