import copy

import pytest

from modulo.adapters.protocols.shared.messages.chat import message, output_fields
from modulo.shared.serialization import loads


@pytest.mark.parametrize(
    "thought",
    [
        {"type": "thinking", "thinking": "先想\r\n ", "signature": "not-model-text"},
        {"type": "reasoning", "reasoning": "先想\r\n ", "extras": {"signature": "not-model-text"}},
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "先想\r\n "}],
            "summary": [],
        },
    ],
)
def test_content_blocks_preserve_text_bytes_sources_and_original_order(thought):
    blocks = [thought, {"type": "text", "text": "答案 e\u0301"}, {"type": "text", "text": "\n"}]
    body = {"role": "assistant", "content": blocks}
    before = copy.deepcopy(body)
    parsed = message(body, path="messages.2.")
    assert parsed.content == "答案 e\u0301\n"
    assert parsed.reasoning_content == "先想\r\n "
    assert parsed.reasoning_source.startswith("messages.2.content.0.")
    assert not parsed.gaps
    assert loads(parsed.structured_content_json) == {"content": blocks}
    assert body == before


@pytest.mark.parametrize("key", ["reasoning_content", "thinking", "reasoning"])
@pytest.mark.parametrize("additional", [False, True])
def test_structured_reasoning_fields_preserve_explicit_empty(key, additional):
    body = {"content": "answer"}
    target = body.setdefault("additional_kwargs", {}) if additional else body
    target[key] = [{"type": "thinking", "thinking": "", "signature": "signature"}]
    fields = output_fields(body)
    assert fields["output_reasoning"] == ""
    assert fields["output_text"] == "answer"
    assert fields["output_reasoning_source"].endswith(key + ".0.thinking")
    assert "signature" in fields["output_structured_content_json"]
    assert not fields.get("output_gaps")


def test_parallel_tools_in_blocks_preserve_numeric_text_and_explicit_false():
    blocks = [
        {"type": "thinking", "thinking": "想"},
        {"type": "text", "text": ""},
        {"type": "tool_use", "id": "a", "name": "f", "input": {"enabled": False, "n": 0}},
        {"type": "tool_call", "id": "b", "name": "g", "args": '{ "n": 1.2300 }'},
    ]
    fields = output_fields({"content": blocks}, finish_reason="tool_use")
    calls = loads(fields["output_tool_calls_json"])
    assert calls[0]["function"]["arguments"] == {"enabled": False, "n": 0}
    assert calls[1]["function"]["arguments"] == '{ "n": 1.2300 }'
    assert not fields.get("output_gaps")
    assert fields["output_finish_reason"] == "tool_use"


@pytest.mark.parametrize(
    "block, reason",
    [
        (
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}]},
            "reasoning_summary_only",
        ),
        ({"type": "redacted_thinking", "data": "opaque"}, "redacted_reasoning"),
        ({"type": "reasoning", "encrypted_content": "opaque"}, "redacted_reasoning"),
        ({"type": "thinking", "thinking": "redacted-by-litellm"}, "redacted_reasoning"),
        ({"type": "thinking", "thinking": 0}, "invalid_content_block_text"),
        (
            {"type": "image_url", "image_url": {"url": "https://never-fetch.invalid/pic"}},
            "output_images_not_supported",
        ),
        ({"type": []}, "invalid_content_block_type"),
    ],
)
def test_unavailable_block_keeps_other_observed_text(block, reason):
    fields = output_fields(
        {"content": [block, {"type": "text", "text": "available"}]}, finish_reason="stop"
    )
    assert fields["output_text"] == "available"
    assert fields["output_reasoning"] is None
    assert reason in fields["output_gaps"]
    assert loads(fields["output_structured_content_json"])["content"][0] == block


def test_interleaved_blocks_retain_their_order_for_the_selected_template():
    fields = output_fields(
        {
            "content": [
                {"type": "text", "text": "before"},
                {"type": "thinking", "thinking": "thought"},
                {"type": "text", "text": "after"},
            ]
        }
    )
    assert fields["output_text"] == "beforeafter"
    assert fields["output_reasoning"] == "thought"
    assert not fields.get("output_gaps")
    assert loads(fields["output_parts_json"]) == [
        {"type": "text", "text": "before"},
        {"type": "reasoning", "text": "thought"},
        {"type": "text", "text": "after"},
    ]


def test_duplicate_block_projection_is_checked_and_not_counted_twice():
    body = {
        "content": [
            {"type": "reasoning", "reasoning": "thought"},
            {"type": "text", "text": "answer"},
        ],
        "reasoning_content": "thought",
    }
    assert output_fields(body)["output_reasoning"] == "thought"
    body["reasoning_content"] = "different"
    fields = output_fields(body)
    assert fields["output_text"] == "answer" and fields["output_reasoning"] is None
    assert fields["output_gaps"] == ["conflicting_reasoning_fields"]


def test_text_only_blocks_and_dify_text_parts_are_normalized_without_separators():
    parsed = message(
        {"role": "user", "text": [{"type": "text", "data": "A\n"}, {"type": "text", "data": " B"}]},
        content_key="text",
    )
    assert parsed.content == "A\n B" and not parsed.gaps
