import copy
import json

import pytest

from modulo.adapters.protocols.shared.api_bodies.chat_stream import assemble_sse
from modulo.adapters.protocols.shared.messages.candidates import (
    expected_candidates,
    response_outputs,
)


def choice(index, text, *, finish="stop"):
    return {
        "index": index,
        "message": {"content": text, "reasoning_content": "thought"},
        "finish_reason": finish,
    }


def test_terminal_choices_keep_their_indices_and_request_count():
    raw = {"choices": [choice(1, "B"), choice(0, "A")]}
    before = copy.deepcopy(raw)
    fields = expected_candidates(response_outputs(raw), ("request", {"n": 2}))
    assert [(item["index"], item["output_text"]) for item in fields["output_candidates"]] == [
        (0, "A"),
        (1, "B"),
    ]
    assert fields["expected_output_candidates"] == 2
    assert fields["expected_output_candidates_sources"] == ["request.n"]
    assert not fields.get("output_gaps") and "output_text" not in fields
    assert raw == before


@pytest.mark.parametrize(
    "choices, reason",
    [
        ([choice(0, "A"), choice(0, "B")], "duplicate_output_choice_index"),
        ([choice(0, "A"), choice(2, "B")], "output_choice_index_gap"),
        ([choice(0, "A"), choice(True, "B")], "invalid_output_choice_index"),
        ([choice(0, "A"), {"message": {"content": "B"}}], "invalid_output_choice_index"),
        ([choice(0, "A"), None], "invalid_output_choice"),
    ],
)
def test_invalid_candidate_identity_does_not_invent_an_index(choices, reason):
    fields = response_outputs({"choices": choices})
    assert reason in fields["output_gaps"]


@pytest.mark.parametrize(
    "root, reason",
    [
        ({"n": 3}, "output_candidate_count_mismatch"),
        ({"n": 0}, "invalid_requested_output_count"),
        ({"n": False}, "invalid_requested_output_count"),
        ({"n": 2, "extra_body": {"n": 1}}, "conflicting_requested_output_count"),
    ],
)
def test_request_count_and_conflicts_are_retained(root, reason):
    fields = expected_candidates(
        response_outputs({"choices": [choice(0, "A"), choice(1, "B")]}), ("request", root)
    )
    assert reason in fields["output_gaps"]
    assert len(fields["output_candidates"]) == 2


def stream(events, *, done=True):
    lines = []
    for event in events:
        lines.extend(["data: " + json.dumps(event, ensure_ascii=False), ""])
    if done:
        lines.extend(["data: [DONE]", ""])
    return lines


def delta(index, body, finish=None):
    return {"index": index, "delta": body, "finish_reason": finish}


def events():
    return [
        {
            "id": "r",
            "choices": [
                delta(1, {"reasoning_content": "think-B"}),
                delta(0, {"thinking": "think-A"}),
            ],
        },
        {
            "id": "r",
            "choices": [
                delta(0, {"content": "A"}),
                delta(
                    1,
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "t",
                                "function": {"name": "f", "arguments": '{ "n": '},
                            }
                        ]
                    },
                ),
            ],
        },
        {
            "id": "r",
            "choices": [
                delta(
                    1,
                    {"tool_calls": [{"index": 0, "function": {"arguments": "1.2300 }"}}]},
                    "tool_calls",
                )
            ],
        },
        {"id": "r", "choices": [delta(0, {"content": "A"}, "stop")]},
        {"id": "r", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 25}},
    ]


def test_interleaved_stream_candidates_do_not_mix_text_tools_or_usage():
    result = assemble_sse(stream(events()))
    assert result.done and not result.output.get("output_gaps")
    first, second = result.output["output_candidates"]
    assert first["output_text"] == "AA" and first["output_reasoning"] == "think-A"
    assert second["output_text"] == "" and second["output_reasoning"] == "think-B"
    assert (
        json.loads(second["output_tool_calls_json"])[0]["function"]["arguments"]
        == '{ "n": 1.2300 }'
    )
    assert (
        first["output_finish_reason"] == "stop" and second["output_finish_reason"] == "tool_calls"
    )
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 25


def test_one_unfinished_candidate_does_not_hide_the_other_candidate():
    data = events()
    data[2]["choices"][0]["finish_reason"] = None
    result = assemble_sse(stream(data))
    assert result.output["output_gaps"] == ["incomplete_output_candidates"]
    first, second = result.output["output_candidates"]
    assert not first.get("output_gaps")
    assert "stream_finish_missing" in second["output_gaps"]


def test_global_stream_truncation_is_not_hidden_by_complete_candidate_finishes():
    result = assemble_sse(stream(events(), done=False))
    assert "stream_done_missing" in result.output["output_gaps"]
    assert all(item["output_finish_reason"] for item in result.output["output_candidates"])


def test_stream_blocks_keep_reasoning_and_text_separate():
    result = assemble_sse(
        stream(
            [
                {
                    "choices": [
                        delta(
                            0,
                            {
                                "content": [
                                    {"type": "thinking", "thinking": "think", "signature": "sig"}
                                ]
                            },
                        )
                    ]
                },
                {"choices": [delta(0, {"content": [{"type": "text", "text": "answer"}]}, "stop")]},
            ]
        )
    )
    assert result.output["output_text"] == "answer"
    assert result.output["output_reasoning"] == "think"
    assert not result.output.get("output_gaps")


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"index": False}, "invalid_stream_choice_index"),
        ({"index": 3}, "output_choice_index_gap"),
    ],
)
def test_stream_invalid_candidate_indexes(changes, reason):
    data = events()
    data[0]["choices"][0].update(changes)
    assert reason in assemble_sse(stream(data)).output["output_gaps"]


def test_duplicate_choice_in_one_event_is_not_concatenated():
    data = [{"choices": [delta(0, {"content": "A"}, "stop"), delta(0, {"content": "B"})]}]
    fields = assemble_sse(stream(data)).output
    assert fields["output_text"] == "A"
    assert "duplicate_stream_choice_index" in fields["output_gaps"]
