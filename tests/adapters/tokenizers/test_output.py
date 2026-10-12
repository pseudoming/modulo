"""Output references from pinned upstream templates/native encoders, not modulo.

Explicit plain-text spellings also anchor prefix and termination boundaries.
GLM stop IDs are from generation_config.json at its pinned asset revision:
154820=endoftext, 154827=user, 154829=observation (all one token).
"""

import copy
import json

import pytest
from tokenizers import Tokenizer

from modulo.adapters.protocols.shared.messages.candidates import (
    expected_candidates,
    response_outputs,
)
from modulo.adapters.protocols.shared.messages.chat import output_fields
from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Usage
from modulo.core.processing.recalculation import Recalculator
from modulo.shared.serialization import bytes_digest, digest, ordered_json
from tests.tools.references.build_family_reference import (
    kimi_reference,
    official_template,
    reference_assets,
)

ASSETS = reference_assets()["assets"]
# Transcribed from each fixed assistant format; no generation-prefix tokens or
# separators belonging to the next turn are included in these outputs.
PLAIN = {
    "qwen": "THOUGHT\n</think>\n\nANSWER<|im_end|>",
    "deepseek": "THOUGHT</think>ANSWER<｜end▁of▁sentence｜>",
    "minimax": "<mm:think>THOUGHT</mm:think>ANSWER[e~[",
    "longcat": "\nTHOUGHT\n</longcat_think>\nANSWER</longcat_s>",
    "glm": "THOUGHT</think>ANSWER<|endoftext|>",
    "mimo": "<think>THOUGHT</think>ANSWER<|im_end|>",
    "stepfun": "THOUGHT\n</think>\nANSWER<|im_end|>",
    "hunyuan": "THOUGHT</think:opensource>ANSWER<｜hy_end:opensource｜>",
    "ling": "THOUGHT</think>ANSWER<|role_end|>",
    "ring": "\n<think>\nTHOUGHT\n</think>\n\nANSWER<|role_end|>",
}
QWEN_TOOL = (
    "\n\n<tool_call>\n<function=lookup>\n<parameter=z>\n2\n</parameter>\n"
    "<parameter=a>\n中\n</parameter>\n</function>\n</tool_call>"
)
MODES = [(asset, {}, {}) for asset in ASSETS] + [
    ("deepseek-v4.1", {"enable_thinking": False}, {"thinking_mode": "chat"}),
    ("minimax-m3", {"thinking_mode": "disabled"}, {"thinking_mode": "disabled"}),
    ("minimax-m3", {"thinking_mode": "enabled"}, {"thinking_mode": "enabled"}),
    ("longcat-2.0", {"enable_thinking": False}, {"enable_thinking": False}),
    ("mimo-v2.6", {"enable_thinking": False}, {"enable_thinking": False}),
    ("hy4-preview", {"reasoning_effort": "no_think"}, {"reasoning_effort": "no_think"}),
    ("ling-3.0", {"thinking_option": "off"}, {"enable_thinking": False}),
    ("longcat-2.0", {"tool_choice": "required"}, {"tool_choice": "required"}),
    ("kimi-k3", {"tool_choice": "required"}, {"tool_choice": "required"}),
    ("qwen3.5-122b", {"enable_thinking": False}, {"enable_thinking": False}),
    ("qwen3.5-397b", {"enable_thinking": False}, {"enable_thinking": False}),
    ("deepseek-v4-pro-0813", {"enable_thinking": False}, {"thinking_mode": "chat"}),
    ("glm-5.2", {"thinking": {"type": "disabled"}}, {"enable_thinking": False}),
    ("kimi-k2.6", {"thinking": {"type": "disabled"}}, {"thinking": False}),
]


@pytest.fixture(scope="module")
def implementations():
    return {}


@pytest.fixture
def backends(official_assets, implementations):
    if not implementations:
        for asset in ASSETS:
            implementations[asset] = builtins().get("tokenizers", asset)(
                official_assets, load_tokenizer_assets()
            )
    return implementations


@pytest.fixture(scope="module")
def references():
    return {}


def reference(directory, entry, response, options, cache):
    family = entry["family"]
    if directory.name not in cache:
        if family == "moonshot":
            cache[directory.name] = kimi_reference(directory)
        else:
            config = json.loads((directory / "tokenizer_config.json").read_text())
            special = {
                k: v["content"] if isinstance(v, dict) else v
                for k, v in config.items()
                if k.endswith("_token") and isinstance(v, (str, dict))
            }
            native = Tokenizer.from_file(str(directory / "tokenizer.json"))
            native.no_padding()
            native.no_truncation()
            cache[directory.name] = official_template(directory, entry), native, special
    # Use a nonempty turn to ensure modulo's scaffold does not affect output.
    messages = [{"role": "user", "content": "独立参照"}]
    response = copy.deepcopy(response)
    responses = response if isinstance(response, list) else [response]
    if family == "moonshot":
        upstream, encoder = cache[directory.name]
        kwargs = {k: v for k, v in options.items() if k != "tool_choice"}
        prefix = upstream.build_chat_segments(messages, add_generation_prompt=True, **kwargs)
        if entry["loader"] == "tiktoken_kimi_k2":
            full = upstream.build_chat_segments(
                [*messages, *responses],
                add_generation_prompt=False,
                **{**kwargs, "preserve_thinking": True},
            )
            prefix_text = "".join(s.text for s in prefix)
            full_text = "".join(s.text for s in full)
            assert full_text.startswith(prefix_text)
            text = full_text[len(prefix_text) :]
            return text, encoder.encode(text, allowed_special="all")
        full = upstream.build_chat_segments(
            [*messages, *responses], add_generation_prompt=False, **kwargs
        )
        assert full[: len(prefix)] == prefix
        segments = full[len(prefix) :]
        text = "".join(s.text for s in segments)
        ids = [
            i
            for s in segments
            for i in encoder.encode(
                s.text,
                **({"allowed_special": "all"} if s.allow_special else {"disallowed_special": ()}),
            )
        ]
        return text, ids
    render, encoder, special = cache[directory.name]
    if family != "mimo":
        for item in responses:
            for tool in item.get("tool_calls", []):
                tool["function"]["arguments"] = json.loads(tool["function"]["arguments"])
    prefix = render.render(
        messages=messages, tools=None, add_generation_prompt=True, **special, **options
    )
    full = render.render(
        messages=[*messages, *responses],
        tools=None,
        add_generation_prompt=False,
        **special,
        **options,
    )
    if (
        family == "minimax"
        and options.get("thinking_mode", "enabled") == "enabled"
        and responses[0]["reasoning_content"] == ""
    ):
        prefix = prefix.removesuffix(
            "<think>\n" if entry["repo"].endswith("M2.7") else "<mm:think>"
        )
    assert full.startswith(prefix)
    text = full[len(prefix) :]
    if family in {"qwen", "minimax", "stepfun"}:
        assert text.endswith("\n")
        text = text[:-1]
    if family == "ring":
        assert text.endswith("\n\n")
        text = text[:-2]
    if family == "glm":
        text += "<|observation|>" if responses[-1].get("tool_calls") else "<|endoftext|>"
    return text, encoder.encode(text, add_special_tokens=False).ids


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_multiple_response_messages_count_real_boundaries_once(
    official_assets, backends, references, asset
):
    from modulo.adapters.protocols.shared.api_bodies.responses import response_fields
    from tests.support.api_bodies import response_body

    responses = [
        {
            "role": "assistant",
            "content": "ANSWER-" + str(i),
            "reasoning_content": "THOUGHT-" + str(i),
        }
        for i in (1, 2)
    ]
    body = response_body("responses", None)
    body["output"] = [
        item
        for r in responses
        for item in response_body("responses", r["content"], r["reasoning_content"])["output"]
    ]
    fields, _, _ = response_fields(body, "response")
    assert len(fields["output_messages"]) == 2
    text, ids = reference(
        official_assets / asset,
        ASSETS[asset],
        responses,
        ASSETS[asset]["template_kwargs"],
        references,
    )
    assert all(value in text for r in responses for value in (r["content"], r["reasoning_content"]))
    actual = backends[asset].count_output(
        Call(call_id="multi-message", protocol="synthetic", model=asset, **fields)
    )
    assert actual["output_execution"] == "completed", actual.get("output_reason")
    assert actual["output_ids_sha256"] == digest(ids)
    assert actual["rendered_output_sha256"] == bytes_digest(text.encode())


@pytest.mark.assets
def test_stop_inside_reconstructed_tool_markers_is_not_counted_twice(backends):
    call = Call(
        call_id="tool-stop",
        protocol="synthetic",
        model="qwen-3.8-2.4t",
        output_text="",
        output_reasoning="THOUGHT",
        output_tool_calls_json=ordered_json(
            [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"z":2}'},
                }
            ]
        ),
        output_finish_reason="stop",
        output_stop_reason="</tool_call>",
        output_stop_reason_source="response.stop_reason",
        generation_options={"stop": ["</tool_call>"]},
    )
    result = backends["qwen-3.8-2.4t"].count_output(call)
    assert result["output_execution"] == "partial"
    assert result["output_reason"] == "output_stop_boundary_ambiguous"


@pytest.mark.assets
@pytest.mark.parametrize("asset,logged,changes", MODES)
@pytest.mark.parametrize(
    "scenario",
    [
        "plain",
        "empty",
        "single_tools",
        "parallel_tools",
        "numeric_tools",
        "whitespace",
        "literal_tokens",
    ],
)
def test_complete_output_against_official_reference(
    official_assets, backends, references, asset, logged, changes, scenario
):
    entry = ASSETS[asset]
    options = {**entry["template_kwargs"], **changes}
    disabled = (
        options.get("enable_thinking") is False
        or options.get("thinking") is False
        or options.get("thinking_mode") in {"chat", "disabled"}
        or options.get("reasoning_effort") == "no_think"
    )
    response = {
        "role": "assistant",
        "content": "" if scenario == "empty" else "ANSWER",
        "reasoning_content": "" if disabled or scenario == "empty" else "THOUGHT",
    }
    if scenario == "whitespace":
        response.update(content="  café 🙂\n ", reasoning_content=" \n中 ")
        if disabled:
            response["reasoning_content"] = ""
    elif scenario == "literal_tokens":
        response["content"] = "<|open|> <|im_end|> <think> <mm:think>"
    if "tools" in scenario:
        arguments = (
            '{"z":1.2300,"a":{"n":2e2,"wide":9007199254740993,"flag":false,"none":null}}'
            if scenario == "numeric_tools"
            else '{"z":2,"a":"中"}'
        )
        response["tool_calls"] = [
            {
                "id": f"t-{i}",
                "type": "function",
                "function": {"name": "lookup", "arguments": arguments},
            }
            for i in range(1 if scenario == "single_tools" else 2)
        ]
    text, ids = reference(official_assets / asset, entry, response, options, references)
    if scenario == "plain" and not changes and entry["family"] != "moonshot":
        expected = (
            "THOUGHT\n</think>\n\nANSWER[e~[" if asset == "minimax-m2.7" else PLAIN[entry["family"]]
        )
        assert text == expected
    call = Call(
        call_id="output-reference",
        protocol="synthetic",
        model=asset,
        output_text=response["content"],
        output_reasoning=response["reasoning_content"],
        output_tool_calls_json=ordered_json(response["tool_calls"])
        if "tool_calls" in response
        else None,
        output_finish_reason="tool_calls" if "tools" in scenario else "stop",
        render_options=logged,
        usage=Usage(output_tokens=len(ids)),
    )
    binding = Binding(aliases=[asset], name=asset)
    calculator = Recalculator(
        Bindings(models=[binding]), lambda _: backends[asset], load_tokenizer_assets()
    )
    measured = calculator.calculate(call)
    assert measured.output_execution == "completed", measured.output_reason
    assert measured.output_tokens == len(ids)
    assert measured.output_ids_sha256 == digest(ids)
    assert measured.rendered_output_sha256 == bytes_digest(text.encode())
    assert measured.output_delta == 0 and measured.output_verdict == "CONSISTENT"
    assert measured.verdict == "UNVERIFIABLE"  # A missing request does not block output.
    if scenario == "single_tools" and asset == "qwen-3.8-2.4t" and not changes:
        assert text == PLAIN["qwen"].removesuffix("<|im_end|>") + QWEN_TOOL + "<|im_end|>"
        native = references["qwen-3.8-2.4t"][1]
        assert len(native.encode(QWEN_TOOL, add_special_tokens=False).ids) == 36
        assert (
            len(native.encode(text.removesuffix("<|im_end|>"), add_special_tokens=False).ids) == 44
        )
        assert measured.output_tokens == 45 and measured.tool_output_tokens == 36
    if disabled:
        omitted = calculator.calculate(call.model_copy(update={"output_reasoning": None}))
        assert omitted.output_tokens == measured.output_tokens
    if "tools" in scenario:
        assert measured.tool_output_tokens > 0
    for reported, expected_delta in [(len(ids) + 3, 3), (0, -len(ids))]:
        result = calculator.calculate(
            call.model_copy(update={"usage": Usage(output_tokens=reported)})
        )
        assert result.output_delta == expected_delta and result.output_verdict == "DEVIATION"
        assert result.input_verdict == "UNVERIFIABLE" and result.verdict == "DEVIATION"
    absent = calculator.calculate(call.model_copy(update={"usage": Usage()}))
    assert absent.output_delta is None and absent.output_reason == "reported_output_missing"
    assert absent.output_verdict == "UNVERIFIABLE"


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("finish", [None, "content_filter", "unexpected"])
def test_unconfirmed_or_truncated_ending_is_not_given_an_eos_or_verdict(backends, asset, finish):
    call = Call(
        call_id="partial",
        protocol="synthetic",
        model=asset,
        output_text="ANSWER",
        output_reasoning="THOUGHT",
        output_finish_reason=finish,
    )
    measured = backends[asset].count_output(call)
    assert measured["output_execution"] == "partial"
    assert measured["visible_output_tokens"] > 0 and measured["reasoning_output_tokens"] > 0
    assert "output_tokens" not in measured and "output_ids_sha256" not in measured


def unfinished_reference(directory, entry, references, *, channel="content", suffix=""):
    """Stop the official serialization inside a channel, before any closing tags."""
    response = {"role": "assistant", "content": "ANSWER", "reasoning_content": "THOUGHT"}
    text, _ = reference(directory, entry, response, entry["template_kwargs"], references)
    marker = response[channel]
    text = text[: text.index(marker) + len(marker)] + suffix
    if entry["family"] != "moonshot":
        native = references[directory.name][1]
        return text, native.encode(text, add_special_tokens=False).ids
    upstream, native = references[directory.name]
    if entry["loader"] == "tiktoken_kimi_k2":
        return text, native.encode(text, allowed_special="all")
    messages = [{"role": "user", "content": "独立参照"}]
    prefix = upstream.build_chat_segments(
        messages, add_generation_prompt=True, **entry["template_kwargs"]
    )
    full = upstream.build_chat_segments(
        [*messages, response], add_generation_prompt=False, **entry["template_kwargs"]
    )
    ids = []
    for segment in full[len(prefix) :]:
        value = segment.text + suffix if segment.text == marker else segment.text
        ids.extend(
            native.encode(
                value,
                **(
                    {"allowed_special": "all"}
                    if segment.allow_special
                    else {"disallowed_special": ()}
                ),
            )
        )
        if segment.text == marker:
            break
    return text, ids


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("channel", ["content", "reasoning_content"])
def test_token_limit_counts_actual_continuation_without_fabricated_closing_tags(
    official_assets, backends, references, asset, channel
):
    expected, ids = unfinished_reference(
        official_assets / asset, ASSETS[asset], references, channel=channel
    )
    call = Call(
        call_id="limit",
        protocol="synthetic",
        output_reasoning="THOUGHT",
        output_text="ANSWER" if channel == "content" else None,
        output_finish_reason="length",
        generation_options={"max_tokens": 100},
    )
    result = backends[asset].count_output(call)
    assert result["output_execution"] == "completed", result
    assert result["output_tokens"] == len(ids)
    assert result["output_ids_sha256"] == digest(ids)
    assert result["rendered_output_sha256"] == bytes_digest(expected.encode())
    assert result["output_ending"] == ""
    assert result["output_termination"] == "token_limit_no_end_marker"


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("included", [False, True])
def test_logged_stop_sequence_is_counted_once_and_never_gets_an_eos(
    official_assets, backends, references, asset, included
):
    text, ids = unfinished_reference(
        official_assets / asset, ASSETS[asset], references, suffix="END"
    )
    result = backends[asset].count_output(
        Call(
            call_id="stop",
            protocol="synthetic",
            output_reasoning="THOUGHT",
            output_text="ANSWEREND" if included else "ANSWER",
            output_finish_reason="stop",
            output_stop_reason="END",
            output_stop_reason_source="response.stop_reason",
            generation_options={"stop": ["END"], "include_stop_str_in_output": included},
        )
    )
    assert result["output_execution"] == "completed", result
    assert result["output_tokens"] == len(ids)
    assert result["output_ids_sha256"] == digest(ids)
    assert result["rendered_output_sha256"] == bytes_digest(text.encode())
    assert result["output_ending"] == ("" if included else "END")


@pytest.mark.assets
def test_stop_ambiguity_does_not_turn_the_request_into_unverifiable(backends):
    backend = backends["qwen-3.8-2.4t"]
    call = Call(
        call_id="stopping",
        protocol="synthetic",
        model="qwen-3.8-2.4t",
        messages=[{"role": "user", "content": "hello"}],
        output_text="ANSWER",
        output_reasoning="THOUGHT",
        output_finish_reason="stop",
        generation_options={"stop": ["END"]},
    )
    binding = Binding(aliases=["qwen-3.8-2.4t"], name="qwen-3.8-2.4t")
    assert backend.count(call, binding)["input_tokens"] > 0
    count = backend.count_output(call)
    assert count["output_reason"] == "output_stop_reason_missing" and "output_tokens" not in count
    explicit_eos = call.model_copy(update={"output_stop_reason_source": "response.stop_reason"})
    assert backend.count_output(explicit_eos)["output_execution"] == "completed"
    invalid = call.model_copy(update={"generation_gaps": ["conflicting_generation_option:stop"]})
    assert backend.count(invalid, binding)["input_tokens"] > 0
    assert backend.count_output(invalid)["output_reason"] == "conflicting_generation_option:stop"


@pytest.mark.assets
def test_stop_token_id_is_checked_in_the_selected_vocabulary(backends, official_assets, references):
    _, ids = unfinished_reference(
        official_assets / "qwen-3.8-2.4t", ASSETS["qwen-3.8-2.4t"], references
    )
    backend = backends["qwen-3.8-2.4t"]
    token = backend.tokenizer.eos_token_id
    call = Call(
        call_id="token",
        protocol="synthetic",
        output_text="ANSWER",
        output_reasoning="THOUGHT",
        output_finish_reason="stop",
        output_stop_reason=token,
        output_stop_reason_source="choice.stop_reason",
        generation_options={"stop_token_ids": [token]},
    )
    result = backend.count_output(call)
    assert result["output_ids_sha256"] == digest([*ids, token])
    assert result["output_termination"] == "logged_stop_token"
    invalid = call.model_copy(update={"output_stop_reason": 10**9})
    assert backend.count_output(invalid)["output_reason"] == "unknown_output_stop_token_id"


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_missing_reasoning_still_counts_complete_tool_fragment(backends, asset):
    base = Call(
        call_id="tool-only",
        protocol="synthetic",
        output_text="",
        output_finish_reason="tool_calls",
        output_tool_calls_json=ordered_json(
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"x":0}'},
                }
            ]
        ),
    )
    result = backends[asset].count_output(base)
    assert result["output_execution"] == "partial"
    assert result["output_reason"] == "missing_output_reasoning"
    assert result["reasoning_output_tokens"] is None
    assert result["tool_output_tokens"] > 0
    assert "output_tokens" not in result and "output_body_tokens" not in result


@pytest.mark.assets
@pytest.mark.parametrize("asset", ["kimi-k2.6", "kimi-k2.7-code"])
def test_kimi_k2_cannot_invent_tool_ids_required_by_its_template(backends, asset):
    result = backends[asset].count_output(
        Call(
            call_id="missing-id",
            protocol="synthetic",
            output_text="",
            output_reasoning="THOUGHT",
            output_finish_reason="tool_calls",
            output_tool_calls_json=ordered_json(
                [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
            ),
        )
    )
    assert result["output_reason"] == "missing_tool_call_id"
    assert result["output_execution"] == "partial"
    assert "output_tokens" not in result


@pytest.mark.assets
@pytest.mark.parametrize(
    "gaps,output_gaps",
    [
        (["conflicting_request_option:reasoning_effort"], []),
        ([], ["conflicting_finish_reason"]),
        ([], ["truncated_output"]),
    ],
)
def test_conflicting_or_incomplete_evidence_never_produces_full_output(backends, gaps, output_gaps):
    call = Call(
        call_id="conflict",
        protocol="synthetic",
        output_text="ANSWER",
        output_reasoning="THOUGHT",
        output_finish_reason="stop",
        gaps=gaps,
        output_gaps=output_gaps,
    )
    result = backends["qwen-3.8-2.4t"].count_output(call)
    assert result["output_execution"] == "partial"
    assert result["visible_output_tokens"] > 0 and result["reasoning_output_tokens"] > 0
    assert result["output_reason"] == (gaps + output_gaps)[0]
    assert "output_tokens" not in result


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("kind", ["thinking", "reasoning", "reasoning_field"])
def test_structured_thoughts_use_the_same_official_output_reference(
    official_assets, backends, references, asset, kind
):
    plain = {"role": "assistant", "content": "ANSWER", "reasoning_content": "THOUGHT"}
    if kind == "reasoning_field":
        body = {"content": "ANSWER", "reasoning": [{"type": "reasoning_text", "text": "THOUGHT"}]}
    else:
        body = {
            "content": [
                {"type": kind, kind: "THOUGHT", "signature": "do-not-tokenize"},
                {"type": "text", "text": "ANSWER"},
            ]
        }
    call = Call(call_id="blocks", protocol="synthetic", **output_fields(body, finish_reason="stop"))
    entry = ASSETS[asset]
    rendered, ids = reference(
        official_assets / asset, entry, plain, entry["template_kwargs"], references
    )
    count = backends[asset].count_output(call)
    assert count["output_execution"] == "completed"
    assert count["output_tokens"] == len(ids)
    assert count["output_ids_sha256"] == digest(ids)
    assert count["rendered_output_sha256"] == bytes_digest(rendered.encode())


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("incomplete", [False, True])
def test_candidates_are_counted_separately_before_request_level_comparison(
    official_assets, backends, references, asset, incomplete
):
    entry = ASSETS[asset]
    responses = [
        {"role": "assistant", "content": text, "reasoning_content": "THOUGHT"}
        for text in ("ANSWER", "另一个答案\n")
    ]
    expected = [
        reference(official_assets / asset, entry, response, entry["template_kwargs"], references)[1]
        for response in responses
    ]
    fields = response_outputs(
        {
            "choices": [
                {
                    "index": index,
                    "message": body,
                    "finish_reason": None if incomplete and index == 1 else "stop",
                }
                for index, body in enumerate(responses)
            ]
        }
    )
    binding = Binding(aliases=[asset], name=asset)
    calculator = Recalculator(
        Bindings(models=[binding]), lambda _: backends[asset], load_tokenizer_assets()
    )
    call = Call(
        call_id="candidates",
        protocol="synthetic",
        model=asset,
        usage=Usage(output_tokens=sum(map(len, expected)) + 3),
        **fields,
    )
    result = calculator.calculate(call)
    assert result.output_candidates[0].tokens == len(expected[0])
    assert result.output_candidates[0].ids_sha256 == digest(expected[0])
    assert result.output_candidates[1].tokens == (None if incomplete else len(expected[1]))
    assert result.output_tokens == (None if incomplete else sum(map(len, expected)))
    assert result.output_verdict == ("UNVERIFIABLE" if incomplete else "DEVIATION")
    assert result.output_delta == (None if incomplete else 3)
    assert result.execution == "not_run" and result.reason == "missing_messages"


@pytest.mark.assets
def test_missing_candidate_blocks_total_but_retains_observed_candidate_counts(backends):
    fields = expected_candidates(
        response_outputs(
            {
                "choices": [
                    {
                        "index": index,
                        "message": {"content": "ANSWER", "reasoning_content": "THOUGHT"},
                        "finish_reason": "stop",
                    }
                    for index in (0, 1)
                ]
            }
        ),
        ("request", {"n": 3}),
    )
    binding = Binding(aliases=["qwen"], name="qwen-3.8-2.4t")
    calculator = Recalculator(
        Bindings(models=[binding]), lambda _: backends["qwen-3.8-2.4t"], load_tokenizer_assets()
    )
    count = calculator.calculate(
        Call(call_id="missing", protocol="synthetic", model="qwen", **fields)
    )
    assert count.output_execution == "partial" and count.output_tokens is None
    assert all(item.execution == "completed" for item in count.output_candidates)
    assert "output_candidate_count_mismatch" in count.output_reason
