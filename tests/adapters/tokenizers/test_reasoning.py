import copy
import json

import pytest

from modulo.adapters.protocols.shared.messages.chat import message
from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Message, Usage
from modulo.core.domain.interfaces import UnsupportedInput
from modulo.core.processing.recalculation import Recalculator
from modulo.shared.serialization import bytes_digest, digest, dumps, loads
from tests.support.paths import ROOT

VECTORS = json.loads(
    (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
)["vectors"]


ASSETS = {v["asset"]: v["backend"] for v in VECTORS}


@pytest.fixture(scope="module")
def backend_cache():
    return {}


@pytest.fixture
def backends(official_assets, backend_cache):
    if not backend_cache:
        registry = builtins()
        backend_cache.update(
            {
                asset: registry.get("tokenizers", asset)(official_assets, load_tokenizer_assets())
                for asset in ASSETS
            }
        )
    return backend_cache


def call_for(vector):
    messages = []
    for original in vector["messages"]:
        body = copy.deepcopy(original)
        if "tool_calls" in body:
            body["tool_calls_json"] = json.dumps(body.pop("tool_calls"), ensure_ascii=False)
        messages.append(Message(**body))
    return Call(
        call_id="reasoning-reference",
        protocol="synthetic",
        model=vector["asset"],
        messages=messages,
        tools_json=json.dumps(vector["tools"]) if vector["tools"] else None,
        render_options=vector["logged_options"],
        output_text=vector["output_text"],
        output_reasoning=vector["output_reasoning"],
        output_reasoning_source="response.reasoning_content",
        usage=Usage(input_tokens=len(vector["input_ids"])),
    )


@pytest.mark.assets
@pytest.mark.parametrize("vector", VECTORS, ids=[v["asset"] + ":" + v["name"] for v in VECTORS])
def test_every_family_against_frozen_official_reasoning_and_tool_vectors(backends, vector):
    call = call_for(vector)
    binding = Binding(aliases=[vector["asset"]], name=vector["asset"])
    backend = backends[vector["asset"]]
    measured = {**backend.count(call, binding), **backend.count_output(call)}
    assert measured["input_tokens"] == len(vector["input_ids"])
    assert measured["input_ids_sha256"] == digest(vector["input_ids"])
    assert measured["rendered_input_sha256"] == bytes_digest(vector["rendered"].encode())
    assert measured["reasoning_output_tokens"] == vector["reasoning_output_tokens"]
    assert measured["visible_output_tokens"] == vector["visible_output_tokens"]
    assert measured["render_options"] == vector["template_options"]
    # JSONL's sorted keys must not alter tool parameter order on replay.
    restored = Call.model_validate(loads(dumps(call.model_dump())))
    assert {**backend.count(restored, binding), **backend.count_output(restored)} == measured


@pytest.mark.assets
@pytest.mark.parametrize("vector", VECTORS, ids=[v["asset"] + ":" + v["name"] for v in VECTORS])
def test_block_messages_match_frozen_input_vectors_without_regenerating_answers(backends, vector):
    messages = []
    for original in vector["messages"]:
        body = copy.deepcopy(original)
        blocks = []
        if body.get("reasoning_content") is not None:
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": body.pop("reasoning_content"),
                    "signature": "metadata-only",
                }
            )
        blocks.append({"type": "text", "text": body["content"]})
        body["content"] = blocks
        messages.append(message(body))
    call = call_for(vector).model_copy(update={"messages": messages})
    measured = backends[vector["asset"]].count(
        call, Binding(aliases=[vector["asset"]], name=vector["asset"])
    )
    assert measured["input_tokens"] == len(vector["input_ids"])
    assert measured["input_ids_sha256"] == digest(vector["input_ids"])
    assert measured["rendered_input_sha256"] == bytes_digest(vector["rendered"].encode())


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_output_missing_empty_and_whitespace_reasoning_are_distinct(backends, asset):
    backend = backends[asset]
    base = Call(call_id="output", protocol="synthetic", output_text="答案")
    missing = backend.count_output(base)
    empty = backend.count_output(base.model_copy(update={"output_reasoning": ""}))
    spaces = backend.count_output(base.model_copy(update={"output_reasoning": " \n "}))
    assert missing["reasoning_output_tokens"] is None
    assert empty["reasoning_output_tokens"] == 0
    assert spaces["reasoning_output_tokens"] > 0
    assert missing["visible_output_tokens"] == empty["visible_output_tokens"]


@pytest.mark.assets
@pytest.mark.parametrize(
    "asset,options",
    [
        ("qwen-3.8-2.4t", {"enable_thinking": False}),
        ("qwen-3.8-2.4t", {"reasoning_effort": "high"}),
        ("deepseek-v4.1", {"reasoning_effort": 0}),
        ("deepseek-v4.1", {"reasoning_effort": 101}),
        ("deepseek-v4.1", {"reasoning_effort": True}),
        ("kimi-k3", {"thinking": False}),
        ("kimi-k3", {"reasoning_effort": "medium"}),
        ("minimax-m3", {"thinking_mode": "invalid"}),
        ("longcat-2.0", {"thinking_budget": -1}),
        ("glm-5.3", {"reasoning_effort": 42}),
        ("mimo-v2.6", {"enable_thinking": "false"}),
        ("step-3.7", {"reasoning_effort": False}),
        ("hy4-preview", {"reasoning_effort": "low"}),
        ("ling-3.0", {"enable_thinking": 0}),
        ("ring-2.6", {"reasoning_effort": "low"}),
    ],
)
def test_invalid_model_specific_mode_has_specific_reason_without_losing_output(
    backends, asset, options
):
    vector = next(v for v in VECTORS if v["asset"] == asset)
    call = call_for(vector).model_copy(update={"render_options": options})
    bindings = Bindings(models=[Binding(aliases=[asset], name=asset)])
    calculator = Recalculator(bindings, lambda _: backends[asset], load_tokenizer_assets())
    result = calculator.calculate(call)
    assert result.execution == "not_run" and result.input_verdict == "UNVERIFIABLE"
    assert result.reason.startswith("invalid_render_option:")
    assert result.output_execution == "partial" and result.reasoning_output_tokens > 0


@pytest.mark.assets
def test_input_and_output_gaps_do_not_block_each_other(backends):
    vector = next(v for v in VECTORS if v["asset"] == "deepseek-v4.1")
    base = call_for(vector)
    binding = Binding(aliases=[base.model], name="deepseek-v4.1")
    calculator = Recalculator(
        Bindings(models=[binding]), lambda _: backends[base.model], load_tokenizer_assets()
    )
    no_output = calculator.calculate(
        base.model_copy(
            update={
                "output_text": None,
                "output_reasoning": None,
                "output_gaps": ["missing_output"],
            }
        )
    )
    assert no_output.input_verdict == "CONSISTENT" and no_output.output_execution == "not_run"
    assert no_output.verdict == "UNVERIFIABLE"
    no_input = calculator.calculate(
        base.model_copy(update={"messages": None, "gaps": ["missing_messages"]})
    )
    assert (
        no_input.input_tokens is None
        and no_input.reasoning_output_tokens == vector["reasoning_output_tokens"]
    )
    assert no_input.input_verdict == "UNVERIFIABLE"


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_complete_inline_reasoning_and_truncated_output(backends, asset):
    backend = backends[asset]
    start, end = backend.inline_think_tags
    call = Call(call_id="inline", protocol="synthetic", output_text=start + "推理\n" + end + "答案")
    inline = backend.count_output(call)
    separate = backend.count_output(
        call.model_copy(update={"output_text": "答案", "output_reasoning": "推理\n"})
    )
    assert inline["reasoning_output_tokens"] == separate["reasoning_output_tokens"]
    assert inline["visible_output_tokens"] == separate["visible_output_tokens"]
    partial = backend.count_output(call.model_copy(update={"output_text": start + "未完成"}))
    assert partial["output_reason"] == "incomplete_inline_reasoning"
    assert partial["output_execution"] == "partial"
    assert partial["reasoning_output_tokens"] > 0
    assert partial["visible_output_tokens"] is None
    assert "output_tokens" not in partial


@pytest.mark.assets
def test_conflicting_aliases_and_unknown_options_are_not_silently_defaulted(backends):
    backend = backends["deepseek-v4.1"]
    call = Call(
        call_id="conflict",
        protocol="synthetic",
        messages=[Message(role="user", content="x")],
        render_options={"enable_thinking": False, "thinking_mode": "thinking"},
    )
    with pytest.raises(UnsupportedInput, match="conflicting_render_option"):
        backend.count(call, Binding(aliases=["deepseek-v4.1"], name="deepseek-v4.1"))
    with pytest.raises(UnsupportedInput, match="unsupported_render_option:unknown_mode"):
        backend.resolve_options(call.model_copy(update={"render_options": {"unknown_mode": True}}))
