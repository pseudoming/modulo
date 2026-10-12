"""Request constraints and configured templates against native token encoders."""

import copy
import json

import pytest
from tokenizers import Tokenizer

from modulo.adapters.protocols.shared.messages.chat import message, output_fields, request_options
from modulo.application.factory import make_calculator
from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Message
from modulo.core.domain.interfaces import AssetError, UnsupportedInput
from modulo.shared.serialization import bytes_digest, digest, dumps, loads, ordered_json
from tests.tools.references.build_family_reference import (
    kimi_reference,
    official_template,
    reference_assets,
)

ASSETS = reference_assets()["assets"]
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Lookup.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"z": {"type": "number"}, "a": {"type": "boolean"}},
            "required": ["z", "a"],
            "additionalProperties": False,
        },
    },
}
CONSTRAINTS = [
    {"response_format": {"type": "json_object"}},
    {"response_format": SCHEMA},
    {"parallel_tool_calls": False, "tool_choice": "auto"},
    {"tool_choice": "none"},
    {"tool_choice": "required"},
    {"tool_choice": {"type": "function", "function": {"name": "lookup"}}},
    {"add_generation_prompt": False},
    {"add_generation_prompt": True},
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


def reference_input(directory, entry, constraints, cache):
    family = entry["family"]
    messages = [{"role": "user", "content": "Return an answer."}]
    tools = [copy.deepcopy(TOOL)]
    options = {"add_generation_prompt": True, **entry["template_kwargs"], **constraints}
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
            cache[directory.name] = (
                official_template(directory, entry),
                Tokenizer.from_file(str(directory / "tokenizer.json")),
                special,
            )
    if family == "moonshot":
        upstream, encoder = cache[directory.name]
        # The fixed upstream Python accepts named arguments rather than **kwargs.
        supplied = {
            k: v
            for k, v in options.items()
            if k
            in {
                "add_generation_prompt",
                "thinking_effort",
                "tool_choice",
                "thinking",
                "preserve_thinking",
            }
        }
        segments = upstream.build_chat_segments(messages, tools=tools, **supplied)
        text = "".join(s.text for s in segments)
        ids = [
            i
            for s in segments
            for i in encoder.encode(
                s.text,
                **({"allowed_special": "all"} if s.allow_special else {"disallowed_special": ()}),
            )
        ]
    else:
        template, encoder, special = cache[directory.name]
        text = template.render(messages=messages, tools=tools, **special, **options)
        ids = encoder.encode(text, add_special_tokens=False).ids
    return text, ids


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
@pytest.mark.parametrize("constraints", CONSTRAINTS)
def test_request_constraints_use_fixed_template_without_inventing_schema_prompts(
    official_assets, backends, references, asset, constraints
):
    call = Call(
        call_id="parameters",
        protocol="synthetic",
        model=asset,
        messages=[Message(role="user", content="Return an answer.")],
        tools_json=ordered_json([TOOL]),
        render_options=constraints,
    )
    text, ids = reference_input(official_assets / asset, ASSETS[asset], constraints, references)
    actual = backends[asset].count(call, Binding(aliases=[asset], name=asset))
    assert actual["input_tokens"] == len(ids)
    assert actual["input_ids_sha256"] == digest(ids)
    assert actual["rendered_input_sha256"] == bytes_digest(text.encode())
    assert all(actual["render_options"][k] == v for k, v in constraints.items())


@pytest.mark.assets
@pytest.mark.parametrize("asset", ["kimi-k2.6", "kimi-k2.7-code"])
@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {
                "z": {"type": ["string", "null"], "enum": ["中", None], "default": "中"},
                "a": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 9,
                        "description": "值\n\n说明",
                    },
                    "minItems": 1,
                    "maxItems": 3,
                },
                "optional": {"type": "boolean", "default": False},
            },
            "required": ["z", "a"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "$defs": {
                "Point": {
                    "type": "object",
                    "description": "坐标",
                    "properties": {"x": {"type": "integer"}},
                }
            },
            "properties": {"point": {"$ref": "#/$defs/Point"}},
        },
        {
            "type": "object",
            "properties": {"next": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}, "flag": True},
        },
        {
            "type": "object",
            "properties": {"x": {"type": ["array", "object", "null"]}},
            "additionalProperties": {"type": "string", "maxLength": 12},
        },
        {"type": "object", "properties": {"x": {"$ref": "https://example.invalid/schema"}}},
        {"type": "object", "properties": {"x": {"type": "integer", "enum": ["invalid"]}}},
    ],
)
def test_kimi_tool_schema_matches_native_converter_including_json_fallback(
    official_assets, backends, references, asset, schema
):
    entry = ASSETS[asset]
    if asset not in references:
        references[asset] = kimi_reference(official_assets / asset)
    upstream, encoder = references[asset]
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "查找\n\n值", "parameters": schema},
        }
    ]
    messages = [{"role": "user", "content": "查询"}]
    text = upstream.build_chat_segments(
        messages, tools=tools, add_generation_prompt=True, **entry["template_kwargs"]
    )[0].text
    actual = backends[asset].count(
        Call(
            call_id="tool-schema",
            protocol="synthetic",
            messages=messages,
            tools_json=ordered_json(tools),
        ),
        Binding(aliases=[asset], name=asset),
    )
    assert actual["rendered_input_sha256"] == bytes_digest(text.encode())
    assert actual["input_ids_sha256"] == digest(encoder.encode(text, allowed_special="all"))


@pytest.mark.assets
@pytest.mark.parametrize(
    "options,reason",
    [
        ({"parallel_tool_calls": "false"}, "invalid_render_option:parallel_tool_calls"),
        ({"response_format": {"type": "unknown"}}, "invalid_render_option:response_format"),
        (
            {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
            "tool_choice_name_not_in_tools",
        ),
        (
            {"chat_template": "{{ raise_exception('execute log data') }}"},
            "logged_chat_template_not_selected_in_binding",
        ),
    ],
)
def test_invalid_parameters_do_not_silently_fall_back(backends, options, reason):
    call = Call(
        call_id="invalid-parameters",
        protocol="synthetic",
        model="qwen-3.8-2.4t",
        messages=[Message(role="user", content="x")],
        tools_json=ordered_json([TOOL]),
        render_options=options,
    )
    with pytest.raises(UnsupportedInput, match=reason):
        backends["qwen-3.8-2.4t"].count(
            call, Binding(aliases=["qwen-3.8-2.4t"], name="qwen-3.8-2.4t")
        )


TEMPLATE = (
    "{% for m in messages %}[{{ m.role }}]"
    "{% if m.content_blocks is defined %}{% for b in m.content_blocks %}"
    "{% if b.type == 'text' %}{{ b.text }}"
    "{% elif b.type == 'reasoning' %}<r>{{ b.text }}</r>"
    "{% elif b.type == 'tool_calls' %}{% for t in b.tool_calls %}"
    "<t>{{ t.function.name }}:{{ t.function.arguments }}</t>{% endfor %}"
    "{% endif %}{% endfor %}"
    "{% else %}{{ m.content }}{% endif %}{{ eos_token }}{% endfor %}"
    "{% if add_generation_prompt %}[assistant]{% endif %}"
)
BLOCKS = [
    {"type": "text", "text": "first"},
    {"type": "thinking", "thinking": "thought"},
    {"type": "tool_call", "id": "t1", "name": "lookup", "args": '{"z":2,"a":1}'},
    {"type": "text", "text": "last"},
]
ORDERED_TEXT = 'first<r>thought</r><t>lookup:{"z":2,"a":1}</t>last'


@pytest.mark.assets
@pytest.mark.parametrize("asset", ASSETS)
def test_configured_template_preserves_interleaving_in_input_output_and_replay(
    official_assets, asset
):
    binding = Binding(aliases=[asset], name=asset, **{"chat-template": TEMPLATE})
    calculator = make_calculator(
        builtins(), official_assets, Bindings(models=[binding]), load_tokenizer_assets()
    )
    parsed = message({"role": "assistant", "content": BLOCKS})
    call = Call(
        call_id="ordered",
        protocol="synthetic",
        model=asset,
        messages=[
            Message(role="user", content="ask"),
            parsed,
            Message(role="user", content="continue"),
        ],
        **output_fields({"content": BLOCKS, "finish_reason": "stop"}),
    )
    entry, directory = ASSETS[asset], official_assets / asset
    if entry["family"] == "moonshot":
        _, encoder = kimi_reference(directory)

        def encode(text):
            return encoder.encode(text, allowed_special="all")

        eos = "<|im_end|>" if entry["loader"] == "tiktoken_kimi_k2" else "<|end_of_msg|>"
    else:
        encoder = Tokenizer.from_file(str(directory / "tokenizer.json"))

        def encode(text):
            return encoder.encode(text, add_special_tokens=False).ids

        eos = json.loads((directory / "tokenizer_config.json").read_text())["eos_token"]
        if isinstance(eos, dict):
            eos = eos["content"]
    expected_input = f"[user]ask{eos}[assistant]{ORDERED_TEXT}{eos}[user]continue{eos}[assistant]"
    expected_output = ORDERED_TEXT + eos
    actual = calculator.calculate(call)
    assert actual.execution == "completed", actual.reason
    assert actual.output_execution == "completed", actual.output_reason
    assert actual.input_ids_sha256 == digest(encode(expected_input))
    assert actual.output_ids_sha256 == digest(encode(expected_output))
    assert actual.template_hash == bytes_digest(TEMPLATE.encode())
    restored = Call.model_validate(loads(dumps(call.model_dump())))
    assert calculator.calculate(restored) == actual
    # A token limit leaves the last body block open; it does not invent an EOS.
    limited = calculator.calculate(call.model_copy(update={"output_finish_reason": "length"}))
    assert limited.output_ids_sha256 == digest(encode(ORDERED_TEXT))


@pytest.mark.assets
def test_order_mismatch_requires_a_template_that_reads_the_ordered_blocks(backends):
    call = Call(
        call_id="interleaved",
        protocol="synthetic",
        model="qwen-3.8-2.4t",
        **output_fields({"content": BLOCKS, "finish_reason": "stop"}),
    )
    result = backends["qwen-3.8-2.4t"].count_output(call)
    assert result["output_execution"] == "partial"
    assert result["output_reason"] == "content_order_not_supported_by_template"


@pytest.mark.assets
def test_configured_template_cache_and_ordered_parameters_are_kept_separate(official_assets):
    template = "{{ response_schema | tojson }}{% for m in messages %}{{ m.content }}{% endfor %}"
    schema = loads('{"z":{"value":1.2300},"a":false}')
    bindings = Bindings(
        models=[
            Binding(
                aliases=["first"],
                name="qwen-3.8-2.4t",
                **{"chat-template": template, "template-kwargs": {"response_schema": schema}},
            ),
            Binding(
                aliases=["second"],
                name="qwen-3.8-2.4t",
                **{
                    "chat-template": "extra" + template,
                    "template-kwargs": {"response_schema": schema},
                },
            ),
        ]
    )
    restored = Bindings.model_validate(loads(dumps(bindings.model_dump(by_alias=True))))
    assert restored.models[0].reference_key == bindings.models[0].reference_key
    assert list(restored.models[0].template_kwargs["response_schema"]) == ["z", "a"]
    calculator = make_calculator(builtins(), official_assets, restored, load_tokenizer_assets())
    native = Tokenizer.from_file(str(official_assets / "qwen-3.8-2.4t/tokenizer.json"))
    for model, prefix in [("first", ""), ("second", "extra"), ("first", "")]:
        call = Call(
            call_id=model,
            protocol="synthetic",
            model=model,
            messages=[Message(role="user", content="x")],
        )
        measured = calculator.calculate(call)
        expected = prefix + '{"z": {"value": 1.2300}, "a": false}x'
        assert measured.input_ids_sha256 == digest(
            native.encode(expected, add_special_tokens=False).ids
        )
        assert measured.render_option_sources["response_schema"] == ["binding.template-kwargs"]
    options = request_options(("request", {"response_schema": schema}))
    call = Call(
        call_id="logged-order",
        protocol="synthetic",
        model="first",
        messages=[Message(role="user", content="x")],
        **options,
    )
    assert calculator.calculate(
        Call.model_validate(loads(dumps(call.model_dump())))
    ) == calculator.calculate(call)


@pytest.mark.assets
@pytest.mark.parametrize("template", ["{{ strftime_now('%Y') }}", "{% invalid %}"])
def test_bad_custom_template_is_rejected_without_using_log_code(official_assets, template):
    backend = builtins().get("tokenizers", "qwen-3.8-2.4t")(
        official_assets, load_tokenizer_assets()
    )
    with pytest.raises(AssetError):
        backend.configure_reference(
            Binding(aliases=["x"], name="qwen-3.8-2.4t", **{"chat-template": template})
        )


@pytest.mark.assets
def test_custom_template_never_fills_a_missing_parameter_with_empty_text(official_assets):
    binding = Binding(
        aliases=["x"], name="qwen-3.8-2.4t", **{"chat-template": "{{ unrecorded_parameter }}"}
    )
    calculator = make_calculator(
        builtins(), official_assets, Bindings(models=[binding]), load_tokenizer_assets()
    )
    measured = calculator.calculate(
        Call(
            call_id="missing-option",
            protocol="synthetic",
            model="x",
            messages=[Message(role="user", content="x")],
        )
    )
    assert measured.execution == "error"
    assert measured.input_tokens is None


@pytest.mark.assets
def test_binding_generation_prefix_source_and_logged_override(official_assets):
    binding = Binding(aliases=["x"], name="qwen-3.8-2.4t", add_generation_prompt=False)
    calculator = make_calculator(
        builtins(), official_assets, Bindings(models=[binding]), load_tokenizer_assets()
    )
    call = Call(
        call_id="prefix",
        protocol="synthetic",
        model="x",
        messages=[Message(role="user", content="x")],
    )
    measured = calculator.calculate(call)
    assert measured.render_options["add_generation_prompt"] is False
    assert measured.render_option_sources["add_generation_prompt"] == [
        "binding.add_generation_prompt"
    ]
    override = calculator.calculate(
        call.model_copy(update={"render_options": {"add_generation_prompt": True}})
    )
    assert override.input_tokens > measured.input_tokens
    assert override.render_option_sources["add_generation_prompt"] == ["call.add_generation_prompt"]
