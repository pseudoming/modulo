"""Explicit offline fixture regeneration; never imports modulo's implementation.

HF text templates are transcribed below from the pinned upstream files. Kimi's
reviewed, hash-checked official segment builder supplies its independent XTML
reference. Native tokenizers/tiktoken are shared with production: this verifies
integration and rendering, not an independent segmentation algorithm.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

if __package__:
    from .assets import reference_assets, verify_reference_files
else:
    from assets import reference_assets, verify_reference_files

ROOT = Path(__file__).resolve().parents[3]


def qwen(messages):
    instruction = (
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
    )
    system = messages[0]["content"].strip() if messages[0]["role"] == "system" else ""
    result = "<|im_start|>system\n" + instruction
    result += ("\n\n" + system if system else "") + "<|im_end|>\n"
    for m in messages:
        if m["role"] == "system":
            continue
        result += "<|im_start|>" + m["role"] + "\n"
        if m["role"] == "assistant":
            result += "<think>\n\n</think>\n\n"
        result += m["content"].strip() + "<|im_end|>\n"
    return result + "<|im_start|>assistant\n<think>\n"


def deepseek(messages):
    merged = []
    for m in messages:
        if m["role"] == "user" and merged and merged[-1]["role"] == "user":
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    last_user = max(i for i, m in enumerate(merged) if m["role"] in {"user", "system"})
    result = (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
    for i, m in enumerate(merged):
        if m["role"] == "system":
            result += ("<｜System｜>" if i else "") + m["content"]
        elif m["role"] == "user":
            result += "<｜User｜>" + m["content"]
        else:
            result += "</think>" if i > last_user else ""
            result += m["content"] + "<｜end▁of▁sentence｜>"
        transition = i == len(merged) - 1 or merged[i + 1]["role"] == "assistant"
        if transition and (m["role"] == "user" or (m["role"] == "system" and i > 0)):
            result += "<｜Assistant｜>" + ("<think>" if i >= last_user else "</think>")
    return result


def minimax(messages):
    system = (
        "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026. "
        "Founded in early 2022, MiniMax is a global AI foundation model company committed to "
        "advancing the frontiers of AI towards AGI.\n\n<thinking_instructions>\n"
        "You have a thinking capability that allows you to reason step by step before responding. "
        "When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags before your "
        "response. When thinking is disabled, begin your response directly after the </mm:think> "
        "prefix. When thinking is adaptive, decide on your own whether to think for the current "
        "turn.\nCurrent thinking mode: adaptive. You are encouraged to think for complex "
        "decision-making, multi-step reasoning, or when analyzing function/tool results.\n"
        "</thinking_instructions>"
    )
    leading = messages[0]["role"] == "system"
    developer = (messages[0]["content"] if leading else "") or "You are a helpful assistant."
    result = "]~!b[]~b]system\n" + system + "[e~[\n]~b]developer\n" + developer + "[e~[\n"
    for m in messages[1:] if leading else messages:
        if m["role"] == "user":
            result += "]~b]user\n" + m["content"] + "[e~[\n"
        elif m["role"] == "assistant":
            result += "]~b]ai\n</mm:think>" + m["content"] + "[e~[\n"
    return result + "]~b]ai\n"


def longcat(messages):
    result = ""
    for i, m in enumerate(messages):
        result += "<longcat_" + m["role"] + ">"
        if m["role"] == "assistant":
            result += "<longcat_think>\n\n</longcat_think>\n" + m["content"] + "</longcat_s>"
        else:
            result += m["content"]
        if m["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            result += " /think_on "
    return result + " /think_on <longcat_assistant><longcat_think>"


def glm(messages):
    result = "[gMASK]<sop><|system|>Reasoning Effort: Max"
    for m in messages:
        result += "<|" + m["role"] + "|>"
        if m["role"] == "assistant":
            result += "<think></think>" + m["content"].strip()
        else:
            result += m["content"]
    return result + "<|assistant|><think>"


def mimo(messages):
    result = ""
    for m in messages:
        result += "<|im_start|>" + m["role"] + "\n"
        result += "<think></think>" if m["role"] == "assistant" else ""
        result += m["content"] + "<|im_end|>"
    return result + "<|im_start|>assistant\n"


def stepfun(messages):
    return (
        "<｜begin▁of▁sentence｜>"
        + "".join(
            "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages
        )
        + "<|im_start|>assistant\n<think>\n"
    )


def hunyuan(messages):
    start, middle, end = (
        "<｜hy_start:opensource｜>",
        "<｜hy_middle:opensource｜>",
        "<｜hy_end:opensource｜>",
    )
    prepared = (
        messages
        if messages[0]["role"] == "system"
        else [{"role": "system", "content": ""}, *messages]
    )
    result = ""
    for i, m in enumerate(prepared):
        result += start + m["role"] + middle
        if m["role"] == "assistant":
            result += "<think:opensource></think:opensource>"
        result += m["content"]
        if i == 0:
            result += "<｜reasoning_mode:opensource｜>reasoning_effort:high"
        result += end
    return result + start + "assistant" + middle + "<think:opensource>"


def ling(messages):
    leading = messages[0]["role"] == "system"
    system = messages[0]["content"] + "\n" if leading else ""
    result = "<role>SYSTEM</role>" + system + "detailed thinking on<|role_end|>"
    for m in messages[1:] if leading else messages:
        role = "HUMAN" if m["role"] == "user" else m["role"].upper()
        result += "<role>" + role + "</role>"
        result += "\n<think></think>" if m["role"] == "assistant" else ""
        result += m["content"] + "<|role_end|>"
    return result + "<role>ASSISTANT</role>\n<think>"


def ring(messages):
    leading = messages[0]["role"] == "system"
    result = "<role>SYSTEM</role>\nReasoning: high\n\n"
    if leading and messages[0]["content"]:
        result += messages[0]["content"] + "<|role_end|>\n\n"
    for m in messages[1:] if leading else messages:
        role = "HUMAN" if m["role"] == "user" else m["role"].upper()
        result += "<role>" + role + "</role>\n" + m["content"] + "<|role_end|>\n\n"
    return result + "<role>ASSISTANT</role>"


RENDERERS = {
    "qwen-3.8-2.4t": qwen,
    "deepseek-v4.1": deepseek,
    "minimax-m3": minimax,
    "longcat-2.0": longcat,
    "glm-5.3": glm,
    "mimo-v2.6": mimo,
    "step-3.7": stepfun,
    "hy4-preview": hunyuan,
    "ling-3.0": ling,
    "ring-2.6": ring,
}


def reviewed_module(path):
    name = "modulo_reference_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def official_template(directory, entry):
    config = json.loads((directory / "tokenizer_config.json").read_text())
    source = (
        (directory / entry["template_file"]).read_text()
        if entry.get("template_file")
        else config["chat_template"]
    )
    if entry.get("template_file", "").endswith(".py"):
        upstream = reviewed_module(directory / entry["template_file"])

        def render(messages, tools=None, add_generation_prompt=True, **kwargs):
            messages = deepcopy(messages)
            schema = kwargs.get("response_schema") or kwargs.get("response_format")
            if tools or schema:
                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": ""})
                messages[0].update(tools=tools, response_format=schema)
            for m in messages:
                for t in m.get("tool_calls", []):
                    if not isinstance(t["function"]["arguments"], str):
                        t["function"]["arguments"] = json.dumps(
                            t["function"]["arguments"], ensure_ascii=False
                        )
            options = {
                k: kwargs[k]
                for k in ("thinking_mode", "drop_thinking", "reasoning_effort")
                if k in kwargs
            }
            text = upstream.encode_messages(messages, **options)
            if not add_generation_prompt and messages[-1]["role"] in {"user", "tool", "developer"}:
                ending = "<｜Assistant｜>" + (
                    "<think>" if options.get("thinking_mode") == "thinking" else "</think>"
                )
                assert text.endswith(ending)
                text = text[: -len(ending)]
            return text

        return SimpleNamespace(render=render)
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = lambda v, **kw: json.dumps(
        v, ensure_ascii=kw.pop("ensure_ascii", False), **kw
    )
    env.filters["fromjson"] = json.loads

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    return env.from_string(source)


def kimi_reference(directory):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    # Explicit developer-only import of the reviewed official renderer. Production
    # never imports it; verify both runtime files and independent test references.
    verify_reference_files(directory, reference_assets()["assets"][directory.name])
    if (directory / "encoding_k3.py").is_file():
        module = reviewed_module(directory / "encoding_k3.py")
    else:
        tools_encoder = reviewed_module(directory / "tool_declaration_ts.py")
        render = official_template(directory, {"template_file": "chat_template.jinja"})

        def sorted_value(value):
            if isinstance(value, dict):
                return {k: sorted_value(value[k]) for k in sorted(value)}
            return [sorted_value(v) for v in value] if isinstance(value, list) else value

        def build_chat_segments(messages, tools=None, **kwargs):
            tools = sorted_value(tools)
            tools_ts = None
            if tools:
                try:
                    tools_ts = tools_encoder.encode_tools_to_typescript_style(tools)
                except Exception:
                    pass
            text = render.render(messages=messages, tools=tools, tools_ts_str=tools_ts, **kwargs)
            return [SimpleNamespace(text=text, allow_special=True)]

        module = SimpleNamespace(build_chat_segments=build_chat_segments)
    source = ast.parse((directory / "tokenization_kimi.py").read_text())
    tokenizer_class = next(n for n in source.body if isinstance(n, ast.ClassDef))
    pattern_node = next(
        n.value
        for n in tokenizer_class.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pat_str" for t in n.targets)
    )
    pattern = "|".join(ast.literal_eval(pattern_node.args[0]))
    ranks = load_tiktoken_bpe(str(directory / "tiktoken.model"))
    config = json.loads((directory / "tokenizer_config.json").read_text())
    decoder = config["added_tokens_decoder"]
    specials = {
        decoder.get(str(i), {}).get("content", f"<|reserved_token_{i}|>"): i
        for i in range(len(ranks), len(ranks) + 256)
    }
    encoder = tiktoken.Encoding(
        name="official-kimi-reference",
        pat_str=pattern,
        mergeable_ranks=ranks,
        special_tokens=specials,
    )
    return module, encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", help="Only replace these assets' frozen vectors")
    args = parser.parse_args()
    lock = reference_assets()
    selected = args.assets or list(lock["assets"])
    assert set(selected) <= lock["assets"].keys()
    inputs = json.loads((ROOT / "tests/fixtures/tokenizers/reference_inputs.json").read_text())
    inputs += [
        {
            "name": "edge_whitespace",
            "messages": [
                {"role": "system", "content": "  指令\n"},
                {"role": "user", "content": " \tCafe\u0301 café 🌍\r\n 123456789  "},
            ],
            "output": "  OK\n",
        },
        {
            "name": "consecutive_users",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ],
            "output": "done",
        },
        {
            "name": "literal_special_tokens",
            "messages": [
                {
                    "role": "user",
                    "content": "literal <|open|> <|sep|> <|end_of_msg|> <|im_start|> <think>",
                }
            ],
            "output": "<|open|>",
        },
    ]
    vectors = []
    for asset, entry in lock["assets"].items():
        if asset not in selected:
            continue
        directory = ROOT / "assets/tokenizers" / asset
        for name, expected in entry["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        if entry["family"] == "moonshot":
            upstream, encoder = kimi_reference(directory)
        else:
            tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            template = official_template(directory, entry)
            config = json.loads((directory / "tokenizer_config.json").read_text())
            special = {
                k: v["content"] if isinstance(v, dict) else v
                for k, v in config.items()
                if k.endswith("_token") and isinstance(v, (str, dict))
            }
        for item in inputs:
            if entry["family"] == "moonshot":
                segments = upstream.build_chat_segments(
                    item["messages"],
                    add_generation_prompt=True,
                    **entry["template_kwargs"],
                )
                rendered = "".join(s.text for s in segments)
                ids = [
                    i
                    for s in segments
                    for i in (
                        encoder.encode(s.text, allowed_special="all")
                        if s.allow_special
                        else encoder.encode(s.text, disallowed_special=())
                    )
                ]
                output_ids = encoder.encode(
                    item["output"],
                    **(
                        {"disallowed_special": ()}
                        if asset == "kimi-k3"
                        else {"allowed_special": "all"}
                    ),
                )
                method = "reviewed pinned official renderer + tiktoken"
            else:
                rendered = (
                    RENDERERS[asset](item["messages"])
                    if asset in RENDERERS
                    else template.render(
                        messages=item["messages"],
                        tools=None,
                        add_generation_prompt=True,
                        **special,
                        **entry["template_kwargs"],
                    )
                )
                ids = tokenizer.encode(rendered, add_special_tokens=False).ids
                output_ids = tokenizer.encode(item["output"], add_special_tokens=False).ids
                method = (
                    "manual text template + native tokenizers"
                    if asset in RENDERERS
                    else "pinned official renderer + native tokenizers"
                )
            vectors.append(
                {
                    **item,
                    "asset": asset,
                    "backend": entry["family"],
                    "rendered": rendered,
                    "input_ids": ids,
                    "input_tokens": len(ids),
                    "visible_output_tokens": len(output_ids),
                    "reference_method": method,
                    "asset_revision": entry["revision"],
                }
            )
    target = ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json"
    if args.assets:
        vectors = [
            v for v in json.loads(target.read_text())["vectors"] if v["asset"] not in selected
        ] + vectors
    asset_order = {asset: i for i, asset in enumerate(lock["assets"])}
    vectors.sort(key=lambda v: asset_order[v["asset"]])
    target.write_text(
        json.dumps(
            {
                "schema_version": "family-reference.v1",
                "limitation": "shares native encoders; not provider qualification",
                "vectors": vectors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"froze {len(vectors)} reference vectors")


if __name__ == "__main__":
    main()
