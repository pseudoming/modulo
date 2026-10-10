"""Explicit offline fixture regeneration; never imports modulo's implementation.

HF text templates are transcribed below from the pinned upstream files. Kimi's
reviewed, hash-checked official segment builder supplies its independent XTML
reference. Native tokenizers/tiktoken are shared with production: this verifies
integration and rendering, not an independent segmentation algorithm.
"""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

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
    "qwen": qwen,
    "deepseek": deepseek,
    "minimax": minimax,
    "longcat": longcat,
    "glm": glm,
    "mimo": mimo,
    "stepfun": stepfun,
    "hunyuan": hunyuan,
    "ling": ling,
    "ring": ring,
}


def kimi_reference(directory):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    # Explicit developer-only import of the reviewed official renderer. Production
    # never imports this file. Caller verifies every file against the shipped lock.
    name = "modulo_reference_official_kimi_k3"
    spec = importlib.util.spec_from_file_location(name, directory / "encoding_k3.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
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
    lock = json.loads((ROOT / "src/modulo/application/assets-lock.json").read_text())
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
        directory = ROOT / "assets/tokenizers" / asset
        for name, expected in entry["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        if entry["family"] == "moonshot":
            upstream, encoder = kimi_reference(directory)
        else:
            tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            tokenizer.no_padding()
            tokenizer.no_truncation()
        for item in inputs:
            if entry["family"] == "moonshot":
                segments = upstream.build_chat_segments(
                    item["messages"],
                    add_generation_prompt=True,
                    thinking=True,
                    thinking_effort="max",
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
                output_ids = encoder.encode(item["output"], disallowed_special=())
                method = "reviewed pinned official XTML segments + tiktoken"
            else:
                rendered = RENDERERS[entry["family"]](item["messages"])
                ids = tokenizer.encode(rendered, add_special_tokens=False).ids
                output_ids = tokenizer.encode(item["output"], add_special_tokens=False).ids
                method = "manual text template + native tokenizers"
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
