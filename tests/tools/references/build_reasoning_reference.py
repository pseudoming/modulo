"""Freeze reasoning/tool fixtures directly from reviewed official assets.

This developer-only generator never imports modulo. HF uses the pinned official
Jinja template and raw tokenizer.json; Kimi uses the hash-checked upstream Python
segment builder. It validates adapter wiring, not an independent BPE algorithm.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from build_family_reference import kimi_reference, official_template, reference_assets
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
DS_REFERENCE_SHA256 = "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"
DS_REFERENCE_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/"
    "2cba9e42aa026125f3ed06c6d98c1db82f7ca027/encoding/encoding.py"
)
REASONING = " 先核对条件，再计算：2 + 3 = 5。\n验证 e\u0301🙂。 "
ANSWER = "结果是 5。\n"
HISTORY = [
    {"role": "system", "content": "请根据事实回答。"},
    {"role": "user", "content": "2 + 3 等于多少？"},
    {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING},
    {"role": "user", "content": "请继续核对。"},
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "查询数值。",
            "parameters": {
                "type": "object",
                "properties": {"z": {"type": "integer"}, "a": {"type": "string"}},
                "required": ["z", "a"],
            },
        },
    }
]

# Each pair is (logged options, expected official renderer kwargs). These are
# explicit source-based expectations, not copied from the production resolver.
MODES = {
    "qwen": [({"reasoning_effort": e}, {"reasoning_effort": e}) for e in ("low", "medium", "xhigh")]
    + [
        ({"preserve_thinking": False}, {"preserve_thinking": False}),
        ({"enable_thinking": True}, {"enable_thinking": True}),
    ],
    "deepseek": [
        ({"reasoning_effort": e}, {"reasoning_effort": e}) for e in (1, 100, "low", "medium", "max")
    ]
    + [
        ({"thinking": {"type": "disabled"}}, {"thinking_mode": "chat"}),
        ({"enable_thinking": True}, {"thinking_mode": "thinking"}),
        ({"drop_thinking": False}, {"drop_thinking": False}),
    ],
    "moonshot": [({"reasoning_effort": e}, {"thinking_effort": e}) for e in ("low", "high", "max")],
    "minimax": [
        ({"thinking": {"type": e}}, {"thinking_mode": e})
        for e in ("enabled", "adaptive", "disabled")
    ],
    "longcat": [
        ({"enable_thinking": False}, {"enable_thinking": False}),
        ({"thinking_budget": 0}, {"thinking_budget": 0}),
        ({"thinking_budget": 512}, {"thinking_budget": 512}),
        ({"thinking_budget": 4096}, {"thinking_budget": 4096}),
        ({"save_history_reasoning_content": False}, {"save_history_reasoning_content": False}),
    ],
    "glm": [
        ({"reasoning_effort": e}, {"reasoning_effort": e}) for e in ("low", "high", "max", "medium")
    ]
    + [({"clear_thinking": True}, {"clear_thinking": True})],
    "mimo": [
        ({"enable_thinking": enabled}, {"enable_thinking": enabled}) for enabled in (True, False)
    ],
    "stepfun": [
        ({"reasoning_effort": e}, {"reasoning_effort": e}) for e in ("low", "medium", "high")
    ],
    "hunyuan": [
        ({"reasoning_effort": "no_think"}, {"reasoning_effort": "no_think"}),
        ({"preserved_thinking": True}, {"preserved_thinking": True}),
    ],
    "ling": [
        ({"enable_thinking": enabled}, {"enable_thinking": enabled}) for enabled in (True, False)
    ]
    + [({"thinking_option": "off"}, {"enable_thinking": False})],
    "ring": [({"reasoning_effort": e}, {"reasoning_effort": e}) for e in ("high", "xhigh")],
}
ASSET_MODES = {
    "qwen3.5-122b": [({"enable_thinking": v}, {"enable_thinking": v}) for v in (True, False)],
    "qwen3.5-397b": [({"enable_thinking": v}, {"enable_thinking": v}) for v in (True, False)],
    "deepseek-v4-pro-0813": [
        ({"reasoning_effort": v}, {"reasoning_effort": v}) for v in ("low", "high", "max")
    ]
    + [
        ({"enable_thinking": False}, {"thinking_mode": "chat"}),
        ({"drop_thinking": False}, {"drop_thinking": False}),
    ],
    "minimax-m2.7": [({"thinking": {"type": "enabled"}}, {"thinking_mode": "enabled"})],
    "glm-5.2": MODES["glm"]
    + [
        ({"thinking": {"type": "disabled"}}, {"enable_thinking": False}),
        ({"clear_thinking": False}, {"clear_thinking": False}),
    ],
    "kimi-k2.6": [
        ({"thinking": {"type": "disabled"}}, {"thinking": False}),
        ({"preserve_thinking": True}, {"preserve_thinking": True}),
    ],
    "kimi-k2.7-code": [({"preserve_thinking": True}, {"preserve_thinking": True})],
}


def tool_messages(parallel=False):
    calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"z":2,"a":"中"}'},
        }
    ]
    results = [{"role": "tool", "tool_call_id": "call-1", "name": "lookup", "content": "5"}]
    if parallel:
        calls.append(
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"z":3,"a":"文"}'},
            }
        )
        results.insert(
            0, {"role": "tool", "tool_call_id": "call-2", "name": "lookup", "content": "6"}
        )
    return [
        *copy.deepcopy(HISTORY[:2]),
        {"role": "assistant", "content": "", "reasoning_content": REASONING, "tool_calls": calls},
        *results,
    ]


def numeric_tool_messages():
    messages = tool_messages()
    messages[2]["tool_calls"][0]["function"]["arguments"] = (
        '{"z":1.2300,"a":{"scale":2e2,"wide":9007199254740993,"flag":true,"none":null}}'
    )
    return messages


def fail(message):
    raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepseek-reference",
        type=Path,
        help="Reviewed upstream encoding.py; exact SHA-256 is required",
    )
    parser.add_argument("--assets", nargs="+", help="Only replace these assets' frozen vectors")
    args = parser.parse_args()
    lock = reference_assets()
    selected = args.assets or list(lock["assets"])
    assert set(selected) <= lock["assets"].keys()
    if "deepseek-v4.1" in selected:
        assert args.deepseek_reference, "--deepseek-reference is required for deepseek-v4.1"
        assert (
            hashlib.sha256(args.deepseek_reference.read_bytes()).hexdigest() == DS_REFERENCE_SHA256
        )
        spec = importlib.util.spec_from_file_location(
            "reviewed_deepseek_reference", args.deepseek_reference
        )
        ds_reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ds_reference)
    vectors = []
    gateway_reference = None
    for asset, entry in lock["assets"].items():
        if asset not in selected:
            continue
        directory = ROOT / "assets/tokenizers" / asset
        for name, expected in entry["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        family = entry["family"]
        cases = [
            ("history", HISTORY, None, {}, {}),
            ("tool_continuation", tool_messages(), TOOLS, {}, {}),
            ("parallel_tools", tool_messages(True), TOOLS, {}, {}),
            (
                "numeric_tool_arguments",
                numeric_tool_messages(),
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "查询合成数值。",
                            "parameters": {
                                "type": "object",
                                "properties": {"z": {"type": "number"}, "a": {"type": "object"}},
                            },
                        },
                    }
                ],
                {},
                {},
            ),
            (
                "tool_history_then_user",
                [
                    *tool_messages(),
                    {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING},
                    {"role": "user", "content": "请重新核对。"},
                ],
                TOOLS,
                {},
                {},
            ),
        ]
        for i, (logged, expected) in enumerate(ASSET_MODES.get(asset, MODES[family])):
            cases.append((f"mode_{i}", HISTORY, None, logged, expected))
        if family == "moonshot":
            upstream, encoder = kimi_reference(directory)

            def encode(text, encoder=encoder, k3=asset == "kimi-k3"):
                return encoder.encode(
                    text, **({"disallowed_special": ()} if k3 else {"allowed_special": "all"})
                )
        else:
            encoder = Tokenizer.from_file(str(directory / "tokenizer.json"))
            encoder.no_padding()
            encoder.no_truncation()

            def encode(text, encoder=encoder):
                return encoder.encode(text, add_special_tokens=False).ids

            config = json.loads((directory / "tokenizer_config.json").read_text())
            render = official_template(directory, entry)
            special = {
                k: v["content"] if isinstance(v, dict) else v
                for k, v in config.items()
                if k.endswith("_token") and isinstance(v, (str, dict))
            }
        if asset == "qwen-3.8-2.4t":
            row = json.loads((ROOT / "examples/litellm/basic.jsonl").read_text().splitlines()[7])
            prompt = render.render(
                messages=row["messages"],
                tools=row["model_parameters"]["tools"],
                add_generation_prompt=True,
                **special,
                **entry["template_kwargs"],
            )
            gateway_reference = {
                "call_id": row["id"],
                "rendered": prompt,
                "input_ids": encode(prompt),
            }
        for name, messages, tools, logged, changes in cases:
            options = {**entry["template_kwargs"], **changes}
            rendering_messages = copy.deepcopy(messages)
            if family not in {"moonshot", "mimo"}:
                for message in rendering_messages:
                    for call in message.get("tool_calls", []):
                        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
            if family == "moonshot":
                segments = upstream.build_chat_segments(
                    rendering_messages, tools=tools, add_generation_prompt=True, **options
                )
                rendered = "".join(s.text for s in segments)
                ids = [
                    i
                    for s in segments
                    for i in (
                        encoder.encode(s.text, allowed_special="all")
                        if s.allow_special
                        else encode(s.text)
                    )
                ]
            else:
                rendered = render.render(
                    messages=rendering_messages,
                    tools=tools,
                    add_generation_prompt=True,
                    **special,
                    **options,
                )
                if asset == "deepseek-v4.1":
                    canonical_messages = copy.deepcopy(messages)
                    if tools:
                        canonical_messages[0]["tools"] = tools
                    canonical_options = options.copy()
                    # The pinned Jinja extension names numeric effort 62 "medium";
                    # upstream Python accepts that numeric budget, not this alias.
                    if canonical_options.get("reasoning_effort") == "medium":
                        canonical_options["reasoning_effort"] = 62
                    official = ds_reference.encode_messages(canonical_messages, **canonical_options)
                    assert official == rendered, f"DeepSeek Jinja/canonical disagreement: {name}"
                    rendered = official
                ids = encode(rendered)
            vectors.append(
                {
                    "name": name,
                    "asset": asset,
                    "backend": family,
                    "asset_revision": entry["revision"],
                    "messages": messages,
                    "tools": tools,
                    "logged_options": logged,
                    "template_options": options,
                    "rendered": rendered,
                    "input_ids": ids,
                    "output_reasoning": REASONING,
                    "output_text": ANSWER,
                    "reasoning_output_tokens": len(encode(REASONING)),
                    "visible_output_tokens": len(encode(ANSWER)),
                }
            )
    target = ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json"
    if args.assets:
        previous = json.loads(target.read_text())
        vectors = [v for v in previous["vectors"] if v["asset"] not in selected] + vectors
        gateway_reference = gateway_reference or previous["gateway_tool_reference"]
    asset_order = {asset: i for i, asset in enumerate(lock["assets"])}
    vectors.sort(key=lambda v: asset_order[v["asset"]])
    target.write_text(
        json.dumps(
            {
                "schema_version": "reasoning-reference.v1",
                "reference": (
                    "pinned official templates/XTML + native encoders; no production imports"
                ),
                "deepseek_canonical_reference": {
                    "url": DS_REFERENCE_URL,
                    "sha256": DS_REFERENCE_SHA256,
                },
                "gateway_tool_reference": gateway_reference,
                "vectors": vectors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"froze {len(vectors)} vectors for {len(lock['assets'])} assets")


if __name__ == "__main__":
    main()
