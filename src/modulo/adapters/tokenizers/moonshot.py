"""Kimi K3 text-only XTML reference, without importing upstream Python.

The pinned official Python remains evidence, never imported at audit runtime.
This renderer follows its text/reasoning/function XTML segment boundaries.
"""

import base64
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from modulo.core.domain.bindings import Binding
from modulo.core.domain.data_models import Call
from modulo.core.domain.interfaces import AssetError, TokenizerBackend
from modulo.shared.serialization import bytes_digest, digest, loads, ordered_json

from .shared.assets import verified_asset
from .shared.rendering import RenderProfile, template_json

_PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


def _tag(name: str, attributes: tuple[tuple[str, str], ...] = (), *, close: bool = False):
    parts = [("<|close|>" if close else "<|open|>", True), (name, False)]
    for key, value in attributes:
        value = str(value).replace("&", "&amp;").replace('"', "&quot;")
        parts.extend([(f" {key}", False), ('="', False), (value, False), ('"', False)])
    return [*parts, ("<|sep|>", True)]


def _sort(value):
    if isinstance(value, dict):
        return {key: _sort(value[key]) for key in sorted(value)}
    return [_sort(v) for v in value] if isinstance(value, list) else value


def _argument_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float, Decimal)):
        return "number"
    return "object" if isinstance(value, dict) else "array"


def _arguments(arguments):
    if isinstance(arguments, dict):
        return [
            (k, _argument_type(v), v if isinstance(v, str) else ordered_json(v))
            for k, v in arguments.items()
        ], None
    if arguments == "":
        return [], None
    try:
        # Validate strictly first; then retain original literal spelling of each value.
        parsed = loads(arguments)
        if not isinstance(parsed, dict):
            return [], arguments
    except ValueError:
        return [], arguments
    decoder = json.JSONDecoder(parse_float=Decimal)
    pos, result = arguments.index("{") + 1, []
    for _ in parsed:
        while arguments[pos].isspace() or arguments[pos] == ",":
            pos += 1
        key, pos = decoder.raw_decode(arguments, pos)
        while arguments[pos].isspace() or arguments[pos] == ":":
            pos += 1
        start = pos
        value, pos = decoder.raw_decode(arguments, pos)
        result.append(
            (key, _argument_type(value), value if isinstance(value, str) else arguments[start:pos])
        )
    return result, None


def _segments(call: Call, options, messages):
    parts = []
    if call.tools_json:
        tools = _sort(template_json(call.tools_json))
        if tools:
            parts.extend(_tag("message", (("role", "system"), ("type", "tool-declare"))))
            parts.append(
                (
                    "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n"
                    + ordered_json(tools, separators=(",", ":"))
                    + "\n```",
                    False,
                )
            )
            parts.extend([*_tag("message", close=True), ("<|end_of_msg|>", True)])
    effort = options["thinking_effort"]
    parts.extend(_tag("message", (("role", "system"), ("type", "thinking-effort"))))
    parts.append(
        (
            "`thinking_effort` guides on how much to think in your "
            "thinking channel (not including the response channel), "
            "supported values include `low`, `medium`, `high`, and `max`.\n"
            f"Now the system is invoked with `thinking_effort={effort}`.",
            False,
        )
    )
    parts.extend([*_tag("message", close=True), ("<|end_of_msg|>", True)])
    # Match official K3: consecutive tool results are ordered by the preceding calls.
    ordered, index, position = [], {}, 0
    while position < len(messages):
        item = messages[position]
        if item["role"] == "assistant":
            index = {c.get("id"): i for i, c in enumerate(item.get("tool_calls", []))}
        if item["role"] == "tool":
            group = []
            while position < len(messages) and messages[position]["role"] == "tool":
                group.append(messages[position])
                position += 1
            ordered.extend(sorted(group, key=lambda m: index[m["tool_call_id"]]))
        else:
            ordered.append(item)
            position += 1
    tool_index = 0
    for message in ordered:
        role = message["role"]
        attrs = (("role", role),)
        if role == "tool":
            tool_index += 1
            attrs += (("tool", message["name"]), ("index", str(tool_index)))
        elif message.get("name"):
            attrs += (("name", message["name"]),)
        parts.extend(_tag("message", attrs))
        if role == "assistant":
            tool_index = 0
            parts.extend(_tag("think"))
            parts.append((message.get("reasoning_content", ""), False))
            parts.extend([*_tag("think", close=True), *_tag("response")])
        parts.append((message["content"], False))
        if role == "assistant":
            parts.extend(_tag("response", close=True))
            calls = message.get("tool_calls")
            if calls:
                parts.extend(_tag("tools"))
                for i, tool in enumerate(calls, 1):
                    function = tool["function"]
                    parts.extend(_tag("call", (("tool", function["name"]), ("index", str(i)))))
                    arguments, raw = _arguments(function["arguments"])
                    if raw is not None:
                        parts.extend(
                            [
                                *_tag("json", (("type", "object"),)),
                                (raw, False),
                                *_tag("json", close=True),
                            ]
                        )
                    else:
                        for key, kind, value in arguments:
                            parts.extend(
                                [
                                    *_tag("argument", (("key", key), ("type", kind))),
                                    (value, False),
                                    *_tag("argument", close=True),
                                ]
                            )
                    parts.extend(_tag("call", close=True))
                parts.extend(_tag("tools", close=True))
        parts.extend([*_tag("message", close=True), ("<|end_of_msg|>", True)])
    choice = options.get("tool_choice")
    if choice in {"required", "none"}:
        parts.extend(_tag("message", (("role", "system"), ("type", "tool-choice"))))
        parts.append(
            (
                f"The system is invoked with `tool_choice={choice}`.\n"
                + (
                    "You MUST call tools in the next message."
                    if choice == "required"
                    else "You MUST NOT call any tools in the next message."
                ),
                False,
            )
        )
        parts.extend([*_tag("message", close=True), ("<|end_of_msg|>", True)])
    parts.extend([*_tag("message", (("role", "assistant"),)), *_tag("think")])
    return parts


def _pieces(text: str):
    # Match upstream's bounded encoder calls; never truncate the request.
    for offset in range(0, len(text), 400_000):
        chunk = text[offset : offset + 400_000]
        start, run = 0, 0
        previous_space = None
        for index, char in enumerate(chunk):
            space = char.isspace()
            run = run + 1 if space == previous_space else 1
            previous_space = space
            if run > 25_000:
                yield chunk[start:index]
                start, run = index, 1
        yield chunk[start:]


class MoonshotTokenizer(RenderProfile, TokenizerBackend):
    name = "moonshot"
    version = "kimi-k3.reasoning-xtml.v2"
    option_rules = {
        "thinking": (True,),
        "thinking_effort": ("low", "high", "max"),
        "tool_choice": ("auto", "none", "required"),
    }
    option_aliases = {"reasoning_effort": "thinking_effort", "enable_thinking": "thinking"}
    parse_tool_arguments = False
    named_messages = True

    def __init__(self, root: Path, asset: str, lock: dict[str, Any]) -> None:
        directory, entry = verified_asset(root, asset, lock, "moonshot", ("moonshotai/Kimi-K3",))
        if entry.get("loader") != "tiktoken_kimi_k3" or entry.get("template_kwargs") != {
            "thinking": True,
            "thinking_effort": "max",
        }:
            raise AssetError("asset_profile_unsupported")
        import tiktoken

        config = loads((directory / "tokenizer_config.json").read_bytes())
        ranks = {}
        for line in (directory / "tiktoken.model").read_bytes().splitlines():
            token, rank = line.split()
            ranks[base64.b64decode(token, validate=True)] = int(rank)
        mapping = config["added_tokens_decoder"]
        special = {
            mapping.get(str(i), {}).get("content", f"<|reserved_token_{i}|>"): i
            for i in range(len(ranks), len(ranks) + 256)
        }
        self.tokenizer = tiktoken.Encoding(
            name="modulo-kimi-k3", pat_str=_PATTERN, mergeable_ranks=ranks, special_tokens=special
        )
        self.entry, self.asset_hash = entry, digest(entry)
        self.template_kwargs = entry["template_kwargs"]
        self.template_hash = bytes_digest((directory / "encoding_k3.py").read_bytes())

    def _encode(self, text: str, special: bool = False) -> list[int]:
        result = []
        for piece in _pieces(text):
            result.extend(
                self.tokenizer.encode(piece, allowed_special="all")
                if special
                else self.tokenizer.encode(piece, disallowed_special=())
            )
        return result

    def count(self, call: Call, binding: Binding) -> dict[str, Any]:
        options, sources = self.resolve_options(call)
        parts = _segments(call, options, self.chat_messages(call))
        ids = [token for text, special in parts for token in self._encode(text, special)]
        rendered = "".join(text for text, _ in parts)
        return {
            "input_tokens": len(ids),
            "input_ids_sha256": digest(ids),
            "rendered_input_sha256": bytes_digest(rendered.encode()),
            **self.count_output(call),
            **self.option_evidence(options, sources),
            "asset_revision": self.entry["revision"],
            "asset_hash": self.asset_hash,
            "template_hash": self.template_hash,
        }

    def count_output(self, call: Call) -> dict[str, Any]:
        return self.output_counts(call, self._encode)
