"""Kimi K3 counting and XTML boundaries from the pinned official segment builder."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from modulo.core.domain.data_models import Call
from modulo.core.domain.interfaces import AssetError
from modulo.shared.serialization import bytes_digest, loads, ordered_json

from ..shared.rendering import template_json
from .moonshot import MoonshotTokenizer


class KimiK3Tokenizer(MoonshotTokenizer):
    name = "kimi-k3"
    repo = "moonshotai/Kimi-K3"
    loader = "tiktoken_kimi_k3"
    end_token = "<|end_of_msg|>"
    encode_special = False
    option_rules = {
        "thinking": (True,),
        "thinking_effort": ("low", "high", "max"),
        "tool_choice": ("auto", "none", "required"),
    }
    option_aliases = {"reasoning_effort": "thinking_effort", "enable_thinking": "thinking"}

    def __init__(self, root: Path, lock: dict[str, Any]) -> None:
        super().__init__(root, lock)
        if self.template_kwargs != {"thinking": True, "thinking_effort": "max"}:
            raise AssetError("asset_profile_unsupported")
        # K3 renders with the local implementation below. The official Python
        # encoder is an independent test reference, not a runtime dependency.
        self.template_source = Path(__file__).read_text(encoding="utf-8")
        self.template_hash = bytes_digest(self.template_source.encode())

    def render_input(self, call, options, messages):
        return segments(call, options, messages)

    def image_prompt(self, width, height):
        return f"<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>"

    @staticmethod
    def literal_output(text):
        return [(text, False)]

    def render_output(self, response, options):
        if self.custom_template:
            body, ending = super().render_output(response, options)
            return [(body, True)], [(ending, True)]
        responses = response if isinstance(response, list) else [response]
        parts = []
        ending = [*_tag("message", close=True), ("<|end_of_msg|>", True)]
        for index, item in enumerate(responses):
            body = assistant_body(item)
            if index == 0 and options.get("add_generation_prompt", True):
                body = body[len(_tag("think")) :]
            else:
                body = [*_tag("message", (("role", "assistant"),)), *body]
            parts.extend([*body, *ending])
        return parts[: -len(ending)], ending


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


def _content_parts(content):
    if isinstance(content, str):
        return [(content, False)]
    return [
        (part["prompt"], True) if part["type"] == "image" else (part["text"], False)
        for part in content
    ]


def assistant_body(message):
    parts = []
    parts.extend(_tag("think"))
    parts.append((message.get("reasoning_content", ""), False))
    parts.extend([*_tag("think", close=True), *_tag("response")])
    parts.extend(_content_parts(message["content"]))
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
    return parts


def segments(call: Call, options, messages):
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
            parts.extend(assistant_body(message))
        else:
            parts.extend(_content_parts(message["content"]))
        parts.extend([*_tag("message", close=True), ("<|end_of_msg|>", True)])
    choice = options.get("tool_choice")
    if choice in ("required", "none"):
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
    if options.get("add_generation_prompt", True):
        parts.extend([*_tag("message", (("role", "assistant"),)), *_tag("think")])
    return parts
