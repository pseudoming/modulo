"""Reviewed per-backend options and prompt preparation; no provider/model guessing."""

from copy import deepcopy
from decimal import Decimal
from math import isfinite

from modulo.core.domain.interfaces import UnsupportedInput
from modulo.shared.serialization import digest, loads


def template_json(value):
    """Match official Python JSON rendering, without altering stored evidence.

    Only prompt rendering projects JSON decimals to the official float spelling.
    It is not an accounting conversion: raw arguments and source values stay exact.
    """

    def project(item):
        if isinstance(item, Decimal):
            number = float(item)
            if not isfinite(number):
                raise UnsupportedInput("numeric_argument_outside_reference_range")
            return number
        if isinstance(item, dict):
            return {k: project(v) for k, v in item.items()}
        if isinstance(item, list):
            return [project(v) for v in item]
        return item

    return project(loads(value))


def valid_option(value, rule):
    if rule is bool:
        return type(value) is bool
    if rule is str:
        return isinstance(value, str) and bool(value)
    if callable(rule):
        return rule(value)
    return any(type(value) is type(item) and value == item for item in rule)


class RenderProfile:
    option_rules = {}
    option_aliases = {}
    thinking_parameter = None
    thinking_values = {}
    roles = {"system", "user", "assistant", "tool"}
    inline_think_tags = ("<think>", "</think>")
    parse_tool_arguments = True
    named_messages = False

    def translate_option(self, key, value):
        if key == "thinking" and self.thinking_parameter:
            if isinstance(value, dict):
                if set(value) != {"type"}:
                    raise UnsupportedInput("unsupported_thinking_fields")
                value = value["type"]
            key = self.thinking_parameter
            value = self.thinking_values.get(str(value), value)
        return self.option_aliases.get(key, key), value

    def resolve_options(self, call):
        explicit, sources = {}, {}
        for original, value in call.render_options.items():
            key, value = self.translate_option(original, value)
            if key == "add_generation_prompt" and value is True:
                continue
            if key in explicit and (
                type(explicit[key]) is not type(value) or explicit[key] != value
            ):
                raise UnsupportedInput(f"conflicting_render_option:{key}")
            if key not in self.option_rules:
                raise UnsupportedInput(f"unsupported_render_option:{key}")
            if not valid_option(value, self.option_rules[key]):
                raise UnsupportedInput(f"invalid_render_option:{key}")
            explicit[key] = value
            sources.setdefault(key, []).extend(
                call.render_option_sources.get(original, [f"call.{original}"])
            )
        effective = {**self.template_kwargs, **explicit}
        for key in effective:
            sources.setdefault(key, ["asset_profile"])
        return effective, sources

    def chat_messages(self, call):
        result = []
        tool_names = {}
        tool_results = set()
        for message in call.messages or []:
            if message.role not in self.roles:
                raise UnsupportedInput(f"unsupported_role:{message.role}")
            body = {"role": message.role, "content": message.content}
            if message.reasoning_content is not None:
                body["reasoning_content"] = message.reasoning_content
            elif message.role == "assistant":
                text, reasoning, _ = self.split_output(message.content, None, None)
                if reasoning is not None:
                    body.update(content=text, reasoning_content=reasoning)
            if message.name is not None:
                if message.role != "tool" and not self.supports_name(message, len(result)):
                    raise UnsupportedInput("named_message_not_supported_by_template")
                body["name"] = message.name
            if message.tool_call_id is not None:
                body["tool_call_id"] = message.tool_call_id
            if message.role == "assistant":
                tool_names = {}
                tool_results = set()
            if message.tool_calls_json:
                calls = deepcopy(template_json(message.tool_calls_json))
                for tool in calls:
                    fn = tool["function"]
                    if self.parse_tool_arguments and isinstance(fn["arguments"], str):
                        try:
                            fn["arguments"] = template_json(fn["arguments"])
                        except ValueError as exc:
                            raise UnsupportedInput("invalid_tool_arguments_json") from exc
                        if not isinstance(fn["arguments"], dict):
                            raise UnsupportedInput("tool_arguments_not_object")
                    if tool.get("id"):
                        if tool["id"] in tool_names:
                            raise UnsupportedInput("duplicate_tool_call_id")
                        tool_names[tool["id"]] = fn["name"]
                body["tool_calls"] = calls
            if message.role == "tool":
                # A name is not a substitute for linking to the matching call.
                if message.tool_call_id not in tool_names:
                    raise UnsupportedInput("unresolved_tool_call_id")
                if message.tool_call_id in tool_results:
                    raise UnsupportedInput("duplicate_tool_result")
                tool_results.add(message.tool_call_id)
                expected = tool_names[message.tool_call_id]
                if message.name is not None and message.name != expected:
                    raise UnsupportedInput("conflicting_tool_name")
                body["name"] = expected
            result.append(body)
        return result

    def supports_name(self, message, index):
        return self.named_messages

    def split_output(self, content, reasoning, source):
        if reasoning is not None or content is None:
            return content, reasoning, source
        opening, closing = self.inline_think_tags
        if content.startswith(opening):
            if closing not in content[len(opening) :]:
                raise UnsupportedInput("incomplete_inline_reasoning")
            reasoning, content = content[len(opening) :].split(closing, 1)
            source = "content:inline_thinking"
        return content, reasoning, source

    def output_counts(self, call, encode):
        if call.output_gaps:
            return {"output_execution": "not_run", "output_reason": ",".join(call.output_gaps)}
        try:
            text, reasoning, source = self.split_output(
                call.output_text, call.output_reasoning, call.output_reasoning_source
            )
        except UnsupportedInput as exc:
            return {"output_execution": "not_run", "output_reason": str(exc)}
        if text is None and reasoning is None:
            return {"output_execution": "not_run", "output_reason": "missing_output"}
        try:
            visible_tokens = None if text is None else len(encode(text))
            reasoning_tokens = None if reasoning is None else len(encode(reasoning))
        except Exception as exc:
            return {
                "output_execution": "error",
                "output_reason": f"tokenizer_error:{type(exc).__name__}",
            }
        return {
            "asset_revision": self.entry["revision"],
            "asset_hash": self.asset_hash,
            "template_hash": self.template_hash,
            "visible_output_tokens": visible_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "reasoning_output_source": source,
            "output_execution": "completed",
            "output_reason": "text_components_only" if call.output_tool_calls_json else None,
        }

    @staticmethod
    def option_evidence(options, sources):
        return {
            "render_options": options,
            "render_options_sha256": digest(options),
            "render_option_sources": sources,
        }
