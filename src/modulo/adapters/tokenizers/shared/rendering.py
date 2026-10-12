"""Reviewed per-backend options and prompt preparation; no provider/model guessing."""

from copy import deepcopy
from decimal import Decimal
from math import isfinite

from modulo.core.domain.interfaces import AssetError, UnsupportedInput
from modulo.shared.serialization import bytes_digest, digest, loads, ordered_json

from .images import IMAGE_OPTIONS


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


def tool_choice(value):
    if isinstance(value, str):
        return value in {"auto", "none", "required"}
    return (
        isinstance(value, dict)
        and set(value) == {"type", "function"}
        and value["type"] == "function"
        and isinstance(value["function"], dict)
        and set(value["function"]) == {"name"}
        and isinstance(value["function"]["name"], str)
        and bool(value["function"]["name"])
    )


def response_format(value):
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    kind = value.get("type")
    if kind in {"text", "json_object"}:
        return set(value) == {"type"}
    schema = value.get("json_schema")
    return (
        kind == "json_schema"
        and set(value) == {"type", "json_schema"}
        and isinstance(schema, dict)
        and isinstance(schema.get("schema"), dict)
        and not (set(schema) - {"name", "description", "schema", "strict"})
        and ("strict" not in schema or type(schema["strict"]) is bool)
        and ("name" not in schema or isinstance(schema["name"], str))
    )


REQUEST_RULES = {
    "tool_choice": tool_choice,
    "parallel_tool_calls": bool,
    "response_format": response_format,
    "response_schema": lambda value: isinstance(value, dict),
    "add_generation_prompt": bool,
}


class RenderProfile:
    image_grid_supported = False
    option_rules = {}
    option_aliases = {}
    thinking_parameter = None
    thinking_values = {}
    roles = {"system", "user", "assistant", "tool"}
    inline_think_tags = ("<think>", "</think>")
    parse_tool_arguments = True
    named_messages = False
    custom_template = None
    uses_content_blocks = False
    configured_options = {}

    def configure_reference(self, binding):
        """Compile only a template explicitly supplied in trusted configuration."""
        if binding.chat_template is not None:
            from jinja2 import StrictUndefined, TemplateError, nodes
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            environment = ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
                extensions=["jinja2.ext.loopcontrols"],
                undefined=StrictUndefined,
            )
            environment.filters.update(tojson=ordered_json, fromjson=template_json)

            def raise_exception(message):
                raise UnsupportedInput("custom_template_rejected_input")

            environment.globals["raise_exception"] = raise_exception
            try:
                tree = environment.parse(binding.chat_template)
                if any(node.name == "strftime_now" for node in tree.find_all(nodes.Name)):
                    raise AssetError("dynamic_template_not_supported")
                self.custom_template = environment.from_string(binding.chat_template)
            except TemplateError as exc:
                raise AssetError("invalid_custom_chat_template") from exc
            self.uses_content_blocks = any(
                node.attr == "content_blocks" for node in tree.find_all(nodes.Getattr)
            ) or any(
                isinstance(node.arg, nodes.Const) and node.arg.value == "content_blocks"
                for node in tree.find_all(nodes.Getitem)
            )
            self.template_source = binding.chat_template
            self.template_hash = bytes_digest(binding.chat_template.encode())
        self.template_kwargs = {**self.template_kwargs, **binding.template_kwargs}
        self.configured_options = dict.fromkeys(binding.template_kwargs, "binding.template-kwargs")
        if not binding.add_generation_prompt:
            self.template_kwargs["add_generation_prompt"] = False
            self.configured_options["add_generation_prompt"] = "binding.add_generation_prompt"

    def thinking_enabled(self, options):
        return (
            options.get("enable_thinking", True)
            and options.get("thinking_mode") not in {"chat", "disabled"}
            and options.get("reasoning_effort") != "no_think"
        )

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
        rules = {**self.option_rules, **REQUEST_RULES}
        for original, value in call.render_options.items():
            if original in IMAGE_OPTIONS:
                continue  # Validated independently when preparing the input images.
            key, value = self.translate_option(original, value)
            if key in explicit and (
                type(explicit[key]) is not type(value) or explicit[key] != value
            ):
                raise UnsupportedInput(f"conflicting_render_option:{key}")
            if key == "chat_template":
                if value != getattr(self, "template_source", None):
                    raise UnsupportedInput("logged_chat_template_not_selected_in_binding")
                explicit[key] = value
                sources[key] = call.render_option_sources.get(original, [f"call.{original}"])
                continue
            if key not in rules and not (self.custom_template and key in self.configured_options):
                raise UnsupportedInput(f"unsupported_render_option:{key}")
            if key in rules and not valid_option(value, rules[key]):
                raise UnsupportedInput(f"invalid_render_option:{key}")
            explicit[key] = value
            sources.setdefault(key, []).extend(
                call.render_option_sources.get(original, [f"call.{original}"])
            )
        effective = {**self.template_kwargs, **explicit}
        for key, value in effective.items():
            if key in self.configured_options:
                if key not in rules and not self.custom_template:
                    raise UnsupportedInput(f"unsupported_render_option:{key}")
                if key in rules and not valid_option(value, rules[key]):
                    raise UnsupportedInput(f"invalid_render_option:{key}")
            if key == "tool_choice" and isinstance(value, dict):
                names = {t["function"]["name"] for t in loads(call.tools_json or "[]")}
                if value["function"]["name"] not in names:
                    raise UnsupportedInput("tool_choice_name_not_in_tools")
            sources.setdefault(
                key,
                [self.configured_options.get(key, "asset_profile")],
            )
        return effective, sources

    def chat_messages(self, call):
        result = []
        tool_names = {}
        tool_results = set()
        for message in call.messages or []:
            if message.gaps:
                raise UnsupportedInput(",".join(message.gaps))
            if message.content is None:
                raise UnsupportedInput("missing_message_content")
            if message.role not in self.roles:
                raise UnsupportedInput(f"unsupported_role:{message.role}")
            body = {"role": message.role, "content": message.content}
            if message.assistant_parts_json:
                parts = loads(message.assistant_parts_json)
                phase = -1
                for part in parts:
                    if part["type"] == "text" and part["text"] == "":
                        continue
                    current = {"reasoning": 0, "text": 1, "image": 1, "tool_calls": 2}[part["type"]]
                    if current < phase and not self.uses_content_blocks:
                        raise UnsupportedInput("content_order_not_supported_by_template")
                    phase = max(phase, current)
                body["content_blocks"] = parts
            if message.content_parts:
                body["content"] = [
                    {"type": "image"} if part.type == "image" else part.model_dump()
                    for part in message.content_parts
                ]
            if message.reasoning_content is not None:
                body["reasoning_content"] = message.reasoning_content
            elif message.role == "assistant" and not message.content_parts:
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

    @staticmethod
    def option_evidence(options, sources):
        return {
            "render_options": options,
            "render_options_json": ordered_json(options),
            "render_options_sha256": digest(options),
            "render_option_sources": sources,
        }
