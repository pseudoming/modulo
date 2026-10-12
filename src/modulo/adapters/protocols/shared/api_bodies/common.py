"""Small helpers for mapping API bodies to the existing call contract."""

from dataclasses import dataclass, field

from modulo.core.domain.data_models import DataError, Usage
from modulo.shared.serialization import ordered_json

from ..messages.chat import message, output_fields, request_options


@dataclass
class StreamResult:
    output: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    response_id: str | None = None
    model: str | None = None
    done: bool = False
    finish_reason: str | None = None


def with_gaps(fields, gaps, key="gaps"):
    combined = sorted(set([*fields.get(key, []), *gaps]))
    return {**fields, key: combined} if combined else fields


def parse_message(body, path, *, original=None):
    parsed = message(body, path=path + ".")
    if original is not None:
        parsed = parsed.model_copy(update={"structured_content_json": ordered_json(original)})
    return parsed


def parse_output(body, path, finish, *, original=None, gaps=()):
    fields = output_fields(body, path=path + ".", finish_reason=finish)
    if original is not None:
        fields["output_structured_content_json"] = ordered_json(original)
    return with_gaps(fields, gaps, "output_gaps")


def remap_sources(parsed, origins):
    """Translate normalized field paths back to their actual log locations."""

    def source(value):
        parts = value.split(",")
        for index, part in enumerate(parts):
            for old, new in sorted(origins.items(), key=lambda pair: len(pair[0]), reverse=True):
                if part == old or part.startswith(old + "."):
                    parts[index] = new + part[len(old) :]
                    break
        return ",".join(parts)

    changes = {}
    if parsed.reasoning_source:
        changes["reasoning_source"] = source(parsed.reasoning_source)
    parts = []
    for part in parsed.content_parts:
        if part.type == "image":
            values = {
                key: source(value)
                for key in ("source", "url_source", "dimensions_source")
                if isinstance(value := getattr(part, key), str)
            }
            values["option_sources"] = {
                key: source(value) for key, value in part.option_sources.items()
            }
            part = part.model_copy(update=values)
        parts.append(part)
    changes["content_parts"] = parts
    return parsed.model_copy(update=changes)


def options_fields(request, path, *, dialect="chat", extra_roots=()):
    """Translate only API spelling; the selected asset validates template options."""
    normalized = dict(request)
    origins, gaps = {}, []
    tools = request.get("tools")
    if dialect != "chat" and isinstance(tools, list):
        converted = []
        for tool in tools:
            if not isinstance(tool, dict):
                gaps.append("invalid_tool_definition")
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                converted.append(tool)
                continue  # A framework may already have normalized this view.
            if dialect == "responses" and tool.get("type") == "function":
                allowed = {"type", "name", "description", "parameters", "strict"}
                function = {key: value for key, value in tool.items() if key != "type"}
            elif dialect == "anthropic" and tool.get("type", "custom") == "custom":
                allowed = {"type", "name", "description", "input_schema", "cache_control", "strict"}
                function = {
                    "parameters" if key == "input_schema" else key: value
                    for key, value in tool.items()
                    if key not in {"type", "cache_control"}
                }
            else:
                gaps.append("unsupported_tool_definition")
                continue
            if set(tool) - allowed:
                gaps.append("unsupported_tool_definition_fields")
            converted.append({"type": "function", "function": function})
        normalized["tools"] = converted
    choice = request.get("tool_choice")
    if isinstance(choice, dict) and dialect != "chat":
        kind = choice.get("type")
        if dialect == "anthropic" and kind in {"auto", "any", "none"}:
            normalized["tool_choice"] = {"any": "required"}.get(kind, kind)
        elif kind == ("tool" if dialect == "anthropic" else "function"):
            normalized["tool_choice"] = {
                "type": "function",
                "function": {"name": choice.get("name")},
            }
        else:
            gaps.append("unsupported_tool_choice")
        if set(choice) - {"type", "name", "disable_parallel_tool_use"}:
            gaps.append("unsupported_tool_choice_fields")
        if "disable_parallel_tool_use" in choice:
            value = choice["disable_parallel_tool_use"]
            if type(value) is not bool:
                gaps.append("invalid_parallel_tool_choice")
            else:
                normalized["parallel_tool_calls"] = not value
                origins["parallel_tool_calls"] = path + ".tool_choice.disable_parallel_tool_use"
    if dialect == "responses" and isinstance(request.get("reasoning"), dict):
        reasoning = request["reasoning"]
        normalized.pop("reasoning")
        if set(reasoning) - {"effort", "summary", "generate_summary"}:
            gaps.append("unsupported_reasoning_options")
        if reasoning.get("effort") is not None:
            normalized["reasoning_effort"] = reasoning["effort"]
            origins["reasoning_effort"] = path + ".reasoning.effort"
            if "reasoning_effort" in request and request["reasoning_effort"] != reasoning["effort"]:
                gaps.append("conflicting_request_option:reasoning_effort")
    if dialect == "responses" and isinstance(request.get("text"), dict):
        text = request["text"]
        if text.get("format") not in (None, {"type": "text"}):
            spec = text["format"]
            normalized["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {k: v for k, v in spec.items() if k != "type"},
                }
                if isinstance(spec, dict)
                and spec.get("type") == "json_schema"
                and "json_schema" not in spec
                else spec
            )
            origins["response_format"] = path + ".text.format"
        if text.get("verbosity") is not None:
            gaps.append("unsupported_text_verbosity")
    if isinstance(request.get("thinking"), dict):
        thinking = request["thinking"]
        # The token budget is a generation limit, like max_tokens. The mode is
        # the part used by the fixed templates; display affects output evidence.
        # A simple Messages request without system/tools has the same shape as
        # Chat, so this option does not depend on automatic dialect detection.
        if not set(thinking) - {"type", "budget_tokens", "display"}:
            normalized["thinking"] = {"type": thinking.get("type")}
            origins["thinking"] = path + ".thinking.type"
    fields = request_options((path, normalized))
    for key, origin in origins.items():
        if key in fields["render_option_sources"]:
            fields["render_option_sources"][key] = [
                origin if v == path + "." + key else v for v in fields["render_option_sources"][key]
            ]
    fields["gaps"].extend(gaps)
    for extra_path, root in extra_roots:
        if root is None:
            continue
        if not isinstance(root, dict):
            fields["gaps"].append("invalid_request_options")
            continue
        extra = options_fields(root, extra_path, dialect=dialect)
        fields["gaps"].extend(extra["gaps"])
        fields["generation_gaps"].extend(extra["generation_gaps"])
        if extra["tools_json"] is not None:
            if fields["tools_json"] is not None and fields["tools_json"] != extra["tools_json"]:
                fields["gaps"].append("conflicting_tools")
            else:
                fields["tools_json"] = extra["tools_json"]
        for kind in ("render", "generation"):
            options, sources = fields[kind + "_options"], fields[kind + "_option_sources"]
            for key, value in extra[kind + "_options"].items():
                if key in options and (
                    type(value) is not type(options[key]) or value != options[key]
                ):
                    target = "gaps" if kind == "render" else "generation_gaps"
                    label = "request" if kind == "render" else "generation"
                    fields[target].append(f"conflicting_{label}_option:{key}")
                else:
                    options[key] = value
                sources.setdefault(key, []).extend(extra[kind + "_option_sources"][key])
    fields["gaps"] = sorted(set(fields["gaps"]))
    fields["generation_gaps"] = sorted(set(fields["generation_gaps"]))
    fields["render_options_json"] = ordered_json(fields["render_options"])
    return fields


def read_messages(values, path):
    if not isinstance(values, list) or not values:
        return None, ["missing_messages"]
    messages, gaps = [], []
    for index, value in enumerate(values):
        try:
            if not isinstance(value, dict):
                raise DataError("invalid_message")
            messages.append(parse_message(value, f"{path}.{index}"))
        except DataError as exc:
            gaps.append(str(exc))
    return messages, gaps
