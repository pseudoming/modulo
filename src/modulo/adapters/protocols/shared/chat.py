"""Lossless text, reasoning and function-call extraction shared by log protocols."""

from typing import Any

from modulo.core.domain.data_models import DataError, Message
from modulo.shared.serialization import loads, ordered_json

from .fields import mapping

REDACTED = {"redacted-by-litellm", "redacted by litellm. 'litellm.turn_off_message_logging=True'"}
OPTION_FIELDS = {
    "thinking",
    "thinking_mode",
    "thinking_option",
    "thinking_effort",
    "thinking_budget",
    "enable_thinking",
    "reasoning",
    "reasoning_effort",
    "preserve_thinking",
    "preserved_thinking",
    "save_history_reasoning_content",
    "save_reasoning_content",
    "clear_thinking",
    "drop_thinking",
    "model_identity",
    "raw_last_assistant",
    "add_generation_prompt",
    "chat_template",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "response_schema",
    "fallback_strategy",
    "parallel_tool_calls",
}
CONTAINERS = (
    "extra_body",
    "model_kwargs",
    "model_parameters",
    "completion_params",
    "chat_template_kwargs",
)


def request_options(*roots: tuple[str, Any]):
    options, sources, gaps, tools = {}, {}, [], None
    pending = list(roots)
    while pending:
        path, current = pending.pop(0)
        if current is None or current == "":
            continue
        if not isinstance(current, dict):
            gaps.append("invalid_request_options")
            continue
        for key, value in current.items():
            if key in CONTAINERS:
                pending.append((f"{path}.{key}", value))
            elif key == "tools" and value is not None:
                if not isinstance(value, list) or any(
                    not isinstance(t, dict)
                    or t.get("type", "function") != "function"
                    or not isinstance(t.get("function"), dict)
                    or not isinstance(t["function"].get("name"), str)
                    or set(t) - {"type", "function"}
                    for t in value
                ):
                    gaps.append("unsupported_tool_definition")
                    continue
                encoded = ordered_json(value)
                if tools is not None and tools != encoded:
                    gaps.append("conflicting_tools")
                tools = encoded
            elif key in OPTION_FIELDS or path.endswith("chat_template_kwargs"):
                if key in options and (
                    type(options[key]) is not type(value) or options[key] != value
                ):
                    gaps.append(f"conflicting_request_option:{key}")
                options[key] = value
                sources.setdefault(key, []).append(f"{path}.{key}")
    return options, sources, tools, gaps


def _reasoning(body, path):
    values = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = body.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise DataError("unsupported structured reasoning; text required")
            if value in REDACTED:
                raise DataError("redacted_reasoning")
            values.append((value, f"{path}{key}"))
    if len({v for v, _ in values}) > 1:
        raise DataError("conflicting_reasoning_fields")
    return values[0] if values else (None, None)


def tool_calls_json(value):
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise DataError("tool_calls must be an array")
    for call in value:
        call = mapping(call, "tool_call")
        fn = mapping(call.get("function"), "tool_call.function")
        if call.get("type", "function") != "function" or not isinstance(fn.get("name"), str):
            raise DataError("unsupported_tool_call")
        if not isinstance(fn.get("arguments"), (str, dict)):
            raise DataError("unsupported_tool_arguments")
        if set(call) - {"id", "type", "function"} or set(fn) - {"name", "arguments"}:
            raise DataError("unsupported_tool_call_fields")
        if call.get("id") is not None and not isinstance(call["id"], str):
            raise DataError("invalid_tool_call_id")
    return ordered_json(value)


def _call_projection(calls):
    """Compare LC's parsed view with raw calls without rewriting raw JSON text."""
    result = []
    if not isinstance(calls, list):
        raise DataError("tool_calls must be an array")
    for call in calls:
        call = mapping(call, "tool_call")
        function = call.get("function", {"name": call.get("name"), "arguments": call.get("args")})
        function = mapping(function, "tool_call.function")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = loads(arguments)
            except ValueError as exc:
                raise DataError("invalid_tool_arguments_json") from exc
        result.append((call.get("id"), function.get("name"), arguments))
    return result


def message(body: dict, *, role=None, content_key="content", path="") -> Message:
    allowed = {
        "role",
        content_key,
        "reasoning_content",
        "reasoning",
        "thinking",
        "tool_calls",
        "tool_call_id",
        "name",
        "additional_kwargs",
        "files",
        "tools",
        "invalid_tool_calls",
        "id",
        "type",
        "response_metadata",
        "usage_metadata",
        "example",
        "usage",
        "finish_reason",
    }
    if any(value is not None and key not in allowed for key, value in body.items()):
        raise DataError("unsupported_message_features")
    role = role or body.get("role")
    content = body.get(content_key)
    reasoning, source = _reasoning(body, path)
    additional = body.get("additional_kwargs") or {}
    if not isinstance(additional, dict):
        raise DataError("unsupported_message_features")
    extra_reasoning, extra_source = _reasoning(additional, path + "additional_kwargs.")
    if reasoning is not None and extra_reasoning is not None and reasoning != extra_reasoning:
        raise DataError("conflicting_reasoning_fields")
    if reasoning is None:
        reasoning, source = extra_reasoning, extra_source
    raw_calls = additional.get("tool_calls", body.get("tool_calls"))
    if additional.get("tool_calls") is not None and body.get("tool_calls") is not None:
        if _call_projection(additional["tool_calls"]) != _call_projection(body["tool_calls"]):
            raise DataError("conflicting_tool_calls")
    # LC's parsed args are a convenience projection; prefer retained raw function arguments.
    if (
        raw_calls
        and isinstance(raw_calls, list)
        and all(isinstance(t, dict) and "args" in t and "function" not in t for t in raw_calls)
    ):
        raw_calls = [
            {
                "id": t.get("id"),
                "type": "function",
                "function": {"name": t.get("name"), "arguments": t["args"]},
            }
            for t in raw_calls
        ]
    calls = tool_calls_json(raw_calls)
    if content is None and role == "assistant" and (calls is not None or reasoning is not None):
        content = ""
    if not isinstance(content, str) or role not in {
        "system",
        "user",
        "assistant",
        "tool",
        "developer",
        "root",
    }:
        raise DataError("unsupported_message_features")
    if content in REDACTED:
        raise DataError("redacted_messages")
    if body.get("files") or body.get("tools") or body.get("invalid_tool_calls"):
        raise DataError("unsupported_message_features")
    if set(additional) - {"reasoning_content", "reasoning", "thinking", "tool_calls"}:
        raise DataError("unsupported_message_features")
    if role != "assistant" and (reasoning is not None or calls):
        raise DataError("reasoning_or_calls_on_non_assistant")
    if role == "tool" and not isinstance(body.get("tool_call_id"), str):
        raise DataError("missing_tool_call_id")
    return Message(
        role=role,
        content=content,
        reasoning_content=reasoning,
        reasoning_source=source,
        tool_calls_json=calls,
        tool_call_id=body.get("tool_call_id"),
        name=body.get("name"),
    )


def output_fields(body, *, content_key="content", path=""):
    if body is None:
        return {"output_gaps": ["missing_output"]}
    try:
        parsed = message(body, role="assistant", content_key=content_key, path=path)
    except (DataError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, DataError) else "invalid_output"
        return {"output_gaps": [reason]}
    return {
        "output_text": parsed.content,
        "output_reasoning": parsed.reasoning_content,
        "output_reasoning_source": parsed.reasoning_source,
        "output_tool_calls_json": parsed.tool_calls_json,
    }


def decode_response(value):
    if isinstance(value, str):
        try:
            value = loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    result = choices[0].get("message")
    return result if isinstance(result, dict) else None
