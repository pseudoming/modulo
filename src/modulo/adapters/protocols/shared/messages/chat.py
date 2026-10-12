"""Lossless text, reasoning and function-call extraction shared by log protocols."""

from typing import Any

from modulo.adapters.protocols.shared.fields import mapping
from modulo.adapters.protocols.shared.messages.content import REDACTED, content_blocks
from modulo.core.domain.data_models import DataError, Message
from modulo.shared.serialization import loads, ordered_json

OPTION_FIELDS = {
    "mm_processor_kwargs",
    "images_kwargs",
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
    "add_vision_id",
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
GENERATION_FIELDS = {
    "stop",
    "stop_sequences",
    "stop_token_ids",
    "include_stop_str_in_output",
    "ignore_eos",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "min_tokens",
}


def request_options(*roots: tuple[str, Any]):
    options, sources, gaps, tools = {}, {}, [], None
    generation, generation_sources, generation_gaps = {}, {}, []
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
            elif key in GENERATION_FIELDS:
                name = "stop" if key == "stop_sequences" else key
                if name == "stop" and isinstance(value, str):
                    value = [value]
                if name in generation and (
                    type(generation[name]) is not type(value) or generation[name] != value
                ):
                    generation_gaps.append(f"conflicting_generation_option:{name}")
                generation[name] = value
                generation_sources.setdefault(name, []).append(f"{path}.{key}")
            elif key in OPTION_FIELDS or path.endswith("chat_template_kwargs"):
                if key in options and (
                    type(options[key]) is not type(value) or options[key] != value
                ):
                    gaps.append(f"conflicting_request_option:{key}")
                options[key] = value
                sources.setdefault(key, []).append(f"{path}.{key}")
    return {
        "render_options": options,
        "render_options_json": ordered_json(options),
        "render_option_sources": sources,
        "tools_json": tools,
        "gaps": gaps,
        "generation_options": generation,
        "generation_option_sources": generation_sources,
        "generation_gaps": generation_gaps,
    }


def _reasoning(body, path):
    values, gaps = [], []
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = body.get(key)
        if value is not None:
            if isinstance(value, str):
                if value in REDACTED:
                    gaps.append("redacted_reasoning")
                else:
                    values.append((value, f"{path}{key}"))
            else:
                blocks = content_blocks(value, f"{path}{key}", reasoning=True)
                gaps.extend(blocks.gaps)
                if blocks.reasoning is not None:
                    values.append(
                        (blocks.reasoning, ",".join(blocks.reasoning_sources) or f"{path}{key}")
                    )
    if len({v for v, _ in values}) > 1:
        gaps.append("conflicting_reasoning_fields")
        values = []
    value, source = values[0] if values else (None, None)
    return value, source, gaps


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
        "stop_reason",
        "stop_sequence",
    }
    if any(value is not None and key not in allowed for key, value in body.items()):
        raise DataError("unsupported_message_features")
    role = role or body.get("role")
    content = body.get(content_key)
    reasoning, source, gaps = _reasoning(body, path)
    structured = {
        key: value
        for key, value in body.items()
        if key in {content_key, "reasoning_content", "reasoning", "thinking"}
        and isinstance(value, (list, dict))
    }
    additional = body.get("additional_kwargs") or {}
    if not isinstance(additional, dict):
        raise DataError("unsupported_message_features")
    extra_reasoning, extra_source, extra_gaps = _reasoning(additional, path + "additional_kwargs.")
    gaps.extend(extra_gaps)
    structured.update(
        {
            "additional_kwargs." + key: value
            for key, value in additional.items()
            if key in {"reasoning_content", "reasoning", "thinking"}
            and isinstance(value, (list, dict))
        }
    )
    if reasoning is not None and extra_reasoning is not None and reasoning != extra_reasoning:
        gaps.append("conflicting_reasoning_fields")
    elif reasoning is None:
        reasoning, source = extra_reasoning, extra_source
    blocks = None
    if isinstance(content, list):
        blocks = content_blocks(content, path + content_key)
        content = blocks.text
        gaps.extend(blocks.gaps)
        if blocks.reasoning is not None:
            if reasoning is not None and reasoning != blocks.reasoning:
                gaps.append("conflicting_reasoning_fields")
            elif reasoning is None:
                reasoning = blocks.reasoning
                source = ",".join(blocks.reasoning_sources)
    if "conflicting_reasoning_fields" in gaps:
        reasoning, source = None, None
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
    if blocks and blocks.calls:
        if raw_calls and _call_projection(raw_calls) != _call_projection(blocks.calls):
            gaps.append("conflicting_tool_calls")
        elif not raw_calls:
            raw_calls = blocks.calls
    calls = None if "conflicting_tool_calls" in gaps else tool_calls_json(raw_calls)
    parts = list(blocks.ordered) if blocks and role == "assistant" else []
    if parts:
        if reasoning is not None and blocks.reasoning is None:
            parts.insert(0, {"type": "reasoning", "text": reasoning})
        if calls and not blocks.calls:
            parts.append({"type": "tool_calls", "tool_calls": loads(calls)})
    if (
        content is None
        and not gaps
        and role == "assistant"
        and (calls is not None or reasoning is not None)
    ):
        content = ""
    if (not isinstance(content, str) and not (content is None and gaps)) or role not in {
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
    if body.get("files"):
        # Dify's saved prompt joins text and moves attachments into a separate
        # list; their original positions cannot be recovered from that list.
        structured["files"] = body["files"]
        gaps.append("attachment_content_order_missing")
        if "[TRUNCATED]" in ordered_json(body["files"]):
            gaps.append("truncated_image_data")
    if body.get("tools") or body.get("invalid_tool_calls"):
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
        structured_content_json=ordered_json(structured) if structured else None,
        assistant_parts_json=ordered_json(parts) if parts else None,
        content_parts=blocks.parts
        if blocks and any(part["type"] == "image" for part in blocks.parts)
        else [],
        gaps=sorted(set(gaps)),
        tool_calls_json=calls,
        tool_call_id=body.get("tool_call_id"),
        name=body.get("name"),
    )


def output_fields(
    body,
    *,
    content_key="content",
    path="",
    finish_reason=None,
    termination=None,
    termination_path="",
):
    if body is None:
        return {"output_gaps": ["missing_output"]}
    try:
        parsed = message(body, role="assistant", content_key=content_key, path=path)
    except (DataError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, DataError) else "invalid_output"
        return {"output_gaps": [reason]}
    result = {
        "output_text": parsed.content,
        "output_reasoning": parsed.reasoning_content,
        "output_reasoning_source": parsed.reasoning_source,
        "output_structured_content_json": parsed.structured_content_json,
        "output_parts_json": parsed.assistant_parts_json,
        "output_tool_calls_json": parsed.tool_calls_json,
    }
    gaps = list(parsed.gaps)
    if parsed.content_parts:
        gaps.append("output_images_not_supported")
        if not any(part.type == "text" for part in parsed.content_parts):
            result["output_text"] = None
    metadata = body.get("response_metadata")
    reasons = [finish_reason, body.get("finish_reason")]
    if isinstance(metadata, dict):
        reasons.append(metadata.get("finish_reason"))
    # LangChain's Anthropic messages retain stop_reason instead of finish_reason.
    roots = [(path, body)]
    if isinstance(metadata, dict):
        roots.append((path + "response_metadata.", metadata))
    if termination is not None:
        roots.append((termination_path, termination))
        reasons.append(termination.get("finish_reason"))
    stop_values = []
    for location, root in roots:
        anthropic_reason = root.get("stop_reason")
        is_finish = all(
            reason is None or reason == anthropic_reason for reason in reasons
        ) and anthropic_reason in (
            "end_turn",
            "tool_use",
            "max_tokens",
            "stop_sequence",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        )
        if is_finish:
            reasons.append(anthropic_reason)
        if "stop_sequence" in root:
            stop_values.append((root["stop_sequence"], location + "stop_sequence"))
        elif "stop_reason" in root and not is_finish:
            stop_values.append((root["stop_reason"], location + "stop_reason"))
    if stop_values:
        value, source = stop_values[0]
        if any(v is not None and type(v) not in (str, int) for v, _ in stop_values):
            gaps.append("invalid_output_stop_reason")
        elif any(type(v) is not type(value) or v != value for v, _ in stop_values):
            gaps.append("conflicting_output_stop_reason")
        else:
            result.update(output_stop_reason=value, output_stop_reason_source=source)
    reasons = [reason for reason in reasons if reason is not None]
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        gaps.append("invalid_finish_reason")
    elif len(set(reasons)) > 1:
        gaps.append("conflicting_finish_reason")
    elif reasons:
        result["output_finish_reason"] = reasons[0]
    if result.get("output_stop_reason") is not None or result.get("output_finish_reason") in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "model_context_window_exceeded",
        "stop_sequence",
    }:
        original = body.get(content_key)
        text_present = isinstance(original, str) or (
            parsed.assistant_parts_json is not None
            and any(part["type"] == "text" for part in loads(parsed.assistant_parts_json))
        )
        if not text_present and parsed.reasoning_content is not None and not parsed.tool_calls_json:
            result["output_text"] = None
    if gaps:
        result["output_gaps"] = sorted(set(gaps))
    return result


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
    if not isinstance(result, dict):
        return None
    # Retain both sources so output_fields can detect conflicting terminal metadata.
    metadata = result.get("response_metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ("finish_reason", "stop_reason", "stop_sequence"):
        if key not in choices[0]:
            continue
        value = choices[0][key]
        if key in metadata and metadata[key] != value:
            return {**result, key: value, "response_metadata": metadata}
        metadata[key] = value
    return {**result, "response_metadata": metadata}
