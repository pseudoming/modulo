"""Dispatch logged API bodies without making API requests or loading plugins."""

from modulo.core.domain.data_models import DataError, Usage
from modulo.shared.serialization import loads

from ..fields import json_object, usage_from
from ..messages.candidates import response_outputs
from . import anthropic, responses
from .chat_stream import assemble_sse as chat_sse
from .common import StreamResult, options_fields, read_messages, with_gaps

FORMATS = {"auto", "openai", "chat_completions", "responses", "anthropic"}


def body_format(body, hint="auto", *, response=False):
    if hint not in FORMATS:
        raise DataError("unsupported_api_body_format")
    if hint in {"responses", "anthropic", "chat_completions"}:
        return hint
    if response:
        if body.get("object") == "response" or isinstance(body.get("output"), list):
            return "responses"
        if body.get("type") in {"message", "error"} and "choices" not in body:
            return "anthropic"
    else:
        if "input" in body and not isinstance(body["input"], dict):
            return "responses"
        if "instructions" in body or body.get("previous_response_id") or body.get("conversation"):
            return "responses"
        tools = body.get("tools")
        messages = body.get("messages")
        if (
            "system" in body
            or (
                isinstance(tools, list)
                and any(isinstance(t, dict) and "input_schema" in t for t in tools)
            )
            or (
                isinstance(messages, list)
                and any(
                    isinstance(m, dict)
                    and isinstance(m.get("content"), list)
                    and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"]
                    )
                    for m in messages
                )
            )
        ):
            return "anthropic"
    return "chat_completions"


def request_fields(body, path="request_body", *, format="auto", extra_roots=()):
    body = json_object(body, path)
    dialect = body_format(body, format)
    if dialect == "responses":
        return responses.request_fields(body, path, extra_roots=extra_roots)
    if dialect == "anthropic":
        return anthropic.request_fields(body, path, extra_roots=extra_roots)
    values, message_path = body.get("messages"), path + ".messages"
    if values is None and isinstance(body.get("input"), dict):
        values, message_path = body["input"].get("messages"), path + ".input.messages"
    fields = options_fields(body, path, extra_roots=extra_roots)
    fields["messages"], gaps = read_messages(values, message_path)
    return with_gaps(fields, gaps)


def response_fields(body, path="response_body", *, format="auto", normalize=None):
    if body is None:
        return {"output_gaps": ["missing_response_body"]}, Usage(), None
    body = json_object(body, path)
    dialect = body_format(body, format, response=True)
    if dialect == "responses":
        return responses.response_fields(body, path)
    if dialect == "anthropic":
        return anthropic.response_fields(body, path)
    if body.get("error"):
        return {"output_gaps": ["gateway_error_response"]}, Usage(), body.get("model")
    return (
        response_outputs(body, path=path + ".choices.", normalize=normalize),
        usage_from(body.get("usage"), "prompt_tokens", "completion_tokens"),
        body.get("model"),
    )


def _events(lines):
    """Accept SSE framing and logger exports that omit blank separator lines."""
    events, gaps, pending, name = [], [], [], None

    def flush():
        if not pending:
            return
        data = "\n".join(pending)
        pending.clear()
        if data == "[DONE]":
            events.append(data)
            return
        try:
            event = loads(data)
            if not isinstance(event, dict):
                raise ValueError()
            if name and event.get("type") not in (None, name):
                gaps.append("conflicting_sse_event_type")
            events.append(event)
        except ValueError:
            gaps.append("invalid_stream_json")

    for block in lines:
        for line in block.splitlines() or [""]:
            if not line or line.startswith("event:"):
                flush()
                name = line[6:].strip() if line else None
            elif line.startswith("data:"):
                if pending:
                    try:
                        loads("\n".join(pending))
                    except ValueError:
                        pass
                    else:
                        flush()
                        name = None
                pending.append(line[5:].removeprefix(" "))
            elif line == "[DONE]":
                flush()
                events.append(line)
            elif not line.startswith((":", "id:", "retry:")):
                gaps.append("unsupported_sse_line")
    flush()
    return events, gaps


def assemble_sse(lines, format="auto", *, require_done=True):
    if format not in FORMATS:
        raise DataError("unsupported_gateway_stream_format")
    events, gaps = _events(lines)
    kinds = {e["type"] for e in events if isinstance(e, dict) and isinstance(e.get("type"), str)}
    dialect = format
    if format in {"openai", "auto"}:
        dialect = (
            "responses"
            if any(str(k).startswith("response.") for k in kinds)
            else (
                "anthropic"
                if kinds & {"message_start", "message_stop", "content_block_delta"}
                else "chat_completions"
            )
        )
    if dialect == "chat_completions":
        return chat_sse(lines, require_done=require_done)
    stream = responses.Stream() if dialect == "responses" else anthropic.Stream()
    for event in events:
        if event == "[DONE]":
            # Some gateways append the Chat sentinel after the native terminator.
            # It cannot substitute for response.completed or message_stop.
            if not stream.done:
                gaps.append("stream_finish_missing")
            continue
        try:
            stream.add(event)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            gaps.append("invalid_stream_event")
    try:
        output, usage, model = stream.finish()
    except (ValueError, TypeError, KeyError, AttributeError):
        return StreamResult(output={"output_gaps": ["invalid_stream_response", *gaps]})
    return StreamResult(
        output=with_gaps(output, gaps, "output_gaps"),
        usage=usage,
        response_id=stream.body.get("id"),
        model=model,
        done=stream.done,
        finish_reason=output.get("output_finish_reason"),
    )
