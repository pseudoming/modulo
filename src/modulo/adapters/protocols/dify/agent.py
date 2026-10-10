"""Dify Agent invocation capture; each attempt is joined only by explicit id."""

from decimal import Decimal, InvalidOperation

from modulo.core.domain.data_models import (
    Call,
    DataError,
    Parsed,
    ProtocolContext,
    ProtocolEmission,
)
from modulo.core.domain.interfaces import ProtocolSession
from modulo.shared.serialization import ordered_json

from ..shared.chat import message, output_fields, request_options, tool_calls_json
from ..shared.fields import identifier, mapping, text_or_none, timestamp, usage_from

SCHEMA = "dify.agent-llm.v1"


def _text_message(body):
    body = mapping(body, "prompt_message")
    content = body.get("content")
    if isinstance(content, list):
        if any(
            not isinstance(part, dict)
            or set(part) != {"type", "data"}
            or part.get("type") != "text"
            or not isinstance(part.get("data"), str)
            for part in content
        ):
            raise DataError("unsupported_dify_content_parts")
        body = {**body, "content": "".join(part["data"] for part in content)}
    return body


def _stream_output(chunks):
    if not isinstance(chunks, list) or not chunks:
        raise DataError("missing_dify_stream_chunks")
    content, reasoning, calls = [], [], {}
    for chunk in chunks:
        delta = mapping(mapping(chunk, "chunk").get("delta"), "chunk.delta")
        body = _text_message(delta.get("message"))
        parsed = message(body)
        content.append(parsed.content)
        if parsed.reasoning_content is not None:
            reasoning.append(parsed.reasoning_content)
        # Dify's plugin runtime emits complete tool calls (its Agent immediately
        # parses their arguments as JSON), not wire-level argument fragments.
        tool_calls_json(body.get("tool_calls"))
        for call in body.get("tool_calls") or []:
            key = identifier(call.get("id"), "tool_call.id")
            if key in calls and ordered_json(calls[key]) != ordered_json(call):
                raise DataError("conflicting_dify_stream_tool_call")
            calls[key] = call
    if not chunks[-1]["delta"].get("finish_reason"):
        raise DataError("missing_dify_stream_finish_reason")
    result = {"role": "assistant", "content": "".join(content), "tool_calls": list(calls.values())}
    if reasoning:
        result["reasoning_content"] = "".join(reasoning)
    return result


def parse_invocation(record):
    phase = record.get("phase")
    if phase not in {"started", "finished"}:
        raise DataError("invalid_dify_invocation_phase")
    request = mapping(record.get("request"), "request")
    response = mapping(record.get("response", {}), "response")
    gaps, messages = [], None
    if isinstance(request.get("messages"), list) and request["messages"]:
        try:
            messages = [
                message(_text_message(m), path="request.messages.") for m in request["messages"]
            ]
        except (DataError, TypeError, ValueError, AttributeError):
            gaps.append("unsupported_dify_invocation_messages")
    else:
        gaps.append("missing_messages")
    tools = request.get("tools")
    converted = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict) or set(tool) - {"name", "description", "parameters"}:
                gaps.append("unsupported_dify_tool_definition")
                continue
            converted.append({"type": "function", "function": tool})
    elif tools is not None:
        gaps.append("unsupported_dify_tool_definition")
    options, sources, tools_json, option_gaps = request_options(
        ("request", {**request, "tools": converted})
    )
    gaps.extend(option_gaps)
    try:
        output = (
            _stream_output(record.get("chunks"))
            if request.get("stream") is True and phase == "finished"
            else _text_message(response["message"])
            if response.get("message") is not None
            else None
        )
        fields = output_fields(output, path="response.message.")
    except DataError as exc:
        fields = {"output_gaps": [str(exc)]}
    status = record.get("status", "unknown")
    if phase == "started":
        status = "unknown"
        fields["output_gaps"] = ["missing_dify_invocation_finish"]
    elif status not in {"success", "failed"}:
        raise DataError("invalid_dify_invocation_status")
    if status == "failed":
        fields["output_gaps"] = [*fields.get("output_gaps", []), "dify_invocation_failed"]
    duration = record.get("duration_ms")
    if duration is not None:
        try:
            if (
                not isinstance(duration, str)
                or not Decimal(duration).is_finite()
                or Decimal(duration) < 0
            ):
                raise InvalidOperation
        except (InvalidOperation, ValueError) as exc:
            raise DataError("invalid_dify_invocation_duration") from exc
    return Parsed(
        [
            Call(
                call_id=identifier(record.get("invocation_id"), "invocation_id"),
                protocol="dify",
                model=text_or_none(record.get("model"), "model"),
                provider=text_or_none(record.get("provider"), "provider"),
                status=status,
                started_at=timestamp(record.get("created_at")),
                duration_ms=duration,
                messages=messages,
                tools_json=tools_json,
                render_options=options,
                render_option_sources=sources,
                usage=usage_from(response.get("usage"), "prompt_tokens", "completion_tokens"),
                gaps=sorted(set(gaps)),
                **fields,
            )
        ],
        contexts=[ProtocolContext(metadata=record.get("metadata"), metadata_path="metadata")],
    )


def parse_thought(row):
    """Native message_agent_thoughts joined to messages for model identity.

    Its `message` column is empty in the pinned runner, and neither `thought`
    nor `tool_input` reconstructs a request. Preserve usage without guessing.
    """
    fields = output_fields({"content": row.get("answer")}, path="answer.")
    if row.get("tool"):
        fields["output_gaps"] = ["missing_agent_tool_call_ids"]
    return Parsed(
        [
            Call(
                call_id=identifier(row.get("id"), "id"),
                protocol="dify",
                model=text_or_none(row.get("model_id"), "model_id"),
                provider=text_or_none(row.get("model_provider"), "model_provider"),
                usage=usage_from(
                    {
                        "input": row.get("message_token"),
                        "output": row.get("answer_token"),
                        "total_tokens": row.get("tokens"),
                    },
                    "input",
                    "output",
                ),
                gaps=["missing_agent_request_body"],
                **fields,
            )
        ]
    )


class DifySession(ProtocolSession):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.invocations = {}

    def feed(self, record):
        if record.payload.get("schema_version") != SCHEMA:
            return super().feed(record)
        key = identifier(record.payload.get("invocation_id"), "invocation_id")
        self.invocations.setdefault(key, []).append(record)
        return []

    def finish(self, *, complete):
        result = []
        for key, records in self.invocations.items():
            try:
                phases = {}
                for record in records:
                    payload = record.payload
                    phase = payload.get("phase")
                    if phase not in {"started", "finished"}:
                        raise DataError("invalid_dify_invocation_phase")
                    if phase in phases and ordered_json(phases[phase]) != ordered_json(payload):
                        raise DataError("conflicting_dify_invocation_event")
                    phases[phase] = payload
                if len(phases) == 2:
                    for field in ("request", "model", "provider", "metadata", "created_at"):
                        if ordered_json(phases["started"].get(field)) != ordered_json(
                            phases["finished"].get(field)
                        ):
                            raise DataError("conflicting_dify_invocation_request")
                selected = phases.get("finished", phases.get("started"))
                parsed = parse_invocation(selected)
                result.append(
                    ProtocolEmission(
                        records,
                        parsed,
                        assembly={
                            "kind": "dify_agent_invocation",
                            "method": "explicit_invocation_id",
                            "invocation_id": key,
                            "phases": sorted(phases),
                            "source_read_complete": complete,
                        },
                    )
                )
            except (DataError, TypeError, ValueError) as exc:
                result.append(ProtocolEmission(records, error=str(exc)))
        self.invocations.clear()
        return result
