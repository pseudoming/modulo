"""Dify Agent invocation capture; each attempt is joined only by explicit id."""

from dataclasses import asdict
from decimal import Decimal, InvalidOperation

from modulo.adapters.protocols.shared.messages.candidates import expected_candidates
from modulo.adapters.protocols.shared.messages.chat import message, output_fields, request_options
from modulo.core.domain.data_models import (
    Call,
    DataError,
    Parsed,
    ProtocolContext,
    ProtocolEmission,
    ProtocolRecord,
)
from modulo.core.domain.interfaces import ProtocolSession
from modulo.shared.serialization import loads, ordered_json

from ..shared.api_bodies.parsing import response_fields
from ..shared.fields import identifier, mapping, text_or_none, timestamp, usage_from

SCHEMA = "dify.agent-llm.v1"


def _stream_output(chunks, finish_reason=None):
    if not isinstance(chunks, list) or not chunks:
        raise DataError("missing_dify_stream_chunks")
    content, reasoning, calls, raw_messages, gaps = [], [], {}, [], []
    reasoning_sources = []
    parts = []
    structured = False
    for index, chunk in enumerate(chunks):
        delta = mapping(mapping(chunk, "chunk").get("delta"), "chunk.delta")
        body = mapping(delta.get("message"), "chunk.delta.message")
        parsed = message(body, path=f"chunks.{index}.delta.message.")
        raw_messages.append(body)
        structured = structured or parsed.structured_content_json is not None
        gaps.extend(parsed.gaps)
        if parsed.assistant_parts_json:
            for part in loads(parsed.assistant_parts_json):
                if part["type"] == "tool_calls":
                    new_calls = [call for call in part["tool_calls"] if call.get("id") not in calls]
                    if new_calls:
                        parts.append({**part, "tool_calls": new_calls})
                else:
                    parts.append(part)
        else:
            if parsed.reasoning_content is not None:
                parts.append({"type": "reasoning", "text": parsed.reasoning_content})
            if parsed.content is not None:
                parts.append({"type": "text", "text": parsed.content})
        if parsed.content is not None:
            content.append(parsed.content)
        if parsed.reasoning_content is not None:
            reasoning.append(parsed.reasoning_content)
            reasoning_sources.append(parsed.reasoning_source)
        # Dify's plugin runtime emits complete tool calls (its Agent immediately
        # parses their arguments as JSON), not wire-level argument fragments.
        for call in loads(parsed.tool_calls_json) if parsed.tool_calls_json else []:
            key = identifier(call.get("id"), "tool_call.id")
            if key in calls and ordered_json(calls[key]) != ordered_json(call):
                raise DataError("conflicting_dify_stream_tool_call")
            if key not in calls and not parsed.assistant_parts_json:
                parts.append({"type": "tool_calls", "tool_calls": [call]})
            calls[key] = call
    if not chunks[-1]["delta"].get("finish_reason"):
        gaps.append("missing_dify_stream_finish_reason")
    result = {
        "role": "assistant",
        "content": "".join(content) if content else None,
        "tool_calls": list(calls.values()),
        "finish_reason": chunks[-1]["delta"].get("finish_reason"),
    }
    if reasoning:
        result["reasoning_content"] = "".join(reasoning)
    fields = output_fields(
        result,
        finish_reason=finish_reason,
        termination=chunks[-1]["delta"],
        termination_path=f"chunks.{len(chunks) - 1}.delta.",
    )
    fields["output_parts_json"] = ordered_json(parts)
    if reasoning_sources:
        fields["output_reasoning_source"] = ",".join(reasoning_sources)
    if structured:
        fields["output_structured_content_json"] = ordered_json({"chunks": raw_messages})
    if gaps:
        fields["output_gaps"] = sorted(set([*fields.get("output_gaps", []), *gaps]))
    return fields


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
                message(mapping(m, "prompt_message"), path=f"request.messages.{i}.")
                for i, m in enumerate(request["messages"])
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
    options = request_options(("request", {**request, "tools": converted}))
    gaps.extend(options.pop("gaps"))
    try:
        if request.get("stream") is True and phase == "finished":
            fields = _stream_output(record.get("chunks"), response.get("finish_reason"))
        else:
            output = (
                mapping(response["message"], "response.message")
                if response.get("message") is not None
                else None
            )
            fields = output_fields(
                output,
                path="response.message.",
                termination=response,
                termination_path="response.",
            )
    except DataError as exc:
        fields = {"output_gaps": [str(exc)]}
    fields = expected_candidates(fields, ("request", request))
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
                **options,
                usage=usage_from(response.get("usage"), "prompt_tokens", "completion_tokens"),
                gaps=sorted(set(gaps)),
                **fields,
            )
        ],
        contexts=[ProtocolContext(metadata=record.get("metadata"), metadata_path="metadata")],
    )


def parse_thought(row):
    """Use structured request data when exported; thought/tool_input are not prompts."""
    request = row.get("message")
    messages, gaps, options = None, [], {}
    path = "message."
    if request in (None, ""):
        gaps.append("missing_agent_request_body")
    else:
        try:
            request = loads(request) if isinstance(request, str) else request
            if isinstance(request, list):
                request = {"messages": request}
            else:
                request = mapping(request, "message")
                path += "messages."
            raw = request.get("messages")
            if not isinstance(raw, list) or not raw:
                raise DataError("missing_agent_request_body")
            messages = [
                message(
                    mapping(m, "message"),
                    path=f"{path}{i}.",
                    content_key="text" if "text" in m and "content" not in m else "content",
                )
                for i, m in enumerate(raw)
            ]
            # A structured export may include tools beside its messages. Bare
            # Dify tool definitions use the same function schema without a wrapper.
            request = dict(request)
            if "tools" in request and isinstance(request["tools"], list):
                request["tools"] = [
                    {"type": "function", "function": t}
                    if isinstance(t, dict) and "name" in t and "type" not in t
                    else t
                    for t in request["tools"]
                ]
            options = request_options(("message", request))
            gaps.extend(options.pop("gaps"))
            if row.get("tool") and "tools" not in request:
                gaps.append("missing_agent_tool_definitions")
        except (ValueError, TypeError, KeyError) as exc:
            gaps.append(
                str(exc) if isinstance(exc, DataError) else "unstructured_agent_request_body"
            )
    reported = usage_from(
        {
            "input": row.get("message_token"),
            "output": row.get("answer_token"),
            "total_tokens": row.get("tokens"),
        },
        "input",
        "output",
    )
    if row.get("response") is not None:
        fields, body_usage, _ = response_fields(row["response"], "response")
        values = reported.model_dump()
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            direct, body = getattr(reported, key), getattr(body_usage, key)
            if direct is None:
                values[key] = body
            elif body is not None and body != direct:
                values[key] = None
                reason = "conflicting_dify_" + key
                if key != "output_tokens":
                    gaps.append(reason)
                if key != "input_tokens":
                    fields.setdefault("output_gaps", []).append(reason)
        reported = type(reported)(**values)
    else:
        fields = output_fields(
            {"content": row.get("answer")}, path="answer.", termination=row, termination_path=""
        )
    if row.get("tool") and not fields.get("output_tool_calls_json"):
        fields["output_gaps"] = [*fields.get("output_gaps", []), "missing_agent_tool_call_ids"]
    fields = expected_candidates(fields, ("message", request))
    return Parsed(
        [
            Call(
                call_id=identifier(row.get("id"), "id"),
                protocol="dify",
                model=text_or_none(row.get("model_id"), "model_id"),
                provider=text_or_none(row.get("model_provider"), "model_provider"),
                messages=messages,
                **options,
                usage=reported,
                gaps=sorted(set(gaps)),
                **fields,
            )
        ],
        contexts=[ProtocolContext(metadata=row.get("metadata"), metadata_path="metadata")],
    )


class DifySession(ProtocolSession):
    def __init__(self, adapter):
        super().__init__(adapter)

    def feed(self, record):
        if record.payload.get("schema_version") != SCHEMA:
            return super().feed(record)
        key = identifier(record.payload.get("invocation_id"), "invocation_id")
        self.events.append(key, {"records": [asdict(record)]})
        return []

    def finish_group(self, key, entries, complete):
        records = [ProtocolRecord(**r) for entry in entries for r in entry["records"]]
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
            yield ProtocolEmission(
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
        except (DataError, TypeError, ValueError) as exc:
            yield ProtocolEmission(records, error=str(exc))
