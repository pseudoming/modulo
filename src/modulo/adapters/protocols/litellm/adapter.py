"""LiteLLM terminal StandardLoggingPayload and individual SpendLogs exports."""

import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from modulo.core.domain.data_models import Call, DataError, Message, Parsed, ProtocolContext
from modulo.core.domain.interfaces import ProtocolAdapter
from modulo.shared.serialization import loads

from ..shared.chat import REDACTED, decode_response, message, output_fields, request_options
from ..shared.fields import (
    identifier,
    json_object,
    mapping,
    text_or_none,
    timestamp,
    usage_from,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_TRUNCATION = re.compile(
    r"\.\.\. \(litellm_truncated skipped \d+ chars\. "
    r"Truncation is a (?:DB storage|stdout logging) safeguard\."
)


def _truncated(value):
    if isinstance(value, str):
        return bool(_TRUNCATION.search(value))
    if isinstance(value, dict):
        return any(_truncated(v) for v in value.values())
    if isinstance(value, list):
        return any(_truncated(v) for v in value)
    return False


def _time(value: Any) -> tuple[str | None, Decimal | None]:
    if value is None:
        return None, None
    if isinstance(value, str) and not _NUMBER.fullmatch(value):
        formatted = timestamp(value)
        delta = datetime.fromisoformat(formatted) - _EPOCH
        micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        return formatted, Decimal(micros) / 1_000_000
    if type(value) not in {int, float, Decimal, str}:
        raise DataError("invalid LiteLLM timestamp type")
    try:
        seconds = Decimal(str(value))
        if not seconds.is_finite():
            raise ValueError("nonfinite timestamp")
        # Keep the exact epoch value for duration; Call timestamp has microsecond
        # resolution. ES can have serialized a JSON decimal as an exact string.
        formatted = (_EPOCH + timedelta(microseconds=int(seconds * 1_000_000))).isoformat()
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise DataError("invalid LiteLLM timestamp") from exc
    return formatted, seconds


def _messages(value: Any, *, path="messages.") -> tuple[list[Message] | None, list[str]]:
    if isinstance(value, str):
        try:
            value = loads(value)
        except ValueError:
            return None, ["unstructured_or_redacted_messages"]
    if value is None or value == [] or value == {}:
        return None, ["missing_messages"]
    if not isinstance(value, list):
        return None, ["unstructured_or_redacted_messages"]
    messages = []
    for item in value:
        item = mapping(item, "message")
        if isinstance(item.get("content"), str) and item["content"] in REDACTED:
            return None, ["redacted_messages"]
        try:
            messages.append(message(item, path=path))
        except DataError as exc:
            return None, [str(exc)]
    return messages, []


def _request_body(record):
    proxy = json_object(record.get("proxy_server_request"), "proxy_server_request")
    if "body" in proxy:
        # A callback wrapper and a native SpendLogs body are distinct shapes.
        if any(key in proxy for key in ("messages", "input", "tools")):
            raise DataError("ambiguous_litellm_request_body")
        return json_object(proxy["body"], "proxy_server_request.body"), "proxy_server_request.body"
    return proxy, "proxy_server_request"


def _empty_messages(value):
    if isinstance(value, str):
        try:
            value = loads(value)
        except ValueError:
            return False
    return value is None or value == [] or value == {}


def _response_message(value):
    body = decode_response(value)
    if body is not None and body.get("provider_specific_fields") in ({}, {"refusal": None}):
        # LiteLLM's stream aggregator inserts this empty provider envelope even
        # when the upstream response has no refusal. Nonempty fields still fail
        # explicitly, so provider-specific content cannot disappear silently.
        body = {k: v for k, v in body.items() if k != "provider_specific_fields"}
    return body


class LiteLLMProtocol(ProtocolAdapter):
    name = "litellm"
    version = "litellm.logging.v4"

    def parse(self, record: dict[str, Any]) -> Parsed:
        call_type = identifier(record.get("call_type"), "call_type")
        # These are two source schemas, not interchangeable attempts in a trace.
        # Reject ambiguous combined records rather than silently choosing an ID.
        if "id" in record and "request_id" in record:
            raise DataError("ambiguous LiteLLM log shape")
        id_field = "request_id" if "request_id" in record else "id"
        call_id = identifier(record.get(id_field), id_field)
        metadata = json_object(record.get("metadata"), "metadata")
        body, body_path = _request_body(record)
        raw_messages = record.get("messages")
        message_path = "messages."
        if _empty_messages(raw_messages) and "messages" in body:
            raw_messages = body["messages"]
            message_path = f"{body_path}.messages."
        messages, gaps = _messages(raw_messages, path=message_path)
        if call_type not in {"completion", "acompletion"}:
            gaps.append("unsupported_call_type")
        parameters = json_object(record.get("model_parameters"), "model_parameters")
        options, option_sources, tools, option_gaps = request_options(
            ("model_parameters", parameters), (body_path, body)
        )
        gaps.extend(option_gaps)
        if any(_truncated(v) for v in (raw_messages, body, parameters)):
            gaps.append("truncated_request")
        status_value = record.get("status")
        if status_value not in {None, "", "success", "failure"}:
            raise DataError("unsupported LiteLLM status")
        status = {"success": "success", "failure": "failed"}.get(status_value, "unknown")
        if status == "failed":
            gaps.append("failed_call")
        cache_hit = record.get("cache_hit")
        if cache_hit is True or cache_hit == "True" or cache_hit == "true":
            gaps.append("cache_hit")
        elif (
            cache_hit is not None
            and cache_hit is not False
            and cache_hit not in {"", "False", "false", "None"}
        ):
            raise DataError("invalid LiteLLM cache_hit")
        start, start_seconds = _time(record.get("startTime"))
        _, end_seconds = _time(record.get("endTime"))
        duration = None
        if start_seconds is not None and end_seconds is not None:
            if end_seconds < start_seconds:
                raise DataError("endTime precedes startTime")
            duration = format((end_seconds - start_seconds) * 1000, "f")
        response = output_fields(
            _response_message(record.get("response")), path="response.choices.message."
        )
        if _truncated(record.get("response")):
            response["output_gaps"] = [*response.get("output_gaps", []), "truncated_output"]
        call = Call(
            call_id=call_id,
            protocol=self.name,
            model=text_or_none(record.get("model"), "model") or None,
            provider=text_or_none(record.get("custom_llm_provider"), "custom_llm_provider") or None,
            status=status,
            started_at=start,
            duration_ms=duration,
            messages=messages,
            **response,
            tools_json=tools,
            render_options=options,
            render_option_sources=option_sources,
            usage=usage_from(record, "prompt_tokens", "completion_tokens"),
            gaps=sorted(set(gaps)),
        )
        return Parsed([call], contexts=[ProtocolContext(metadata, "metadata")])
