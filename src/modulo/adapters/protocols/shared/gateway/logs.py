"""Pinned New API / One API debug log grammars; no time-based correlation."""

import re
from dataclasses import dataclass, field

from modulo.core.domain.data_models import DataError, ProtocolRecord
from modulo.shared.serialization import loads

PREFIX = r"\[(?P<level>[A-Z]+)\] (?P<time>\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2})"
NEW = re.compile(PREFIX + r" \| (?P<id>[^|]+) \| (?P<text>[\s\S]*)\Z")
ONE = re.compile(
    PREFIX
    + r"(?: \| (?P<id>[^|]+))? \| (?P<file>[^|]+\.go:\d+) "
    + r"\[(?P<function>[^\]]+)\] (?P<text>[\s\S]*)\Z"
)
ONE_ACCESS = re.compile(
    r"\[GIN\] \d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2}"
    r" \| [^|]+ \| \d{3} \| [^|]+ \| [^|]+ \|\s+[A-Z]+ /[^\r\n]*\Z"
)
LABELS = {
    "newapi": {
        "requestBody:": ("upstream_request", "openai"),
        "text request body:": ("upstream_request", "openai"),
        "upstream response body:": ("response", "openai"),
        "responseBody:": ("response", "auto"),
        "stream scanner data:": ("sse", "openai"),
    },
    "oneapi": {
        "request body:": ("ingress_request", "openai"),
        "converted request:": ("upstream_request", "auto"),
        "response body:": ("response", "ali"),
        "ollama response:": ("response", "ollama"),
        # Supplied opt-in instrumentation patch, explicitly identified in docs.
        "upstream response body:": ("response", "openai"),
        "stream response data:": ("sse", "openai"),
    },
}


@dataclass
class BodyEvent:
    request_id: str
    kind: str
    body: dict | str
    format: str
    records: list[ProtocolRecord] = field(default_factory=list)
    attempt_id: str | None = None
    event_id: str | None = None
    sequence: int | None = None
    sequence_stream: str | None = None


def native_header(record: ProtocolRecord, gateway: str):
    text = record.payload.get("message", record.payload.get("log"))
    if not isinstance(text, str):
        raise DataError("expected gateway message or exchange record")
    # Remove only the logger's final separator, not whitespace inside JSON text.
    text = text.rstrip("\r\n").removesuffix(" ")
    if gateway == "oneapi" and ONE_ACCESS.fullmatch(text):
        return "ignored", text
    match = (NEW if gateway == "newapi" else ONE).fullmatch(text)
    if not match:
        return None, text
    body = match["text"]
    for label, (kind, format) in LABELS[gateway].items():
        if body.startswith(label):
            request_id = (match["id"] or "").strip()
            if not request_id or request_id == "SYSTEM":
                raise DataError("gateway_request_id_missing")
            payload = body[len(label) :]
            payload = payload[1:] if payload.startswith(" ") else payload
            envelope = {**record.original, **record.payload}
            attempt = envelope.get("attempt_id")
            event = envelope.get("event") or {}
            if not isinstance(event, dict):
                raise DataError("invalid_gateway_event_metadata")
            event_id = envelope.get("event_id", event.get("id"))
            sequence = envelope.get("sequence", event.get("sequence"))
            sequence_stream = "event.sequence" if sequence is not None else None
            log = envelope.get("log")
            if sequence is None and isinstance(log, dict) and "offset" in log:
                file = log.get("file") or {}
                if isinstance(file, dict) and isinstance(file.get("path"), str):
                    sequence, sequence_stream = log["offset"], "file:" + file["path"]
            for value in (attempt, event_id):
                if value is not None and (not isinstance(value, str) or not value):
                    raise DataError("invalid_gateway_event_identity")
            if sequence is not None and (type(sequence) is not int or sequence < 0):
                raise DataError("invalid_gateway_event_sequence")
            return BodyEvent(
                request_id,
                kind,
                payload,
                format,
                [record],
                attempt,
                event_id,
                sequence,
                sequence_stream,
            ), None
    return "ignored", body


def body_object(value):
    value = loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise DataError("gateway_body_not_object")
    return value
