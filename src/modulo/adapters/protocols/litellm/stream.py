"""Assemble already logged LiteLLM callback chunks within one import."""

from dataclasses import asdict

from modulo.adapters.protocols.shared.api_bodies.common import remap_sources
from modulo.adapters.protocols.shared.api_bodies.parsing import assemble_sse
from modulo.core.domain.data_models import (
    DataError,
    ProtocolContext,
    ProtocolEmission,
    ProtocolRecord,
)
from modulo.core.domain.interfaces import ProtocolSession
from modulo.shared.serialization import digest, loads, ordered_json

from ..shared.fields import identifier, mapping


def callback_record(record):
    """Read a serialized CustomLogger argument pair; no callback is installed here."""
    if "kwargs" not in record or "response_obj" not in record:
        return record
    kwargs = mapping(record["kwargs"], "kwargs")
    params = mapping(kwargs.get("litellm_params") or {}, "litellm_params")
    call_id = kwargs.get("litellm_call_id", params.get("litellm_call_id"))
    return {
        **{k: v for k, v in record.items() if k not in {"kwargs", "response_obj"}},
        "id": identifier(call_id, "litellm_call_id"),
        "litellm_call_id": call_id,
        "call_type": kwargs.get("call_type"),
        "model": kwargs.get("model"),
        "messages": kwargs.get("messages"),
        "model_parameters": kwargs.get("optional_params", {}),
        "proxy_server_request": kwargs.get(
            "proxy_server_request", params.get("proxy_server_request")
        ),
        "metadata": params.get("metadata"),
        "custom_llm_provider": kwargs.get("custom_llm_provider", params.get("custom_llm_provider")),
        "cache_hit": kwargs.get("cache_hit"),
        "response": record["response_obj"],
        "startTime": record.get("start_time"),
        "endTime": record.get("end_time"),
    }


def callback_sources(call, context, original):
    if "kwargs" not in original or "response_obj" not in original:
        return call, context
    kwargs = original["kwargs"]
    origins = {
        "messages": "kwargs.messages",
        "model_parameters": "kwargs.optional_params",
        "response": "response_obj",
        "proxy_server_request": (
            "kwargs.proxy_server_request"
            if "proxy_server_request" in kwargs
            else "kwargs.litellm_params.proxy_server_request"
        ),
    }

    def path(value):
        if value is None:
            return None
        parts = value.split(",")
        for i, part in enumerate(parts):
            for old, new in origins.items():
                if part == old or part.startswith(old + "."):
                    parts[i] = new + part[len(old) :]
                    break
        return ",".join(parts)

    changes = {
        "messages": [remap_sources(m, origins) for m in call.messages] if call.messages else None,
        "expected_output_candidates_sources": [
            path(p) for p in call.expected_output_candidates_sources
        ],
        "output_reasoning_source": path(call.output_reasoning_source),
        "output_stop_reason_source": path(call.output_stop_reason_source),
        "output_candidates": [
            c.model_copy(
                update={
                    "output_reasoning_source": path(c.output_reasoning_source),
                    "output_stop_reason_source": path(c.output_stop_reason_source),
                }
            )
            for c in call.output_candidates
        ],
    }
    for field in ("render_option_sources", "generation_option_sources"):
        changes[field] = {
            key: [path(p) for p in paths] for key, paths in getattr(call, field).items()
        }
    return call.model_copy(update=changes), ProtocolContext(
        context.metadata, "kwargs.litellm_params.metadata"
    )


def is_stream(value):
    if isinstance(value, list):
        return True
    if isinstance(value, str):
        return not value.strip() or value.lstrip().startswith(
            ("data:", "event:", "[DONE]", ":", "id:", "retry:")
        )
    if not isinstance(value, dict):
        return False
    kind = value.get("type", "")
    return (
        value.get("object") == "chat.completion.chunk"
        or isinstance(kind, str)
        and (
            kind.startswith("response.")
            or kind
            in {
                "message_start",
                "message_delta",
                "message_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "ping",
            }
        )
        or any(isinstance(c, dict) and "delta" in c for c in value.get("choices", []) or [])
    )


def stream_response(value):
    events = value if isinstance(value, list) else [value]
    lines = []
    decoded = all(isinstance(event, dict) for event in events)
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("choices"), list):
            choices = []
            for choice in event["choices"]:
                choice = dict(mapping(choice, "choice"))
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    delta = {k: v for k, v in delta.items() if v is not None}
                    if delta.get("provider_specific_fields") in ({}, {"refusal": None}):
                        delta.pop("provider_specific_fields", None)
                    choice["delta"] = delta
                choices.append(choice)
            event = {**event, "choices": choices}
        if isinstance(event, dict):
            lines.extend(["data: " + ordered_json(event), ""])
        elif isinstance(event, str):
            lines.extend(event.splitlines())
            lines.append("")
        else:
            raise DataError("invalid_litellm_stream_event")
    return assemble_sse(lines, require_done=not decoded)


class LiteLLMSession(ProtocolSession):
    def __init__(self, adapter):
        super().__init__(adapter)

    def feed(self, record):
        payload = callback_record(record.payload)
        key = payload.get("litellm_call_id") or payload.get("request_id", payload.get("id"))
        identifier(key, "litellm_call_id/id/request_id")
        attempt = payload.get("attempt_id")
        if attempt is not None:
            identifier(attempt, "attempt_id")
        self.events.append((key, attempt), {"records": [asdict(record)], "payload": payload})
        return []

    def finish_group(self, key, values, complete):
        entries = [(ProtocolRecord(**entry["records"][0]), entry["payload"]) for entry in values]
        if not any(is_stream(payload.get("response")) for _, payload in entries):
            for record, _payload in entries:
                try:
                    yield ProtocolEmission([record], self.adapter.parse(record.payload))
                except (ValueError, TypeError, KeyError) as exc:
                    yield ProtocolEmission(
                        [record],
                        error=str(exc) if isinstance(exc, DataError) else "invalid_litellm_record",
                    )
            return
        try:
            yield self._assemble(key, entries, complete)
        except (ValueError, TypeError, KeyError) as exc:
            yield ProtocolEmission(
                [r for r, _ in entries],
                error=str(exc) if isinstance(exc, DataError) else "invalid_litellm_stream",
            )

    def _assemble(self, key, entries, complete):
        fragments, terminals, seen, gaps = [], [], {}, []
        duplicates = 0
        for record, payload in entries:
            response = payload.get("response")
            if not is_stream(response):
                if response is not None:
                    terminals.append(payload)
                continue
            envelope = {**record.original, **payload}
            event = mapping(envelope.get("event") or {}, "event")
            sequence = envelope.get("sequence", event.get("sequence"))
            basis = "event.sequence" if sequence is not None else None
            if sequence is None and isinstance(response, dict):
                sequence = response.get("sequence_number")
                if sequence is not None:
                    basis = "response.sequence_number"
            log = envelope.get("log")
            if sequence is None and isinstance(log, dict) and "offset" in log:
                path = mapping(log.get("file") or {}, "log.file").get("path")
                if isinstance(path, str):
                    sequence, basis = log["offset"], "file:" + path
            if sequence is not None and (type(sequence) is not int or sequence < 0):
                raise DataError("invalid_litellm_stream_sequence")
            event_id = envelope.get("event_id", event.get("id"))
            if event_id is not None:
                identifier(event_id, "event_id")
            identity = ("id", event_id) if event_id else ((basis, sequence) if basis else None)
            if identity is not None and identity in seen:
                if ordered_json(seen[identity]) != ordered_json(response):
                    gaps.append("conflicting_litellm_stream_event")
                duplicates += 1
                continue
            if identity is not None:
                seen[identity] = response
            fragments.append((record, response, sequence, basis))
        if len(fragments) == 1 and isinstance(fragments[0][1], (str, list)):
            basis = "response_array_or_sse_order"
        elif (
            fragments
            and all(f[2] is not None for f in fragments)
            and len({f[3] for f in fragments}) == 1
        ):
            fragments.sort(key=lambda f: f[2])
            basis = fragments[0][3]
            if len({f[2] for f in fragments}) != len(fragments):
                gaps.append("conflicting_litellm_stream_sequence")
        elif all(f[0].source_ordered for f in fragments):
            fragments.sort(key=lambda f: f[0].position)
            basis = "source_file_order"
        else:
            gaps.append("stream_order_unknown")
            basis = None
        if sum(len(ordered_json(f[1]).encode()) for f in fragments) > self.max_group_bytes:
            gaps.append("litellm_stream_size_limit")
        if not complete:
            gaps.append("source_incomplete")
        payloads = [p for _, p in entries]
        # Prefer a record with actual request data; response-only events do not
        # overwrite it. Every other request snapshot must agree.
        parsed = [self.adapter.parse({**p, "response": None}).calls[0] for p in payloads]
        request_indexes = [i for i, call in enumerate(parsed) if call.messages is not None]
        primary = request_indexes[0] if request_indexes else 0
        request_gaps = []
        for field in (
            "model",
            "messages",
            "tools_json",
            "render_options_json",
            "generation_options",
            "cache_hit",
        ):
            values = {ordered_json(parsed[i].model_dump()[field]) for i in request_indexes}
            if len(values) > 1:
                request_gaps.append("conflicting_litellm_stream_request")
        response = None
        if not gaps:
            response = []
            for _, value, _, _ in fragments:
                response.extend(value if isinstance(value, list) else [value])
        merged = {**payloads[primary], "response": response}
        # Intermediate stream logging objects can carry progressive/zero usage.
        # Only terminal logging payloads report the whole request; raw stream
        # usage is read from the ordered response events by the shared parser.
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            merged.pop(field, None)
            values = {p[field] for p in terminals if p.get(field) is not None}
            if len(values) > 1:
                request_gaps.append("conflicting_litellm_" + field)
                merged.pop(field, None)
            elif values:
                merged[field] = values.pop()
        terminal_values = {ordered_json(p["response"]) for p in terminals}
        if len(terminal_values) > 1:
            gaps.append("conflicting_litellm_terminal_response")
        elif terminal_values:
            terminal = terminals[0]["response"]
            # A complete terminal response is usable on its own. If both forms
            # are complete, their generated contents must agree.
            if response is not None:
                streamed = self.adapter.parse(merged).calls[0]
                final = self.adapter.parse({**merged, "response": terminal}).calls[0]
                if not streamed.output_gaps and _output_content(streamed) != _output_content(final):
                    gaps.append("conflicting_litellm_terminal_response")
            merged["response"] = terminal
            if gaps == ["stream_order_unknown"]:
                gaps.clear()
        result = self.adapter.parse(merged)
        call = result.calls[0]
        call = call.model_copy(
            update={
                "call_id": str(key[0]) if key[1] is None else str(key[0]) + "/attempt/" + key[1],
                "gaps": sorted(set([*call.gaps, *request_gaps])),
                "output_gaps": sorted(set([*call.output_gaps, *gaps, *request_gaps])),
            }
        )
        call, context = callback_sources(call, result.contexts[0], entries[primary][0].payload)
        return ProtocolEmission(
            [entries[primary][0], *[r for i, (r, _) in enumerate(entries) if i != primary]],
            type(result)([call], contexts=[context]),
            assembly={
                "kind": "litellm_stream",
                "call_id": key[0],
                "attempt_id": key[1],
                "link_basis": (
                    "litellm_call_id"
                    if payloads[primary].get("litellm_call_id")
                    else "request_id"
                    if "request_id" in payloads[primary]
                    else "id"
                ),
                "sequence_basis": basis,
                "duplicate_events": duplicates,
                "source_read_complete": complete,
                "terminal_response_present": bool(terminals),
                "response_complete": not call.output_gaps,
                "association_id": digest(key),
            },
        )


def _output_content(call):
    result = []
    for candidate in call.output_candidates or [call]:
        result.append(
            (
                candidate.output_text,
                candidate.output_reasoning,
                _tool_content(candidate.output_tool_calls_json),
                candidate.output_finish_reason,
                candidate.output_stop_reason,
                [
                    (m.role, m.content, m.reasoning_content, _tool_content(m.tool_calls_json))
                    for m in candidate.output_messages
                ],
            )
        )
    return result


def _tool_content(raw):
    # HTTP/SDK wrapper key order is not generated text. Argument strings remain
    # byte-exact; object arguments also keep their original key order.
    result = []
    for tool in loads(raw) if raw else []:
        function = tool["function"]
        arguments = function["arguments"]
        result.append(
            (
                tool.get("id"),
                tool.get("type", "function"),
                function["name"],
                arguments if isinstance(arguments, str) else ordered_json(arguments),
            )
        )
    return result
