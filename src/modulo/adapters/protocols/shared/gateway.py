"""Batch-local gateway body correlation shared by the two native log adapters."""

from modulo.core.domain.data_models import (
    DataError,
    Parsed,
    ProtocolContext,
    ProtocolEmission,
    Usage,
)
from modulo.core.domain.interfaces import MAX_RECORD_BYTES, ProtocolSession
from modulo.shared.serialization import digest, loads, ordered_json

from .gateway_body import build_call, exchange, response_fields
from .gateway_logs import body_object, native_header
from .sse import assemble_sse
from .usage_log import UsageLogProtocol


class GatewayProtocol(UsageLogProtocol):
    def decode(self, data):
        try:
            value = loads(data)
        except ValueError:
            return {"message": data.decode("utf-8")}
        if isinstance(value, str):
            return {"message": value}
        if not isinstance(value, dict):
            raise DataError("expected gateway log object or text")
        return value

    def parse(self, record):
        if record.get("schema_version") == "gateway.exchange.v1":
            call, context, _ = exchange(record, self.name)
            return Parsed([call], contexts=[context])
        return super().parse(record)

    def new_session(self):
        return GatewaySession(self)


class GatewaySession(ProtocolSession):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.groups = {}
        self.pending = None

    def _add(self, event):
        self.groups.setdefault((event.request_id, event.attempt_id), []).append(event)

    def _body(self, event):
        if event.kind == "sse":
            self._add(event)
            return []
        try:
            event.body = body_object(event.body)
        except (ValueError, TypeError):
            self.pending = event
            return []
        self._add(event)
        return []

    def feed(self, record):
        payload = record.payload
        if (
            self.pending
            and record.source_ordered
            and not any(key in payload for key in ("type", "schema_version", "message", "log"))
        ):
            # One API's converted request JSON can occupy its own physical line.
            pending, self.pending = self.pending, None
            pending.records.append(record)
            pending.body += "\n" + ordered_json(payload)
            if len(pending.body.encode()) > MAX_RECORD_BYTES:
                return [ProtocolEmission(pending.records, error="gateway_body_size_limit")]
            return self._body(pending)
        if "type" in payload or "schema_version" in payload:
            emitted = []
            if self.pending is not None:
                emitted.append(
                    ProtocolEmission(self.pending.records, error="incomplete_gateway_body")
                )
                self.pending = None
            try:
                if payload.get("schema_version") == "gateway.exchange.v1":
                    call, context, assembly = exchange(payload, self.adapter.name)
                    return [
                        *emitted,
                        ProtocolEmission(
                            [record], Parsed([call], contexts=[context]), assembly=assembly
                        ),
                    ]
                return [*emitted, *super().feed(record)]
            except (ValueError, TypeError, KeyError) as exc:
                reason = str(exc) if isinstance(exc, DataError) else "invalid_gateway_body"
                return [*emitted, ProtocolEmission([record], error=reason)]
        try:
            event, text = native_header(record, self.adapter.name)
        except DataError as exc:
            emitted = []
            if self.pending:
                emitted.append(
                    ProtocolEmission(self.pending.records, error="incomplete_gateway_body")
                )
                self.pending = None
            return [*emitted, ProtocolEmission([record], error=str(exc))]
        emitted = []
        if self.pending is not None:
            pending = self.pending
            self.pending = None
            # Continuations only inherit a header within an ordered file. ES
            # multiline events must have been assembled by the collector.
            if (
                event is None
                and record.source_ordered
                and all(r.source_ordered for r in pending.records)
            ):
                pending.records.append(record)
                pending.body += "\n" + text
                if len(pending.body.encode()) > MAX_RECORD_BYTES:
                    return [ProtocolEmission(pending.records, error="gateway_body_size_limit")]
                return self._body(pending)
            emitted.append(ProtocolEmission(pending.records, error="incomplete_gateway_body"))
        if event is None or event == "ignored":
            if event is None and text.lstrip().startswith(("{", "[")):
                emitted.append(ProtocolEmission([record], error="unrecognized_gateway_log"))
            else:
                emitted.append(ProtocolEmission([record], Parsed([], "non_body_log")))
        else:
            emitted.extend(self._body(event))
        return emitted

    @staticmethod
    def _unique(events):
        unique, seen, conflicts, duplicates = [], {}, [], 0
        for event in events:
            identity = (
                ("id", event.event_id)
                if event.event_id is not None
                else (
                    ("sequence", event.sequence_stream, event.sequence)
                    if event.sequence is not None
                    else None
                )
            )
            if identity is None or identity not in seen:
                unique.append(event)
                if identity is not None:
                    seen[identity] = event
                continue
            prior = seen[identity]
            if (prior.kind, prior.format, prior.body) != (event.kind, event.format, event.body):
                conflicts.append("conflicting_gateway_event_identity")
            prior.records.extend(event.records)
            duplicates += 1
        return unique, conflicts, duplicates

    def _assemble(self, key, events):
        request_id, attempt_id = key
        events, gaps, duplicates = self._unique(events)
        outbound = [e for e in events if e.kind == "upstream_request"]
        inbound = [e for e in events if e.kind == "ingress_request"]
        requests = outbound or inbound
        primary = requests[0] if requests else events[0]
        if len(outbound) > 1 or len(inbound) > 1:
            gaps.append("ambiguous_gateway_attempts")
        responses = [e for e in events if e.kind == "response"]
        stream = [e for e in events if e.kind == "sse"]
        output = {"output_gaps": ["missing_response_body"]}
        usage, model = Usage(), None
        sequence_basis = None
        response_complete = False
        if len(responses) > 1 or (responses and stream):
            gaps.append("ambiguous_gateway_responses")
        elif responses:
            output, usage, model = response_fields(responses[0].body, responses[0].format)
            response_complete = not output.get("output_gaps")
        elif stream:
            if (
                all(e.sequence is not None for e in stream)
                and len({e.sequence_stream for e in stream}) == 1
            ):
                stream.sort(key=lambda e: e.sequence)
                sequence_basis = stream[0].sequence_stream
            elif all(r.source_ordered for e in stream for r in e.records):
                stream.sort(key=lambda e: e.records[0].position)
                sequence_basis = "source_file_order"
            else:
                output = {"output_gaps": ["stream_order_unknown"]}
            if sequence_basis:
                if sum(len(e.body.encode()) for e in stream) > MAX_RECORD_BYTES:
                    output = {"output_gaps": ["gateway_stream_size_limit"]}
                else:
                    result = assemble_sse([e.body for e in stream])
                    output, usage, model = result.output, result.usage, result.model
                    response_complete = result.done and not output.get("output_gaps")
        if gaps:
            # A request ID may span retries. Never choose an attempt by matching
            # usage, model, text similarity, timestamp or arrival proximity.
            usage = Usage()
            output["output_gaps"] = sorted(set([*output.get("output_gaps", []), *gaps]))
            response_complete = False
        request = requests[0].body if requests else None
        call = build_call(
            self.adapter.name, request_id, attempt_id, request, output, usage, model, gaps=gaps
        )
        records = [*primary.records, *[r for e in events if e is not primary for r in e.records]]
        primary_record = primary.records[0]
        context = ProtocolContext(
            primary_record.payload.get("metadata", primary_record.original.get("metadata")),
            "metadata",
            "payload" if "metadata" in primary_record.payload else "record",
        )
        assembly = {
            "schema_version": "gateway-assembly.v1",
            "kind": "native_debug_log",
            "request_id": request_id,
            "attempt_id": attempt_id,
            "link_basis": "gateway_request_id"
            if attempt_id is None
            else "gateway_request_and_attempt_id",
            "request_stage": "upstream" if outbound else "ingress" if inbound else "missing",
            "request_source_refs": [r.source_ref for e in requests for r in e.records],
            "response_source_refs": [
                r.source_ref for e in [*responses, *stream] for r in e.records
            ],
            "response_kind": "sse" if stream else responses[0].format if responses else "missing",
            "sequence_basis": sequence_basis,
            "duplicate_events": duplicates,
            "response_complete": response_complete,
            "gaps": sorted(set(gaps)),
            "association_id": digest([self.adapter.name, request_id, attempt_id]),
        }
        return ProtocolEmission(records, Parsed([call], contexts=[context]), assembly=assembly)

    def finish(self, *, complete):
        emitted = []
        if self.pending:
            emitted.append(ProtocolEmission(self.pending.records, error="incomplete_gateway_body"))
            self.pending = None
        for key, events in self.groups.items():
            try:
                emitted.append(self._assemble(key, events))
            except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
                reason = str(exc) if isinstance(exc, DataError) else "invalid_gateway_body"
                emitted.append(
                    ProtocolEmission([r for e in events for r in e.records], error=reason)
                )
        self.groups.clear()
        return emitted
