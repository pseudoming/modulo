"""Batch-local gateway body correlation shared by the two native log adapters."""

from dataclasses import asdict

from modulo.adapters.protocols.shared.api_bodies.parsing import assemble_sse
from modulo.adapters.protocols.shared.gateway.body import build_call, exchange, response_fields
from modulo.adapters.protocols.shared.gateway.logs import BodyEvent, body_object, native_header
from modulo.adapters.protocols.shared.gateway.usage import UsageLogProtocol
from modulo.core.domain.data_models import (
    DataError,
    Parsed,
    ProtocolContext,
    ProtocolEmission,
    ProtocolRecord,
    Usage,
)
from modulo.core.domain.interfaces import ProtocolSession
from modulo.shared.serialization import digest, loads, ordered_json


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

    def report_warnings(self, calls, diagnostics):
        findings = {item["reason"] for item in diagnostics}
        counts = [0] * len(_REPORT_WARNINGS)
        samples = [set() for _ in _REPORT_WARNINGS]
        for row in calls:
            call = row["call"]
            gaps = [*call["gaps"], *call["output_gaps"]]
            if (
                not call["output_gaps"]
                and not any(
                    call.get(k)
                    for k in (
                        "output_text",
                        "output_reasoning",
                        "output_tool_calls_json",
                        "output_candidates",
                    )
                )
                and call.get("output_text") is None
            ):
                gaps.append("missing_response_body")
            findings.update(gaps)
            for index, (_, reasons, *_) in enumerate(_REPORT_WARNINGS):
                if reasons.intersection(gaps):
                    counts[index] += 1
                    if len(samples[index]) < 20:
                        samples[index].add(call["call_id"])
        warnings = []
        for index, (code, reasons, fields, message, action) in enumerate(_REPORT_WARNINGS):
            observed = sorted(reasons & findings)
            if not observed:
                continue
            warnings.append(
                {
                    "level": "warning",
                    "reason": code,
                    "protocol": self.name,
                    "scope": "imported_records",
                    "observed_reasons": observed,
                    "missing_fields": fields,
                    "call_ids": sorted(samples[index]),
                    "call_count": counts[index],
                    "call_ids_truncated": counts[index] > len(samples[index]),
                    "message": message,
                    "action": action,
                }
            )
        return warnings


_REPORT_WARNINGS = (
    (
        "gateway_request_body_required",
        {"missing_messages", "missing_request_body", "incomplete_gateway_body"},
        ["request body: messages/input, tools, model parameters"],
        "缺少完整请求正文，输入复算不完整；消费记录中的 usage 不能代替请求正文。",
        "如需完整计算，请在已有网关日志中启用并导出请求正文，保留消息、工具定义及模型参数；补齐后重新导入。",
    ),
    (
        "gateway_response_body_required",
        {"missing_response_body", "missing_output", "response_capture_incomplete"},
        ["response body or complete stream"],
        "缺少完整响应正文，输出复算不完整。",
        "如需完整计算，请导出已有响应正文或全部流片段（包括 usage 与结束事件），"
        "与请求日志一起重新导入。",
    ),
    (
        "gateway_correlation_id_required",
        {"gateway_request_id_missing"},
        ["request_id"],
        "请求和响应缺少共同的请求 ID，无法可靠关联，相关记录未参与完整复算。",
        "请导出含同一 request_id 的请求和响应日志，"
        "或将已确认关联的正文放入支持的 exchange 记录后重新导入；不要按时间或 token 数凑配。",
    ),
    (
        "gateway_attempt_id_required",
        {
            "ambiguous_gateway_attempts",
            "ambiguous_gateway_responses",
            "conflicting_gateway_event_identity",
        },
        ["attempt_id", "event_id"],
        "同一请求存在重试或冲突记录，无法确定本次响应对应哪次请求，计算不完整。",
        "请保留每次尝试独立的 attempt_id，以及事件 ID；"
        "请求、响应和流片段使用相同的尝试 ID，排除导出冲突后重新导入。",
    ),
    (
        "gateway_stream_order_required",
        {"stream_order_unknown"},
        ["event.sequence or log.file.path + log.offset"],
        "流片段缺少可靠顺序，无法恢复完整回答和工具参数，输出复算不完整。",
        "请保留 event.sequence，或同一日志文件的 log.file.path 与 log.offset；"
        "也可按原始顺序导入完整日志文件，再重新计算。",
    ),
    (
        "gateway_stream_end_required",
        {"stream_done_missing", "stream_finish_missing", "stream_incomplete"},
        ["stream finish event", "usage"],
        "流日志缺少结束事件，不能把已有片段当作完整输出。",
        "请扩大导出的时间范围，包含结束事件和 usage，再将整次调用的日志一起重新导入。",
    ),
)


class GatewaySession(ProtocolSession):
    def __init__(self, adapter):
        super().__init__(adapter)
        self.pending = None

    def _add(self, event):
        self.events.append((event.request_id, event.attempt_id), asdict(event))

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
            if len(pending.body.encode()) > self.max_group_bytes:
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
                if len(pending.body.encode()) > self.max_group_bytes:
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
                if sum(len(e.body.encode()) for e in stream) > self.max_group_bytes:
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
        if self.pending:
            yield ProtocolEmission(self.pending.records, error="incomplete_gateway_body")
            self.pending = None
        yield from super().finish(complete=complete)

    def finish_group(self, key, entries, complete):
        events = [
            BodyEvent(**{**entry, "records": [ProtocolRecord(**r) for r in entry["records"]]})
            for entry in entries
        ]
        try:
            yield self._assemble(key, events)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
            reason = str(exc) if isinstance(exc, DataError) else "invalid_gateway_body"
            yield ProtocolEmission([r for e in events for r in e.records], error=reason)
