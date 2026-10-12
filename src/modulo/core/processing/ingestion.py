"""Read through source interfaces, normalize calls and retain every observed identity variant."""

from collections import Counter
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from modulo.shared.serialization import digest, loads

from ..domain.bindings import Bindings
from ..domain.data_models import (
    DataError,
    ProtocolContext,
    ProtocolEmission,
    ProtocolRecord,
    SourceError,
)
from ..domain.interfaces import ProtocolAdapter, ProtocolSession, RawArchive, SourceAdapter
from ..domain.platforms import PlatformConfig


@dataclass
class CollectedBatch:
    calls: MutableMapping[str, dict]
    dimensions: MutableMapping[str, list]
    counters: Counter
    diagnostics: Sequence[dict]
    complete: bool


def record_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataError("expected object: record")
    return value


def extract_payload(data: bytes | dict, field: str | None, decoder=None) -> dict[str, Any]:
    parsed = loads(data) if isinstance(data, bytes) else data
    if field:
        for part in field.split("."):
            if not isinstance(parsed, dict) or part not in parsed:
                raise DataError("payload_field_missing")
            parsed = parsed[part]
        if isinstance(parsed, str):
            parsed = decoder(parsed.encode()) if decoder else loads(parsed)
    return record_object(parsed)


def collect_calls(
    source: SourceAdapter,
    protocol: ProtocolAdapter,
    archive: RawArchive,
    *,
    namespace: str,
    payload_field: str | None,
    max_records: int,
    platform_config: PlatformConfig,
    bindings: Bindings | None = None,
    working=None,
    max_record_bytes: int = 8 * 1024 * 1024,
) -> CollectedBatch:
    bindings = bindings or Bindings()
    diagnostics = working.sequence("diagnostics") if working is not None else []
    counters: Counter = Counter(
        raw_records=0, ignored_records=0, invalid_records=0, duplicate_calls=0, identity_conflicts=0
    )
    calls = working.mapping("calls") if working is not None else {}
    observed_dimensions = working.mapping("dimensions") if working is not None else {}
    complete = False
    decoder = getattr(protocol, "decode", lambda data: record_object(loads(data)))
    session = (
        protocol.new_session() if hasattr(protocol, "new_session") else ProtocolSession(protocol)
    )
    if working is not None:
        session.configure_storage(working, max_record_bytes)

    def invalid(references, reason="invalid_record", error_type=None):
        counters["invalid_records"] += len(references)
        for reference in references:
            diagnostics.append(
                {
                    "level": "error",
                    "reason": reason,
                    "error_type": error_type,
                    "source_ref": reference,
                }
            )

    def consume(emission: ProtocolEmission):
        references = [r.source_ref for r in emission.records]
        if not references:
            raise DataError("protocol emission has no source records")
        if emission.error:
            invalid(references, emission.error)
            return
        parsed = emission.parsed
        if parsed is None:
            raise DataError("protocol emission is missing parsed result")
        contexts = parsed.contexts or [ProtocolContext() for _ in parsed.calls]
        if len(contexts) != len(parsed.calls):
            raise DataError("protocol context count differs from call count")
        if parsed.ignored:
            counters["ignored_records"] += len(references)
            for reference in references:
                diagnostics.append(
                    {"level": "info", "reason": parsed.ignored, "source_ref": reference}
                )
        # The primary request supplies platform context. Other fragments remain
        # source evidence; response-only fields never overwrite the request tag.
        primary = emission.records[0]
        for call, context in zip(parsed.calls, contexts, strict=True):
            resolution = bindings.resolve(call, platform_config, primary.original, context)
            tag = resolution.tag
            call = call.model_copy(update={"platform": tag.value, "platform_source": tag.source})
            key = digest([namespace, protocol.name, call.call_id])
            body = call.model_dump()
            pairs = observed_dimensions.get(key, [])
            pair = [call.platform, call.model]
            if pair not in pairs:
                pairs.append(pair)
                observed_dimensions[key] = pairs
            tag_records = [{**item, "source_ref": primary.source_ref} for item in resolution.tags]
            for item in tag_records:
                if not item["reason"] or item["source"] == "none":
                    continue
                diagnostics.append(
                    {
                        "level": "warning",
                        "reason": item["reason"],
                        "call_key": key,
                        "source_ref": primary.source_ref,
                    }
                )
            if resolution.error:
                diagnostics.append(
                    {
                        "level": "error",
                        "reason": resolution.error,
                        "call_key": key,
                        "binding_matches": resolution.matches,
                        "platform_tags": tag_records,
                        "source_ref": primary.source_ref,
                    }
                )
            if key in calls:
                prior = calls[key]
                prior["source_refs"].extend(references)
                prior["platform_tags"].extend(tag_records)
                if (
                    prior["call"] == body
                    and prior["binding_matches"] == resolution.matches
                    and prior["binding_error"] == resolution.error
                ):
                    counters["duplicate_calls"] += 1
                else:
                    if not prior["identity_conflict"]:
                        counters["identity_conflicts"] += 1
                    prior["identity_conflict"] = True
                    diagnostics.append(
                        {
                            "level": "error",
                            "reason": "identity_conflict",
                            "call_key": key,
                            "source_ref": primary.source_ref,
                        }
                    )
                if emission.assembly:
                    prior.setdefault("assemblies", []).append(emission.assembly)
                calls[key] = prior
            else:
                calls[key] = {
                    "call_key": key,
                    "call": body,
                    "source_refs": references,
                    "identity_conflict": False,
                    "platform_tags": tag_records,
                    "binding_matches": resolution.matches,
                    "binding_error": resolution.error,
                }
                if emission.assembly:
                    row = calls[key]
                    row["assemblies"] = [emission.assembly]
                    calls[key] = row

    records = source.read()
    try:
        for record in records:
            if max_records and counters["raw_records"] >= max_records:
                raise SourceError("record_budget_exceeded")
            counters["raw_records"] += 1
            reference = archive(record)
            try:
                original = record_object(decoder(record.data))
                current = ProtocolRecord(
                    extract_payload(original, payload_field, decoder),
                    original,
                    reference,
                    counters["raw_records"],
                    getattr(source, "ordered_records", False),
                )
                for emission in session.feed(current):
                    consume(emission)
            except (ValueError, TypeError, KeyError, ValidationError, RecursionError) as exc:
                invalid([reference], error_type=type(exc).__name__)
        complete = True
    except (SourceError, OSError) as exc:
        diagnostics.append(
            {
                "level": "error",
                "reason": "source_incomplete",
                "error_type": type(exc).__name__,
                "detail": str(exc) if isinstance(exc, SourceError) else "io_error",
            }
        )
    finally:
        try:
            close = getattr(records, "close", None)
            if close is not None:
                close()
        except (SourceError, OSError) as exc:
            complete = False
            diagnostics.append(
                {
                    "level": "error",
                    "reason": "source_cleanup_failed",
                    "error_type": type(exc).__name__,
                }
            )

    try:
        for emission in session.finish(complete=complete):
            consume(emission)
    except (ValueError, TypeError, KeyError, ValidationError, RecursionError) as exc:
        complete = False
        diagnostics.append(
            {
                "level": "error",
                "reason": "protocol_finalize_failed",
                "error_type": type(exc).__name__,
            }
        )

    explain = getattr(protocol, "report_warnings", None)
    if explain:
        diagnostics.extend(explain(calls.values(), diagnostics))
    return CollectedBatch(calls, observed_dimensions, counters, diagnostics, complete)
