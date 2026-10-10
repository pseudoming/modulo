"""Native consumption rows shared by the One API family, without inventing prompts."""

from typing import Any, ClassVar, Literal

from modulo.core.domain.data_models import Call, DataError, Parsed, ProtocolContext
from modulo.core.domain.interfaces import ProtocolAdapter

from .fields import json_object, text_or_none, usage_from


class UsageLogProtocol(ProtocolAdapter):
    call_types: ClassVar[dict[int, Literal["failed", "unknown"]]]
    ignored_types: ClassVar[frozenset[int]]
    duration_field: ClassVar[str]
    duration_scale: ClassVar[int]
    metadata_field: ClassVar[str | None] = None

    def parse(self, record: dict[str, Any]) -> Parsed:
        kind = record.get("type")
        if type(kind) is not int:
            raise DataError("expected integer log type")
        if kind in self.ignored_types:
            return Parsed([], "non_consumption_log")
        if kind not in self.call_types:
            raise DataError("unsupported log type")
        # Only stable database/admin-export row IDs are supported. User-facing
        # New API lists can replace IDs with display positions; see input docs.
        row_id = record.get("id")
        if type(row_id) is not int or row_id <= 0:
            raise DataError("expected positive stable log row id")
        duration = record.get(self.duration_field)
        if duration is not None and (type(duration) is not int or duration < 0):
            raise DataError(f"expected nonnegative integer: {self.duration_field}")
        context = ProtocolContext()
        if self.metadata_field:
            context = ProtocolContext(
                json_object(record.get(self.metadata_field), self.metadata_field),
                self.metadata_field,
            )
        call = Call(
            call_id=str(row_id),
            protocol=self.name,
            model=text_or_none(record.get("model_name"), "model_name") or None,
            status=self.call_types[kind],
            duration_ms=str(duration * self.duration_scale) if duration is not None else None,
            # created_at is log creation time, not request start. content is an
            # accounting/error description, not a serialized conversation.
            usage=usage_from(record, "prompt_tokens", "completion_tokens"),
            gaps=["missing_messages"],
        )
        return Parsed([call], contexts=[context])
