"""Strict field helpers shared by protocol implementations."""

from datetime import UTC, datetime
from typing import Any

from modulo.core.domain.data_models import DataError, Usage
from modulo.shared.serialization import loads


def mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataError(f"expected object: {field}")
    return value


def json_object(value: Any, field: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = loads(value)
        except ValueError as exc:
            raise DataError(f"invalid JSON object: {field}") from exc
    return mapping(value, field)


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataError(f"missing or invalid identifier: {field}")
    return value


def text_or_none(value: Any, field: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise DataError(f"expected text: {field}")
    return value


def usage_from(value: Any, input_key: str, output_key: str) -> Usage:
    value = mapping(value, "usage") if value is not None else {}
    values = [value.get(input_key), value.get(output_key), value.get("total_tokens")]
    if any(v is not None and (type(v) is not int or v < 0) for v in values):
        raise DataError("usage counters must be nonnegative integers or null")
    return Usage(input_tokens=values[0], output_tokens=values[1], total_tokens=values[2])


def timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is int:
        try:
            return datetime.fromtimestamp(value, UTC).isoformat()
        except (OverflowError, OSError, ValueError) as exc:
            raise DataError("invalid timestamp") from exc
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DataError("invalid timestamp") from exc
        if parsed.tzinfo is None:
            raise DataError("timestamp must include timezone")
        return parsed.astimezone(UTC).isoformat()
    raise DataError("invalid timestamp type")
