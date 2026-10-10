"""Strict JSON encoding and content digests shared across the package."""

import hashlib
import json
from decimal import Decimal
from typing import Any


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def loads(data: str | bytes) -> Any:
    result = json.loads(
        data, object_pairs_hook=_object, parse_float=Decimal, parse_constant=_invalid
    )
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            value.encode("utf-8")  # Reject unpaired surrogate escapes at the input boundary.
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return result


def dumps(value: Any) -> str:
    # Decimal is stored as an exact string, never routed through float.
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_default,
    )


def _default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def bytes_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordered_json(
    value: Any, *, ensure_ascii: bool = False, indent=None, separators=None, **_
) -> str:
    """JSON for prompt structures: retain insertion order and exact Decimal values.

    Artifact JSON remains sorted separately. Prompt JSON must not turn a Decimal
    into a quoted string or reorder tool parameters when it is saved and read.
    """
    comma, colon = separators or (", ", ": ")
    if indent is not None:
        raise ValueError("indented prompt JSON is not supported")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite JSON number")
        return str(value)
    if isinstance(value, dict):
        return (
            "{"
            + comma.join(
                json.dumps(k, ensure_ascii=ensure_ascii)
                + colon
                + ordered_json(v, ensure_ascii=ensure_ascii, separators=(comma, colon))
                for k, v in value.items()
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return (
            "["
            + comma.join(
                ordered_json(v, ensure_ascii=ensure_ascii, separators=(comma, colon)) for v in value
            )
            + "]"
        )
    return json.dumps(value, ensure_ascii=ensure_ascii, allow_nan=False)
