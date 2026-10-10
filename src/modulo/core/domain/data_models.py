"""Data structures for normalized calls, measurements and log records."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class Message(Contract):
    role: Literal["system", "user", "assistant", "tool", "developer", "root"]
    content: str
    reasoning_content: str | None = None
    reasoning_source: str | None = None
    # JSON strings preserve nested key order and numeric text across bundle replay.
    tool_calls_json: str | None = None
    tool_call_id: str | None = None
    name: str | None = None


class Usage(Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provenance: Literal["source_reported"] = "source_reported"


class Call(Contract):
    schema_version: Literal["call.v3"] = "call.v3"
    call_id: str
    protocol: str
    model: str | None = None
    provider: str | None = None
    platform: str | None = None
    platform_source: str = "none"
    status: Literal["success", "failed", "unknown"] = "unknown"
    started_at: str | None = None
    duration_ms: str | None = None
    messages: list[Message] | None = None
    output_text: str | None = None
    output_reasoning: str | None = None
    output_reasoning_source: str | None = None
    output_tool_calls_json: str | None = None
    output_gaps: list[str] = Field(default_factory=list)
    tools_json: str | None = None
    render_options: dict[str, Any] = Field(default_factory=dict)
    render_option_sources: dict[str, list[str]] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    gaps: list[str] = Field(default_factory=list)


class Measurement(Contract):
    schema_version: Literal["measurement.v2"] = "measurement.v2"
    execution: Literal["completed", "not_run", "error"] = "not_run"
    applicability: Literal["supported", "unsupported", "unknown"] = "unknown"
    verdict: Literal["CONSISTENT", "DEVIATION", "UNVERIFIABLE"] = "UNVERIFIABLE"
    scope: str = "local_reference_input"
    qualification: str = "reference_fixture_only"
    model_identity: str = "UNVERIFIED"
    reason: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    asset_revision: str | None = None
    asset_hash: str | None = None
    template_hash: str | None = None
    rendered_input_sha256: str | None = None
    input_ids_sha256: str | None = None
    input_tokens: int | None = None
    reported_input_tokens: int | None = None
    input_delta: int | None = None
    visible_output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    reasoning_output_source: str | None = None
    output_execution: Literal["completed", "not_run", "error"] = "not_run"
    output_reason: str | None = None
    # Counts above are separate text components, not an output billing verdict.
    render_options: dict[str, Any] = Field(default_factory=dict)
    render_options_sha256: str | None = None
    render_option_sources: dict[str, list[str]] = Field(default_factory=dict)


@dataclass(frozen=True)
class RawRecord:
    locator: str
    data: bytes


@dataclass(frozen=True)
class ProtocolContext:
    # Ephemeral parsing context: only the selected platform value and its path
    # are persisted, not arbitrary metadata copied into every normalized call.
    metadata: Any = None
    metadata_path: str | None = None
    metadata_root: Literal["payload", "record"] = "payload"


@dataclass(frozen=True)
class Parsed:
    calls: list[Call]
    ignored: str | None = None
    contexts: list[ProtocolContext] = field(default_factory=list)


@dataclass(frozen=True)
class ProtocolRecord:
    payload: dict[str, Any]
    original: dict[str, Any]
    source_ref: dict[str, Any]
    position: int
    source_ordered: bool = False


@dataclass(frozen=True)
class ProtocolEmission:
    records: list[ProtocolRecord]
    parsed: Parsed | None = None
    error: str | None = None
    assembly: dict[str, Any] | None = None


class DataError(ValueError):
    """A record cannot be interpreted within the declared input contract."""


class SourceError(RuntimeError):
    """The reader cannot establish completion of the requested batch."""


@dataclass(frozen=True)
class BatchResult:
    """Semantic output of one bounded batch; persistence is handled outside core."""

    rows: list[dict]
    diagnostics: list[dict]
    summary: dict
