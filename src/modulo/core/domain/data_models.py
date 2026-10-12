"""Data structures for normalized calls, measurements and log records."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modulo.shared.serialization import digest, loads


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class TextPart(Contract):
    type: Literal["text"] = "text"
    text: str


class ImageDownload(Contract):
    url_sha256: str
    final_url_sha256: str
    sha256: str
    byte_size: int = Field(gt=0)
    downloaded_at: str
    cache_file: str
    historical_content: Literal["unverified"] = "unverified"


class ImagePart(Contract):
    type: Literal["image"] = "image"
    source: str
    url: str | None = None
    url_source: str | None = None
    download: ImageDownload | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    dimensions_source: str | None = None
    sha256: str | None = None
    orientation: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    option_sources: dict[str, str] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)


class Message(Contract):
    role: Literal["system", "user", "assistant", "tool", "developer", "root"]
    content: str | None
    reasoning_content: str | None = None
    reasoning_source: str | None = None
    structured_content_json: str | None = None
    assistant_parts_json: str | None = None
    # Populated for image-bearing messages; retains text/image positions.
    content_parts: list[TextPart | ImagePart] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    # JSON strings preserve nested key order and numeric text across bundle replay.
    tool_calls_json: str | None = None
    tool_call_id: str | None = None
    name: str | None = None


class Usage(Contract):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    provenance: Literal["source_reported"] = "source_reported"


class InputBatch(Contract):
    """One input's position in a Run; aggregate usage is never per-input usage."""

    run_id: str
    index: int = Field(ge=0)
    size: int = Field(ge=1)
    usage: Usage = Field(default_factory=Usage)


class OutputContent(Contract):
    output_text: str | None = None
    output_reasoning: str | None = None
    output_reasoning_source: str | None = None
    output_structured_content_json: str | None = None
    output_parts_json: str | None = None
    output_messages: list[Message] = Field(default_factory=list)
    output_tool_calls_json: str | None = None
    output_finish_reason: str | None = None
    output_stop_reason: str | int | None = None
    output_stop_reason_source: str | None = None
    output_gaps: list[str] = Field(default_factory=list)


class OutputCandidate(OutputContent):
    index: int = Field(ge=0)


class Call(OutputContent):
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
    input_batch: InputBatch | None = None
    cache_hit: bool | None = None
    # Multi-choice responses stay within one call; top-level text is not a
    # concatenation or an arbitrary first candidate. Single responses stay flat.
    output_candidates: list[OutputCandidate] = Field(default_factory=list)
    expected_output_candidates: int | None = Field(default=None, ge=1)
    expected_output_candidates_sources: list[str] = Field(default_factory=list)
    tools_json: str | None = None
    render_options: dict[str, Any] = Field(default_factory=dict)
    render_options_json: str | None = None
    render_option_sources: dict[str, list[str]] = Field(default_factory=dict)
    generation_options: dict[str, Any] = Field(default_factory=dict)
    generation_option_sources: dict[str, list[str]] = Field(default_factory=dict)
    generation_gaps: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def restore_option_order(cls, value):
        if isinstance(value, dict) and value.get("render_options_json"):
            options = loads(value["render_options_json"])
            if digest(options) == digest(value.get("render_options", {})):
                value = {**value, "render_options": options}
        return value


class CandidateMeasurement(Contract):
    index: int = Field(ge=0)
    execution: Literal["completed", "partial", "not_run", "error"]
    tokens: int | None = None
    body_tokens: int | None = None
    visible_tokens: int | None = None
    reasoning_tokens: int | None = None
    tool_tokens: int | None = None
    reason: str | None = None
    reasoning_source: str | None = None
    rendered_sha256: str | None = None
    ids_sha256: str | None = None
    ending: str | None = None
    termination: str | None = None
    render_options: dict[str, Any] = Field(default_factory=dict)
    render_option_sources: dict[str, list[str]] = Field(default_factory=dict)


class ImageMeasurement(Contract):
    message_index: int
    part_index: int
    source: str
    execution: Literal["completed", "not_run"] = "not_run"
    reason: str | None = None
    width: int | None = None
    height: int | None = None
    dimensions_source: str | None = None
    sha256: str | None = None
    download: ImageDownload | None = None
    visual_tokens: int | None = None
    input_tokens: int | None = None
    layout: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_sources: dict[str, list[str]] = Field(default_factory=dict)


class Measurement(Contract):
    schema_version: Literal["measurement.v2"] = "measurement.v2"
    execution: Literal["completed", "not_run", "error"] = "not_run"
    applicability: Literal["supported", "unsupported", "unknown"] = "unknown"
    verdict: Literal["CONSISTENT", "DEVIATION", "UNVERIFIABLE"] = "UNVERIFIABLE"
    scope: str = "local_reference_input_output"
    input_verdict: Literal["CONSISTENT", "DEVIATION", "UNVERIFIABLE"] = "UNVERIFIABLE"
    input_scope: str = "local_reference_input"
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
    input_ids_scope: str = "reference_input_ids"
    input_images: list[ImageMeasurement] = Field(default_factory=list)
    image_input_tokens: int | None = None
    input_tokens: int | None = None
    reported_input_tokens: int | None = None
    input_delta: int | None = None
    visible_output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    reasoning_output_source: str | None = None
    tool_output_tokens: int | None = None
    output_body_tokens: int | None = None
    output_tokens: int | None = None
    reported_output_tokens: int | None = None
    output_delta: int | None = None
    output_verdict: Literal["CONSISTENT", "DEVIATION", "UNVERIFIABLE"] = "UNVERIFIABLE"
    output_scope: str = "local_reference_output"
    rendered_output_sha256: str | None = None
    output_ids_sha256: str | None = None
    output_ending: str | None = None
    output_termination: str | None = None
    output_render_options: dict[str, Any] = Field(default_factory=dict)
    output_render_options_json: str | None = None
    output_render_option_sources: dict[str, list[str]] = Field(default_factory=dict)
    output_execution: Literal["completed", "partial", "not_run", "error"] = "not_run"
    output_reason: str | None = None
    output_candidates: list[CandidateMeasurement] = Field(default_factory=list)
    # Component re-encodings are diagnostic; their sum need not equal output_tokens.
    render_options: dict[str, Any] = Field(default_factory=dict)
    render_options_json: str | None = None
    render_options_sha256: str | None = None
    render_option_sources: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def derive_verdict(self) -> Self:
        # A known difference remains a difference even if the other side is
        # unavailable. Only two complete, consistent comparisons pass overall.
        sides = (self.input_verdict, self.output_verdict)
        if "DEVIATION" in sides:
            verdict = "DEVIATION"
        elif sides == ("CONSISTENT", "CONSISTENT"):
            verdict = "CONSISTENT"
        else:
            verdict = "UNVERIFIABLE"
        object.__setattr__(self, "verdict", verdict)
        return self


@dataclass(frozen=True)
class RawRecord:
    locator: str
    data: bytes
    checkpoint: dict | None = None


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

    rows: Sequence[dict]
    diagnostics: Sequence[dict]
    summary: dict
