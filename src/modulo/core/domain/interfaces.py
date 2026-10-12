"""Interfaces for log sources, call parsers, tokenizers and core calculation."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from typing import Any, ClassVar, Protocol

from modulo.shared.serialization import loads, ordered_json

from .bindings import Binding, Bindings
from .data_models import (
    Call,
    DataError,
    Measurement,
    Parsed,
    ProtocolEmission,
    ProtocolRecord,
    RawRecord,
)

MAX_RECORD_BYTES = 8 * 1024 * 1024


class SourceAdapter(ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    ordered_records: ClassVar[bool] = False

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Describe the source without credentials."""

    @abstractmethod
    def read(self) -> Iterator[RawRecord]:
        """Yield records or raise SourceError when the batch is incomplete."""


class ProtocolAdapter(ABC):
    name: ClassVar[str]
    version: ClassVar[str]

    def decode(self, data: bytes) -> dict[str, Any]:
        value = loads(data)
        if not isinstance(value, dict):
            raise DataError("expected object: record")
        return value

    def new_session(self):
        return ProtocolSession(self)

    def report_warnings(self, calls: list[dict], diagnostics: list[dict]) -> list[dict]:
        """Explain observed protocol-specific gaps and how to supply missing evidence."""
        return []

    @abstractmethod
    def parse(self, record: dict[str, Any]) -> Parsed:
        """Normalize calls and their per-call context without performing I/O."""


class ProtocolSession:
    """One batch's parsing state; the default keeps existing row plugins stateless."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.events = MemoryEvents()
        self.max_group_bytes = MAX_RECORD_BYTES

    def configure_storage(self, storage, max_group_bytes=MAX_RECORD_BYTES):
        self.events = storage.groups("protocol_events")
        self.max_group_bytes = max_group_bytes

    def feed(self, record: ProtocolRecord) -> list[ProtocolEmission]:
        return [ProtocolEmission([record], self.adapter.parse(record.payload))]

    def finish(self, *, complete: bool):
        for key, entries in self.events.items():
            size = getattr(entries, "byte_size", None)
            if size is None:
                size = sum(len(ordered_json(item).encode()) for item in entries)
            # Buffered records retain both raw wrapper and parsed payload.
            if size > self.max_group_bytes * 4:
                for entry in entries:
                    yield ProtocolEmission(
                        [ProtocolRecord(**r) for r in entry["records"]],
                        error="protocol_group_size_limit",
                    )
                continue
            yield from self.finish_group(key, list(entries), complete)
        self.events.clear()

    def finish_group(self, key, entries, complete):
        return []


class MemoryEvents:
    """Small in-memory implementation for direct parser use and unit tests."""

    def __init__(self):
        self.values = {}

    def append(self, key, value):
        self.values.setdefault(key, []).append(value)

    def items(self):
        return self.values.items()

    def clear(self):
        self.values.clear()


class WorkingStorage(Protocol):
    """Inject replayable collections; core never opens their backing files."""

    def mapping(self, name: str) -> MutableMapping: ...
    def sequence(self, name: str) -> Sequence: ...
    def groups(self, name: str): ...


class TokenizerBackend(ABC):
    name: ClassVar[str]
    family: ClassVar[str]
    version: ClassVar[str]

    @abstractmethod
    def count(self, call: Call, binding: Binding) -> dict[str, Any]:
        """Count input from frozen assets and explicit binding options."""

    def count_output(self, call: Call) -> dict[str, Any]:
        """Count output independently; return partial components when evidence is missing."""
        return {}


class AssetError(ValueError):
    pass


class UnsupportedInput(ValueError):
    """A documented capability/parameter gap, not an implementation failure."""

    def __init__(self, reason, *, evidence=None):
        super().__init__(reason)
        self.evidence = evidence or {}


class Calculator(Protocol):
    bindings: Bindings

    def calculate(self, call: Call, binding_matches: list[int] | None = None) -> Measurement: ...


BackendLoader = Callable[[Binding], TokenizerBackend]
RawArchive = Callable[[RawRecord], dict[str, str]]
