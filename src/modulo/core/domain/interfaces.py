"""Interfaces for log sources, call parsers, tokenizers and core calculation."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Protocol

from modulo.shared.serialization import loads

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

    @abstractmethod
    def parse(self, record: dict[str, Any]) -> Parsed:
        """Normalize calls and their per-call context without performing I/O."""


class ProtocolSession:
    """One batch's parsing state; the default keeps existing row plugins stateless."""

    def __init__(self, adapter):
        self.adapter = adapter

    def feed(self, record: ProtocolRecord) -> list[ProtocolEmission]:
        return [ProtocolEmission([record], self.adapter.parse(record.payload))]

    def finish(self, *, complete: bool) -> list[ProtocolEmission]:
        return []


class TokenizerBackend(ABC):
    name: ClassVar[str]
    version: ClassVar[str]

    @abstractmethod
    def count(self, call: Call, binding: Binding) -> dict[str, Any]:
        """Count from frozen assets and explicit binding options."""

    def count_output(self, call: Call) -> dict[str, Any]:
        """Optional separate output-component count; older plugins remain valid."""
        return {}


class AssetError(ValueError):
    pass


class UnsupportedInput(ValueError):
    """A documented capability/parameter gap, not an implementation failure."""


class Calculator(Protocol):
    bindings: Bindings

    def calculate(self, call: Call, binding_matches: list[int] | None = None) -> Measurement: ...


BackendLoader = Callable[[Binding], TokenizerBackend]
RawArchive = Callable[[RawRecord], dict[str, str]]
