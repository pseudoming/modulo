"""The core flow works with memory ports and has no evidence-store dependency."""

import pytest

from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Message, Parsed, RawRecord, SourceError, Usage
from modulo.core.domain.interfaces import ProtocolAdapter, SourceAdapter, TokenizerBackend
from modulo.core.main import run
from modulo.core.processing.recalculation import Recalculator
from modulo.shared.serialization import bytes_digest


class MemorySource(SourceAdapter):
    name = "memory-test"
    version = "memory.v1"

    def __init__(self, fail):
        self.fail, self.closed = fail, False

    def describe(self):
        return {"kind": self.name}

    def read(self):
        try:
            yield RawRecord("memory:1", b'{"id":"one","text":"hello world"}')
            if self.fail:
                raise SourceError("injected_read_failure")
        finally:
            self.closed = True


class MemoryProtocol(ProtocolAdapter):
    name = "memory-test"
    version = "memory.v1"

    def parse(self, record):
        return Parsed(
            [
                Call(
                    call_id=record["id"],
                    protocol=self.name,
                    model="memory-model",
                    messages=[Message(role="user", content=record["text"])],
                    usage=Usage(input_tokens=5),
                )
            ]
        )


class FixedReference(TokenizerBackend):
    name = "memory-test"
    version = "synthetic-vector.v1"

    def count(self, call, binding):
        assert call.messages[0].content == "hello world"
        return {"input_tokens": 2}  # Explicit synthetic reference, not a cloud tokenizer.


@pytest.mark.parametrize("fail", [False, True])
def test_core_processes_and_retains_partial_evidence_entirely_in_memory(fail):
    archived = []
    saved = []

    def archive(record):
        archived.append(record)
        return {"locator": record.locator, "sha256": bytes_digest(record.data)}

    calculator = Recalculator(
        Bindings(models=[Binding(aliases=["memory-model"], name="synthetic")]),
        lambda binding: FixedReference(),
        {"assets": {"synthetic": {"family": "memory-test"}}},
    )
    source = MemorySource(fail)

    def save(result):
        assert source.closed
        saved.append(result)
        return "saved-result"

    result_location = run(
        source,
        MemoryProtocol(),
        calculator,
        archive=archive,
        save=save,
        namespace="synthetic/memory",
    )
    assert result_location == "saved-result" and len(saved) == 1
    result = saved[0]
    assert source.closed
    assert len(archived) == len(result.rows) == 1
    assert result.rows[0]["measurement"]["input_delta"] == 3
    assert result.summary["verdicts"] == {"DEVIATION": 1}
    assert result.summary["input_verdicts"] == {"DEVIATION": 1}
    assert result.summary["output_verdicts"] == {"UNVERIFIABLE": 1}
    assert result.summary["batch_read_complete"] is not fail
    assert result.summary["run_status"] == ("PARTIAL" if fail else "COMPLETE")
    assert [item["reason"] for item in result.diagnostics] == (
        ["source_incomplete"] if fail else []
    )


def test_output_backend_failure_makes_batch_partial_without_discarding_input():
    class BrokenOutput(FixedReference):
        def count_output(self, call):
            raise RuntimeError("private output must not be echoed")

    calculator = Recalculator(
        Bindings(models=[Binding(aliases=["memory-model"], name="synthetic")]),
        lambda _: BrokenOutput(),
        {"assets": {"synthetic": {"family": "memory-test"}}},
    )
    result = run(
        MemorySource(False),
        MemoryProtocol(),
        calculator,
        archive=lambda r: {"locator": r.locator},
        save=lambda result: result,
        namespace="synthetic/memory",
    )
    measurement = result.rows[0]["measurement"]
    assert measurement["execution"] == "completed" and measurement["input_delta"] == 3
    assert measurement["output_execution"] == "error"
    assert result.summary["run_status"] == "PARTIAL"
    assert result.summary["output_execution"] == {"error": 1}
    assert result.diagnostics[0]["reason"] == "tokenizer_error:RuntimeError"
    assert result.diagnostics[0]["component"] == "output"
    group = result.summary["by_model"]["memory-model"]
    assert group["reasoning_output_token_sum"] is None
    assert group["reasoning_output_recalculated"] == 0
