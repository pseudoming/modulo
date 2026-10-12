import copy

import httpx
import pytest

from modulo.adapters.protocols import DifyProtocol, NewAPIProtocol
from modulo.adapters.sources import ElasticsearchSource, FileSource
from modulo.application.continuation import JournalSource
from modulo.application.importer import run_batch
from modulo.core.domain.data_models import SourceError
from modulo.core.domain.sampling import SamplingOptions
from modulo.shared.serialization import dumps, loads
from modulo.storage.result_bundle import verify_bundle
from modulo.storage.working_set import WorkingSet
from tests.support.mock_es import MockElasticsearch


def snapshot(tmp_path, source, calculator, protocol=None, **kwargs):
    return run_batch(
        source,
        protocol or DifyProtocol(),
        calculator,
        namespace="continue",
        output_root=tmp_path / "runs",
        state=tmp_path / "state",
        **kwargs,
    )


def rows(path):
    return [loads(line) for line in (path / "results.jsonl").read_text().splitlines()]


def test_sampling_windows_reuse_journal_and_can_replay_an_earlier_window(
    tmp_path, unit_calculator, dify_record
):
    path = tmp_path / "access.jsonl"
    records = []
    for minute in (1, 6):
        for i in range(4):
            record = copy.deepcopy(dify_record)
            record.update(id=f"{minute}-{i}", created_at=f"2026-10-13T10:0{minute}:00+08:00")
            records.append(record)
    path.write_text("".join(dumps(record) + "\n" for record in records))
    first_options = SamplingOptions(
        mode="window", window_minutes=5, count=2, before="2026-10-13T10:05:00+08:00"
    )
    first, _ = snapshot(tmp_path, FileSource(path), unit_calculator, sampling=first_options)
    second, _ = snapshot(
        tmp_path,
        FileSource(path),
        unit_calculator,
        sampling=SamplingOptions(
            mode="window", window_minutes=5, count=2, before="2026-10-13T10:10:00+08:00"
        ),
    )
    assert len(rows(first)) == len(rows(second)) == 2
    assert all(row["call"]["call_id"].startswith("1-") for row in rows(first))
    assert all(row["call"]["call_id"].startswith("6-") for row in rows(second))
    replay, reused = snapshot(tmp_path, FileSource(path), unit_calculator, sampling=first_options)
    assert replay == first and reused


def test_append_reuses_completed_calculations_and_preserves_old_result(
    tmp_path, unit_calculator, dify_record, monkeypatch
):
    path = tmp_path / "access.log"
    path.write_text(dumps(dify_record) + "\n")
    calculated = []
    calculate = unit_calculator.calculate

    def count(call, binding_matches=None):
        calculated.append(call.call_id)
        return calculate(call, binding_matches)

    monkeypatch.setattr(unit_calculator, "calculate", count)
    first, _ = snapshot(tmp_path, FileSource(path), unit_calculator)
    first_manifest = verify_bundle(first)
    second_record = copy.deepcopy(dify_record)
    second_record["id"] = "second"
    with path.open("a") as stream:
        stream.write(dumps(second_record) + "\n")
    second, reused = snapshot(tmp_path, FileSource(path), unit_calculator)
    assert not reused and first != second
    assert calculated == ["call-1", "second"]
    assert len(rows(second)) == 2
    assert verify_bundle(first) == first_manifest
    assert snapshot(tmp_path, FileSource(path), unit_calculator) == (second, True)
    assert calculated == ["call-1", "second"]


def test_separate_imports_join_request_response_and_late_conflict(tmp_path, unit_calculator):
    path = tmp_path / "events.jsonl"
    request = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
    response = {
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 13, "completion_tokens": 1},
    }
    prefix = "[DEBUG] 2026/10/10 - 12:00:00 | request | "
    path.write_text(prefix + "requestBody: " + dumps(request) + "\n")
    first, _ = snapshot(tmp_path, FileSource(path), unit_calculator, NewAPIProtocol())
    assert len(rows(first)) == 1
    assert rows(first)[0]["call"]["output_text"] is None
    with path.open("a") as stream:
        stream.write(prefix + "responseBody: " + dumps(response) + "\n")
    second, _ = snapshot(tmp_path, FileSource(path), unit_calculator, NewAPIProtocol())
    assert len(rows(second)) == 1
    assert rows(second)[0]["call"]["output_text"] == "hi"
    assert len(rows(second)[0]["source_refs"]) == 2
    assert verify_bundle(first)["files"] != verify_bundle(second)["files"]
    response["choices"][0]["message"]["content"] = "conflicting response"
    with path.open("a") as stream:
        stream.write(prefix + "responseBody: " + dumps(response) + "\n")
    third, _ = snapshot(tmp_path, FileSource(path), unit_calculator, NewAPIProtocol())
    assert "ambiguous_gateway_responses" in rows(third)[0]["call"]["gaps"]
    assert rows(third)[0]["measurement"]["verdict"] == "UNVERIFIABLE"


def test_file_budget_resume_partial_line_and_rotation(tmp_path):
    path = tmp_path / "active.log"
    path.write_bytes(b"one\ntwo\nthree")
    state = tmp_path / "state"
    with WorkingSet(state) as working:
        first = JournalSource(FileSource(path), working, 1)
        first.capture()
        assert first.error
        assert working.raw_count() == 1
    with WorkingSet(state) as working:
        second = JournalSource(FileSource(path), working, 0)
        second.capture()
        assert "complete_log_line" in str(second.error)
        assert working.raw_count() == 2
    with path.open("ab") as stream:
        stream.write(b"\n")
    path.rename(tmp_path / "old.log")
    path.write_bytes(b"four\n")
    with WorkingSet(state) as working:
        third = JournalSource(FileSource(tmp_path, pattern="*.log"), working, 0)
        third.capture()
        assert third.error is None
        assert sorted(row.data for row in working.raw_records()) == [
            b"four\n",
            b"one\n",
            b"three\n",
            b"two\n",
        ]


@pytest.mark.parametrize("changed", [b"", b"BAD\nnew\n"])
def test_consumed_file_prefix_cannot_be_overwritten_silently(tmp_path, changed):
    path = tmp_path / "source.log"
    path.write_bytes(b"one\n")
    with WorkingSet(tmp_path / "state") as working:
        JournalSource(FileSource(path), working, 0).capture()
        path.write_bytes(changed)
        second = JournalSource(FileSource(path), working, 0)
        second.capture()
        assert second.error is not None
        assert working.raw_count() == 1


def test_live_file_append_after_selection_is_deferred_without_failing(tmp_path):
    path = tmp_path / "live.log"
    path.write_bytes(b"one\n")
    selected = FileSource(path)
    with path.open("ab") as stream:
        stream.write(b"two\n")
    with WorkingSet(tmp_path / "state") as working:
        first = JournalSource(selected, working, 0)
        first.capture()
        assert first.error is None and working.raw_count() == 1
        second = JournalSource(FileSource(path), working, 0)
        second.capture()
        assert second.error is None and working.raw_count() == 2


@pytest.mark.parametrize("expired", [False, True])
def test_es_resume_mid_page_or_expired_pit_does_not_lose_or_duplicate_records(tmp_path, expired):
    mock = MockElasticsearch([{"n": i} for i in range(5)])
    searches = []
    expire_next = False

    def handle(request):
        nonlocal expire_next
        if request.url.path == "/_search":
            searches.append(loads(request.content))
            if expire_next:
                expire_next = False
                return httpx.Response(404)
        return mock.handle(request)

    def source(working):
        result = ElasticsearchSource(
            "https://es.invalid", "logs", page_size=4, transport=httpx.MockTransport(handle)
        )
        result.seen = working.mapping("saved:seen")
        return result

    with WorkingSet(tmp_path) as working:
        first = JournalSource(source(working), working, 2)
        first.capture()
        assert first.error and not mock.closed
        assert working.raw_count() == 2
    expire_next = expired
    with WorkingSet(tmp_path) as working:
        second = JournalSource(source(working), working, 0)
        second.capture()
        assert second.error is None and mock.closed
        assert [loads(row.data)["n"] for row in working.raw_records()] == list(range(5))
        assert searches[1]["search_after"] == [1]
        assert not working.mapping("saved:cursors")


def test_es_page_memory_limit_reports_incomplete_source():
    source = ElasticsearchSource(
        "https://es.invalid",
        "logs",
        transport=MockElasticsearch([{"text": "x" * 1000}]).transport(),
    )
    source.max_page_bytes = 512
    with pytest.raises(SourceError, match="memory budget"):
        list(source.read())


def test_es_failed_page_keeps_rotated_pit_but_does_not_skip_uncommitted_hits(tmp_path):
    mock = MockElasticsearch([{"n": i} for i in range(5)], fail_page=2)

    def source(working):
        reader = ElasticsearchSource(
            "https://es.invalid", "logs", page_size=2, transport=mock.transport()
        )
        reader.seen = working.mapping("saved:seen")
        return reader

    with WorkingSet(tmp_path) as working:
        first = JournalSource(source(working), working, 0)
        first.capture()
        assert "incomplete page" in str(first.error)
        assert working.raw_count() == 2
    with WorkingSet(tmp_path) as working:
        second = JournalSource(source(working), working, 0)
        second.capture()
        assert second.error is None
        assert [loads(row.data)["n"] for row in working.raw_records()] == list(range(5))


def test_failed_calculation_is_retried_even_when_source_is_unchanged(
    tmp_path, unit_calculator, dify_record, monkeypatch
):
    path = tmp_path / "access.log"
    path.write_text(dumps(dify_record) + "\n")
    calculate = unit_calculator.calculate

    def fail(*args, **kwargs):
        raise RuntimeError("temporary failure")

    monkeypatch.setattr(unit_calculator, "calculate", fail)
    first, _ = snapshot(tmp_path, FileSource(path), unit_calculator)
    assert verify_bundle(first)["status"] == "PARTIAL"
    monkeypatch.setattr(unit_calculator, "calculate", calculate)
    second, _ = snapshot(tmp_path, FileSource(path), unit_calculator)
    assert second != first and verify_bundle(second)["status"] == "COMPLETE"
