import copy

import pytest

from modulo.adapters.protocols import DifyProtocol, NewAPIProtocol
from modulo.adapters.sources import FileSource
from modulo.core.main import run
from modulo.shared.serialization import bytes_digest, dumps, iter_dumps, loads
from modulo.storage.working_set import WorkingSet


def test_disk_processing_matches_memory_for_conflicts_filters_and_exact_values(
    tmp_path, unit_calculator, dify_record
):
    records = [dify_record, copy.deepcopy(dify_record)]
    records[1]["outputs"]["usage"]["prompt_tokens"] += 1
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(dumps(row) + "\n" for row in records) + "invalid json\n")

    def execute(working=None):
        return run(
            FileSource(path),
            DifyProtocol(),
            unit_calculator,
            archive=lambda raw: {"locator": raw.locator, "sha256": bytes_digest(raw.data)},
            save=lambda result: {
                "rows": loads("".join(iter_dumps(result.rows))),
                "diagnostics": loads("".join(iter_dumps(result.diagnostics))),
                "summary": loads("".join(iter_dumps(result.summary))),
            },
            namespace="parity",
            working=working,
        )

    expected = execute()
    with WorkingSet(tmp_path / "working", memory_mib=8) as working:
        assert execute(working) == expected


def test_oversized_event_group_retains_every_raw_reference(tmp_path, unit_calculator):
    path = tmp_path / "events.log"
    prefix = "[DEBUG] 2026/10/10 - 12:00:00 | same-request | stream scanner data: "
    path.write_text((prefix + 'data: {"padding":"' + "x" * 80 + '"}\n') * 8)
    with WorkingSet(tmp_path / "working") as working:
        result = run(
            FileSource(path),
            NewAPIProtocol(),
            unit_calculator,
            archive=lambda raw: {"locator": raw.locator},
            save=lambda result: result,
            namespace="group-limit",
            working=working,
            max_record_bytes=300,
        )
        diagnostics = list(result.diagnostics)
        assert result.summary["run_status"] == "FAILED"
        assert len([d for d in diagnostics if d["reason"] == "protocol_group_size_limit"]) == 8
        assert len({d["source_ref"]["locator"] for d in diagnostics if "source_ref" in d}) == 8


@pytest.mark.parametrize("limit,expected", [(0, 10001), (3, 3)])
def test_default_is_not_limited_to_ten_thousand_records(tmp_path, unit_calculator, limit, expected):
    path = tmp_path / "logs.jsonl"
    # Valid non-LLM records avoid spending tokenizer time in a read-budget test.
    path.write_text('{"node_type":"start"}\n' * 10001)
    with WorkingSet(tmp_path / "working") as working:
        result = run(
            FileSource(path),
            DifyProtocol(),
            unit_calculator,
            archive=lambda raw: {"locator": raw.locator},
            save=lambda result: result,
            namespace="budget",
            working=working,
            max_records=limit,
        )
        assert result.summary["counters"]["raw_records"] == expected
        assert result.summary["batch_read_complete"] is (limit == 0)
