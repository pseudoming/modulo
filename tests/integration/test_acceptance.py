import json

import pytest

from modulo.adapters.sources import ElasticsearchSource, FileSource
from modulo.application.importer import run_batch
from modulo.application.registry import builtins
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT


@pytest.mark.assets
@pytest.mark.parametrize("protocol_name", ["dify", "langchain"])
@pytest.mark.parametrize("physical", ["file", "es"])
def test_two_by_two_real_recount_matrix(tmp_path, calculator, protocol_name, physical):
    expected = json.loads((ROOT / "tests/fixtures/expected/basic.json").read_text())
    path = ROOT / "examples" / protocol_name / "basic.jsonl"
    mock = None
    if physical == "file":
        source = FileSource(path)
    else:
        mock = MockElasticsearch.from_file(path)
        source = ElasticsearchSource(
            "https://mock.invalid", "logs", page_size=3, transport=mock.transport()
        )
    bundle, _ = run_batch(
        source,
        builtins().get("protocols", protocol_name),
        calculator,
        namespace="synthetic/app",
        output_root=tmp_path,
    )
    summary = json.loads((bundle / "summary.json").read_text())
    assert verify_bundle(bundle)["status"] == "COMPLETE"
    assert summary["observed_calls"] == expected["expected_calls"]
    assert summary["counters"]["raw_records"] == expected["raw_records_per_protocol"]
    assert summary["counters"]["duplicate_calls"] == expected["expected_duplicate_calls"]
    assert summary["counters"]["ignored_records"] == expected["expected_ignored_records"]
    assert summary["verdicts"] == expected["expected_input_verdicts"]
    rows = details(bundle)
    assert sum(row["measurement"]["input_tokens"] is not None for row in rows) == 4
    deviations = [row for row in rows if row["measurement"]["verdict"] == "DEVIATION"]
    assert len(deviations) == 1 and deviations[0]["measurement"]["input_delta"] == 3
    assert all(row["measurement"]["model_identity"] == "UNVERIFIED" for row in rows)
    assert sum(m["compared_calls"] for m in summary["by_model"].values()) == 3
    if mock:
        assert mock.searches == 5 and mock.closed
