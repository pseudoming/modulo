import json
import runpy
from collections import Counter
from datetime import datetime

import pytest

from modulo.adapters.sources import FileSource
from modulo.application.importer import run_batch
from modulo.application.registry import builtins
from modulo.storage.query_index import details
from tests.support.paths import ROOT

LAB = runpy.run_path(str(ROOT / "tests/local-dev/elastic_lab.py"))


def test_six_daily_indices_have_unique_calls_and_four_models():
    rows = LAB["build_documents"]()
    expected = LAB["expected"]()
    counts = Counter(row["index"] for row in rows)
    assert len(counts) == expected["indices"]
    assert set(counts.values()) == {expected["documents_per_index"]}
    assert len(rows) == len({row["id"] for row in rows}) == expected["documents_total"]
    for index in counts:
        batch = [row["document"] for row in rows if row["index"] == index]
        assert {doc["model"] for doc in batch} == set(expected["models"])
        for doc in batch:
            assert datetime.fromisoformat(doc["@timestamp"]).strftime("%Y%m%d") == index[-8:]
            assert doc["synthetic"] is True


@pytest.mark.assets
@pytest.mark.parametrize("protocol", ["dify", "langchain"])
def test_real_es_seed_inputs_match_frozen_expectations(tmp_path, calculator, protocol):
    rows = [
        row["document"]
        for row in LAB["build_documents"]()
        if row["document"]["log_source"] == protocol
    ]
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    bundle, _ = run_batch(
        FileSource(path),
        builtins().get("protocols", protocol),
        calculator,
        namespace="synthetic/es-lab",
        output_root=tmp_path / "runs",
        payload_field="payload",
    )
    expected = LAB["expected"]()["per_protocol"]
    records = details(bundle)
    assert len(records) == expected["calls"]
    assert Counter(row["measurement"]["verdict"] for row in records) == expected["verdicts"]
    assert (
        sum(row["measurement"]["input_tokens"] is not None for row in records)
        == (expected["recalculated_calls"])
    )
    assert (
        sorted(
            row["measurement"]["input_delta"]
            for row in records
            if row["measurement"]["input_delta"] is not None
        )
        == expected["deltas"]
    )
