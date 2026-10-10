"""Actual deployed Dify Agent capture through the common import pipeline."""

import hashlib
import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.native_logs.dify_agent_verify import check_rows

FIXTURES = ROOT / "tests/fixtures/dify-agent"


def test_actual_agent_fixture_provenance():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for item in provenance["files"]:
        assert hashlib.sha256((FIXTURES / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    runs = json.loads((FIXTURES / "run-manifest.json").read_text())
    assert {run["stream"] for run in runs} == {True, False}
    assert all(run["model_requests"] == 2 and run["tool_executions"] == 1 for run in runs)


@pytest.mark.assets
@pytest.mark.parametrize("source", ["log", "es"])
@pytest.mark.parametrize("label", ["agent-llm", "agent-thoughts"])
def test_real_agent_rounds_through_cli(tmp_path, capsys, official_assets, mock_es, source, label):
    path = FIXTURES / f"{label}.jsonl"
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        "dify",
        "--bindings",
        str(ROOT / "examples/dify/reasoning-tools.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        args += ["--path", str(path)]
    else:
        # Deliberately reverse all ES events; joining must use invocation id,
        # never adjacency, output usage or accidental retrieval order.
        unordered = tmp_path / "es.jsonl"
        unordered.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
        mock_es(MockElasticsearch.from_file(unordered))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs", "--page-size", "3"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    bundle = Path(result["run"])
    verify_bundle(bundle)
    actual = details(bundle)
    assert len(actual) == 4
    assert result["summary"]["counters"]["invalid_records"] == 0
    if label == "agent-llm":
        check_rows(actual, json.loads((FIXTURES / "reference.json").read_text()))
    else:
        assert all(r["measurement"]["verdict"] == "UNVERIFIABLE" for r in actual)
        assert all(r["call"]["usage"]["input_tokens"] == 123 for r in actual)
