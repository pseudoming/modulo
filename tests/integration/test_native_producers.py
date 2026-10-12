"""Regress actual native producer outputs, not manually reconstructed shapes."""

import hashlib
import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT

FIXTURES = ROOT / "tests/fixtures/native-logs"


def test_captured_native_fixtures_match_provenance():
    manifest = json.loads((FIXTURES / "provenance.json").read_text())
    for item in manifest["files"]:
        assert hashlib.sha256((FIXTURES / item["file"]).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.assets
@pytest.mark.parametrize("source", ["log", "es"])
@pytest.mark.parametrize(
    "label,adapter,count",
    [
        ("dify", "dify", 11),
        ("newapi", "newapi", 4),
        ("oneapi-patched", "oneapi", 4),
        ("oneapi", "oneapi", 4),
        ("litellm-standard", "litellm", 4),
        ("litellm-spend", "litellm", 4),
    ],
)
def test_native_capture_through_cli(
    tmp_path, capsys, official_assets, mock_es, source, label, adapter, count
):
    path = FIXTURES / f"{label}.{'log' if adapter in {'newapi', 'oneapi'} else 'jsonl'}"
    binding = "request-response" if adapter in {"newapi", "oneapi"} else "reasoning-tools"
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--bindings",
        str(ROOT / f"examples/{adapter}/{binding}.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        args += ["--path", str(path)]
    else:
        rows = []
        for i, line in enumerate(path.read_text().splitlines()):
            rows.append(
                json.loads(line)
                if path.suffix == ".jsonl"
                else {"message": line, "event": {"sequence": i}}
            )
        mock_es(MockElasticsearch(rows))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs", "--page-size", "3"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    verify_bundle(Path(result["run"]))
    rows = details(Path(result["run"]))
    assert len(rows) == count
    assert result["summary"]["counters"]["invalid_records"] == 0
    assert result["summary"]["input_verdicts"] == {
        "UNVERIFIABLE" if label == "oneapi" else "CONSISTENT": count
    }
    family = json.loads(
        (ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json").read_text()
    )["vectors"]
    reasoning = json.loads(
        (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
    )["vectors"]
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    for row in rows:
        call, measurement = row["call"], row["measurement"]
        candidates = [
            v
            for v in (family if label == "dify" else reasoning)
            if assets[v["asset"]]["repo"] == call["model"]
        ]
        vector = (
            next(v for v in candidates if v["name"] == "basic_zh")
            if label == "dify"
            else candidates[0]
        )
        assert measurement["input_tokens"] == len(vector["input_ids"])
        if label == "oneapi":
            assert measurement["output_execution"] == "not_run"
            assert call["output_gaps"] == ["missing_response_body"]
        else:
            assert not call["gaps"] and not call["output_gaps"]
            assert measurement["output_execution"] == (
                "partial" if label == "dify" else "completed"
            )
            assert measurement["visible_output_tokens"] == vector["visible_output_tokens"]
            if label != "dify":
                assert call["output_reasoning"] == vector["output_reasoning"]
                assert measurement["reasoning_output_tokens"] == vector["reasoning_output_tokens"]
