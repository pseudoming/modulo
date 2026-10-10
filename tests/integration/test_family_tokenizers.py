import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT


@pytest.mark.assets
@pytest.mark.parametrize("source", ["log", "es"])
def test_family_references_through_the_shared_cli(
    tmp_path, official_assets, capsys, mock_es, source
):
    fixture = ROOT / "examples/litellm/model-families.jsonl"
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        "litellm",
        "--bindings",
        str(ROOT / "examples/litellm/model-families.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
        "--platform-tag-source",
        "metadata:platform",
        "--platforms",
        "synthetic-reference",
    ]
    if source == "log":
        args += ["--path", str(fixture)]
    else:
        mock_es(MockElasticsearch.from_file(fixture))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs", "--page-size", "3"]
    assert main(args) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["summary"]["verdicts"] == {"CONSISTENT": 11, "DEVIATION": 1, "UNVERIFIABLE": 3}
    assert imported["summary"]["batch_read_complete"]
    bundle = Path(imported["run"])
    verify_bundle(bundle)
    rows = details(bundle)
    assert len(rows) == 15
    calls = {row["call"]["call_id"]: row for row in rows}
    assert calls["synthetic-qwen-deviation"]["measurement"]["input_delta"] == 3
    missing = calls["synthetic-kimi-missing-usage"]["measurement"]
    assert missing["execution"] == "completed" and missing["input_tokens"] == 112
    assert (
        missing["reported_input_tokens"] is None and missing["reason"] == "reported_input_missing"
    )
    assert (
        calls["synthetic-unbound-model"]["measurement"]["reason"]
        == "platform_model_binding_missing"
    )
    assert calls["synthetic-qwen-explicit-mode"]["measurement"]["execution"] == "not_run"
    assert all(row["measurement"]["model_identity"] == "UNVERIFIED" for row in rows)
    if source == "log":
        assert main(args) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["reused"] and repeated["run"] == imported["run"]
