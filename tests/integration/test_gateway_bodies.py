import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT

VECTORS = json.loads(
    (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
)["vectors"]
EXPECTED = {f"synthetic-{v['asset']}-{v['name']}": v for v in VECTORS}


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["newapi", "oneapi"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_native_bodies_all_models_through_cli_assets_and_bundle(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    directory = ROOT / "examples" / adapter
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--bindings",
        str(directory / "request-response.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        args += ["--path", str(directory / "request-response.log")]
    else:
        mock_es(MockElasticsearch.from_file(directory / "request-response-events.jsonl"))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs", "--page-size", "7"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["verdicts"] == {"CONSISTENT": len(VECTORS)}
    assert result["summary"]["output_execution"] == {"completed": len(VECTORS)}
    assert result["summary"]["counters"]["invalid_records"] == 0
    bundle = Path(result["run"])
    verify_bundle(bundle)
    rows = details(bundle)
    assert len(rows) == len(VECTORS)
    assert (
        sum(len(row["source_refs"]) for row in rows) == result["summary"]["counters"]["raw_records"]
    )
    for row in rows:
        call, measurement = row["call"], row["measurement"]
        assembly = row["assemblies"][0]
        expected = EXPECTED[assembly["request_id"]]
        assert call["output_reasoning"] == expected["output_reasoning"]
        assert call["output_text"] == expected["output_text"]
        assert (
            call["messages"][2]["reasoning_content"] == expected["messages"][2]["reasoning_content"]
        )
        assert measurement["input_tokens"] == len(expected["input_ids"])
        assert measurement["reasoning_output_tokens"] == expected["reasoning_output_tokens"]
        assert measurement["visible_output_tokens"] == expected["visible_output_tokens"]
        assert measurement["render_options"] == expected["template_options"]
        assert measurement["output_execution"] == "completed"
        assert assembly["response_complete"]
        assert call["usage"]["output_tokens"] is None
    if source == "log":
        assert main(args) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["reused"] and repeated["run"] == result["run"]
