import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.native_logs.native_langchain import serialized_run

VECTORS = json.loads(
    (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
)["vectors"]
ASSETS = json.loads((ROOT / "src/modulo/application/assets-lock.json").read_text())["assets"]


@pytest.mark.assets
@pytest.mark.parametrize("source", ["log", "es"])
@pytest.mark.parametrize("stream", [False, True])
def test_framework_runs_through_import_and_storage(
    tmp_path, official_assets, capsys, mock_es, source, stream
):
    rows = [serialized_run(v, ASSETS[v["asset"]]["repo"], stream=stream) for v in VECTORS]
    expected = {row["id"]: v for row, v in zip(rows, VECTORS, strict=True)}
    fixture = tmp_path / "langchain.jsonl"
    fixture.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        "langchain",
        "--bindings",
        str(ROOT / "examples/langchain/reasoning-tools.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        args += ["--path", str(fixture)]
    else:
        mock_es(MockElasticsearch.from_file(fixture))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    bundle = Path(result["run"])
    verify_bundle(bundle)
    saved = details(bundle)
    assert len(saved) == len(VECTORS)
    for row in saved:
        call, measurement = row["call"], row["measurement"]
        vector = expected[call["call_id"]]
        assert call["gaps"] == []
        assert call["output_gaps"] == []
        assert call["output_reasoning"] == vector["output_reasoning"]
        assert measurement["input_tokens"] == len(vector["input_ids"])
        assert measurement["reasoning_output_tokens"] == vector["reasoning_output_tokens"]
        assert measurement["output_execution"] == "completed"
        assert measurement["render_options"] == vector["template_options"]
    assert result["summary"]["verdicts"] == {"CONSISTENT": len(VECTORS)}
