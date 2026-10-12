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
# These historical vectors intentionally pair disabled thinking with nonempty
# output reasoning. Input expectations stay valid; full output is inconsistent.
OUTPUT_MODE_CONFLICTS = {
    ("deepseek-v4.1", "mode_5"),
    ("minimax-m3", "mode_2"),
    ("longcat-2.0", "mode_0"),
    ("mimo-v2.6", "mode_1"),
    ("hy4-preview", "mode_0"),
    ("ling-3.0", "mode_1"),
    ("ling-3.0", "mode_2"),
    ("qwen3.5-122b", "mode_1"),
    ("qwen3.5-397b", "mode_1"),
    ("deepseek-v4-pro-0813", "mode_3"),
    ("glm-5.2", "mode_5"),
    ("kimi-k2.6", "mode_0"),
}


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_all_models_with_reasoning_through_protocol_source_and_bundle(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    fixture = ROOT / f"examples/{adapter}/reasoning-tools.jsonl"
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--bindings",
        str(ROOT / f"examples/{adapter}/reasoning-tools.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        args += ["--path", str(fixture)]
    else:
        mock_es(MockElasticsearch.from_file(fixture))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs", "--page-size", "7"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["input_verdicts"] == {"CONSISTENT": len(VECTORS)}
    assert result["summary"]["output_execution"] == (
        {
            "completed": len(VECTORS) - len(OUTPUT_MODE_CONFLICTS),
            "partial": len(OUTPUT_MODE_CONFLICTS),
        }
        if adapter == "dify"
        else {"partial": len(VECTORS)}
    )
    assert result["summary"]["output_scope"] == "local_reference_output"
    groups = result["summary"]["by_model"].values()
    assert sum(g["reasoning_output_recalculated"] for g in groups) == len(VECTORS)
    assert sum(g["reasoning_output_token_sum"] for g in groups) == sum(
        v["reasoning_output_tokens"] for v in VECTORS
    )
    bundle = Path(result["run"])
    verify_bundle(bundle)
    rows = details(bundle)
    assert len(rows) == len(VECTORS)
    for row in rows:
        call, measurement = row["call"], row["measurement"]
        expected = EXPECTED[call["call_id"]]
        assert call["schema_version"] == "call.v3"
        assert call["output_reasoning"] == expected["output_reasoning"]
        assert (
            call["messages"][2]["reasoning_content"] == expected["messages"][2]["reasoning_content"]
        )
        assert measurement["reasoning_output_tokens"] == expected["reasoning_output_tokens"]
        assert measurement["render_options"] == expected["template_options"]
        assert measurement["output_execution"] == (
            "completed"
            if adapter == "dify"
            and (expected["asset"], expected["name"]) not in OUTPUT_MODE_CONFLICTS
            else "partial"
        )
        assert measurement["model_identity"] == "UNVERIFIED"
        assert call["usage"]["output_tokens"] is None
        assert measurement["output_verdict"] == "UNVERIFIABLE"
        assert measurement["output_delta"] is None
    if source == "log":
        assert main(args) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["reused"] and repeated["run"] == result["run"]
