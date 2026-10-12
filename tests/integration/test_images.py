import json
from pathlib import Path

import duckdb
import pytest

from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.fixtures.generate_image_fixtures import image_rows

VECTORS = [
    v
    for v in json.loads(
        (ROOT / "tests/fixtures/tokenizers/image_reference_vectors.json").read_text()
    )["vectors"]
    if v["name"] in {"640x480", "two_images", "explicit_options"}
]


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm", "newapi", "oneapi"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_images_through_protocol_source_calculation_and_storage(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    assets = load_tokenizer_assets()["assets"]
    rows = []
    expected = {}
    for vector in VECTORS:
        model = assets[vector["asset"]]["repo"]
        scenarios = (
            ["complete"]
            if vector["name"] != "640x480"
            else ["embedded", "more", "url-only", "detail"]
        )
        if adapter == "dify" and vector["name"] == "640x480":
            scenarios.append("saved-files")
        for scenario in scenarios:
            records = image_rows(vector, adapter, model, scenario)
            rows.extend(records)
            key = "images" if adapter in {"newapi", "oneapi"} else "synthetic"
            key += f"-{vector['asset']}-{vector['name']}-{scenario}"
            expected[key] = (vector, scenario)
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--bindings",
        str(ROOT / "examples/litellm/model-families.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        path = tmp_path / "images.jsonl"
        path.write_text(
            "".join(row if isinstance(row, str) else json.dumps(row) + "\n" for row in rows)
        )
        args += ["--path", str(path)]
    else:
        mock_es(
            MockElasticsearch(
                [
                    {"message": row, "event": {"sequence": i}} if isinstance(row, str) else row
                    for i, row in enumerate(rows)
                ]
            )
        )
        args += ["--path", "https://mock.invalid", "--index", "image-logs", "--page-size", "7"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    bundle = Path(result["run"])
    verify_bundle(bundle)
    saved = details(bundle)
    assert len(saved) == len(expected)
    image_total = 0
    for row in saved:
        call_id = (
            row["assemblies"][0]["request_id"]
            if adapter in {"newapi", "oneapi"}
            else row["call"]["call_id"]
        )
        vector, scenario = expected[call_id]
        count = row["measurement"]
        assert count["output_execution"] == "completed"
        assert count["visible_output_tokens"] > 0 and count["reasoning_output_tokens"] > 0
        if scenario in {"complete", "embedded", "more"}:
            assert count["input_tokens"] == vector["input_tokens"]
            assert count["input_delta"] == (7 if scenario == "more" else 0)
            assert count["input_verdict"] == ("DEVIATION" if scenario == "more" else "CONSISTENT")
            assert count["image_input_tokens"] == sum(vector["image_input_tokens"])
            prefix = {
                "dify": "process_data.prompts.0.text.",
                "langchain": "inputs.messages.0.0.data.content.",
                "litellm": "messages.0.content.",
                "newapi": "request_body.messages.0.content.",
                "oneapi": "request_body.messages.0.content.",
            }[adapter]
            assert count["input_images"][0]["source"] == prefix + "1"
            image_total += count["image_input_tokens"]
            if scenario == "embedded":
                assert count["input_images"][0]["sha256"]
                assert count["input_images"][0]["dimensions_source"].endswith(":image_header")
        else:
            assert count["input_tokens"] is None and count["input_delta"] is None
            assert count["input_verdict"] == "UNVERIFIABLE"
            reason = {
                "url-only": "missing_image_dimensions",
                "detail": "unsupported_image_detail",
                "saved-files": "attachment_content_order_missing",
            }[scenario]
            assert reason in count["reason"]
    assert (
        sum(g["image_input_token_sum"] for g in result["summary"]["by_model"].values())
        == image_total
    )
    with duckdb.connect(str(bundle / "query.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM input_images").fetchone()[0] == sum(
            len(row["measurement"]["input_images"]) for row in saved
        )
        assert connection.execute(
            "SELECT count(*) FROM input_images WHERE reason IS NOT NULL"
        ).fetchone()[0] == 2 * sum(vector["name"] == "640x480" for vector in VECTORS)
