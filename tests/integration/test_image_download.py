import json
from pathlib import Path

import httpx
import pytest

from modulo.adapters.sources import ElasticsearchSource, FileSource
from modulo.application.image_cache import ImageDownloadOptions
from modulo.application.importer import run_batch
from modulo.application.registry import builtins
from modulo.cli import main
from modulo.shared.serialization import dumps, loads
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.image_http import mock_image_http, png_bytes
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.fixtures.generate_image_fixtures import image_rows

VECTORS = [
    item
    for item in loads(
        (ROOT / "tests/fixtures/tokenizers/image_reference_vectors.json").read_bytes()
    )["vectors"]
    if item["name"] == "640x480"
]


@pytest.mark.assets
@pytest.mark.parametrize(
    "adapter,source_kind",
    [
        ("dify", "log"),
        ("langchain", "log"),
        ("litellm", "log"),
        ("newapi", "log"),
        ("oneapi", "log"),
        ("litellm", "es"),
    ],
)
def test_url_images_recount_against_frozen_vectors_with_cache_and_failure_isolation(
    tmp_path, monkeypatch, calculator, adapter, source_kind
):
    def response(request):
        return httpx.Response(
            404 if request.url.path == "/missing.png" else 200,
            stream=httpx.ByteStream(png_bytes()),
        )

    requests = mock_image_http(monkeypatch, response)
    rows, expected = [], {}
    for vector in VECTORS:
        model = calculator.lock["assets"][vector["asset"]]["repo"]
        rows.extend(image_rows(vector, adapter, model, "url-only"))
        expected[model] = vector
    first = VECTORS[0]
    failed = image_rows({**first, "name": "missing"}, adapter, next(iter(expected)), "url-only")
    for row in failed:
        encoded = row if isinstance(row, str) else dumps(row)
        encoded = encoded.replace("/image.png", "/missing.png")
        rows.append(encoded if isinstance(row, str) else loads(encoded))
    # A URL in an unbound model must never be fetched.
    rows.extend(image_rows({**first, "name": "unbound"}, adapter, "unbound-model", "url-only"))
    log = tmp_path / "images.jsonl"
    log.write_text(
        "".join(row + "\n" if isinstance(row, str) else dumps(row) + "\n" for row in rows)
    )

    def run(config):
        source = (
            FileSource(log)
            if source_kind == "log"
            else ElasticsearchSource(
                "https://mock.invalid", "images-*", transport=MockElasticsearch(rows).transport()
            )
        )
        return run_batch(
            source,
            builtins().get("protocols", adapter),
            calculator,
            namespace="test/image-url",
            output_root=tmp_path / "runs",
            image_download=config,
        )

    config = ImageDownloadOptions(enabled=True, cache_dir=tmp_path / "cache")
    disabled, _ = run(config.model_copy(update={"enabled": False}))
    assert requests == [] and not config.cache_dir.exists()
    assert all(row["measurement"]["input_tokens"] is None for row in details(disabled))
    bundle, _ = run(config)
    assert verify_bundle(bundle)["plan"]["image_download"]["enabled"] is True
    assert len(requests) == 2  # One shared URL + one failed URL, independent of model count.
    saved = details(bundle)
    assert len(saved) == len(VECTORS) + 2
    for row in saved:
        count = row["measurement"]
        if row["call"]["model"] == "unbound-model":
            assert count["reason"] == "model_binding_missing"
            continue
        assert count["output_execution"] == "completed"
        image = count["input_images"][0]
        if image["reason"]:
            assert count["input_verdict"] == "UNVERIFIABLE"
            assert "image_http_status:404" in image["reason"]
            assert count["input_tokens"] is None
            continue
        vector = expected[row["call"]["model"]]
        assert count["input_verdict"] == "CONSISTENT" and count["input_delta"] == 0
        assert count["input_tokens"] == vector["input_tokens"]
        assert count["input_ids_sha256"] == vector["input_ids_sha256"]
        assert image["download"]["historical_content"] == "unverified"
        assert image["download"]["sha256"] == image["sha256"]
        assert row["call"]["messages"][0]["content_parts"][1]["download"] == image["download"]
    # Cache hits retain original retrieval evidence, keeping reruns reproducible.
    repeated, reused = run(config)
    assert reused and repeated == bundle
    assert len(requests) == 3  # Failed downloads are retried rather than cached as success.


@pytest.mark.assets
def test_cli_config_and_model_filter_control_downloads(
    tmp_path, monkeypatch, official_assets, capsys
):
    requests = mock_image_http(monkeypatch)
    vector = VECTORS[0]
    # Both aliases are bound, but only "included" may trigger retrieval.
    records = []
    for alias in ("included", "excluded"):
        row = image_rows({**vector, "name": alias}, "litellm", alias, "url-only")[0]
        records.append(loads(dumps(row).replace("/image.png", f"/{alias}.png")))
    log = tmp_path / "images.jsonl"
    log.write_text("".join(dumps(row) + "\n" for row in records))
    (tmp_path / "source.yaml").write_text("path: images.jsonl\n")
    config = tmp_path / "modulo.yaml"
    config.write_text(
        "source: log\nsource-config: source.yaml\nadapter: litellm\n"
        f"models: [{{aliases: [included, excluded], name: {vector['asset']}}}]\n"
        f"assets: {official_assets}\noutput: runs\n"
        "image-download:\n  enabled: true\n  cache-dir: image-cache\n"
    )
    args = ["import", "--config", str(config), "--models", "included"]
    assert main([*args, "--no-download-images"]) == 0
    capsys.readouterr()
    assert requests == []
    assert main(args) == 0
    bundle = Path(json.loads(capsys.readouterr().out)["run"])
    assert len(requests) == 1 and requests[0].url.path == "/included.png"
    assert details(bundle)[0]["measurement"]["input_verdict"] == "CONSISTENT"
    assert (tmp_path / "image-cache").is_dir()
