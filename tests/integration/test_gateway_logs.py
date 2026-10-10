import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_gateway_logs_share_sources_platform_filter_bindings_and_storage(
    tmp_path, official_assets, mock_es, capsys, adapter, source
):
    fixture = ROOT / f"examples/{adapter}/basic.jsonl"
    provenance = json.loads((ROOT / "tests/fixtures/expected/gateway-logs.json").read_text())
    expected = provenance["expected_after_model_filter"][adapter]
    records = [
        {
            "platform": "synthetic-gateway",
            "model": "misleading-wrapper-model",
            "payload": json.loads(line),
        }
        for line in fixture.read_text().splitlines()
    ]
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    mock = MockElasticsearch(records)
    if source == "es":
        mock_es(mock)
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--path",
        str(path) if source == "log" else "https://mock.invalid",
        "--payload-field",
        "payload",
        "--platform-tag-source",
        "field:platform",
        "--platforms",
        "synthetic-gateway",
        "--models",
        *provenance["selected_models"],
        "--bindings",
        str(ROOT / f"examples/{adapter}/basic.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "es":
        args.extend(["--index", f"{adapter}-access-log-*", "--page-size", "2"])
    assert main(args) == 0
    imported = json.loads(capsys.readouterr().out)
    summary = imported["summary"]
    assert summary["run_status"] == "COMPLETE" and summary["batch_read_complete"]
    for field in ("raw_records", "ignored_records"):
        assert summary["counters"][field] == expected[field]
    for field in ("parsed_calls", "selected_calls", "excluded_calls"):
        assert summary["selection"][field] == expected[field]
    assert summary["selection"]["unresolved_calls"] == 0
    assert summary["verdicts"] == expected["verdicts"]
    bundle = Path(imported["run"])
    rows = details(bundle)
    assert len(rows) == expected["selected_calls"]
    assert {row["call"]["platform"] for row in rows} == {"synthetic-gateway"}
    assert {row["call"]["model"] for row in rows} == set(provenance["selected_models"])
    assert all(row["source_refs"] and row["platform_tags"] for row in rows)
    verify_bundle(bundle)
    if adapter == "litellm":
        measured = [row["measurement"] for row in rows]
        tool_reference = json.loads(
            (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
        )["gateway_tool_reference"]
        tool_tokens = len(tool_reference["input_ids"])
        assert sorted(m["input_delta"] for m in measured if m["input_delta"] is not None) == sorted(
            [0, 3, 66 - tool_tokens]
        )
        assert sorted(m["input_tokens"] for m in measured if m["input_tokens"] is not None) == [
            55,
            66,
            66,
            tool_tokens,
        ]
    else:
        assert all(row["measurement"]["execution"] == "not_run" for row in rows)
        assert all(row["measurement"]["input_tokens"] is None for row in rows)
        assert any(row["call"]["usage"]["input_tokens"] == 58 for row in rows)
    if source == "log":
        assert main(args) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["run"] == imported["run"] and repeated["reused"]
    else:
        assert mock.closed and mock.searches > 1


@pytest.mark.parametrize("adapter", ["newapi", "oneapi"])
def test_gateway_row_identity_not_request_id_or_usage(tmp_path, capsys, adapter):
    record = json.loads((ROOT / f"examples/{adapter}/basic.jsonl").read_text().splitlines()[0])
    other = {**record, "id": record["id"] + 1000}
    conflict = {**record, "prompt_tokens": record["prompt_tokens"] + 1}
    path = tmp_path / "identity.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in (record, record, other, conflict)))
    assert (
        main(
            [
                "import",
                "--source",
                "log",
                "--path",
                str(path),
                "--adapter",
                adapter,
                "--bindings",
                str(ROOT / f"examples/{adapter}/basic.bindings.yaml"),
                "--output",
                str(tmp_path / "runs"),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["observed_calls"] == 2
    assert result["summary"]["counters"]["duplicate_calls"] == 1
    assert result["summary"]["counters"]["identity_conflicts"] == 1
    assert result["summary"]["verdicts"] == {"UNVERIFIABLE": 2}
