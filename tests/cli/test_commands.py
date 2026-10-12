import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from modulo.cli import main
from modulo.shared.serialization import bytes_digest
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT


@pytest.mark.parametrize(
    "region,override,expected",
    [("cn", [], "modelscope.cn"), ("cn", ["--asset-source", "global"], "huggingface.co")],
)
def test_prepare_assets_reads_main_config_and_reports_verified_files(
    tmp_path, monkeypatch, capsys, region, override, expected
):
    path = tmp_path / "modulo.yaml"
    path.write_text(f"assets: prepared\nasset-source: {region}\n")
    data = b"fixed tokenizer"
    catalog = {
        "assets": {
            "sample": {
                "repo": "official/model",
                "revision": "fixed",
                "files": {"tokenizer.json": bytes_digest(data)},
                "modelscope": {"repo": "cn/model", "revision": "cn-fixed"},
            }
        }
    }
    requests = []

    def download(request, **kwargs):
        requests.append(urlsplit(request.full_url).netloc)
        return io.BytesIO(data)

    monkeypatch.setattr(
        "modulo.application.tokenizer_assets.load_tokenizer_assets", lambda: catalog
    )
    monkeypatch.setattr("modulo.application.tokenizer_assets.urlopen", download)
    assert main(["prepare-assets", "--config", str(path), "--only", "sample", *override]) == 0
    assert requests == [expected]
    assert (tmp_path / "prepared/sample/tokenizer.json").read_bytes() == data
    assert json.loads(capsys.readouterr().out) == {"verified_assets": ["sample/tokenizer.json"]}


MODELS = ["Qwen/Qwen3.8-2.4T-A95B", "deepseek-ai/DeepSeek-V4.1-Flash"]


def test_prometheus_command_uses_config_and_cli_override_without_loading_plugins(
    tmp_path, monkeypatch, capsys
):
    from modulo.cli.configuration import prometheus_options
    from modulo.cli.parser import parser

    config = tmp_path / "export.yaml"
    config.write_text("url: http://localhost/api/v1/write\nscope: old\nca-file: cert.pem\n")
    args = ["export-prometheus", str(tmp_path / "run"), "--config", str(config), "--scope", "new"]
    options = prometheus_options(parser().parse_args(args))
    assert options.scope == "new" and options.ca_file == tmp_path / "cert.pem"
    captured = []
    monkeypatch.setattr(
        "modulo.cli.commands.export_prometheus",
        lambda run, options: captured.append((run, options)) or {"export": "sent"},
    )
    monkeypatch.setattr(
        "modulo.cli.commands.build_registry",
        lambda _: pytest.fail("export must not discover model plugins"),
    )
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"export": "sent"}
    assert captured[0][1] == options


def test_prometheus_command_reports_failed_writes_and_missing_config(tmp_path, monkeypatch, capsys):
    assert main(["export-prometheus", str(tmp_path)]) == 1
    assert "Field required" in capsys.readouterr().err

    def fail(*_args):
        raise ValueError("Prometheus Remote Write returned HTTP 401")

    monkeypatch.setattr("modulo.cli.commands.export_prometheus", fail)
    assert (
        main(
            [
                "export-prometheus",
                str(tmp_path),
                "--url",
                "http://localhost/write",
                "--scope",
                "test",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "HTTP 401" in output.err and not output.out


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_unified_cli_filters_and_recounts_both_protocols(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    fixture = ROOT / f"examples/{adapter}/basic.jsonl"
    # Wrapper metadata is deliberately wrong: filtering must use the adapter's
    # normalized model, including when LangChain nests calls under a chain.
    records = [
        {"model": "misleading-wrapper-model", "payload": json.loads(line)}
        for line in fixture.read_text().splitlines()
    ]
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
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
        "--models",
        *MODELS,
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "es":
        args.extend(["--index", f"{adapter}-access-log-*", "--page-size", "3"])
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    bundle = Path(output["run"])
    summary = output["summary"]
    assert summary["run_status"] == "COMPLETE"
    assert summary["batch_read_complete"] is True
    assert summary["counters"]["raw_records"] == 10
    assert summary["counters"]["duplicate_calls"] == 1
    assert summary["selection"]["parsed_calls"] == 8
    assert summary["selection"]["selected_calls"] == summary["observed_calls"] == 7
    assert summary["selection"]["excluded_calls"] == 1
    assert summary["selection"]["unresolved_calls"] == 0
    assert summary["selection"]["unmatched_models"] == []
    assert summary["input_verdicts"] == {"CONSISTENT": 2, "DEVIATION": 1, "UNVERIFIABLE": 4}
    rows = details(bundle)
    assert {row["call"]["model"] for row in rows} == set(MODELS)
    assert sum(row["measurement"]["input_tokens"] is not None for row in rows) == 4
    assert sorted(
        row["measurement"]["input_delta"]
        for row in rows
        if row["measurement"]["input_delta"] is not None
    ) == [0, 0, 3]
    manifest = verify_bundle(bundle)
    assert manifest["plan"]["selection"]["models"] == MODELS
    if source == "es":
        assert mock.searches == 5 and mock.closed
        assert manifest["plan"]["source"]["index"] == f"{adapter}-access-log-*"
    assert main(["report", str(bundle), "--details"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 7


def test_legacy_flags_reuse_the_same_bundle_and_no_matches_are_explicit(tmp_path, capsys):
    common = [
        "--models",
        "not-present",
        "--output",
        str(tmp_path / "runs"),
    ]
    path = str(ROOT / "examples/dify/basic.jsonl")
    assert main(["import", "--source", "log", "--path", path, "--adapter", "dify", *common]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["summary"]["selection"]["unmatched_models"] == ["not-present"]
    assert first["summary"]["selection"]["excluded_calls"] == 8
    assert first["summary"]["observed_calls"] == 0
    assert first["summary"]["input_verdicts"] == {}
    assert first["summary"]["verdicts"] == first["summary"]["output_verdicts"] == {}
    assert details(Path(first["run"])) == []
    assert (
        main(
            [
                "import",
                "--source",
                "file",
                "--file",
                path,
                "--protocol",
                "dify",
                "--namespace",
                "local",
                *common,
            ]
        )
        == 0
    )
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["run"] == first["run"] and repeated["reused"]


@pytest.mark.parametrize(
    "args, message",
    [
        (["--source", "es", "--path", "http://localhost:9200"], "--index"),
        (["--source", "log"], "--path"),
        (["--source", "log", "--path", "any.jsonl", "--index", "logs"], "ES options"),
        (["--source", "es", "--file", "any.jsonl"], "--file"),
    ],
)
def test_incompatible_or_missing_source_options_fail_before_reading(args, message, capsys):
    assert main(["import", "--adapter", "dify", *args]) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert captured.out == ""


def test_conflicting_location_flags_are_rejected(capsys):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "import",
                "--source",
                "log",
                "--adapter",
                "dify",
                "--path",
                "first.jsonl",
                "--file",
                "second.jsonl",
            ]
        )
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
