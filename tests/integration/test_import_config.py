import base64
import json
from pathlib import Path

import pytest

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch


@pytest.fixture
def configured_import(tmp_path, unit_registry, monkeypatch):
    directory = tmp_path / "config"
    directory.mkdir()
    models = "models: [{modelnames: [test-model, internal-model], asset: test-only}]\n"
    (directory / "bindings.yaml").write_text(models)
    (directory / "modulo.yaml").write_text(
        "source: log\nsource-config: import-log-file.yaml\nadapter: dify\n"
        + models
        + "output: ../results\nmax-records: 10\n"
    )
    monkeypatch.setattr("modulo.cli.commands.build_registry", lambda _plugins: unit_registry)
    monkeypatch.setattr(
        "modulo.cli.commands.asset_lock",
        lambda: {
            "assets": {"test-only": {"family": "test-only"}},
        },
    )
    monkeypatch.chdir(tmp_path)
    return directory


def test_dated_files_config_matches_cli_and_keeps_one_batch_budget(
    tmp_path, configured_import, dify_record, capsys
):
    logs = tmp_path / "logs"
    logs.mkdir()
    originals = {}
    for day in ["02", "01"]:
        filename = f"dify-202610{day}.jsonl"
        data = (json.dumps({**dify_record, "id": day}, ensure_ascii=False) + "\r\n").encode()
        (logs / filename).write_bytes(data)
        originals[f"file:{filename}:line:1"] = data
    (logs / "dify-20260930.jsonl").write_text("excluded bad log\n")
    config = configured_import / "modulo.yaml"
    (configured_import / "import-log-file.yaml").write_text(
        "path: ../logs\npattern: dify-202610*.jsonl\n"
    )
    assert main(["import", "--config", str(config)]) == 0
    first = json.loads(capsys.readouterr().out)
    bundle = Path(first["run"])
    assert first["summary"]["verdicts"] == {"CONSISTENT": 2}
    assert len(details(bundle)) == 2
    manifest = verify_bundle(bundle)
    assert [item["path"] for item in manifest["plan"]["source"]["files"]] == [
        "dify-20261001.jsonl",
        "dify-20261002.jsonl",
    ]
    raw = [json.loads(line) for line in (bundle / "raw.jsonl").read_text().splitlines()]
    for row in raw:
        assert base64.b64decode(row["data"]) == originals[row["locator"]]
    assert (
        main(
            [
                "import",
                "--source",
                "log",
                "--path",
                str(logs),
                "--pattern",
                "dify-202610*.jsonl",
                "--adapter",
                "dify",
                "--bindings",
                str(configured_import / "bindings.yaml"),
                "--output",
                str(tmp_path / "results"),
                "--max-records",
                "10",
            ]
        )
        == 0
    )
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["run"] == first["run"] and repeated["reused"]
    assert main(["import", "--config", str(config), "--max-records", "1"]) == 2
    limited = json.loads(capsys.readouterr().out)
    assert limited["summary"]["counters"]["raw_records"] == 1
    assert limited["summary"]["batch_read_complete"] is False
    assert limited["run"] != first["run"]
    verify_bundle(bundle)


@pytest.mark.parametrize("auth", ["basic", "api-key", "none"])
def test_es_config_reaches_selected_cluster_pattern_and_environment_credentials(
    configured_import,
    dify_record,
    capsys,
    monkeypatch,
    auth,
):
    config = configured_import / "modulo.yaml"
    config.write_text(
        config.read_text()
        .replace("source: log", "source: es")
        .replace("import-log-file.yaml", "import-es.yaml")
    )
    credentials = {
        "basic": "username: audit-reader\npassword-env: TEST_ES_PASSWORD\n",
        "api-key": "api-key-env: TEST_ES_KEY\n",
        "none": "",
    }
    (configured_import / "import-es.yaml").write_text(
        "path: https://es.example.test:9243\nindex: dify-202610*\npage-size: 1\n"
        "query-file: query.json\n" + credentials[auth]
    )
    (configured_import / "query.json").write_text('{"match_all": {}}')
    monkeypatch.setenv("TEST_ES_KEY", "synthetic-test-key")
    monkeypatch.setenv("TEST_ES_PASSWORD", "synthetic-test-password")
    # An ambient key must not silently change the selected authentication method.
    monkeypatch.setenv("MODULO_ES_API_KEY", "ambient-key-must-not-be-used")
    mock = MockElasticsearch([dify_record])
    requests = []
    handle = mock.handle

    def capture(request):
        requests.append(request)
        return handle(request)

    monkeypatch.setattr(mock, "handle", capture)
    with mock.patch_source():
        assert main(["import", "--config", str(config)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["observed_calls"] == 1
    assert mock.closed and mock.searches == 2
    assert requests[0].url.host == "es.example.test" and requests[0].url.port == 9243
    assert requests[0].url.path == "/dify-202610*/_pit"
    expected_auth = {
        "basic": "Basic " + base64.b64encode(b"audit-reader:synthetic-test-password").decode(),
        "api-key": "ApiKey synthetic-test-key",
        "none": None,
    }
    assert all(request.headers.get("Authorization") == expected_auth[auth] for request in requests)
    bundle = Path(output["run"])
    plan = verify_bundle(bundle)["plan"]
    assert plan["source"]["index"] == "dify-202610*"
    assert json.loads(plan["source"]["query_json"]) == {"match_all": {}}
    for name in ["manifest.json", "execution.json", "diagnostics.jsonl"]:
        content = (bundle / name).read_text()
        assert "synthetic-test-key" not in content
        assert "synthetic-test-password" not in content


def test_main_config_maps_internal_aliases_and_filters_platforms(
    configured_import,
    dify_record,
    tmp_path,
    capsys,
):
    config = configured_import / "modulo.yaml"
    config.write_text(
        config.read_text() + "platform-tag-source: field:platform\nplatforms: [supplier-a]\n"
    )
    rows = []
    for name in ["supplier-a", "supplier-b"]:
        rows.append(
            {
                **dify_record,
                "id": name,
                "platform": name,
                "process_data": {**dify_record["process_data"], "model_name": "internal-model"},
            }
        )
    (tmp_path / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (configured_import / "import-log-file.yaml").write_text("path: ../logs.jsonl\n")
    assert main(["import", "--config", str(config)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["verdicts"] == {"CONSISTENT": 1}
    assert result["summary"]["selection"]["excluded_calls"] == 1
    bundle = Path(result["run"])
    (row,) = details(bundle)
    assert row["call"]["model"] == "internal-model"
    assert row["call"]["platform"] == "supplier-a"
    assert json.loads((bundle / "bindings.json").read_text())["models"][0]["asset"] == "test-only"
    assert verify_bundle(bundle)["plan"]["selection"]["platforms"] == ["supplier-a"]
    assert main(["import", "--config", str(config), "--platforms", "supplier-b"]) == 0
    other = json.loads(capsys.readouterr().out)
    assert details(Path(other["run"]))[0]["call"]["platform"] == "supplier-b"


@pytest.mark.parametrize(
    "credentials",
    [
        "username: audit-reader\npassword-env: MISSING_ES_PASSWORD\n",
        "api-key-env: MISSING_ES_KEY\n",
        "username: audit-reader\n",
        "username: audit-reader\npassword-env: TEST_ES_PASSWORD\napi-key-env: TEST_ES_KEY\n",
    ],
)
def test_invalid_credentials_stop_before_reading_logs(
    configured_import,
    credentials,
    capsys,
    monkeypatch,
):
    config = configured_import / "modulo.yaml"
    config.write_text(
        config.read_text()
        .replace("source: log", "source: es")
        .replace("import-log-file.yaml", "import-es.yaml")
    )
    (configured_import / "import-es.yaml").write_text(
        "path: https://es.test\nindex: logs-*\n" + credentials
    )
    monkeypatch.delenv("MISSING_ES_PASSWORD", raising=False)
    monkeypatch.delenv("MISSING_ES_KEY", raising=False)
    monkeypatch.setenv("TEST_ES_PASSWORD", "synthetic-test-password")
    monkeypatch.setenv("TEST_ES_KEY", "synthetic-test-key")
    assert main(["import", "--config", str(config)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "modulo:" in captured.err
    assert "synthetic-test-password" not in captured.err
    assert "synthetic-test-key" not in captured.err
