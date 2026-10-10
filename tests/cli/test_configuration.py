import json

import pytest

from modulo.cli.configuration import import_options, read_config
from modulo.cli.parser import parser
from modulo.core.domain.platforms import PlatformConfig


@pytest.mark.parametrize(
    "content",
    [
        "platform-tag-source: none\nplatform-tag-source: literal:other\n",
        "a: &anchor value\nb: *anchor\n",
        "!!python/object/apply:os.system ['exit 1']\n",
        "platform-tag-sorce: none\n",
        "platform-tag-source: endpoint\n",
        "platform-tag-source: field:\n",
        "platforms: []\n",
    ],
)
def test_invalid_yaml_configuration_is_rejected(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        PlatformConfig.model_validate(read_config(path))


def test_yaml_does_not_convert_version_or_boolean_like_names(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("platform-tag-source: field:platform\nplatforms: [on, off, 1.10]\n")
    config = PlatformConfig.model_validate(read_config(path))
    assert config.platforms == ["1.10", "off", "on"]


def test_import_config_paths_and_explicit_cli_overrides(tmp_path, monkeypatch):
    directory = tmp_path / "settings"
    directory.mkdir()
    path = directory / "import.yaml"
    source = directory / "import-log-file.yaml"
    source.write_text("path: ../logs\npattern: '*.jsonl'\n")
    path.write_text(
        "source: log\nsource-config: import-log-file.yaml\nadapter: dify\n"
        "models: [{modelnames: [configured-model], asset: test-only}]\noutput: ../results\n"
        "max-records: 12\n"
    )
    monkeypatch.chdir(tmp_path)
    configured = import_options(parser().parse_args(["import", "--config", str(path)]))
    assert configured.path == str(tmp_path / "logs")
    assert configured.bindings.models[0].modelnames == ["configured-model"]
    assert configured.models is None  # Declaring a mapping does not select a scan scope.
    assert configured.output == tmp_path / "results"
    assert configured.max_records == 12
    overridden = import_options(
        parser().parse_args(
            [
                "import",
                "--config",
                str(path),
                "--path",
                "other",
                "--models",
                "chosen-model",
                "--output",
                "other-results",
                "--max-records",
                "3",
            ]
        )
    )
    assert overridden.path == "other"
    assert str(overridden.output) == "other-results"
    assert overridden.models == ["chosen-model"]
    assert overridden.max_records == 3
    assert overridden.bindings == configured.bindings


def test_es_config_preserves_url_and_cli_location_alias_replaces_it(tmp_path):
    (tmp_path / "es.yaml").write_text(
        "path: https://es.example.test:9243\nindex: langchain-202610*\npage-size: 3\n"
    )
    path = tmp_path / "import.json"
    path.write_text(
        json.dumps(
            {
                "source": "es",
                "adapter": "langchain",
                "source-config": "es.yaml",
                "models": [],
            }
        )
    )
    configured = import_options(parser().parse_args(["import", "--config", str(path)]))
    assert configured.path == "https://es.example.test:9243"
    assert configured.es_index == "langchain-202610*"
    assert configured.page_size == 3
    overridden = import_options(
        parser().parse_args(
            ["import", "--config", str(path), "--es-url", "https://other.example.test"]
        )
    )
    assert overridden.path is None
    assert overridden.es_url == "https://other.example.test"


@pytest.mark.parametrize(
    "values",
    [
        {"soruce": "log"},
        {"source": ["log"]},
        {"source-config": ["source.yaml"]},
        {"models": "not-an-array"},
        {"models": ["not-a-binding"]},
        {"models": None},
        {"max-records": 0},
        {"path": "belongs-in-source-config"},
        {"index": "belongs-in-source-config"},
        {"bindings": "models-are-inline.yaml"},
    ],
)
def test_invalid_import_config_fails_before_opening_sources(tmp_path, values):
    path = tmp_path / "import.json"
    path.write_text(json.dumps({"source": "log", "adapter": "dify", "models": [], **values}))
    with pytest.raises(ValueError):
        import_options(parser().parse_args(["import", "--config", str(path)]))


@pytest.mark.parametrize(
    "kind, values",
    [
        ("es", {"adapter": "dify"}),
        ("es", {"assets": "assets"}),
        ("es", {"output": "runs"}),
        ("es", {"page-size": True}),
        ("es", {"page-size": 1.5}),
        ("es", {"username": ""}),
        ("es", {"password": "must-use-environment-variable"}),
        ("log", {"pattern": ""}),
        ("log", {"index": "logs-*"}),
    ],
)
def test_source_configuration_rejects_misplaced_or_invalid_options(tmp_path, kind, values):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(values))
    path = tmp_path / "modulo.yaml"
    path.write_text(f"source: {kind}\nsource-config: source.json\nadapter: dify\nmodels: []\n")
    with pytest.raises(ValueError):
        import_options(parser().parse_args(["import", "--config", str(path)]))


def test_changing_source_does_not_inherit_file_pattern_or_credentials(tmp_path):
    (tmp_path / "source.yaml").write_text("path: ../logs\npattern: '*.jsonl'\n")
    path = tmp_path / "modulo.yaml"
    path.write_text("source: log\nsource-config: source.yaml\nadapter: dify\nmodels: []\n")
    configured = import_options(
        parser().parse_args(
            [
                "import",
                "--config",
                str(path),
                "--source",
                "es",
                "--path",
                "https://es.test",
                "--index",
                "logs-*",
            ]
        )
    )
    assert configured.path == "https://es.test"
    assert configured.pattern is None and configured.source_config is None


def test_source_paths_are_relative_to_source_file_and_explicit_binding_replaces_models(tmp_path):
    (tmp_path / "connections").mkdir()
    (tmp_path / "connections/es.yaml").write_text(
        "path: https://es.test\nindex: logs-*\nquery-file: query.json\n"
    )
    path = tmp_path / "modulo.yaml"
    path.write_text("source: es\nsource-config: connections/es.yaml\nadapter: dify\nmodels: []\n")
    binding = tmp_path / "aliases.yaml"
    binding.write_text("models: [{modelnames: [on, 1.10], asset: test-only}]\n")
    options = import_options(
        parser().parse_args(
            [
                "import",
                "--config",
                str(path),
                "--bindings",
                str(binding),
            ]
        )
    )
    assert options.es_query == tmp_path / "connections/query.json"
    assert options.bindings.models[0].modelnames == ["on", "1.10"]
