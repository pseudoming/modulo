import json

import pytest

from modulo.cli.configuration import asset_options, import_options, read_config
from modulo.cli.parser import parser
from modulo.core.domain.platforms import PlatformConfig


def test_sampling_configuration_cli_overrides_and_switching_modes(tmp_path):
    path = tmp_path / "modulo.yaml"
    path.write_text(
        "source: log\nadapter: dify\nmodels: []\nsampling:\n"
        "  mode: window\n  window-minutes: 5\n  count: 100\n  seed: buyer\n"
    )
    args = ["import", "--config", str(path)]
    configured = import_options(parser().parse_args(args)).sampling
    assert configured.mode == "window" and configured.count == 100
    overridden = import_options(parser().parse_args([*args, "--sample-count", "30"])).sampling
    assert overridden.count == 30 and overridden.window_minutes == 5
    switched = import_options(
        parser().parse_args([*args, "--sample-mode", "ratio", "--sample-ratio", "0.1"])
    ).sampling
    assert str(switched.ratio) == "0.1" and switched.count is None and switched.seed == "buyer"
    assert (
        import_options(parser().parse_args([*args, "--sample-mode", "all"])).sampling.mode == "all"
    )
    assert asset_options(parser().parse_args(["prepare-assets", "--config", str(path)]))


def test_asset_region_configuration_and_explicit_cli_overrides(tmp_path):
    path = tmp_path / "modulo.yaml"
    path.write_text(
        "source: log\nadapter: dify\nmodels: []\nassets: local-assets\nasset-source: CN\n"
    )
    args = ["prepare-assets", "--config", str(path)]
    configured = asset_options(parser().parse_args(args))
    assert configured.assets == tmp_path / "local-assets"
    assert configured.asset_source == "cn"
    overridden = asset_options(
        parser().parse_args([*args, "--assets", "cli-assets", "--asset-source", "Global"])
    )
    assert str(overridden.assets) == "cli-assets"
    assert overridden.asset_source == "global"
    imported = import_options(parser().parse_args(["import", "--config", str(path)]))
    assert imported.asset_source == "cn"


def test_asset_preparation_does_not_require_log_source_or_bindings(tmp_path):
    path = tmp_path / "modulo.yaml"
    path.write_text("assets: local-assets\nasset-source: cn\n")
    assert (
        asset_options(parser().parse_args(["prepare-assets", "--config", str(path)])).asset_source
        == "cn"
    )
    default = asset_options(parser().parse_args(["prepare-assets"]))
    assert default.asset_source == "global"
    assert str(default.assets) == "assets/tokenizers"


@pytest.mark.parametrize("value", ["cn-mirror", "", None, False, ["cn"]])
def test_invalid_asset_region_is_rejected_by_both_commands(tmp_path, value):
    path = tmp_path / "modulo.json"
    path.write_text(
        json.dumps({"source": "log", "adapter": "dify", "models": [], "asset-source": value})
    )
    for command, resolve in [("import", import_options), ("prepare-assets", asset_options)]:
        with pytest.raises(ValueError):
            resolve(parser().parse_args([command, "--config", str(path)]))


def test_template_parameters_are_typed_but_model_names_stay_strings(tmp_path):
    path = tmp_path / "modulo.yaml"
    path.write_text(
        "source: log\nadapter: litellm\nmodels:\n"
        "  - aliases: [on, 1.10]\n    name: qwen-3.8-2.4t\n"
        "    add_generation_prompt: false\n"
        "    chat-template: '{{ messages[0].content }}'\n"
        "    template-kwargs:\n      enable_thinking: true\n"
        "      parallel_tool_calls: false\n      count: 0\n"
        "      schema: {properties: {z: [1.2300, null], a: 'false'}}\n"
    )
    binding = import_options(
        parser().parse_args(["import", "--config", str(path)])
    ).bindings.models[0]
    assert binding.aliases == ["on", "1.10"]
    assert binding.add_generation_prompt is False
    options = binding.template_kwargs
    assert options["enable_thinking"] is True and options["parallel_tool_calls"] is False
    assert type(options["count"]) is int and options["count"] == 0
    assert str(options["schema"]["properties"]["z"][0]) == "1.2300"
    assert options["schema"]["properties"]["z"][1] is None
    assert options["schema"]["properties"]["a"] == "false"


def test_image_download_configuration_paths_and_explicit_enable_disable(tmp_path):
    path = tmp_path / "modulo.yaml"
    path.write_text(
        "source: log\nadapter: dify\nmodels: []\n"
        "image-download:\n  enabled: false\n  cache-dir: image-cache\n"
        "  max-file-bytes: 1024\n  max-cache-bytes: 8192\n  ttl-seconds: 3600\n"
    )
    args = ["import", "--config", str(path)]
    configured = import_options(parser().parse_args(args)).image_download
    assert configured.enabled is False
    assert configured.cache_dir == tmp_path / "image-cache"
    assert (configured.max_file_bytes, configured.max_cache_bytes, configured.ttl_seconds) == (
        1024,
        8192,
        3600,
    )
    enabled = import_options(parser().parse_args([*args, "--download-images"])).image_download
    assert enabled.enabled is True and enabled.cache_dir == configured.cache_dir
    path.write_text(path.read_text().replace("enabled: false", "enabled: true"))
    assert import_options(parser().parse_args(args)).image_download.enabled is True
    assert (
        import_options(parser().parse_args([*args, "--no-download-images"])).image_download.enabled
        is False
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"enabled": "maybe"},
        {"enabled": 1},
        {"cache-dir": ["bad"]},
        {"max-file-bytes": 0},
        {"max-cache-bytes": 1000},
        {"max-file-bytes": True},
        {"ttl-seconds": -1},
        {"max-cache-bytes": 1.5},
        {"unkown": 12},
        {"allow-private-hosts": ["*"]},
        {"allow-private-hosts": ["https://example.test"]},
        {"allow-private-hosts": "example.test"},
        {"allow-private-hosts": ["example.test:8080"]},
    ],
)
def test_invalid_image_download_configuration_fails(tmp_path, value):
    path = tmp_path / "modulo.json"
    path.write_text(
        json.dumps(
            {
                "source": "log",
                "adapter": "dify",
                "models": [],
                "image-download": value,
            }
        )
    )
    with pytest.raises(ValueError):
        import_options(parser().parse_args(["import", "--config", str(path)]))


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
        "models: [{aliases: [configured-model], name: test-only}]\noutput: ../results\n"
        "max-records: 12\n"
    )
    monkeypatch.chdir(tmp_path)
    configured = import_options(parser().parse_args(["import", "--config", str(path)]))
    assert configured.path == str(tmp_path / "logs")
    assert configured.bindings.models[0].aliases == ["configured-model"]
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
        {"max-records": -1},
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
    binding.write_text("models: [{aliases: [on, 1.10], name: test-only}]\n")
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
    assert options.bindings.models[0].aliases == ["on", "1.10"]
