import copy
import json
from pathlib import Path

import duckdb
import pytest

from modulo.adapters.protocols import DifyProtocol, LangChainProtocol
from modulo.adapters.sources import FileSource
from modulo.application.importer import run_batch
from modulo.cli import main
from modulo.cli.configuration import read_config
from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.interfaces import AssetError
from modulo.core.domain.platforms import PlatformConfig, resolve_platform
from modulo.shared.serialization import dumps, loads
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT


def configuration(**values):
    return PlatformConfig.model_validate(
        {key.replace("_", "-"): value for key, value in values.items()}
    )


def test_nested_langchain_metadata_and_dify_envelope_are_call_scoped(dify_record):
    base = loads((ROOT / "examples/langchain/basic.jsonl").read_bytes().splitlines()[0])
    child = copy.deepcopy(base)
    child["id"] = "another-child"
    base["extra"]["metadata"]["platform"] = "volcengine"
    child["extra"]["metadata"]["platform"] = "huaweicloud"
    nested = {
        "run_type": "chain",
        "extra": {"metadata": {"platform": "wrong-parent"}},
        "child_runs": [base, child],
    }
    parsed = LangChainProtocol().parse(nested)
    config = configuration(platform_tag_source="metadata:platform")
    tags = [resolve_platform(config, nested, context) for context in parsed.contexts]
    assert [tag.value for tag in tags] == ["volcengine", "huaweicloud"]
    assert tags[1].field_path == "payload:child_runs.1.extra.metadata.platform"
    dify_record["metadata"] = {"platform": "volcengine"}
    envelope = {
        "event": "node_finished",
        "data": dify_record,
        "metadata": {"platform": "wrong-envelope"},
    }
    context = DifyProtocol().parse(envelope).contexts[0]
    assert resolve_platform(config, envelope, context).value == "volcengine"
    assert context.metadata_path == "data.metadata"


def run_platform_records(tmp_path, records, calculator, config):
    path = tmp_path / "logs.jsonl"
    path.write_text("".join(dumps(record) + "\n" for record in records))
    return run_batch(
        FileSource(path),
        DifyProtocol(),
        calculator,
        namespace="test",
        output_root=tmp_path / "runs",
        platform_config=config,
    )


@pytest.fixture
def rule_calculator(unit_calculator):
    class PlatformReference:
        version = "explicit-test-vector"

        def count(self, call, binding):
            return {"input_tokens": 17}

    unit_calculator.lock["assets"]["platform-reference"] = {"family": "test-only"}
    load = unit_calculator.load_backend
    unit_calculator.load_backend = lambda binding: (
        PlatformReference() if binding.asset == "platform-reference" else load(binding)
    )
    return unit_calculator


@pytest.mark.parametrize("reverse", [False, True])
def test_per_rule_sources_precede_global_without_changing_model_names(
    tmp_path, rule_calculator, dify_record, reverse
):
    rules = [
        {"modelnames": ["test-model", "local-alias"], "asset": "test-only"},
        {
            "modelnames": ["test-model", "local-alias"],
            "asset": "platform-reference",
            "platform": "volcengine",
            "platform-tag-source": "metadata:platform",
        },
    ]
    rule_calculator.bindings = Bindings.model_validate(
        {"models": rules[::-1] if reverse else rules}
    )
    records = []
    for index, platform in enumerate(["volcengine", "huaweicloud", None]):
        record = copy.deepcopy(dify_record)
        record.update(id=str(index), metadata={"platform": platform})
        record["process_data"]["model_name"] = "local-alias"
        records.append(record)
    # An explicit entry source is not replaced by a batch default.
    config = configuration(platform_tag_source="literal:batch-default")
    bundle, _ = run_platform_records(tmp_path, records, rule_calculator, config)
    rows = {row["call"]["platform"]: row for row in details(bundle)}
    assert set(rows) == {"volcengine", "huaweicloud", None}
    assert rows["volcengine"]["measurement"]["input_tokens"] == 17
    assert rows["huaweicloud"]["measurement"]["input_tokens"] == 13
    assert rows[None]["measurement"]["input_tokens"] == 13
    assert all(row["call"]["model"] == "local-alias" for row in rows.values())
    assert rows[None]["platform_tags"][0]["reason"] == "platform_field_missing"
    assert rows["volcengine"]["platform_tags"][0]["field_path"] == "payload:metadata.platform"
    frozen = loads((bundle / "bindings.json").read_bytes())
    assert verify_bundle(bundle)["plan"]["bindings"] == frozen
    assert Bindings.model_validate(frozen) == rule_calculator.bindings

    filtered, _ = run_platform_records(
        tmp_path, records, rule_calculator, configuration(platforms=["volcengine"])
    )
    summary = loads((filtered / "summary.json").read_bytes())
    assert summary["selection"]["selected_calls"] == 1
    assert summary["selection"]["excluded_calls"] == 1
    assert summary["selection"]["unresolved_calls"] == 1
    assert details(filtered)[0]["measurement"]["input_tokens"] == 17


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("second_platform", ["volcengine", "huaweicloud"])
def test_conflicting_matching_rules_preserve_evidence_and_never_recount(
    tmp_path, rule_calculator, dify_record, reverse, second_platform
):
    rules = [
        {"modelnames": ["test-model"], "asset": "test-only"},
        {
            "modelnames": ["test-model"],
            "asset": "test-only",
            "platform": "volcengine",
            "platform-tag-source": "field:platform",
        },
        {
            "modelnames": ["test-model"],
            "asset": "platform-reference",
            "platform": second_platform,
            "platform-tag-source": "metadata:platform",
        },
    ]
    rule_calculator.bindings = Bindings.model_validate(
        {"models": rules[::-1] if reverse else rules}
    )

    def forbidden(_):
        pytest.fail("a conflicting binding must not load any tokenizer")

    rule_calculator.load_backend = forbidden
    record = {**dify_record, "platform": "volcengine", "metadata": {"platform": second_platform}}
    bundle, _ = run_platform_records(tmp_path, [record], rule_calculator, PlatformConfig())
    row = details(bundle)[0]
    assert row["measurement"]["reason"] == "binding_rule_conflict"
    assert row["measurement"]["execution"] == "error"
    assert row["measurement"]["input_tokens"] is None
    assert row["measurement"]["output_execution"] == "not_run"
    assert row["binding_matches"] == [0, 1, 2]
    assert {tag["source"] for tag in row["platform_tags"]} == {
        "field:platform",
        "metadata:platform",
    }
    assert all(tag["source_ref"] and tag["matched"] for tag in row["platform_tags"])
    assert loads((bundle / "summary.json").read_bytes())["run_status"] == "PARTIAL"
    assert len((bundle / "raw.jsonl").read_text().splitlines()) == 1


def test_equivalent_rules_are_allowed_and_endpoint_is_configured_in_the_rule(
    tmp_path, rule_calculator, dify_record
):
    base = {"modelnames": ["test-model"], "asset": "platform-reference", "platform": "volcengine"}
    rule_calculator.bindings = Bindings.model_validate(
        {
            "models": [
                {**base, "platform-tag-source": "metadata:platform"},
                {
                    **base,
                    "platform-tag-source": "endpoint",
                    "endpoint-source": "field:endpoint",
                    "endpoint-map": {"https://synthetic.test/v1": "volcengine"},
                },
            ]
        }
    )
    record = {
        **dify_record,
        "endpoint": "https://synthetic.test/v1",
        "metadata": {"platform": "volcengine"},
    }
    bundle, _ = run_platform_records(tmp_path, [record], rule_calculator, PlatformConfig())
    row = details(bundle)[0]
    assert row["measurement"]["input_tokens"] == 17 and row["binding_error"] is None
    assert row["binding_matches"] == [0, 1]
    assert row["platform_tags"][1]["mapping_hash"]


def test_matching_platform_asset_failure_does_not_retry_global(
    tmp_path, rule_calculator, dify_record
):
    rule_calculator.bindings = Bindings.model_validate(
        {
            "models": [
                {"modelnames": ["test-model"], "asset": "test-only"},
                {
                    "modelnames": ["test-model"],
                    "asset": "platform-reference",
                    "platform": "volcengine",
                    "platform-tag-source": "literal:volcengine",
                },
            ]
        }
    )

    def missing(binding):
        assert binding.asset == "platform-reference"
        raise AssetError("asset_files_missing")

    rule_calculator.load_backend = missing
    bundle, _ = run_platform_records(tmp_path, [dify_record], rule_calculator, PlatformConfig())
    row = details(bundle)[0]
    assert row["call"]["platform"] == "volcengine"
    assert row["measurement"]["input_tokens"] is None
    assert row["measurement"]["reason"] == "asset_files_missing"


def test_same_model_is_grouped_and_bound_separately(tmp_path, unit_calculator, dify_record):
    binding = Binding(modelnames=["test-model"], asset="test-only")
    unit_calculator.bindings = Bindings(
        models=[binding, binding.model_copy(update={"platform": "volcengine"})],
    )
    records = []
    for index, platform in enumerate(["volcengine", "huaweicloud", None]):
        record = copy.deepcopy(dify_record)
        record.update(id=f"call-{index}", platform=platform)
        records.append(record)
    config = configuration(platform_tag_source="field:platform")
    bundle, _ = run_platform_records(tmp_path, records, unit_calculator, config)
    frozen_bindings = loads((bundle / "bindings.json").read_bytes())
    assert set(frozen_bindings) == {"models"}
    assert verify_bundle(bundle)["plan"]["bindings"] == frozen_bindings
    rows = {row["call"]["platform"]: row for row in details(bundle)}
    assert rows["volcengine"]["measurement"]["input_tokens"] == 13
    assert rows["huaweicloud"]["measurement"]["input_tokens"] == 13
    assert rows[None]["measurement"]["input_tokens"] == 13
    assert rows["volcengine"]["binding_matches"] == [0, 1]
    assert rows["huaweicloud"]["binding_matches"] == rows[None]["binding_matches"] == [0]
    summary = loads((bundle / "summary.json").read_bytes())
    groups = {group["platform"]: group for group in summary["by_platform_model"]}
    assert set(groups) == {"volcengine", "huaweicloud", None}
    assert all(group["calls"] == 1 for group in groups.values())
    assert groups["volcengine"]["input_recalculated"] == 1
    assert groups["huaweicloud"]["recalculated_input_sum"] == 13
    with duckdb.connect(str(bundle / "query.duckdb"), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM calls WHERE platform = 'volcengine'"
            ).fetchone()[0]
            == 1
        )


def test_platform_filter_counts_unknown_and_does_not_invent_cross_variant_match(
    tmp_path, unit_calculator, dify_record
):
    first = copy.deepcopy(dify_record)
    first["platform"] = "volcengine"
    first["process_data"]["model_name"] = "model-a"
    conflict = copy.deepcopy(first)
    conflict["platform"] = "huaweicloud"
    conflict["process_data"]["model_name"] = "model-b"
    missing = copy.deepcopy(dify_record)
    missing["id"] = "missing-platform"
    missing["process_data"]["model_name"] = "model-b"
    config = configuration(
        platform_tag_source="field:platform", platforms=["volcengine"], models=["model-b"]
    )
    bundle, _ = run_platform_records(tmp_path, [first, conflict, missing], unit_calculator, config)
    summary = loads((bundle / "summary.json").read_bytes())
    assert summary["selection"]["selected_calls"] == 0
    assert summary["selection"]["excluded_calls"] == 1
    assert summary["selection"]["unresolved_calls"] == 1
    assert summary["counters"]["identity_conflicts"] == 1
    assert summary["run_status"] == "FAILED"


@pytest.mark.parametrize("reverse", [False, True])
def test_platform_filter_keeps_conflicting_identity_and_provenance(
    tmp_path, unit_calculator, dify_record, reverse
):
    first = {**dify_record, "platform": "volcengine"}
    conflict = {**dify_record, "platform": "huaweicloud"}
    records = [first, conflict]
    config = configuration(platform_tag_source="field:platform", platforms=["volcengine"])
    bundle, _ = run_platform_records(
        tmp_path, records[::-1] if reverse else records, unit_calculator, config
    )
    row = details(bundle)[0]
    assert row["identity_conflict"] and row["measurement"]["reason"] == "identity_conflict"
    assert {tag["value"] for tag in row["platform_tags"]} == {"volcengine", "huaweicloud"}
    assert all(tag["source_ref"] for tag in row["platform_tags"])
    group = loads((bundle / "summary.json").read_bytes())["by_platform_model"][0]
    assert group["platform"] is None and group["identity_conflicts"] == 1
    assert group["reported_input_sum"] is None


def test_platform_configuration_is_frozen_for_replay(tmp_path, unit_calculator, dify_record):
    source = {**dify_record, "endpoint": "synthetic-endpoint"}
    config = configuration(platform_tag_source="endpoint", endpoint_map={"synthetic-endpoint": "a"})
    first, _ = run_platform_records(tmp_path, [source], unit_calculator, config)
    same, reused = run_platform_records(tmp_path, [source], unit_calculator, config)
    assert same == first and reused
    changed = configuration(
        platform_tag_source="endpoint", endpoint_map={"synthetic-endpoint": "b"}
    )
    second, _ = run_platform_records(tmp_path, [source], unit_calculator, changed)
    assert first != second
    assert verify_bundle(first)["plan"]["platform_config"]["endpoint-map"] == {
        "synthetic-endpoint": "a"
    }
    assert (
        details(first)[0]["platform_tags"][0]["mapping_hash"]
        != details(second)[0]["platform_tags"][0]["mapping_hash"]
    )


@pytest.mark.parametrize("source", ["none", "literal:volcengine"])
def test_aliases_share_a_reference_but_preserve_logged_names_and_filters(
    tmp_path, unit_calculator, dify_record, source
):
    binding = Binding(modelnames=["test-model", "local-alias"], asset="test-only")
    unit_calculator.bindings = Bindings(
        models=[binding, binding.model_copy(update={"platform": "volcengine"})]
    )
    records = []
    for name in ["test-model", "local-alias", "LOCAL-ALIAS"]:
        record = copy.deepcopy(dify_record)
        record["id"] = name
        record["process_data"]["model_name"] = name
        records.append(record)
    bundle, _ = run_platform_records(
        tmp_path, records, unit_calculator, configuration(platform_tag_source=source)
    )
    rows = {row["call"]["model"]: row for row in details(bundle)}
    assert set(rows) == {"test-model", "local-alias", "LOCAL-ALIAS"}
    for name in binding.modelnames:
        assert rows[name]["measurement"]["input_tokens"] == 13
        assert rows[name]["measurement"]["verdict"] == "CONSISTENT"
    assert rows["LOCAL-ALIAS"]["measurement"]["execution"] == "not_run"
    frozen = verify_bundle(bundle)["plan"]["bindings"]
    entries = frozen["models"]
    assert entries[0]["modelnames"] == ["test-model", "local-alias"]

    filtered, _ = run_platform_records(
        tmp_path,
        records,
        unit_calculator,
        configuration(platform_tag_source=source, models=["local-alias"]),
    )
    assert [row["call"]["model"] for row in details(filtered)] == ["local-alias"]
    summary = loads((filtered / "summary.json").read_bytes())
    assert summary["selection"]["excluded_calls"] == 2


@pytest.mark.assets
@pytest.mark.parametrize("protocol", ["dify", "langchain"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_unified_cli_platform_model_selection_and_yaml_equivalence(
    tmp_path, official_assets, capsys, mock_es, protocol, source
):
    fixture = ROOT / f"examples/{protocol}/platforms.jsonl"
    common = [
        "import",
        "--source",
        source,
        "--adapter",
        protocol,
        "--path",
        str(fixture) if source == "log" else "https://mock.invalid",
        "--payload-field",
        "payload",
        "--page-size",
        "3",
        "--bindings",
        str(ROOT / f"examples/{protocol}/platforms.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "es":
        mock_es(MockElasticsearch.from_file(fixture))
        common += ["--index", "mock-logs"]
    assert (
        main(
            [
                *common,
                "--platform-tag-source",
                "metadata:platform",
                "--platforms",
                "volcengine",
                "--models",
                "deepseek-ai/DeepSeek-V4.1-Flash",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    summary = result["summary"]
    assert summary["selection"]["parsed_calls"] == 8
    assert summary["selection"]["selected_calls"] == 1
    assert summary["selection"]["excluded_calls"] == 6
    assert summary["selection"]["unresolved_calls"] == 1
    assert summary["verdicts"] == {"DEVIATION": 1}
    row = details(Path(result["run"]))[0]
    assert row["call"]["platform"] == "volcengine"
    assert row["measurement"]["input_tokens"] == 55
    assert row["measurement"]["input_delta"] == 3
    assert row["platform_tags"][0]["provenance"] == "source_reported"
    assert summary["by_platform_model"][0]["platform"] == "volcengine"
    assert (
        main(
            [
                *common,
                "--platform-config",
                str(ROOT / f"examples/{protocol}/platforms-metadata.yaml"),
            ]
        )
        == 0
    )
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["run"] == result["run"] and repeated["reused"]

    # The same extraction and selection work from rule-local configuration,
    # with neither --platform-config nor --platform-tag-source.
    local = read_config(ROOT / f"examples/{protocol}/platforms.bindings.yaml")
    for rule in local["models"]:
        rule["platform-tag-source"] = "metadata:platform"
    binding_file = tmp_path / "bindings.json"
    binding_file.write_text(dumps(local))
    per_rule = common.copy()
    per_rule[per_rule.index("--bindings") + 1] = str(binding_file)
    assert (
        main(
            [*per_rule, "--platforms", "volcengine", "--models", "deepseek-ai/DeepSeek-V4.1-Flash"]
        )
        == 0
    )
    result_local = json.loads(capsys.readouterr().out)
    assert result_local["summary"] == summary
    local_row = details(Path(result_local["run"]))[0]
    assert local_row["call"] == row["call"]
    assert local_row["measurement"] == row["measurement"]


@pytest.mark.assets
def test_cli_endpoint_mode_and_explicit_yaml_overrides(tmp_path, official_assets, capsys):
    common = [
        "import",
        "--source",
        "log",
        "--adapter",
        "dify",
        "--path",
        str(ROOT / "examples/dify/platforms.jsonl"),
        "--payload-field",
        "payload",
        "--bindings",
        str(ROOT / "examples/dify/platforms.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    assert (
        main(
            [
                *common,
                "--platform-tag-source",
                "endpoint",
                "--endpoint-map",
                str(ROOT / "examples/dify/platforms-endpoints.yaml"),
                "--platforms",
                "volcengine",
                "--models",
                "deepseek-ai/DeepSeek-V4.1-Flash",
            ]
        )
        == 0
    )
    mapped = json.loads(capsys.readouterr().out)
    assert mapped["summary"]["selection"]["unresolved_calls"] == 2
    assert mapped["summary"]["selection"]["selected_calls"] == 1
    row = details(Path(mapped["run"]))[0]
    assert row["platform_tags"][0]["raw_value"] == "https://volcengine.example.test/v1"
    assert (
        main([*common, "--platform-config", str(ROOT / "examples/dify/platforms-endpoint.yaml")])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["run"] == mapped["run"]
    assert (
        main(
            [
                *common,
                "--platform-config",
                str(ROOT / "examples/dify/platforms-metadata.yaml"),
                "--platforms",
                "huaweicloud",
                "--models",
                "Qwen/Qwen3.8-2.4T-A95B",
            ]
        )
        == 0
    )
    overridden = json.loads(capsys.readouterr().out)
    assert overridden["summary"]["verdicts"] == {"CONSISTENT": 1}
    assert overridden["summary"]["by_platform_model"][0]["platform"] == "huaweicloud"


@pytest.mark.assets
@pytest.mark.parametrize("protocol", ["dify", "langchain"])
def test_mixed_platform_examples_match_frozen_expectations(
    tmp_path, official_assets, capsys, protocol
):
    assert (
        main(
            [
                "import",
                "--source",
                "log",
                "--path",
                str(ROOT / f"examples/{protocol}/platforms.jsonl"),
                "--adapter",
                protocol,
                "--payload-field",
                "payload",
                "--platform-tag-source",
                "field:platform",
                "--bindings",
                str(ROOT / f"examples/{protocol}/platforms.bindings.yaml"),
                "--assets",
                str(official_assets),
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)["summary"]
    expected = json.loads((ROOT / "tests/fixtures/expected/platforms.json").read_text())
    assert summary["verdicts"] == expected["expected_with_field_or_metadata_source"]
    assert summary["selection"]["selected_calls"] == 8
    groups = {(row["platform"], row["model"]): row for row in summary["by_platform_model"]}
    for platform in expected["platforms"]:
        assert groups[platform, "deepseek-ai/DeepSeek-V4.1-Flash"]["input_recalculated"] == 1
    assert groups["volcengine", "deepseek-ai/DeepSeek-V4.1-Flash"]["signed_input_delta_sum"] == 3
