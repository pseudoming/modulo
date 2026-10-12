import json
from pathlib import Path

import pytest

from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.fixtures.generate_reasoning_fixtures import protocol_row
from tests.tools.references.build_family_reference import kimi_reference

ADDED_OUTPUTS = {
    "qwen3.5-122b": "THOUGHT\n</think>\n\nANSWER<|im_end|>",
    "qwen3.5-397b": "THOUGHT\n</think>\n\nANSWER<|im_end|>",
    "deepseek-v4-pro-0813": "THOUGHT</think>ANSWER<｜end▁of▁sentence｜>",
    "glm-5.3-flash": "THOUGHT</think>ANSWER<|endoftext|>",
    "glm-5.2": "THOUGHT</think>ANSWER<|endoftext|>",
    "minimax-m2.7": "THOUGHT\n</think>\n\nANSWER[e~[",
    "kimi-k2.6": "THOUGHT</think>ANSWER<|im_end|>",
    "kimi-k2.7-code": "THOUGHT</think>ANSWER<|im_end|>",
}


@pytest.mark.assets
@pytest.mark.parametrize("source", ["log", "es"])
def test_family_references_through_the_shared_cli(
    tmp_path, official_assets, capsys, mock_es, source
):
    fixture = ROOT / "examples/litellm/model-families.jsonl"
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        "litellm",
        "--bindings",
        str(ROOT / "examples/litellm/model-families.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
        "--platform-tag-source",
        "metadata:platform",
        "--platforms",
        "synthetic-reference",
    ]
    if source == "log":
        args += ["--path", str(fixture)]
    else:
        mock_es(MockElasticsearch.from_file(fixture))
        args += ["--path", "https://mock.invalid", "--index", "mock-logs", "--page-size", "3"]
    assert main(args) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["summary"]["input_verdicts"] == {
        "CONSISTENT": 19,
        "DEVIATION": 1,
        "UNVERIFIABLE": 3,
    }
    assert imported["summary"]["batch_read_complete"]
    bundle = Path(imported["run"])
    verify_bundle(bundle)
    rows = details(bundle)
    assert len(rows) == 23
    calls = {row["call"]["call_id"]: row for row in rows}
    assert calls["synthetic-qwen-deviation"]["measurement"]["input_delta"] == 3
    missing = calls["synthetic-kimi-missing-usage"]["measurement"]
    assert missing["execution"] == "completed" and missing["input_tokens"] == 112
    assert (
        missing["reported_input_tokens"] is None and missing["reason"] == "reported_input_missing"
    )
    assert (
        calls["synthetic-unbound-model"]["measurement"]["reason"]
        == "platform_model_binding_missing"
    )
    assert calls["synthetic-qwen-explicit-mode"]["measurement"]["execution"] == "not_run"
    assert all(row["measurement"]["model_identity"] == "UNVERIFIED" for row in rows)
    if source == "log":
        assert main(args) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["reused"] and repeated["run"] == imported["run"]


@pytest.fixture(scope="module")
def added_output_cache():
    return {}


@pytest.fixture
def added_output_counts(official_assets, added_output_cache):
    from tokenizers import Tokenizer

    if added_output_cache:
        return added_output_cache
    result = added_output_cache
    for asset, text in ADDED_OUTPUTS.items():
        directory = official_assets / asset
        if asset.startswith("kimi"):
            _, encoder = kimi_reference(directory)
            ids = encoder.encode(text, allowed_special="all")
        else:
            ids = (
                Tokenizer.from_file(str(directory / "tokenizer.json"))
                .encode(text, add_special_tokens=False)
                .ids
            )
        result[asset] = len(ids)
    return result


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain", "newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_added_models_keep_both_directions_through_every_adapter(
    tmp_path, official_assets, capsys, mock_es, adapter, source, added_output_counts
):
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
    )["vectors"]
    entries = load_tokenizer_assets()["assets"]
    rows = []
    for asset in ADDED_OUTPUTS:
        vector = {
            **next(v for v in vectors if v["asset"] == asset and v["name"] == "tool_continuation"),
            "output_text": "ANSWER",
            "output_reasoning": "THOUGHT",
        }
        model = entries[asset]["repo"]
        usage = {
            "prompt_tokens": len(vector["input_ids"]),
            "completion_tokens": added_output_counts[asset],
        }
        terminal = {
            "message": {"role": "assistant", "content": "ANSWER", "reasoning_content": "THOUGHT"},
            "finish_reason": "stop",
        }
        if adapter in {"newapi", "oneapi"}:
            row = {
                "schema_version": "gateway.exchange.v1",
                "request_id": asset,
                "request_body": {
                    "model": model,
                    "messages": vector["messages"],
                    "tools": vector["tools"],
                },
                "response_body": {"choices": [terminal], "usage": usage},
            }
        else:
            row = protocol_row(vector, adapter, model)
            if adapter == "dify":
                row["outputs"].update(usage=usage)
            elif adapter == "langchain":
                row["outputs"]["generations"][0][0]["generation_info"] = {"finish_reason": "stop"}
                row["outputs"]["llm_output"]["token_usage"] = usage
            else:
                row.update(usage)
                row["response"] = {"choices": [terminal]}
        rows.append(row)
    fixture = tmp_path / "added.jsonl"
    fixture.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
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
        args += ["--path", str(fixture)]
    else:
        mock_es(MockElasticsearch.from_file(fixture))
        args += ["--path", "https://mock.invalid", "--index", "added", "--page-size", "3"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    for field in ("verdicts", "input_verdicts", "output_verdicts"):
        assert result["summary"][field] == {"CONSISTENT": len(ADDED_OUTPUTS)}
    verify_bundle(Path(result["run"]))
