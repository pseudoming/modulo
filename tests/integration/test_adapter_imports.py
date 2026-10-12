"""Requested adapter additions through import, recount, saved details and reports."""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT

MODEL = "Qwen/Qwen3.8-2.4T-A95B"
# Literal official template expansion; independent of modulo's renderer.
PROMPT = (
    "<|im_start|>system\nReasoning effort is set to xhigh. "
    "Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize correctness, "
    "consistency, and clarity in the final answer.<|im_end|>\n"
    "<|im_start|>user\n查询<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
ANSWER = "THOUGHT\n</think>\n\nANSWER<|im_end|>"


def import_records(tmp_path, capsys, mock_es, adapter, records, source, official_assets):
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--assets",
        str(official_assets),
        "--bindings",
        str(ROOT / "examples/litellm/basic.bindings.yaml"),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        path = tmp_path / "input.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        args.extend(["--path", str(path)])
    else:
        mock_es(MockElasticsearch(list(reversed(records))))
        args.extend(["--path", "https://mock.invalid", "--index", "logs"])
    code = main(args)
    result = json.loads(capsys.readouterr().out)
    bundle = Path(result["run"])
    verify_bundle(bundle)
    assert main(["report", str(bundle)]) == 0
    assert json.loads(capsys.readouterr().out) == result["summary"]
    return code, result["summary"], details(bundle)


@pytest.mark.assets
@pytest.mark.parametrize("source", ["log", "es"])
@pytest.mark.parametrize(
    "adapter,filename",
    [
        ("dify", "thought-exports.jsonl"),
        ("langchain", "input-batches.jsonl"),
        ("litellm", "stream-and-cache.jsonl"),
    ],
)
def test_requested_adapter_examples_are_recounted_and_reported(
    tmp_path, capsys, mock_es, official_assets, source, adapter, filename
):
    tokenizer = Tokenizer.from_file(str(official_assets / "qwen-3.8-2.4t/tokenizer.json"))
    assert len(tokenizer.encode(PROMPT, add_special_tokens=False).ids) == 53
    assert len(tokenizer.encode(ANSWER, add_special_tokens=False).ids) == 9
    records = [
        json.loads(line)
        for line in (ROOT / "examples" / adapter / filename).read_text().splitlines()
    ]
    code, summary, rows = import_records(
        tmp_path, capsys, mock_es, adapter, records, source, official_assets
    )
    assert code == 0
    for row in rows:
        count = row["measurement"]
        assert count["input_tokens"] == 53 and count["execution"] == "completed"
        assert count["output_tokens"] == 9 and count["output_execution"] == "completed"
    if adapter == "langchain":
        assert len(rows) == 4
        assert summary["verdicts"] == {"CONSISTENT": 2, "UNVERIFIABLE": 2}
        assert len(summary["input_batches"]) == 2
        for batch in summary["input_batches"]:
            assert batch["input_tokens"] == 106 and batch["output_tokens"] == 18
            assert batch["input_verdict"] == batch["output_verdict"] == "CONSISTENT"
        # Only the Run with per-input usage contributes to the per-call sums.
        assert summary["by_model"][MODEL]["reported_input_sum"] == 106
    elif adapter == "litellm":
        assert len(rows) == 2
        assert summary["by_model"][MODEL]["cache_hit_calls"] == 1
        cached = next(row for row in rows if row["call"]["cache_hit"])
        assert cached["measurement"]["input_delta"] is None
        assert cached["measurement"]["output_delta"] is None
        assert cached["measurement"]["verdict"] == "UNVERIFIABLE"
        live = next(row for row in rows if not row["call"]["cache_hit"])
        assert live["measurement"]["verdict"] == "CONSISTENT"
        assert len(live["source_refs"]) == 4
    else:
        assert summary["verdicts"] == {"CONSISTENT": 1}


@pytest.mark.assets
@pytest.mark.parametrize("gateway", ["newapi", "oneapi"])
def test_gateway_missing_correlation_warnings_survive_report_with_no_calls(
    tmp_path, capsys, mock_es, official_assets, gateway
):
    code, summary, rows = import_records(
        tmp_path,
        capsys,
        mock_es,
        gateway,
        [{"schema_version": "gateway.exchange.v1", "request_id": None}],
        "log",
        official_assets,
    )
    assert code == 2 and not rows
    warning = next(
        w for w in summary["warnings"] if w["reason"] == "gateway_correlation_id_required"
    )
    assert warning["missing_fields"] == ["request_id"]
    assert "无法可靠关联" in warning["message"] and "重新导入" in warning["action"]


@pytest.mark.assets
@pytest.mark.parametrize("gateway", ["newapi", "oneapi"])
def test_consumption_only_reports_explain_body_requirements(
    tmp_path, capsys, mock_es, official_assets, gateway
):
    code, summary, rows = import_records(
        tmp_path,
        capsys,
        mock_es,
        gateway,
        [{"type": 2, "id": 1, "model_name": MODEL, "prompt_tokens": 53, "completion_tokens": 9}],
        "log",
        official_assets,
    )
    assert code == 0 and rows[0]["measurement"]["verdict"] == "UNVERIFIABLE"
    assert {w["reason"] for w in summary["warnings"]} == {
        "gateway_request_body_required",
        "gateway_response_body_required",
    }
