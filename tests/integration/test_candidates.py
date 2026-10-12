"""Block messages and candidate outputs through CLI, sources and stored reports."""

import copy
import json
from pathlib import Path

import duckdb
import pytest
from tokenizers import Tokenizer

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.fixtures.generate_gateway_body_fixtures import encoded, native

MODEL = "Qwen/Qwen3.8-2.4T-A95B"
# Independently expanded pinned Qwen formats; no modulo renderer used for answers.
INPUT = (
    "<|im_start|>system\nReasoning effort is set to xhigh. "
    "Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize correctness, "
    "consistency, and clarity in the final answer.<|im_end|>\n"
    "<|im_start|>user\n查询<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
OUTPUT = "THOUGHT\n</think>\n\nANSWER<|im_end|>"


def record(adapter, scenario="complete", *, multiple=True, streamed=False):
    count = 2 if multiple else 1
    request = {
        "model": MODEL,
        "n": count,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "查询"}]}],
    }
    blocks = [
        {"type": "thinking", "thinking": "THOUGHT", "signature": "not-a-token"},
        {"type": "text", "text": "ANSWER"},
    ]
    choices = [
        {
            "index": index,
            "message": {"role": "assistant", "content": copy.deepcopy(blocks)},
            "finish_reason": "stop",
        }
        for index in range(count)
    ]
    if scenario == "truncated":
        choices[-1]["finish_reason"] = "length"
    if scenario == "missing-candidate":
        request["n"] = count + 1
    if scenario == "summary":
        choices[-1]["message"]["content"][0] = {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "THOUGHT"}],
        }
    usage = {"prompt_tokens": 53, "completion_tokens": count * 9 + (3 if scenario == "more" else 0)}
    response = {"id": "blocks-" + scenario, "choices": choices, "usage": usage}
    if adapter in {"newapi", "oneapi"}:
        request_label = "text request body:" if adapter == "newapi" else "converted request:"
        rows = [native(adapter, "blocks-" + scenario, request_label, encoded(request))]
        if streamed:
            stream_label = (
                "stream scanner data:" if adapter == "newapi" else "stream response data:"
            )
            # Candidate 1 can arrive before candidate 0; each contains blocks.
            events = [
                {
                    "id": "blocks-" + scenario,
                    "choices": [
                        {
                            "index": choice["index"],
                            "delta": choice["message"],
                            "finish_reason": choice["finish_reason"],
                        }
                    ],
                }
                for choice in reversed(choices)
            ]
            events.append({"id": "blocks-" + scenario, "choices": [], "usage": usage})
            for event in events:
                rows.append(
                    native(
                        adapter,
                        "blocks-" + scenario,
                        stream_label,
                        "data: " + encoded(event),
                    )
                )
                rows.append(native(adapter, "blocks-" + scenario, stream_label, ""))
            rows.append(native(adapter, "blocks-" + scenario, stream_label, "data: [DONE]"))
        else:
            rows.append(
                native(adapter, "blocks-" + scenario, "upstream response body:", encoded(response))
            )
        return rows
    if adapter == "litellm":
        return [
            {
                "id": "blocks-" + scenario,
                "call_type": "completion",
                "model": MODEL,
                "model_parameters": {"n": request["n"]},
                "messages": request["messages"],
                "response": response,
                **usage,
            }
        ]
    if adapter == "langchain":
        # ChatOpenAI repeats request usage on all candidates, and llm_output.
        generations = [
            {
                "message": {
                    "type": "ai",
                    "data": {
                        **choice["message"],
                        "usage_metadata": {
                            "input_tokens": 53,
                            "output_tokens": usage["completion_tokens"],
                        },
                    },
                },
                "generation_info": {"finish_reason": choice["finish_reason"]},
            }
            for choice in choices
        ]
        return [
            {
                "id": "blocks-" + scenario,
                "run_type": "llm",
                "inputs": {"messages": [[{"type": "human", "data": request["messages"][0]}]]},
                "extra": {"invocation_params": {"model": MODEL, "n": request["n"]}},
                "outputs": {"generations": [generations], "llm_output": {"token_usage": usage}},
            }
        ]
    assert adapter == "dify" and not multiple
    return [
        {
            "id": "blocks-" + scenario,
            "node_type": "llm",
            "process_data": {
                "model_name": MODEL,
                "model_mode": "chat",
                "model_parameters": {"n": request["n"]},
                "prompts": [{"role": "user", "text": request["messages"][0]["content"]}],
            },
            "outputs": {
                "text": choices[0]["message"]["content"],
                "finish_reason": choices[0]["finish_reason"],
                "usage": usage,
            },
        }
    ]


def run_import(tmp_path, capsys, mock_es, rows, adapter, source, official_assets):
    args = [
        "import",
        "--source",
        source,
        "--adapter",
        adapter,
        "--bindings",
        str(ROOT / "examples/litellm/basic.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        path = tmp_path / "records.log"
        path.write_text(
            "".join(
                row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            )
        )
        args += ["--path", str(path)]
    else:
        docs = [
            {"message": row, "event": {"sequence": i}} if isinstance(row, str) else row
            for i, row in enumerate(rows)
        ]
        mock_es(MockElasticsearch(docs))
        args += ["--path", "https://mock.invalid", "--index", "block-logs"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    bundle = Path(result["run"])
    verify_bundle(bundle)
    return result["summary"], bundle, details(bundle)


@pytest.mark.assets
@pytest.mark.parametrize(
    "adapter,streamed",
    [
        ("langchain", False),
        ("litellm", False),
        ("newapi", False),
        ("oneapi", False),
        ("newapi", True),
        ("oneapi", True),
    ],
)
@pytest.mark.parametrize("source", ["log", "es"])
def test_multi_candidate_totals_are_not_duplicated(
    tmp_path, capsys, mock_es, official_assets, adapter, streamed, source
):
    native_encoder = Tokenizer.from_file(str(official_assets / "qwen-3.8-2.4t/tokenizer.json"))
    assert len(native_encoder.encode(INPUT, add_special_tokens=False).ids) == 53
    assert len(native_encoder.encode(OUTPUT, add_special_tokens=False).ids) == 9
    scenarios = ["complete", "more", "truncated", "missing-candidate", "summary"]
    rows = [row for scenario in scenarios for row in record(adapter, scenario, streamed=streamed)]
    summary, bundle, saved = run_import(
        tmp_path, capsys, mock_es, rows, adapter, source, official_assets
    )
    assert len(saved) == 5
    assert summary["input_verdicts"] == {"CONSISTENT": 5}
    assert summary["output_verdicts"] == {"CONSISTENT": 1, "DEVIATION": 2, "UNVERIFIABLE": 2}
    assert summary["verdicts"] == {"CONSISTENT": 1, "DEVIATION": 2, "UNVERIFIABLE": 2}
    group = summary["by_model"][MODEL]
    assert group["reported_input_sum"] == group["recalculated_input_sum"] == 265
    assert group["reported_output_sum"] == 93
    assert group["recalculated_output_sum"] == 53 and group["output_compared_calls"] == 3
    assert group["signed_output_delta_sum"] == 4
    for row in saved:
        call, count = row["call"], row["measurement"]
        assert call["output_text"] is None
        assert len(call["output_candidates"]) == len(count["output_candidates"]) == 2
        assert count["output_candidates"][0]["tokens"] == 9
        assert call["output_candidates"][0]["output_structured_content_json"]
    with duckdb.connect(str(bundle / "query.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM calls").fetchone() == (5,)
        assert connection.execute("SELECT COUNT(*) FROM output_candidates").fetchone() == (10,)
        assert connection.execute(
            "SELECT COUNT(*) FROM output_candidates WHERE execution='completed'"
        ).fetchone() == (9,)


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm", "newapi", "oneapi"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_single_structured_response_all_log_adapters(
    tmp_path, capsys, mock_es, official_assets, adapter, source
):
    rows = record(adapter, multiple=False)
    summary, _, saved = run_import(
        tmp_path, capsys, mock_es, rows, adapter, source, official_assets
    )
    assert summary["input_verdicts"] == {"CONSISTENT": 1}
    assert summary["output_verdicts"] == {"CONSISTENT": 1}
    assert summary["verdicts"] == {"CONSISTENT": 1}
    assert saved[0]["measurement"]["output_tokens"] == 9
    assert saved[0]["call"]["output_reasoning"] == "THOUGHT"
    assert saved[0]["call"]["output_structured_content_json"]


def test_public_example_matches_the_validated_synthetic_cases():
    rows = [
        json.loads(line)
        for line in (ROOT / "examples/litellm/content-blocks.jsonl").read_text().splitlines()
    ]
    assert rows == [
        record("litellm", scenario)[0]
        for scenario in ("complete", "more", "truncated", "missing-candidate", "summary")
    ]
