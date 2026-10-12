"""Complete output through all five log adapters, both sources and saved queries."""

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
from tests.tools.fixtures.generate_reasoning_fixtures import protocol_row

# 45 tokens: pinned Qwen template, THOUGHT/ANSWER and one lookup(z=2,a=中).
# Its complete spelling and native-encoder agreement are checked in test_output.py.
SCENARIOS = [
    ("same", 45, "tool_calls"),
    ("more", 48, "tool_calls"),
    ("less", 42, "tool_calls"),
    ("no-usage", None, "tool_calls"),
    ("no-finish", 45, None),
    ("truncated", 45, "length"),
]


def records(adapter):
    rows = []
    for name, reported, finish in SCENARIOS:
        vector = {
            "asset": "qwen-3.8-2.4t",
            "name": name,
            "messages": [{"role": "user", "content": "查询"}],
            "logged_options": {},
            "tools": None,
            "input_ids": [],
            "output_text": "ANSWER",
            "output_reasoning": "THOUGHT",
        }
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"z":2,"a":"中"}'},
            }
        ]
        usage = {} if reported is None else {"completion_tokens": reported}
        response = {
            "role": "assistant",
            "content": "ANSWER",
            "reasoning_content": "THOUGHT",
            "tool_calls": tool_calls,
        }
        if adapter in {"newapi", "oneapi"}:
            request_id = "output-" + name
            request = {"model": "Qwen/Qwen3.8-2.4T-A95B", "messages": vector["messages"]}
            label = "text request body:" if adapter == "newapi" else "converted request:"
            rows.append(native(adapter, request_id, label, encoded(request)))
            rows.append(
                native(
                    adapter,
                    request_id,
                    "upstream response body:",
                    encoded(
                        {
                            "id": request_id,
                            "choices": [{"message": response, "finish_reason": finish}],
                            "usage": usage,
                        }
                    ),
                )
            )
        else:
            row = protocol_row(vector, adapter, "Qwen/Qwen3.8-2.4T-A95B")
            if adapter == "dify":
                row["outputs"].update(tool_calls=tool_calls, finish_reason=finish, usage=usage)
            elif adapter == "langchain":
                generation = row["outputs"]["generations"][0][0]
                generation["message"]["data"]["additional_kwargs"]["tool_calls"] = tool_calls
                generation["generation_info"] = {"finish_reason": finish}
                row["outputs"]["llm_output"]["token_usage"] = usage
            else:
                row.pop("prompt_tokens")
                row.update(usage)
                row["response"] = {"choices": [{"message": response, "finish_reason": finish}]}
            rows.append(row)
    return rows


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain", "newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_output_verdicts_sums_and_duckdb(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    rows = records(adapter)
    if adapter == "litellm":
        sample = ROOT / "examples/litellm/output-audit.jsonl"
        assert [json.loads(line) for line in sample.read_text().splitlines()] == rows
    fixture = tmp_path / "output.log"
    fixture.write_text(
        "".join(
            row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        )
    )
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
        args += ["--path", str(fixture)]
    else:
        mock_es(
            MockElasticsearch([{"message": row} if isinstance(row, str) else row for row in rows])
        )
        args += ["--path", "https://mock.invalid", "--index", "output-logs"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    summary = result["summary"]
    assert summary["output_execution"] == {"completed": 5, "partial": 1}
    assert summary["output_verdicts"] == {"CONSISTENT": 1, "DEVIATION": 3, "UNVERIFIABLE": 2}
    assert summary["input_verdicts"] == {"UNVERIFIABLE": 6}
    assert summary["verdicts"] == {"DEVIATION": 3, "UNVERIFIABLE": 3}
    group = summary["by_model"]["Qwen/Qwen3.8-2.4T-A95B"]
    assert group["output_compared_calls"] == 4
    assert group["signed_output_delta_sum"] == 1 and group["absolute_output_delta_sum"] == 7
    assert group["recalculated_output_sum"] == 224 and group["output_recalculated"] == 5
    assert group["reported_output_sum"] == 225 and group["output_present"] == 5
    assert group["signed_input_delta_sum"] is None
    bundle = Path(result["run"])
    verify_bundle(bundle)
    saved = details(bundle)
    assert all(row["measurement"]["tool_output_tokens"] == 36 for row in saved)
    assert all(row["measurement"]["output_body_tokens"] == 44 for row in saved)
    assert sum(row["measurement"]["output_tokens"] is None for row in saved) == 1
    with duckdb.connect(str(bundle / "query.duckdb"), read_only=True) as connection:
        assert connection.execute(
            "SELECT verdict, input_verdict, output_verdict, count(*) "
            "FROM calls GROUP BY ALL ORDER BY verdict, output_verdict"
        ).fetchall() == [
            ("DEVIATION", "UNVERIFIABLE", "DEVIATION", 3),
            ("UNVERIFIABLE", "UNVERIFIABLE", "CONSISTENT", 1),
            ("UNVERIFIABLE", "UNVERIFIABLE", "UNVERIFIABLE", 2),
        ]
        assert connection.execute(
            "SELECT output_delta FROM calls WHERE output_verdict='DEVIATION' ORDER BY output_delta"
        ).fetchall() == [("-3",), ("1",), ("3",)]
    assert main(["report", str(bundle), "--details"]) == 0
    assert json.loads(capsys.readouterr().out) == saved


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["dify", "langchain", "newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_request_parameters_and_stops_reach_saved_comparisons(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    native_encoder = Tokenizer.from_file(str(official_assets / "qwen-3.8-2.4t/tokenizer.json"))
    output = "THOUGHT\n</think>\n\nANSWER"
    rows = []
    options = {
        "response_format": {"type": "json_object"},
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stop": ["END"],
    }
    for scenario, ending, stop in [
        ("limit", "length", None),
        ("matched", "stop", "END"),
        ("unknown", "stop", None),
    ]:
        expected = len(native_encoder.encode(output + (stop or ""), add_special_tokens=False).ids)
        vector = {
            "name": scenario,
            "asset": "qwen-3.8-2.4t",
            "messages": [{"role": "user", "content": "查询"}],
            "tools": None,
            "logged_options": options,
            "input_ids": [],
            "output_text": "ANSWER",
            "output_reasoning": "THOUGHT",
        }
        response = {"content": "ANSWER", "reasoning_content": "THOUGHT"}
        terminal = {"finish_reason": ending, **({"stop_reason": stop} if stop else {})}
        usage = {"completion_tokens": expected}
        if adapter in {"newapi", "oneapi"}:
            row = {
                "schema_version": "gateway.exchange.v1",
                "request_id": scenario,
                "request_body": {
                    "model": "Qwen/Qwen3.8-2.4T-A95B",
                    "messages": vector["messages"],
                    **options,
                },
                "response_body": {"choices": [{"message": response, **terminal}], "usage": usage},
            }
        else:
            row = protocol_row(vector, adapter, "Qwen/Qwen3.8-2.4T-A95B")
            if adapter == "dify":
                row["outputs"].update(**terminal, usage=usage)
            elif adapter == "langchain":
                row["outputs"]["generations"][0][0]["generation_info"] = terminal
                row["outputs"]["llm_output"]["token_usage"] = usage
            else:
                row.pop("prompt_tokens")
                row.update(usage)
                row["response"] = {"choices": [{"message": response, **terminal}]}
        rows.append(row)
    logfile = tmp_path / "parameters.jsonl"
    logfile.write_text("".join(json.dumps(row) + "\n" for row in rows))
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
        args += ["--path", str(logfile)]
    else:
        mock_es(MockElasticsearch.from_file(logfile))
        args += ["--path", "https://mock.invalid", "--index", "parameters"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["output_verdicts"] == {"CONSISTENT": 2, "UNVERIFIABLE": 1}
    saved = details(Path(result["run"]))
    for row in saved:
        measured = row["measurement"]
        assert measured["execution"] == "completed", measured["reason"]
        assert measured["render_options"]["response_format"] == options["response_format"]
        assert measured["render_options"]["parallel_tool_calls"] is False
        assert row["call"]["generation_options"]["stop"] == ["END"]
        assert row["call"]["generation_option_sources"]["stop"]
        if measured["output_execution"] == "partial":
            assert measured["output_reason"] == "output_stop_reason_missing"
        else:
            assert measured["output_delta"] == 0
    matched = next(row for row in saved if row["call"]["output_stop_reason"] == "END")
    assert matched["call"]["output_stop_reason_source"]
    assert matched["measurement"]["output_termination"] == "logged_stop_sequence_excluded"
