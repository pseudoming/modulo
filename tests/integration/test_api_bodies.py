"""Raw API bodies through adapters, file/ES, fixed assets and saved results."""

import json
from functools import cache
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.api_bodies import request_body, response_body, sse, stream_events
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.references.build_family_reference import kimi_reference

VECTORS = json.loads(
    (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
)["vectors"]
LOCK = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]


@cache
def kimi_encoder(asset):
    return kimi_reference(ROOT / "assets/tokenizers" / asset)[1]


def expected_input(vector, dialect):
    asset = vector["asset"]
    if dialect != "anthropic" or asset not in {"mimo-v2.6", "kimi-k2.6", "kimi-k2.7-code"}:
        return len(vector["input_ids"])
    # MiMo's pinned template lines 47-53 renders object arguments as parameter
    # tags, unlike string arguments. Expand those tags in the frozen spelling
    # and use the upstream encoder directly; do not use modulo's renderer.
    rendered = vector["rendered"]
    for message in vector["messages"]:
        for call in message.get("tool_calls", []):
            raw = call["function"]["arguments"]
            value = json.loads(raw)
            # K2's official Jinja preserves string arguments verbatim, but renders
            # Anthropic object arguments with its tojson filter (including spaces).
            tags = (
                json.dumps(value, ensure_ascii=False)
                if asset.startswith("kimi-")
                else "".join(f"<parameter={key}>{item}</parameter>" for key, item in value.items())
            )
            rendered = rendered.replace(raw, tags)
    if asset.startswith("kimi-"):
        return len(kimi_encoder(asset).encode(rendered, allowed_special="all"))
    tokenizer = Tokenizer.from_file(str(ROOT / "assets/tokenizers/mimo-v2.6/tokenizer.json"))
    return len(tokenizer.encode(rendered, add_special_tokens=False).ids)


def record(adapter, dialect, request, response, identity, streamed=False):
    if adapter == "litellm":
        return {
            "request_id": identity,
            "call_type": "aresponses" if dialect == "responses" else "anthropic_messages",
            "model": request["model"],
            "proxy_server_request": request,
            "response": response,
            "stream": streamed,
        }
    return {
        "schema_version": "gateway.exchange.v1",
        "request_id": identity,
        "request_body": request,
        **(
            {"response_sse": sse(stream_events(dialect, response))}
            if streamed
            else {"response_body": response}
        ),
    }


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_multiple_messages_and_token_limits_survive_file_es_and_streams(
    tmp_path, official_assets, capsys, mock_es, adapter, source
):
    model = LOCK["qwen-3.8-2.4t"]["repo"]
    request = request_body(
        "responses",
        [{"role": "user", "content": "查询"}],
        model=model,
        options={"text": {"format": {"type": "json_object"}}, "max_output_tokens": 80},
    )
    first = response_body("responses", "FIRST", "THOUGHT-1", model=model)
    second = response_body("responses", "SECOND", "THOUGHT-2", model=model)
    for item in second["output"]:
        item["id"] += "-second"
    # Expand both fixed Qwen messages explicitly; the first generation prefix
    # is input, while the second assistant header is generated output.
    prefix = (
        "THOUGHT-1\n</think>\n\nFIRST<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nTHOUGHT-2\n</think>\n\nSECOND"
    )
    encoder = Tokenizer.from_file(str(official_assets / "qwen-3.8-2.4t/tokenizer.json"))
    rows = []
    for limited in (False, True):
        expected = len(
            encoder.encode(prefix + ("" if limited else "<|im_end|>"), add_special_tokens=False).ids
        )
        response = {
            **first,
            "output": [*first["output"], *second["output"]],
            "usage": {"input_tokens": 53, "output_tokens": expected},
        }
        if limited:
            response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        rows.append(
            record(
                adapter, "responses", request, response, "multi-" + str(limited), streamed=limited
            )
        )
    logfile = tmp_path / "multi.jsonl"
    logfile.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = [
        "import",
        "--adapter",
        adapter,
        "--source",
        source,
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
        args += ["--path", "https://mock.invalid", "--index", "multi-messages"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["verdicts"] == {"CONSISTENT": 2}
    for detail in details(Path(result["run"])):
        call, count = detail["call"], detail["measurement"]
        assert len(call["output_messages"]) == 2
        assert not call["output_candidates"]
        assert count["output_delta"] == 0
        assert call["generation_options"]["max_output_tokens"] == 80


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
@pytest.mark.parametrize("source", ["log", "es"])
def test_frozen_history_and_tool_inputs_all_models(
    tmp_path, official_assets, capsys, mock_es, adapter, dialect, source
):
    vectors = [
        v
        for v in VECTORS
        if v["name"] in {"history", "tool_continuation", "parallel_tools", "tool_history_then_user"}
    ]
    rows = []
    for i, vector in enumerate(vectors):
        request = request_body(
            dialect,
            vector["messages"],
            tools=vector["tools"],
            options=vector["logged_options"],
            model=LOCK[vector["asset"]]["repo"],
        )
        response = response_body(
            dialect,
            vector["output_text"],
            vector["output_reasoning"],
            input_tokens=expected_input(vector, dialect),
        )
        rows.append(record(adapter, dialect, request, response, str(i), streamed=i % 2 == 1))
    logfile = tmp_path / "bodies.jsonl"
    logfile.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    args = [
        "import",
        "--adapter",
        adapter,
        "--source",
        source,
        "--bindings",
        str(ROOT / "examples/newapi/request-response.bindings.yaml"),
        "--assets",
        str(official_assets),
        "--output",
        str(tmp_path / "runs"),
    ]
    if source == "log":
        args += ["--path", str(logfile)]
    else:
        mock_es(MockElasticsearch.from_file(logfile))
        args += ["--path", "https://mock.invalid", "--index", "body-logs", "--page-size", "5"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["counters"]["invalid_records"] == 0
    assert result["summary"]["input_verdicts"] == {"CONSISTENT": len(rows)}
    bundle = Path(result["run"])
    verify_bundle(bundle)
    actual = details(bundle)
    for row in actual:
        identity = (
            row["call"]["call_id"] if adapter == "litellm" else row["assemblies"][0]["request_id"]
        )
        vector = vectors[int(identity)]
        measurement = row["measurement"]
        assert measurement["input_tokens"] == expected_input(vector, dialect)
        assert measurement["reasoning_output_tokens"] == vector["reasoning_output_tokens"]
        assert measurement["visible_output_tokens"] == vector["visible_output_tokens"]
        assert measurement["output_execution"] == "completed"
        assert row["call"]["output_text"] == vector["output_text"]
        assert not row["call"]["output_candidates"]
    if source == "log":
        assert main(args) == 0
        repeated = json.loads(capsys.readouterr().out)
        assert repeated["reused"] and repeated["run"] == result["run"]


@pytest.mark.assets
@pytest.mark.parametrize("adapter", ["newapi", "oneapi", "litellm"])
@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
def test_full_tool_output_and_signed_difference(
    tmp_path, official_assets, capsys, adapter, dialect
):
    # Independent pinned Qwen spelling in test_output: THOUGHT + ANSWER +
    # lookup(z=2,a=中) + the tool terminator is exactly 45 output tokens.
    calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"z":2,"a":"中"}'},
        }
    ]
    request = request_body(
        dialect, [{"role": "user", "content": "查询"}], model=LOCK["qwen-3.8-2.4t"]["repo"]
    )
    response = response_body(
        dialect, "ANSWER", "THOUGHT", calls=calls, input_tokens=53, output_tokens=48
    )
    row = record(adapter, dialect, request, response, "tool-output", streamed=True)
    path = tmp_path / "body.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert (
        main(
            [
                "import",
                "--source",
                "log",
                "--adapter",
                adapter,
                "--path",
                str(path),
                "--bindings",
                str(ROOT / "examples/newapi/request-response.bindings.yaml"),
                "--assets",
                str(official_assets),
                "--output",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    row = details(Path(result["run"]))[0]
    assert row["measurement"]["input_tokens"] == 53
    assert row["measurement"]["output_tokens"] == 45
    assert row["measurement"]["output_delta"] == 3
    assert row["measurement"]["output_verdict"] == "DEVIATION"
