import copy
import json

import pytest

from modulo.adapters.protocols import NewAPIProtocol, OneAPIProtocol
from modulo.adapters.protocols.shared.api_bodies.chat_stream import assemble_sse
from modulo.core.domain.data_models import RawRecord, SourceError
from modulo.core.domain.interfaces import SourceAdapter
from modulo.core.domain.platforms import PlatformConfig
from modulo.core.processing.ingestion import collect_calls
from tests.support.api_bodies import request_body, response_body, stream_events
from tests.support.api_bodies import sse as api_sse

REQUEST = {"model": "volc-deepseek", "messages": [{"role": "user", "content": "  中文🙂\n"}]}
RESPONSE = {
    "model": "deepseek-server",
    "choices": [
        {"message": {"role": "assistant", "content": "结果", "reasoning_content": "想一想"}}
    ],
    "usage": {"prompt_tokens": 17, "completion_tokens": 3},
}


def native(gateway, label, body, request_id="r1"):
    prefix = f"[DEBUG] 2026/10/10 - 12:00:00 | {request_id} | "
    if gateway == "oneapi":
        prefix += "relay/controller/text.go:112 [getRequestBody] "
    return prefix + label + " " + (body if isinstance(body, str) else json.dumps(body)) + " \n"


class MemoryLogs(SourceAdapter):
    name, version = "memory", "synthetic.v1"

    def __init__(self, rows, ordered=True, fail=False):
        self.rows, self.ordered_records, self.fail = rows, ordered, fail

    def describe(self):
        return {"kind": "synthetic"}

    def read(self):
        for i, row in enumerate(self.rows):
            yield RawRecord(str(i), (row if isinstance(row, str) else json.dumps(row)).encode())
        if self.fail:
            raise SourceError("injected")


def collect(rows, gateway="newapi", *, ordered=True, fail=False, payload_field=None, platform=None):
    return collect_calls(
        MemoryLogs(rows, ordered, fail),
        NewAPIProtocol() if gateway == "newapi" else OneAPIProtocol(),
        lambda raw: {"locator": raw.locator},
        namespace="synthetic",
        payload_field=payload_field,
        max_records=1000,
        platform_config=platform or PlatformConfig(),
    )


def single(batch):
    assert not [d for d in batch.diagnostics if d["level"] != "warning"]
    assert len(batch.calls) == 1
    return next(iter(batch.calls.values()))


def sse(*deltas, finish=True, done=True, usage=None):
    events = [
        {
            "id": "s1",
            "model": "volc-deepseek",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    if finish:
        events.append({"id": "s1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage is not None:
        events.append({"id": "s1", "choices": [], "usage": usage})
    lines = [part for event in events for part in ("data: " + json.dumps(event), "")]
    return lines + (["data: [DONE]", ""] if done else [])


@pytest.mark.parametrize("gateway", ["newapi", "oneapi"])
def test_native_interleaving_association_is_independent_of_model_and_usage(gateway):
    label = "text request body:" if gateway == "newapi" else "request body:"
    rows = [
        native(gateway, label, REQUEST, "a"),
        native(gateway, label, REQUEST, "b"),
        native(gateway, "upstream response body:", RESPONSE, "b"),
        native(gateway, "upstream response body:", RESPONSE, "a"),
    ]
    batch = collect(rows, gateway)
    assert batch.complete and len(batch.calls) == 2
    before = {r["assemblies"][0]["request_id"]: r for r in batch.calls.values()}
    assert [s["locator"] for s in before["a"]["source_refs"]] == ["0", "3"]
    assert [s["locator"] for s in before["b"]["source_refs"]] == ["1", "2"]
    changed = copy.deepcopy(RESPONSE)
    changed["usage"]["prompt_tokens"] = 999
    changed["model"] = "unrelated-model"
    rows[3] = native(gateway, "upstream response body:", changed, "a")
    after = {r["assemblies"][0]["request_id"]: r for r in collect(rows, gateway).calls.values()}
    for key in before:
        assert before[key]["call_key"] == after[key]["call_key"]
        assert before[key]["source_refs"] == after[key]["source_refs"]
    assert after["a"]["call"]["model"] == "volc-deepseek"
    assert after["a"]["call"]["usage"]["input_tokens"] == 999
    assert after["a"]["call"]["messages"][0]["content"] == "  中文🙂\n"


@pytest.mark.parametrize("gateway", ["newapi", "oneapi"])
@pytest.mark.parametrize("dialect", ["responses", "anthropic"])
@pytest.mark.parametrize("streamed", [False, True])
def test_api_bodies_in_native_wrappers_and_unordered_es(gateway, dialect, streamed):
    request = request_body(
        dialect, [{"role": "system", "content": "系统"}, {"role": "user", "content": "你好"}]
    )
    response = response_body(dialect, "回答", "思考", input_tokens=10, output_tokens=8)
    rows = [native(gateway, "requestBody:" if gateway == "newapi" else "request body:", request)]
    if streamed:
        label = "stream scanner data:" if gateway == "newapi" else "stream response data:"
        rows.extend(
            native(gateway, label, line)
            for line in api_sse(stream_events(dialect, response)).splitlines()
        )
    else:
        label = "responseBody:" if gateway == "newapi" else "upstream response body:"
        rows.append(native(gateway, label, response))
    wrapped = [{"message": row, "event": {"sequence": i}} for i, row in enumerate(rows)]
    result = single(collect(list(reversed(wrapped)), gateway, ordered=False))
    assert not result["call"]["gaps"] and not result["call"]["output_gaps"]
    assert result["call"]["output_text"] == "回答" and result["call"]["output_reasoning"] == "思考"
    assert result["call"]["usage"]["input_tokens"] == 10
    assert len(result["source_refs"]) == len(rows)
    if streamed:
        assert result["assemblies"][0]["sequence_basis"] == "event.sequence"


def test_oneapi_multiline_prefers_actual_converted_request_and_ali_response():
    converted = {
        "model": "qwen",
        "input": {"messages": REQUEST["messages"]},
        "parameters": {"enable_thinking": True},
    }
    rows = [
        native("oneapi", "request body:", REQUEST),
        native("oneapi", "converted request:", "\n" + json.dumps(converted)),
        native(
            "oneapi",
            "response body:",
            {"output": {"text": "结果"}, "usage": {"input_tokens": 17, "output_tokens": 3}},
        ),
    ]
    row = single(collect("".join(rows).splitlines(), "oneapi"))
    assert row["call"]["model"] == "qwen"
    assert row["call"]["render_options"] == {"enable_thinking": True}
    assert row["call"]["output_text"] == "结果"
    assert row["call"]["usage"]["input_tokens"] == 17
    assert len(row["source_refs"]) == 4
    assert row["assemblies"][0]["request_stage"] == "upstream"


def test_ollama_requires_correlation_and_retains_native_usage():
    body = {
        "message": RESPONSE["choices"][0]["message"],
        "prompt_eval_count": 17,
        "eval_count": 3,
        "done": True,
    }
    row = single(
        collect(
            [
                native("oneapi", "request body:", REQUEST),
                native("oneapi", "ollama response:", body),
            ],
            "oneapi",
        )
    )
    assert row["call"]["output_reasoning"] == "想一想"
    assert row["call"]["usage"] == {
        "input_tokens": 17,
        "output_tokens": 3,
        "total_tokens": None,
        "provenance": "source_reported",
    }
    missing = native("oneapi", "ollama response:", body).replace(" | r1", "")
    batch = collect([missing], "oneapi")
    assert not batch.calls and batch.diagnostics[0]["reason"] == "gateway_request_id_missing"


def test_sse_reasoning_repeated_text_and_interleaved_tool_arguments():
    first = {
        "index": 0,
        "id": "tool-a",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"city":"'},
    }
    second = {
        "index": 1,
        "id": "tool-b",
        "type": "function",
        "function": {"name": "calc", "arguments": '{"n":'},
    }
    lines = sse(
        {"reasoning_content": "先"},
        {"thinking": "想"},
        {"content": "哈"},
        {"content": "哈"},
        {"tool_calls": [first, second]},
        {
            "tool_calls": [
                {"index": 1, "function": {"arguments": "1.20}"}},
                {"index": 0, "function": {"arguments": '杭州"}'}},
            ]
        },
        usage={"prompt_tokens": 17, "completion_tokens": 3},
    )
    # An earlier snapshot is replaced, never summed into the terminal snapshot.
    lines = ['data: {"id":"s1","choices":[],"usage":{"prompt_tokens":10}}', "", *lines]
    result = assemble_sse(lines)
    assert result.output.get("output_gaps", []) == []
    assert result.output["output_reasoning"] == "先想"
    assert result.output["output_text"] == "哈哈"
    calls = json.loads(result.output["output_tool_calls_json"])
    assert [c["function"]["arguments"] for c in calls] == ['{"city":"杭州"}', '{"n":1.20}']
    assert result.usage.input_tokens == 17 and result.usage.output_tokens == 3


@pytest.mark.parametrize(
    "lines, reason",
    [
        (sse({"content": "partial"}, done=False), "stream_done_missing"),
        (sse({"content": "partial"}, finish=False), "stream_finish_missing"),
        (["data: broken", "", *sse({"content": "x"})], "invalid_stream_json"),
        (sse({"reasoning": "a", "reasoning_content": "b"}), "conflicting_stream_reasoning"),
        (
            sse(
                {
                    "tool_calls": [
                        {"index": 0, "id": "a", "function": {"name": "f", "arguments": "{"}}
                    ]
                }
            ),
            "invalid_stream_tool_arguments",
        ),
        ([*sse({"content": "a"}), *sse({"content": "b"})], "data_after_stream_done"),
        (
            ['data: {"id":"other","choices":[]}', "", *sse({"content": "x"})],
            "conflicting_stream_id",
        ),
        (sse({"role": []}), "unsupported_stream_role"),
    ],
)
def test_broken_stream_is_not_completed_output(lines, reason):
    assert reason in assemble_sse(lines).output["output_gaps"]


@pytest.mark.parametrize("payload_field", [None, "message"])
def test_es_order_uses_collector_sequence_and_request_metadata(payload_field):
    rows = [
        {
            "message": native("newapi", "text request body:", REQUEST),
            "metadata": {"platform": "volc"},
            "event": {"sequence": 0},
        }
    ]
    rows += [
        {
            "message": native("newapi", "stream scanner data:", line),
            "event": {"sequence": i + 1},
            "metadata": {"platform": "wrong-response-tag"},
        }
        for i, line in enumerate(
            sse({"reasoning_content": "想"}, {"content": "答"}, usage={"prompt_tokens": 17})
        )
    ]
    config = PlatformConfig.model_validate({"platform-tag-source": "metadata:platform"})
    row = single(
        collect(list(reversed(rows)), ordered=False, platform=config, payload_field=payload_field)
    )
    assert row["call"]["output_text"] == "答" and row["call"]["output_reasoning"] == "想"
    assert row["call"]["platform"] == "volc"
    root = "record" if payload_field else "payload"
    assert row["platform_tags"][0]["field_path"] == f"{root}:metadata.platform"
    assert row["assemblies"][0]["sequence_basis"] == "event.sequence"
    for item in rows:
        del item["event"]
    row = single(collect(rows, ordered=False))
    assert row["call"]["output_gaps"] == ["stream_order_unknown"]
    assert row["call"]["usage"]["input_tokens"] is None
    assert row["call"]["messages"] is not None


def test_duplicate_events_require_identity_and_conflicts_block_comparison():
    request = {"message": native("newapi", "text request body:", REQUEST), "event_id": "a"}
    response = {"message": native("newapi", "upstream response body:", RESPONSE), "event_id": "b"}
    row = single(collect([request, request, response]))
    assert row["assemblies"][0]["duplicate_events"] == 1 and len(row["source_refs"]) == 3
    assert row["call"]["gaps"] == []
    changed = {
        **request,
        "message": native("newapi", "text request body:", {**REQUEST, "model": "other"}),
    }
    row = single(collect([request, changed, response]))
    assert "conflicting_gateway_event_identity" in row["call"]["gaps"]
    assert row["call"]["usage"]["input_tokens"] is None
    row = single(collect([request["message"], request["message"], response["message"]]))
    assert "ambiguous_gateway_attempts" in row["call"]["gaps"]
    assert row["call"]["usage"]["input_tokens"] is None
    assert (
        len(
            collect([{**e, "attempt_id": a} for a in ("1", "2") for e in (request, response)]).calls
        )
        == 2
    )


def test_missing_response_still_retains_input_and_partial_source_evidence():
    rows = [native("newapi", "text request body:", REQUEST)]
    row = single(collect(rows))
    assert row["call"]["messages"] and row["call"]["gaps"] == []
    assert row["call"]["output_gaps"] == ["missing_response_body"]
    partial = collect(rows, fail=True)
    assert not partial.complete and len(partial.calls) == 1
    assert partial.diagnostics[0]["reason"] == "source_incomplete"


def test_incomplete_multiline_and_bad_record_preserve_all_diagnostics():
    batch = collect(
        [
            native("oneapi", "converted request:", "{"),
            {"schema_version": "gateway.exchange.v1", "request_id": None},
            native("oneapi", "request body:", REQUEST),
        ],
        "oneapi",
    )
    assert batch.counters["invalid_records"] == 2 and len(batch.calls) == 1
    assert {d["source_ref"]["locator"] for d in batch.diagnostics if d["level"] == "error"} == {
        "0",
        "1",
    }


@pytest.mark.parametrize("gateway", ["newapi", "oneapi"])
def test_paired_exchange_with_complete_and_truncated_stream(gateway):
    record = {
        "schema_version": "gateway.exchange.v1",
        "gateway": gateway,
        "request_id": "r",
        "request_body": REQUEST,
        "request_stage": "upstream",
        "status_code": 200,
        "response_sse": "\n".join(sse({"reasoning_content": "想"}, {"content": "答"})),
    }
    row = single(collect([{"payload": record}], gateway, payload_field="payload"))
    assert row["call"]["status"] == "success" and row["call"]["output_reasoning"] == "想"
    record["response_complete"] = False
    row = single(collect([record], gateway))
    assert "response_capture_incomplete" in row["call"]["output_gaps"]
    assert row["call"]["status"] == "unknown"


def test_orphan_response_keeps_output_and_does_not_invent_input():
    row = single(collect([native("newapi", "upstream response body:", RESPONSE)]))
    assert row["call"]["messages"] is None
    assert row["call"]["gaps"] == ["missing_messages"]
    assert row["call"]["output_reasoning"] == "想一想"
    assert row["assemblies"][0]["request_stage"] == "missing"


def test_exchange_done_marker_does_not_hide_invalid_content():
    record = {
        "schema_version": "gateway.exchange.v1",
        "request_id": "r",
        "request_body": REQUEST,
        "response_sse": "\n".join(sse({"content": {"invalid": "object"}})),
    }
    row = single(collect([record]))
    assert "unsupported_stream_content" in row["call"]["output_gaps"]
    assert not row["assemblies"][0]["response_complete"]


def test_unknown_json_is_invalid_while_unrelated_native_log_is_ignored():
    batch = collect(
        [
            '{"type":2,"type":5}',
            native("newapi", "server info:", "hello"),
            native("newapi", "text request body:", REQUEST),
        ]
    )
    assert batch.counters["invalid_records"] == 1
    assert batch.counters["ignored_records"] == 1
    assert len(batch.calls) == 1


def test_same_file_offset_can_deduplicate_but_different_files_cannot_order_a_stream():
    rows = [
        {
            "message": native("newapi", "text request body:", REQUEST),
            "log": {"offset": 0, "file": {"path": "a"}},
        }
    ]
    rows += [
        {
            "message": native("newapi", "stream scanner data:", line),
            "log": {"offset": 100 + i, "file": {"path": "a"}},
        }
        for i, line in enumerate(sse({"content": "答"}))
    ]
    row = single(collect([*reversed(rows), rows[-1]], ordered=False))
    assert row["call"]["output_text"] == "答" and row["call"]["output_gaps"] == []
    assert row["assemblies"][0]["duplicate_events"] == 1
    rows[-1]["log"]["file"]["path"] = "b"
    row = single(collect(rows, ordered=False))
    assert row["call"]["output_gaps"] == ["stream_order_unknown"]


@pytest.mark.parametrize("gateway", ["newapi", "oneapi"])
@pytest.mark.parametrize(
    "missing", ["request", "response", "request_id", "attempt_id", "order", "end"]
)
def test_report_warnings_explain_observed_missing_evidence(gateway, missing):
    record = {
        "schema_version": "gateway.exchange.v1",
        "gateway": gateway,
        "request_id": "r",
        "request_body": REQUEST,
        "response_body": RESPONSE,
    }
    ordered = True
    if missing == "request":
        record.pop("request_body")
        reason = "gateway_request_body_required"
    elif missing == "response":
        record.pop("response_body")
        reason = "gateway_response_body_required"
    elif missing == "request_id":
        record.pop("request_id")
        reason = "gateway_correlation_id_required"
    elif missing == "attempt_id":
        label = "text request body:" if gateway == "newapi" else "converted request:"
        rows = [native(gateway, label, REQUEST), native(gateway, label, REQUEST)]
        reason = "gateway_attempt_id_required"
    else:
        label = "text request body:" if gateway == "newapi" else "converted request:"
        stream_label = "stream scanner data:" if gateway == "newapi" else "stream response data:"
        events = sse({"content": "answer"})
        if missing == "end":
            events = [line for line in events if "[DONE]" not in line]
        ordered = missing != "order"
        rows = [
            native(gateway, label, REQUEST),
            *[native(gateway, stream_label, line) for line in events],
        ]
        reason = (
            "gateway_stream_order_required" if missing == "order" else "gateway_stream_end_required"
        )
    batch = collect(
        rows if missing in {"attempt_id", "order", "end"} else [record], gateway, ordered=ordered
    )
    warning = next(d for d in batch.diagnostics if d["reason"] == reason)
    assert warning["level"] == "warning"
    assert warning["missing_fields"] and warning["message"] and warning["action"]
