"""Gateway body examples from frozen references; no model or tokenizer execution."""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stream_lines(response):
    message = response["choices"][0]["message"]
    deltas = [{"role": "assistant", "content": ""}]
    for field in ("reasoning_content", "content"):
        value = message[field]
        middle = len(value) // 2
        deltas.extend([{field: value[:middle]}, {field: value[middle:]}])
    chunks = [
        {
            "id": response["id"],
            "model": response["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    chunks.append(
        {
            "id": response["id"],
            "model": response["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    chunks.append(
        {
            "id": response["id"],
            "model": response["model"],
            "choices": [],
            "usage": response["usage"],
        }
    )
    return list(
        itertools.chain.from_iterable(
            [("data: " + encoded(chunk), "") for chunk in chunks] + [("data: [DONE]", "")]
        )
    )


def native(gateway, request_id, label, body):
    locations = {
        "request body:": "controller/relay.go:49 [Relay] ",
        "converted request:": "relay/controller/text.go:112 [getRequestBody] ",
        "upstream response body:": "relay/adaptor/openai/main.go:106 [Handler] ",
        "stream response data:": "relay/adaptor/openai/main.go:38 [StreamHandler] ",
    }
    suffix = "" if gateway == "newapi" else locations[label]
    return f"[DEBUG] 2026/10/10 - 12:00:00 | {request_id} | {suffix}{label} {body} \n"


def main():
    source = ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json"
    vectors = json.loads(source.read_text())["vectors"]
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    bindings = {
        "models": [{"name": asset, "aliases": [entry["repo"]]} for asset, entry in assets.items()],
    }
    counts = {}
    for gateway in ("newapi", "oneapi"):
        directory = ROOT / "examples" / gateway
        directory.mkdir(parents=True, exist_ok=True)
        groups, exchanges = [], []
        for index, vector in enumerate(vectors):
            request_id = f"synthetic-{vector['asset']}-{vector['name']}"
            model = assets[vector["asset"]]["repo"]
            request = {
                "model": model,
                "messages": copy.deepcopy(vector["messages"]),
                **copy.deepcopy(vector["logged_options"]),
            }
            if vector["tools"]:
                request["tools"] = vector["tools"]
            response = {
                "id": "response-" + request_id,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": vector["output_text"],
                            "reasoning_content": vector["output_reasoning"],
                        },
                    }
                ],
                "usage": {"prompt_tokens": len(vector["input_ids"])},
            }
            streaming = index % 2 == 1
            request["stream"] = streaming
            events = []
            if gateway == "oneapi":
                events.append(native(gateway, request_id, "request body:", encoded(request)))
                # True native formatting: the JSON starts on a second physical line.
                events.append(
                    native(gateway, request_id, "converted request:", "\n" + encoded(request))
                )
            else:
                events.append(native(gateway, request_id, "text request body:", encoded(request)))
            exchange = {
                "schema_version": "gateway.exchange.v1",
                "gateway": gateway,
                "request_id": request_id,
                "attempt_id": "synthetic-attempt-1",
                "request_stage": "upstream",
                "request_body": request,
                "status_code": 200,
                "response_complete": True,
            }
            if streaming:
                lines = stream_lines(response)
                label = "stream scanner data:" if gateway == "newapi" else "stream response data:"
                events.extend(native(gateway, request_id, label, line) for line in lines)
                exchange["response_sse"] = "\n".join(lines) + "\n"
            else:
                events.append(
                    native(gateway, request_id, "upstream response body:", encoded(response))
                )
                exchange["response_body"] = response
            groups.append(events)
            exchanges.append(exchange)
        # Interleave requests while keeping each individual log statement atomic.
        events = [e for row in itertools.zip_longest(*groups) for e in row if e is not None]
        (directory / "request-response.log").write_text("".join(events))
        wrappers = [
            {
                "message": event.rstrip("\n"),
                "event": {"sequence": i},
                "metadata": {"platform": "synthetic-gateway"},
            }
            for i, event in enumerate(events)
        ]
        # ES document retrieval order deliberately differs from stream order.
        (directory / "request-response-events.jsonl").write_text(
            "".join(encoded(row) + "\n" for row in reversed(wrappers))
        )
        (directory / "request-response-exchanges.jsonl").write_text(
            "".join(encoded(row) + "\n" for row in exchanges)
        )
        counts[gateway] = {
            "calls": len(vectors),
            "events": len(events),
            "physical_lines": sum(len(e.splitlines()) for e in events),
            "stream_calls": len(vectors) // 2,
        }
        (directory / "request-response.bindings.yaml").write_text(
            yaml.safe_dump(bindings, allow_unicode=True, sort_keys=False)
        )
    provenance = {
        "synthetic": True,
        "generator": "tests/tools/fixtures/generate_gateway_body_fixtures.py",
        "reference": str(source.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "models": len(assets),
        "counts": counts,
        "newapi_profile": "native-debug@1d4328e97417a043a161a0dd30a5b129be3ace49",
        "oneapi_profile": (
            "native-debug@8df4a2670b98266bd287c698243fff327d9748cf"
            " + historical-response-log-extension"
        ),
        "oneapi_note": (
            "Synthetic response/SSE labels follow a historical logging extension,"
            " not stock One API output. This generator only writes log fixtures;"
            " modulo does not modify gateways."
        ),
        "input_usage": "frozen official reference input_ids length; no production counter invoked",
        "output_usage": "absent; local reasoning/answer text counts are not billing totals",
    }
    (ROOT / "tests/fixtures/provenance/request-response.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    print(encoded(counts))


if __name__ == "__main__":
    main()
