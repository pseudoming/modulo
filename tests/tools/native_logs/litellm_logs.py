"""Exercise LiteLLM's actual callback and SpendLogs serializers with a local stub."""

import argparse
import asyncio
import json
import os
from pathlib import Path

# Set before importing LiteLLM. No cost-map download, telemetry, or remote model.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["STORE_PROMPTS_IN_SPEND_LOGS"] = "true"
os.environ["MAX_STRING_LENGTH_PROMPT_IN_DB"] = "100000"

import litellm  # noqa: E402
from litellm.integrations.custom_logger import CustomLogger  # noqa: E402
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "runs/native-logs"


def json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


class Collector(CustomLogger):
    def __init__(self):
        self.standard, self.spend, self.errors = [], [], []
        self.ready = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self.standard.append(kwargs["standard_logging_object"])
            self.spend.append(get_logging_payload(kwargs, response_obj, start_time, end_time))
        except Exception as exc:
            self.errors.append(exc)
        finally:
            self.ready.set()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
    )["vectors"]
    collector = Collector()
    litellm.callbacks = [collector]
    for asset in ("qwen-3.8-2.4t", "deepseek-v4.1"):
        index, vector = next((i, v) for i, v in enumerate(vectors) if v["asset"] == asset)
        for stream in (False, True):
            request = {
                "model": assets[asset]["repo"],
                "messages": vector["messages"],
                "tools": vector["tools"],
                "stream": stream,
                "user": f"modulo-case-{index}",
                **vector["logged_options"],
            }
            collector.ready.clear()
            response = await litellm.acompletion(
                model="openai/" + request["model"],
                messages=request["messages"],
                tools=request["tools"],
                stream=stream,
                user=request["user"],
                extra_body=vector["logged_options"],
                api_base="http://127.0.0.1:18080/v1",
                api_key="local-synthetic-only",
                proxy_server_request={"body": request},
                extra_headers={"x-modulo-case": str(index)},
                stream_options={"include_usage": True} if stream else None,
            )
            if stream:
                async for _ in response:
                    pass
            await asyncio.wait_for(collector.ready.wait(), 30)
            if collector.errors:
                raise collector.errors[-1]
    for kind, rows in (("standard", collector.standard), ("spend", collector.spend)):
        path = output / f"litellm-{kind}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, default=json_default) + "\n" for row in rows
            )
        )
        print(json.dumps({"producer": f"LiteLLM {kind}", "calls": len(rows), "path": str(path)}))


if __name__ == "__main__":
    asyncio.run(main())
