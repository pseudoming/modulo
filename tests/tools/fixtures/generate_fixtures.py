"""Generate synthetic Dify exports and actual LangChain FakeModel trace exports.

No provider calls or model weights. Gold reference counts are read, never computed
with the implementation under test. Deliberate gaps/deviations are listed below.
"""

import copy
import importlib.metadata
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

ROOT = Path(__file__).resolve().parents[3]
MODELS = {"qwen": "Qwen/Qwen3.8-2.4T-A95B", "deepseek": "deepseek-ai/DeepSeek-V4.1-Flash"}
ASSETS = {"qwen": "qwen-3.8-2.4t", "deepseek": "deepseek-v4.1"}


def main() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json").read_text()
    )["vectors"]
    by_name = {(v["asset"], v["name"]): v for v in vectors}
    cases = [
        ("qwen", "basic_zh", "consistent"),
        ("deepseek", "history_unicode", "delta_plus_three"),
        ("qwen", "no_system", "missing_usage"),
        ("qwen", "basic_zh", "unknown_model"),
        ("qwen", "basic_zh", "missing_messages"),
        ("deepseek", "basic_zh", "multimodal"),
        ("qwen", "basic_zh", "same_content_new_id"),
        ("deepseek", "basic_zh", "failed_missing_messages"),
    ]
    dify, langchain = [], []
    for asset, vector_name, scenario in cases:
        vector = by_name[ASSETS[asset], vector_name]
        model = "vendor/unknown" if scenario == "unknown_model" else MODELS[asset]
        input_tokens = vector["input_tokens"] + (3 if scenario == "delta_plus_three" else 0)
        output_tokens = vector["visible_output_tokens"]
        run_id = uuid5(NAMESPACE_URL, "modulo/synthetic/" + scenario)
        messages = copy.deepcopy(vector["messages"])
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        # The default collector flattens messages into display strings. This pinned,
        # internal schema option preserves the actual message objects for fixtures.
        collector = RunCollectorCallbackHandler(_schema_format="original+chat")
        fake = FakeMessagesListChatModel(
            responses=[AIMessage(content=vector["output"], usage_metadata=usage)]
        )
        fake.invoke(
            [(m["role"], m["content"]) for m in messages],
            config={
                "callbacks": [collector],
                "run_id": run_id,
                "metadata": {"ls_model_name": model, "ls_provider": asset, "synthetic": True},
            },
        )
        trace = json.loads(collector.traced_runs[0].model_dump_json())
        # Freeze volatile clock/UUID fields; preserve the framework's serialization shape.
        trace["start_time"] = "2026-10-08T00:00:00+00:00"
        trace["end_time"] = "2026-10-08T00:00:00.250000+00:00"
        trace["events"] = []
        trace["dotted_order"] = f"20261008T000000000000Z{run_id}"
        trace["trace_id"] = str(run_id)
        output_body = trace["outputs"]["generations"][0][0]["message"]["kwargs"]
        output_body["id"] = "synthetic-response"
        node = {
            "id": str(run_id),
            "node_id": "llm",
            "node_type": "llm",
            "status": "succeeded",
            "created_at": 1791417600,
            "elapsed_time": "0.250",
            "process_data": {
                "model_mode": "chat",
                "model_provider": asset,
                "model_name": model,
                "prompts": [
                    {"role": m["role"], "text": m["content"], "files": []} for m in messages
                ],
            },
            "outputs": {
                "text": vector["output"],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "finish_reason": "stop",
            },
        }
        if scenario in {"missing_messages", "failed_missing_messages"}:
            node["process_data"].pop("prompts")
            trace["inputs"].pop("messages")
        if scenario in {"missing_usage", "failed_missing_messages"}:
            node["outputs"].pop("usage")
            output_body["usage_metadata"] = None
            trace["outputs"]["llm_output"] = None
        if scenario == "multimodal":
            node["process_data"]["prompts"][-1]["files"] = [{"type": "image", "synthetic": True}]
            trace["inputs"]["messages"][0][-1]["kwargs"]["content"] = [
                {"type": "text", "text": "synthetic image prompt"},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/mock.png"}},
            ]
        if scenario == "failed_missing_messages":
            node["status"] = "failed"
            node["outputs"].pop("text")
            trace["error"] = "synthetic timeout"
            trace["outputs"] = None
        dify.append(
            {"event": "node_finished", "workflow_run_id": "synthetic-workflow", "data": node}
        )
        langchain.append(trace)
    dify += [{"node_type": "start", "id": "ignored-start"}, copy.deepcopy(dify[0])]
    langchain += [
        {"run_type": "chain", "id": "ignored-chain", "child_runs": []},
        copy.deepcopy(langchain[0]),
    ]
    for name, records in (("dify", dify), ("langchain", langchain)):
        target = ROOT / "examples" / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "basic.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for r in records
            ),
            encoding="utf-8",
        )
    (ROOT / "tests/fixtures/expected/basic.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "reference_vectors": "tests/fixtures/tokenizers/family_reference_vectors.json",
                "reference_assets": ASSETS,
                "generated_with": {"langchain-core": importlib.metadata.version("langchain-core")},
                "dify_source_revision": "2bb466c09aa16ebce7a9d0770ededb80ac2de05c",
                "cases": [case[2] for case in cases],
                "raw_records_per_protocol": 10,
                "expected_calls": 8,
                "expected_duplicate_calls": 1,
                "expected_ignored_records": 1,
                "expected_input_verdicts": {"CONSISTENT": 2, "DEVIATION": 1, "UNVERIFIABLE": 5},
                "expected_compared_calls": 3,
                "expected_recalculated_calls": 4,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("generated 10 Dify and 10 LangChain records; all synthetic")


if __name__ == "__main__":
    main()
