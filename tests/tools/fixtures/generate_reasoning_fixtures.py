"""Synthetic protocol rows using frozen expectations, without running a model/tokenizer."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def protocol_row(vector, protocol, model):
    call_id = f"synthetic-{vector['asset']}-{vector['name']}"
    messages = copy.deepcopy(vector["messages"])
    parameters = copy.deepcopy(vector["logged_options"])
    if vector["tools"]:
        parameters["tools"] = vector["tools"]
    # Output billing is deliberately absent: separate component counts do not
    # establish a provider's generation/EOS accounting rule.
    usage = {
        "prompt_tokens": vector["input_tokens"]
        if "input_tokens" in vector
        else len(vector["input_ids"])
    }
    response = {
        "role": "assistant",
        "content": vector["output_text"],
        "reasoning_content": vector["output_reasoning"],
    }
    if protocol == "dify":
        for message in messages:
            message["text"] = message.pop("content")
        return {
            "id": call_id,
            "node_type": "llm",
            "status": "succeeded",
            "process_data": {
                "model_mode": "chat",
                "model_name": model,
                "prompts": messages,
                "model_parameters": parameters,
            },
            "outputs": {
                "text": response["content"],
                "reasoning_content": response["reasoning_content"],
                "usage": usage,
                "finish_reason": "stop",
            },
        }
    if protocol == "langchain":

        def lc_message(message):
            body = copy.deepcopy(message)
            role = {"user": "human", "assistant": "ai"}.get(body.pop("role"), message["role"])
            additional = {
                key: body.pop(key) for key in ("reasoning_content", "tool_calls") if key in body
            }
            if additional:
                body["additional_kwargs"] = additional
            return {"type": role, "data": body}

        return {
            "id": call_id,
            "run_type": "chat_model",
            "inputs": {"messages": [[lc_message(m) for m in messages]]},
            "extra": {"metadata": {"ls_model_name": model}, "invocation_params": parameters},
            "outputs": {
                "generations": [[{"text": response["content"], "message": lc_message(response)}]],
                "llm_output": {"token_usage": usage},
            },
        }
    return {
        "id": call_id,
        "call_type": "acompletion",
        "status": "success",
        "cache_hit": False,
        "model": model,
        "messages": messages,
        "model_parameters": parameters,
        "response": {"choices": [{"message": response}]},
        **usage,
    }


def main():
    source = ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json"
    vectors = json.loads(source.read_text())["vectors"]
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    bindings = {
        "models": [{"name": asset, "aliases": [entry["repo"]]} for asset, entry in assets.items()],
    }
    for protocol in ("dify", "langchain", "litellm"):
        directory = ROOT / "examples" / protocol
        directory.mkdir(parents=True, exist_ok=True)
        rows = [protocol_row(v, protocol, assets[v["asset"]]["repo"]) for v in vectors]
        (directory / "reasoning-tools.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        (directory / "reasoning-tools.bindings.yaml").write_text(
            yaml.safe_dump(bindings, allow_unicode=True, sort_keys=False)
        )
    provenance = {
        "synthetic": True,
        "reference": "tests/fixtures/tokenizers/reasoning_reference_vectors.json",
        "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator": "tests/tools/fixtures/generate_reasoning_fixtures.py",
        "models": len(assets),
        "rows_per_protocol": len(vectors),
        "input_usage": "frozen official reference IDs; no gateway/model service was run",
        "output_usage": (
            "not supplied; reasoning and answer counts are separate local text components"
        ),
    }
    (ROOT / "tests/fixtures/provenance/reasoning-tools.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {len(vectors)} rows per protocol for {len(assets)} models")


if __name__ == "__main__":
    main()
