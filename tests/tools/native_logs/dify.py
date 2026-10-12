"""Execute Dify's pinned Graphon LLM node with a synthetic local model runtime.

Requires graphon==0.7.0 in an isolated interpreter. The Dify saving serializer
is loaded from the reviewed source file; its sole Dify-only enum import is
replaced by the identical Graphon chat/completion enum. No parsing code is
copied into the test harness. This is node-level validation, not a deployed UI.
"""

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

from graphon.model_runtime.entities.llm_entities import LLMMode, LLMResult, LLMUsage
from graphon.model_runtime.entities.message_entities import AssistantPromptMessage
from graphon.nodes.llm.entities import LLMNodeData
from graphon.nodes.llm.node import GraphInitParams, GraphRuntimeState, LLMNode
from graphon.runtime import VariablePool

ROOT = Path(__file__).resolve().parents[3]


def load_serializer(path):
    source = path.read_text()
    tree = ast.parse(source, str(path))
    # Only substitute the import of a two-value enum. Execute all of the
    # upstream saving serializer, including its tool and truncation behavior.
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module == "core.prompt.simple_prompt_transform")
    ]
    namespace = {"ModelMode": LLMMode}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["PromptMessageUtil"]


class SyntheticRuntime:
    provider = "modulo-synthetic"
    parameters = {}
    stop = None

    def __init__(self, vector, model):
        self.vector, self.model_name = vector, model

    def get_model_schema(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_properties={"mode": "chat"},
            parameter_rules=[],
            supports_prompt_content_type=lambda kind: kind == "text",
        )

    def invoke_llm(self, *, prompt_messages, **kwargs):
        expected = self.vector["messages"]
        assert [
            {"role": m.role.value, "content": m.get_text_content()} for m in prompt_messages
        ] == expected
        return LLMResult(
            model=self.model_name,
            prompt_messages=prompt_messages,
            message=AssistantPromptMessage(content=self.vector["output"]),
            usage=LLMUsage.from_metadata(
                {
                    "prompt_tokens": self.vector["input_tokens"],
                    "completion_tokens": self.vector["visible_output_tokens"],
                }
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/native-logs")
    args = parser.parse_args()
    util = load_serializer(args.serializer)

    class Serializer:
        def serialize(self, *, model_mode, prompt_messages):
            return util.prompt_messages_to_prompt_for_saving(model_mode, prompt_messages)

    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json").read_text()
    )["vectors"]
    rows, expected = [], {}
    # Dify's plain LLM node accepts system/user/assistant prompt templates.
    # Tool invocations are covered separately; do not relabel Agent calls LLM.
    for vector in vectors:
        if vector["name"] != "basic_zh":
            continue
        model = assets[vector["asset"]]["repo"]
        node_id = "native-dify-" + vector["asset"]
        data = LLMNodeData.model_validate(
            {
                "title": "Synthetic native log acceptance",
                "type": "llm",
                "model": {"provider": "modulo-synthetic", "name": model, "mode": "chat"},
                "prompt_template": [
                    {"role": m["role"], "text": m["content"]} for m in vector["messages"]
                ],
                "context": {"enabled": False},
            }
        )
        node = LLMNode(
            node_id,
            data,
            graph_init_params=GraphInitParams(
                workflow_id="synthetic", graph_config={}, run_context={}, call_depth=0
            ),
            graph_runtime_state=GraphRuntimeState(
                variable_pool=VariablePool(), start_at=time.perf_counter()
            ),
            model_instance=SyntheticRuntime(vector, model),
            llm_file_saver=None,
            prompt_message_serializer=Serializer(),
        )
        node.bind_execution_id(node_id)
        events = list(node.run())
        terminal = events[-1]
        result = terminal.node_run_result.model_dump(mode="json")
        assert result["status"] == "succeeded", result
        rows.append({"id": node.execution_id, "node_type": "llm", **result})
        expected[node.execution_id] = {
            "input_tokens": vector["input_tokens"],
            "visible_output_tokens": vector["visible_output_tokens"],
            "model": model,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dify.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (args.output / "dify-expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2)
    )
    (args.output / "dify-provenance.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "producer": "graphon==0.7.0 LLMNode.run + Dify PromptMessageUtil",
                "serializer_sha256": hashlib.sha256(args.serializer.read_bytes()).hexdigest(),
                "scope": "executed LLM node; no full Dify application deployment",
                "calls": len(rows),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "producer": "Dify LLM node",
                "calls": len(rows),
                "path": str(args.output / "dify.jsonl"),
            }
        )
    )


if __name__ == "__main__":
    main()
