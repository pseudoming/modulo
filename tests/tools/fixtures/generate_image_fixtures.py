"""Synthetic image logs, with usage taken from frozen upstream references."""

import base64
import copy
import io
import json
from pathlib import Path

from PIL import Image

from tests.tools.fixtures.generate_gateway_body_fixtures import encoded, native
from tests.tools.fixtures.generate_reasoning_fixtures import protocol_row

ROOT = Path(__file__).resolve().parents[3]


def image_rows(vector, adapter, model, scenario="complete"):
    vector = copy.deepcopy(vector)
    vector.update(
        name=vector["name"] + "-" + scenario,
        tools=None,
        logged_options={"mm_processor_kwargs": vector["overrides"]} if vector["overrides"] else {},
        output_text="图片已核对。",
        output_reasoning="先检查图片内容。",
    )
    block = vector["messages"][0]["content"][1]
    if scenario == "embedded":
        buffer = io.BytesIO()
        Image.new("RGB", (block["width"], block["height"]), "white").save(buffer, "PNG")
        block.clear()
        block.update(
            type="image_url",
            image_url={
                "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            },
        )
    elif scenario == "url-only":
        block.clear()
        block.update(type="image_url", image_url={"url": "https://example.invalid/image.png"})
    elif scenario == "more":
        vector["input_tokens"] += 7
    elif scenario == "detail":
        block["detail"] = "high"
    elif scenario == "saved-files":
        assert adapter == "dify"
        text = "".join(p["text"] for p in vector["messages"][0]["content"] if p["type"] == "text")
        vector["messages"][0].update(
            content=text,
            files=[{"type": "image", "data": "data:image...[TRUNCATED]...AA", "detail": "high"}],
        )
    if adapter in {"newapi", "oneapi"}:
        request = {"model": model, "messages": vector["messages"], **vector["logged_options"]}
        response = {
            "id": "images-" + vector["name"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": vector["output_text"],
                        "reasoning_content": vector["output_reasoning"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": vector["input_tokens"]},
        }
        call_id = f"images-{vector['asset']}-{vector['name']}"
        request_label = "text request body:" if adapter == "newapi" else "converted request:"
        return [
            native(adapter, call_id, request_label, encoded(request)),
            native(adapter, call_id, "upstream response body:", encoded(response)),
        ]
    row = protocol_row(vector, adapter, model)
    if adapter == "langchain":
        row["outputs"]["generations"][0][0]["generation_info"] = {"finish_reason": "stop"}
    elif adapter == "litellm":
        row["response"]["choices"][0]["finish_reason"] = "stop"
    return [row]


def main():
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/image_reference_vectors.json").read_text()
    )["vectors"]
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    rows = []
    for vector in vectors:
        if vector["name"] == "640x480":
            rows.extend(image_rows(vector, "litellm", assets[vector["asset"]]["repo"], "embedded"))
    first = next(v for v in vectors if v["name"] == "640x480")
    for scenario in ("more", "url-only", "detail"):
        rows.extend(image_rows(first, "litellm", assets[first["asset"]]["repo"], scenario))
    path = ROOT / "examples/litellm/images.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} synthetic image logs to {path}")


if __name__ == "__main__":
    main()
