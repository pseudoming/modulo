"""Produce real LangChain Run JSON from local synthetic model invocations."""

import argparse
import json
from pathlib import Path

from native_langchain import serialized_run

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/native-logs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
    )["vectors"]
    expected = {}
    for stream in (False, True):
        label = "langchain-stream" if stream else "langchain"
        rows = [serialized_run(v, assets[v["asset"]]["repo"], stream=stream) for v in vectors]
        for row, vector in zip(rows, vectors, strict=True):
            expected[row["id"]] = {
                "input_tokens": len(vector["input_ids"]),
                "visible_output_tokens": vector["visible_output_tokens"],
                "reasoning_output_tokens": vector["reasoning_output_tokens"],
            }
        (args.output / f"{label}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        print(json.dumps({"producer": label, "calls": len(rows)}))
    (args.output / "langchain-expected.json").write_text(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
