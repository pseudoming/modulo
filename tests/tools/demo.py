"""Offline CLI smoke test: run `python -m tests.tools.demo` from the repository root."""

import json
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path

from modulo.cli import main as cli
from modulo.storage.query_index import details
from tests.support.mock_es import MockElasticsearch

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    expected = json.loads((ROOT / "tests/fixtures/expected/basic.json").read_text())
    report = []
    for protocol in ("dify", "langchain"):
        path = ROOT / "examples" / protocol / "basic.jsonl"
        for physical in ("log", "es"):
            args = [
                "import",
                "--source",
                physical,
                "--adapter",
                protocol,
                "--path",
                str(path) if physical == "log" else "https://mock.invalid",
                "--namespace",
                "synthetic/demo",
                "--page-size",
                "3",
                "--output",
                str(ROOT / "runs/demo"),
            ]
            injection = nullcontext()
            if physical == "es":
                args += ["--index", "mock-logs"]
                injection = MockElasticsearch.from_file(path).patch_source()
            with injection, redirect_stdout(StringIO()) as output:
                status = cli(args)
            if status != 0:
                raise RuntimeError(f"demo import failed: {physical}/{protocol}, exit {status}")
            imported = json.loads(output.getvalue())
            bundle = Path(imported["run"])
            summary = imported["summary"]
            if (
                summary["run_status"] != "COMPLETE"
                or summary["input_verdicts"] != expected["expected_input_verdicts"]
            ):
                raise RuntimeError(
                    f"demo acceptance failed: {physical}/{protocol}; inspect {bundle}"
                )
            rows = details(bundle)
            if len(rows) != expected["expected_calls"]:
                raise RuntimeError("detail reload count differs from frozen fixture expectation")
            report.append(
                {
                    "source": physical,
                    "protocol": protocol,
                    "run": str(bundle),
                    "reused": imported["reused"],
                    "calls": len(rows),
                    "verdicts": summary["verdicts"],
                    "input_verdicts": summary["input_verdicts"],
                    "output_verdicts": summary["output_verdicts"],
                    "batch_read_complete": summary["batch_read_complete"],
                }
            )
    target = ROOT / "runs/demo/index.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target.read_text())


if __name__ == "__main__":
    main()
