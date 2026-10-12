"""Measure local import memory/disk growth with an explicitly synthetic counter.

Use a new directory and separate processes to compare peak RSS.
This is not tokenizer accuracy validation.
python tests/tools/large_import.py --calls 10000 --directory /tmp/modulo-scale-10k
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from modulo.adapters.protocols import DifyProtocol
from modulo.adapters.sources import FileSource
from modulo.application import importer
from modulo.application.importer import run_batch
from modulo.core import main as core_flow
from modulo.core.domain.bindings import Bindings
from modulo.core.processing.recalculation import Recalculator


class SyntheticCounter:
    version = "synthetic-scale-check"

    def count(self, call, binding):
        return {"input_tokens": 13}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--memory-mib", type=int, default=64)
    parser.add_argument("--phases", action="store_true")
    options = parser.parse_args()
    if options.phases:
        for module, names in (
            (core_flow, ("collect_calls", "selection_summary", "calculate_calls", "summarize")),
            (importer, ("rebuild_index",)),
        ):
            for name in names:
                original = getattr(module, name)

                def measured(*args, _name=name, _function=original, **kwargs):
                    result = _function(*args, **kwargs)
                    print(
                        json.dumps(
                            {
                                "phase": _name,
                                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                / 1024,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    return result

                setattr(module, name, measured)
    options.directory.mkdir(parents=True, exist_ok=False)
    source = options.directory / "logs.jsonl"
    with source.open("w") as stream:
        for index in range(options.calls):
            stream.write(
                json.dumps(
                    {
                        "id": str(index),
                        "node_type": "llm",
                        "status": "succeeded",
                        "process_data": {
                            "model_name": "synthetic",
                            "model_mode": "chat",
                            "prompts": [{"role": "user", "text": "scale test"}],
                        },
                        "outputs": {
                            "text": "ok",
                            "usage": {"prompt_tokens": 13, "completion_tokens": 1},
                        },
                    }
                )
                + "\n"
            )
    calculator = Recalculator(
        Bindings.model_validate({"models": [{"name": "synthetic", "aliases": ["synthetic"]}]}),
        lambda _: SyntheticCounter(),
        {"assets": {"synthetic": {"family": "synthetic"}}},
    )
    started = time.monotonic()
    path, _ = run_batch(
        FileSource(source),
        DifyProtocol(),
        calculator,
        namespace="scale-test",
        output_root=options.directory / "results",
        state=options.directory / "state",
        memory_mib=options.memory_mib,
    )
    summary = json.loads((path / "summary.json").read_bytes())
    assert summary["observed_calls"] == options.calls
    assert summary["input_verdicts"] == {"CONSISTENT": options.calls}
    assert summary["run_status"] == "COMPLETE"
    print(
        json.dumps(
            {
                "calls": options.calls,
                "seconds": round(time.monotonic() - started, 2),
                "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                "disk_mib": round(
                    sum(p.stat().st_size for p in options.directory.rglob("*") if p.is_file())
                    / 1024**2,
                    2,
                ),
                "memory_mib": options.memory_mib,
                "result": str(path),
            }
        )
    )


if __name__ == "__main__":
    main()
