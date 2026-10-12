"""Replay frozen Dify logs through the public import CLI.

Expected values remain frozen from the independent official-template reference.
This module reads log fixtures; it does not collect from or modify Dify.
"""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "runs/dify-agent"
FIXTURES = ROOT / "tests/fixtures/dify-agent"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_rows(actual, expected):
    assert {r["call"]["call_id"] for r in actual} == set(expected)
    for row in actual:
        call, measured = row["call"], row["measurement"]
        ref = expected[call["call_id"]]
        assert not call["gaps"] and not call["output_gaps"]
        for field in ("input_tokens", "visible_output_tokens", "reasoning_output_tokens"):
            assert measured[field] == ref[field], (
                call["call_id"],
                field,
                measured[field],
                ref[field],
            )
        ids_digest = hashlib.sha256(
            json.dumps(ref["input_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        assert measured["input_ids_sha256"] == ids_digest
        assert (
            measured["rendered_input_sha256"]
            == hashlib.sha256(ref["rendered_input"].encode()).hexdigest()
        )
        # Only the streamed final answer retains both inline reasoning and finish.
        complete_output = call["output_finish_reason"] == "stop"
        assert measured["output_execution"] == ("completed" if complete_output else "partial")
        assert measured["output_tokens"] == (21 if complete_output else None)
        assert measured["input_delta"] == 123 - ref["input_tokens"]
        assert measured["input_verdict"] == "DEVIATION"
        assert json.loads(call["output_tool_calls_json"] or "[]") == ref["tool_calls"]
        assert len(row["source_refs"]) == 2
        assert row["assemblies"][0]["method"] == "explicit_invocation_id"


def verify(es=None):
    from modulo.storage.query_index import details
    from modulo.storage.result_bundle import verify_bundle

    if __package__:
        from .verify import seed_es
    else:
        from verify import seed_es

    expected = json.loads((FIXTURES / "reference.json").read_text())
    projections, report = [], []
    for label in ("agent-llm", "agent-thoughts"):
        for source in ["log", "es"] if es else ["log"]:
            path = FIXTURES / f"{label}.jsonl"
            command = [
                str(ROOT / ".venv/bin/modulo"),
                "import",
                "--source",
                source,
                "--adapter",
                "dify",
                "--bindings",
                str(ROOT / "examples/dify/reasoning-tools.bindings.yaml"),
                "--output",
                str(DATA / "results"),
            ]
            index = None
            if source == "log":
                command += ["--path", str(path)]
            else:
                index = seed_es(es, "dify-" + label, path)
                command += [
                    "--path",
                    es,
                    "--index",
                    index,
                    "--payload-field",
                    "payload",
                    "--page-size",
                    "3",
                ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            imported = json.loads(result.stdout)
            bundle = Path(imported["run"])
            verify_bundle(bundle)
            actual = details(bundle)
            assert len(actual) == len(expected)
            assert imported["summary"]["counters"]["invalid_records"] == 0
            if label == "agent-llm":
                check_rows(actual, expected)
                projections.append(
                    sorted(
                        [{"call": r["call"], "measurement": r["measurement"]} for r in actual],
                        key=lambda r: r["call"]["call_id"],
                    )
                )
            else:
                assert all(r["measurement"]["input_verdict"] == "UNVERIFIABLE" for r in actual)
                assert all(r["call"]["usage"]["input_tokens"] == 123 for r in actual)
            report.append(
                {
                    "case": label,
                    "source": source,
                    "calls": len(actual),
                    "index": index,
                    "run": str(bundle),
                    "verdicts": imported["summary"]["verdicts"],
                    "input_verdicts": imported["summary"]["input_verdicts"],
                    "output_verdicts": imported["summary"]["output_verdicts"],
                    "output_execution": imported["summary"]["output_execution"],
                }
            )
            print(json.dumps(report[-1]), flush=True)
    if es:
        assert projections[0] == projections[1]
    target = DATA / "validation.json"
    if target.exists():
        replaced = {(r["case"], r["source"]) for r in report}
        old = json.loads(target.read_text())
        report = [r for r in old if (r["case"], r["source"]) not in replaced] + report
    save(target, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--es")
    args = parser.parse_args()
    verify(args.es)
