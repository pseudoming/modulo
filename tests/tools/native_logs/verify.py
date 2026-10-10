"""Replay producer-generated files through the CLI and optionally a real ES."""

import hashlib
import json
import subprocess
from pathlib import Path

import httpx

from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[3]
CASES = {
    "dify": ("dify", "jsonl", 11),
    "langchain": ("langchain", "jsonl", 96),
    "langchain-stream": ("langchain", "jsonl", 96),
    "litellm-standard": ("litellm", "jsonl", 4),
    "litellm-spend": ("litellm", "jsonl", 4),
    "newapi": ("newapi", "log", 4),
    "oneapi-patched": ("oneapi", "log", 4),
    "oneapi": ("oneapi", "log", 4),
}


def expected_counts(data, label, call):
    if label.startswith("langchain"):
        return json.loads((data / "langchain-expected.json").read_text())[call["call_id"]]
    if label == "dify":
        return json.loads((data / "dify-expected.json").read_text())[call["call_id"]]
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
    )["vectors"]
    assets = json.loads((ROOT / "src/modulo/application/assets-lock.json").read_text())["assets"]
    vector = next(v for v in vectors if assets[v["asset"]]["repo"] == call["model"])
    return {
        "input_tokens": len(vector["input_ids"]),
        "visible_output_tokens": vector["visible_output_tokens"],
        "reasoning_output_tokens": vector["reasoning_output_tokens"],
    }


def seed_es(url, label, path):
    raw = path.read_bytes()
    index = f"modulo-native-{label}-{hashlib.sha256(raw).hexdigest()[:12]}"
    with httpx.Client(base_url=url, trust_env=False, timeout=30) as client:
        if client.head(f"/{index}").status_code == 404:
            response = client.put(
                f"/{index}",
                json={
                    "settings": {"number_of_replicas": 0},
                    "mappings": {"properties": {"payload": {"type": "object", "enabled": False}}},
                },
            )
            response.raise_for_status()
        bulk = []
        for i, line in enumerate(raw.splitlines()):
            payload = (
                json.loads(line)
                if path.suffix == ".jsonl"
                else {"message": line.decode(), "event": {"sequence": i}}
            )
            bulk.extend(
                [
                    json.dumps({"index": {"_index": index, "_id": str(i)}}),
                    json.dumps({"payload": payload}, ensure_ascii=False),
                ]
            )
        response = client.post(
            "/_bulk",
            params={"refresh": "wait_for"},
            content="\n".join(bulk) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        assert not response.json()["errors"], "ES bulk write failed"
    return index


def verify(data, es=None, cases=None):
    report = []
    for label, (adapter, extension, count) in CASES.items():
        if cases and label not in cases:
            continue
        path = data / f"{label}.{extension}"
        binding = "request-response" if adapter in {"newapi", "oneapi"} else "reasoning-tools"
        projections = []
        for source in ["log", "es"] if es else ["log"]:
            command = [
                str(ROOT / ".venv/bin/modulo"),
                "import",
                "--source",
                source,
                "--adapter",
                adapter,
                "--bindings",
                str(ROOT / f"examples/{adapter}/{binding}.bindings.yaml"),
                "--output",
                str(ROOT / "runs/native-logs/results"),
            ]
            index = None
            if source == "log":
                command += ["--path", str(path)]
            else:
                index = seed_es(es, label, path)
                command += [
                    "--path",
                    es,
                    "--index",
                    index,
                    "--payload-field",
                    "payload",
                    "--page-size",
                    "7",
                ]
            result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
            assert result.returncode == 0, (label, source, result.stdout, result.stderr)
            imported = json.loads(result.stdout)
            bundle = Path(imported["run"])
            verify_bundle(bundle)
            rows = details(bundle)
            assert len(rows) == count, (label, len(rows), count)
            for row in rows:
                call, measurement = row["call"], row["measurement"]
                expected = expected_counts(data, label, call)
                assert measurement["input_tokens"] == expected["input_tokens"], (
                    label,
                    call["call_id"],
                )
                if label == "oneapi":
                    assert measurement["output_execution"] == "not_run"
                    assert "missing_response_body" in call["output_gaps"]
                else:
                    assert not call["gaps"] and not call["output_gaps"], (
                        label,
                        call["gaps"],
                        call["output_gaps"],
                    )
                    for key in ("visible_output_tokens", "reasoning_output_tokens"):
                        if key in expected:
                            assert measurement[key] == expected[key], (
                                label,
                                key,
                                measurement[key],
                                expected[key],
                            )
                    assert measurement["output_execution"] == "completed"
            projection = sorted(
                [{"call": row["call"], "measurement": row["measurement"]} for row in rows],
                key=lambda row: row["call"]["call_id"],
            )
            projections.append(projection)
            summary = imported["summary"]
            assert summary["counters"]["invalid_records"] == 0
            item = {
                "producer": label,
                "source": source,
                "calls": count,
                "run": str(bundle),
                "index": index,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "verdicts": summary["verdicts"],
                "output_execution": summary["output_execution"],
            }
            report.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        if es:
            assert projections[0] == projections[1], f"file/ES semantic mismatch: {label}"
    output = ROOT / "runs/native-logs/validation.json"
    if cases and output.exists():
        old = json.loads(output.read_text())
        replaced = {(r["producer"], r["source"]) for r in report}
        report = [r for r in old if (r["producer"], r["source"]) not in replaced] + report
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
