"""Seed the local Compose lab, without model calls or Kubernetes.

Mutations are limited to the six explicitly named synthetic indices and a dedicated
Kibana space. Existing indices/documents with different ownership/content are not
overwritten. The legacy audit smoke test delegates all imports to `modulo import`.
"""

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import NAMESPACE_URL, uuid5

import httpx

from modulo.shared.serialization import dumps
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[2]
OWNER = "modulo-local-lab.v1"
SPACE = "modulo"
EXPECTED = ROOT / "tests/local-dev/expected.json"
TIME_RANGE = {"from": "2026-09-30T16:00:00.000Z", "to": "2026-10-03T16:00:00.000Z"}
COLUMNS = ["@timestamp", "log_source", "model", "scenario", "usage.input_tokens", "request_text"]


def expected() -> dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def build_documents() -> list[dict]:
    """Use frozen, independently built reference vectors, not the recount implementation."""
    spec = expected()
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/family_reference_vectors.json").read_text()
    )["vectors"]
    references = {(vector["asset"], vector["name"]): vector for vector in vectors}
    templates = {
        protocol: json.loads(
            (ROOT / f"examples/{protocol}/basic.jsonl").read_text().splitlines()[0]
        )
        for protocol in ("dify", "langchain")
    }
    documents = []
    for protocol in ("dify", "langchain"):
        for day, date in enumerate(spec["dates"]):
            index = f"{protocol}-access-log-{date}"
            for slot, model in enumerate(spec["models"]):
                family = "qwen" if slot in (0, 2) else "deepseek"
                reference = references[
                    spec["reference_assets"][family],
                    ["basic_zh", "history_unicode", "no_system"][day],
                ]
                start = datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC) + timedelta(
                    hours=2, minutes=slot * 10
                )
                duration = 120 + day * 35 + slot * 40
                end = start + timedelta(milliseconds=duration)
                call_id = str(uuid5(NAMESPACE_URL, f"modulo/elastic-lab/{protocol}/{date}/{model}"))
                delta = 3 if (day, slot) == (0, 1) else -2 if (day, slot) == (1, 0) else 0
                missing = (day, slot) == (2, 0)
                input_tokens = None if missing else reference["input_tokens"] + delta
                output_tokens = None if missing else reference["visible_output_tokens"]
                total = None if missing else input_tokens + output_tokens
                scenario = (
                    "unbound_alias"
                    if slot >= 2
                    else "missing_usage"
                    if missing
                    else "injected_plus_3"
                    if delta == 3
                    else "injected_minus_2"
                    if delta == -2
                    else "reference_consistent"
                )
                messages = copy.deepcopy(reference["messages"])
                payload = copy.deepcopy(templates[protocol])
                if protocol == "dify":
                    payload["workflow_run_id"] = f"synthetic-workflow-{date}"
                    node = payload["data"]
                    node["id"] = call_id
                    node["created_at"] = int(start.timestamp())
                    node["elapsed_time"] = f"{duration // 1000}.{duration % 1000:03d}"
                    node["process_data"].update(
                        {
                            "model_name": model,
                            "model_provider": family,
                            "prompts": [
                                {"role": m["role"], "text": m["content"], "files": []}
                                for m in messages
                            ],
                        }
                    )
                    node["outputs"] = {
                        "text": reference["output"],
                        "finish_reason": "stop",
                        "usage": None
                        if missing
                        else {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": total,
                        },
                    }
                else:
                    payload.update(
                        {
                            "id": call_id,
                            "trace_id": call_id,
                            "start_time": start.isoformat(),
                            "end_time": end.isoformat(),
                            "dotted_order": start.strftime("%Y%m%dT%H%M%S%fZ") + call_id,
                        }
                    )
                    payload["extra"]["metadata"].update(
                        {"ls_model_name": model, "ls_provider": family, "synthetic": True}
                    )
                    payload["serialized"]["repr"] = "Synthetic fixture; no model was invoked"
                    names = {
                        "system": "SystemMessage",
                        "user": "HumanMessage",
                        "assistant": "AIMessage",
                    }
                    types = {"system": "system", "user": "human", "assistant": "ai"}
                    payload["inputs"] = {
                        "messages": [
                            [
                                {
                                    "lc": 1,
                                    "type": "constructor",
                                    "id": ["langchain", "schema", "messages", names[m["role"]]],
                                    "kwargs": {"content": m["content"], "type": types[m["role"]]},
                                }
                                for m in messages
                            ]
                        ]
                    }
                    generation = payload["outputs"]["generations"][0][0]
                    generation["text"] = reference["output"]
                    generation["message"]["kwargs"].update(
                        {
                            "id": f"synthetic-response-{call_id}",
                            "content": reference["output"],
                            "usage_metadata": None
                            if missing
                            else {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total,
                            },
                        }
                    )
                document = {
                    "@timestamp": start.isoformat(),
                    "log_source": protocol,
                    "service": {"name": f"synthetic-{protocol}-app"},
                    "call_id": call_id,
                    "model": model,
                    "provider": family,
                    "scenario": scenario,
                    "synthetic": True,
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "duration_ms": duration,
                    "request_text": "\n".join(f"[{m['role']}] {m['content']}" for m in messages),
                    "response_text": reference["output"],
                    "payload": payload,
                }
                documents.append({"index": index, "id": call_id, "document": document})
    return documents


def index_definition() -> dict:
    properties = {
        key: {"type": "keyword"}
        for key in ("log_source", "call_id", "model", "provider", "scenario")
    }
    properties.update(
        {
            "@timestamp": {"type": "date"},
            "synthetic": {"type": "boolean"},
            "duration_ms": {"type": "long"},
            "request_text": {"type": "text"},
            "response_text": {"type": "text"},
            "payload": {"type": "object", "enabled": False},
            "service": {"properties": {"name": {"type": "keyword"}}},
            "usage": {
                "properties": {"input_tokens": {"type": "long"}, "output_tokens": {"type": "long"}}
            },
        }
    )
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"dynamic": "strict", "_meta": {"owner": OWNER}, "properties": properties},
    }


def checked(response: httpx.Response) -> dict:
    response.raise_for_status()
    return response.json()


def configure_kibana(client: httpx.Client, public_url: str) -> dict:
    space = client.get(f"/api/spaces/space/{SPACE}")
    if space.status_code == 404:
        checked(
            client.post(
                "/api/spaces/space",
                json={
                    "id": SPACE,
                    "name": "Modulo 日志实验",
                    "description": OWNER,
                    "initials": "M",
                    "color": "#2563eb",
                    "disabledFeatures": [],
                },
            )
        )
    elif checked(space).get("description") != OWNER:
        raise ValueError("Kibana space 'modulo' is not owned by this lab; left unchanged")
    prefix = f"/s/{SPACE}"
    views = (
        ("all", "*-access-log-2026100*"),
        ("dify", "dify-access-log-2026100*"),
        ("langchain", "langchain-access-log-2026100*"),
    )
    links = {}
    for label, pattern in views:
        view_id = f"modulo-{label}"
        current = client.get(f"{prefix}/api/data_views/data_view/{view_id}")
        if current.status_code == 404:
            checked(
                client.post(
                    f"{prefix}/api/data_views/data_view",
                    json={
                        "data_view": {
                            "id": view_id,
                            "title": pattern,
                            "name": f"Modulo · {label} · 2026-10-01~03",
                            "timeFieldName": "@timestamp",
                        }
                    },
                )
            )
        elif checked(current)["data_view"]["title"] != pattern:
            raise ValueError(f"data view {view_id} has changed; left unchanged")
        global_state = "(time:(from:'" + TIME_RANGE["from"] + "',to:'" + TIME_RANGE["to"] + "'))"
        app_state = (
            "(columns:!(" + ",".join("'" + c + "'" for c in COLUMNS) + "),"
            "dataSource:(dataViewId:'" + view_id + "',type:dataView),"
            "index:'" + view_id + "',query:(language:kuery,query:''),"
            "sort:!(!('@timestamp',desc)))"
        )
        links[label] = (
            f"{public_url.rstrip('/')}{prefix}/app/discover#/?"
            f"_g={quote(global_state, safe='')}&_a={quote(app_state, safe='')}"
        )
    checked(
        client.post(
            f"{prefix}/api/data_views/default", json={"data_view_id": "modulo-all", "force": True}
        )
    )
    return links


def seed(es_url: str, kibana_url: str) -> dict:
    documents = build_documents()
    indices = sorted({row["index"] for row in documents})
    with httpx.Client(base_url=es_url, timeout=30, trust_env=False) as client:
        info = checked(client.get("/"))
        if info.get("cluster_name") != "modulo-local":
            raise ValueError("seed requires the dedicated 'modulo-local' cluster")
        for index in indices:
            mapping = client.get(f"/{index}/_mapping")
            if mapping.status_code == 404:
                checked(client.put(f"/{index}", json=index_definition()))
            elif checked(mapping)[index]["mappings"].get("_meta", {}).get("owner") != OWNER:
                raise ValueError(f"index {index} is not owned by this lab; left unchanged")
        bulk = "".join(
            json.dumps({"create": {"_index": row["index"], "_id": row["id"]}})
            + "\n"
            + json.dumps(row["document"], ensure_ascii=False)
            + "\n"
            for row in documents
        )
        result = checked(
            client.post(
                "/_bulk", content=bulk.encode(), headers={"Content-Type": "application/x-ndjson"}
            )
        )
        if len(result["items"]) != len(documents):
            raise ValueError("bulk response has an unexpected number of items")
        created = 0
        for row, item in zip(documents, result["items"], strict=True):
            status = item["create"]["status"]
            if status == 201:
                created += 1
            elif status == 409:
                existing = checked(client.get(f"/{row['index']}/_doc/{row['id']}"))
                if existing["_source"] != row["document"]:
                    raise ValueError(f"fixture {row['id']} was changed; no overwrite performed")
            else:
                raise ValueError(f"bulk create failed for {row['index']}: HTTP {status}")
        checked(client.post("/" + ",".join(indices) + "/_refresh"))
        counts = {index: checked(client.get(f"/{index}/_count"))["count"] for index in indices}
        if any(count != expected()["documents_per_index"] for count in counts.values()):
            raise ValueError(f"unexpected document counts; no data deleted: {counts}")
    with httpx.Client(
        base_url=kibana_url, headers={"kbn-xsrf": "modulo-lab"}, timeout=60, trust_env=False
    ) as client:
        status = checked(client.get("/api/status"))
        links = configure_kibana(client, kibana_url)
    output = ROOT / "runs/elastic-lab"
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-snapshot.jsonl").write_text(
        "".join(dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    report = {
        "indices": counts,
        "documents": sum(counts.values()),
        "created": created,
        "existing": len(documents) - created,
        "elasticsearch_version": info["version"]["number"],
        # Public status can be redacted when security is disabled; keep unknown
        # metadata missing rather than substituting the expected image version.
        "kibana_version": status.get("version", {}).get("number"),
        "kibana_status": status["status"]["overall"]["level"],
        "discover": links,
    }
    (output / "seed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def audit(es_url: str) -> list[dict]:
    spec = expected()["per_protocol"]
    report = []
    for protocol in ("dify", "langchain"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "modulo.cli",
                "import",
                "--source",
                "es",
                "--path",
                es_url,
                "--adapter",
                protocol,
                "--index",
                f"{protocol}-access-log-2026100*",
                "--namespace",
                f"synthetic/elastic-lab/{protocol}",
                "--payload-field",
                "payload",
                "--page-size",
                "3",
                "--output",
                str(ROOT / "runs/elastic-lab/audit"),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        imported = json.loads(result.stdout)
        bundle = Path(imported["run"])
        verify_bundle(bundle)
        rows = details(bundle)
        summary = imported["summary"]
        actual = Counter(row["measurement"]["verdict"] for row in rows)
        deltas = sorted(
            row["measurement"]["input_delta"]
            for row in rows
            if row["measurement"]["input_delta"] is not None
        )
        recalculated = sum(row["measurement"]["execution"] == "completed" for row in rows)
        if (
            len(rows) != spec["calls"]
            or recalculated != spec["recalculated_calls"]
            or len(deltas) != spec["compared_calls"]
            or actual != spec["verdicts"]
            or deltas != spec["deltas"]
            or summary["run_status"] != "COMPLETE"
            or not summary["batch_read_complete"]
        ):
            raise ValueError(f"live ES acceptance failed for {protocol}; inspect {bundle}")
        report.append(
            {
                "protocol": protocol,
                "run": str(bundle),
                "reused": imported["reused"],
                "calls": len(rows),
                "recalculated_calls": recalculated,
                "compared_calls": len(deltas),
                "verdicts": dict(actual),
                "deltas": deltas,
                "batch_read_complete": summary["batch_read_complete"],
            }
        )
    output = ROOT / "runs/elastic-lab/audit-index.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 ES/Kibana 合成日志实验，无需 Kubernetes")
    parser.add_argument(
        "command",
        choices=["seed", "audit"],
        help="seed 准备数据；audit 是开发兼容回归，日常使用 modulo import",
    )
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--kibana-url", default="http://localhost:5601")
    args = parser.parse_args()
    for url in (args.es_url, args.kibana_url):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            parser.error("use an http(s) URL without embedded credentials")
    result = seed(args.es_url, args.kibana_url) if args.command == "seed" else audit(args.es_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
