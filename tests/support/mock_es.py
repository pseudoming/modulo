"""Test-only ES responses, injected into the real reader without a running server."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx

from modulo.adapters.sources import ElasticsearchSource
from modulo.shared.serialization import loads, ordered_json


class MockElasticsearch:
    def __init__(self, records: list[dict], *, fail_page: int | None = None) -> None:
        self.records, self.fail_page = records, fail_page
        self.searches = 0
        self.closed = False
        self.pit_id = "mock-pit-0"

    @classmethod
    def from_file(cls, path: Path) -> "MockElasticsearch":
        records = [loads(line) for line in path.read_bytes().splitlines() if line.strip()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("mock ES input requires JSON objects")
        return cls(records)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def patch_source(self):
        """Inject this transport for in-process CLI tests; restore on context exit."""
        original_init = ElasticsearchSource.__init__

        def mock_init(source, *args, **kwargs):
            original_init(source, *args, **kwargs, transport=self.transport())

        return patch.object(ElasticsearchSource, "__init__", mock_init)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/_pit"):
            return httpx.Response(200, json={"id": self.pit_id})
        body = json.loads(request.content) if request.content else {}
        if request.method == "DELETE" and request.url.path == "/_pit":
            if body.get("id") != self.pit_id:
                return httpx.Response(400, json={"error": "stale PIT"})
            self.closed = True
            return httpx.Response(200, json={"succeeded": True, "num_freed": 1})
        if request.method == "POST" and request.url.path == "/_search":
            if body.get("pit", {}).get("id") != self.pit_id:
                return httpx.Response(400, json={"error": "stale PIT"})
            if body.get("query") != {"match_all": {}}:
                return httpx.Response(400, json={"error": "mock supports match_all only"})
            self.searches += 1
            self.pit_id = f"mock-pit-{self.searches}"
            offset = body.get("search_after", [-1])[0] + 1
            size = body["size"]
            rows = [
                {"_index": "mock-logs", "_id": str(i), "_source": self.records[i], "sort": [i]}
                for i in range(offset, min(offset + size, len(self.records)))
            ]
            return httpx.Response(
                200,
                content=ordered_json(
                    {
                        "pit_id": self.pit_id,
                        "timed_out": self.searches == self.fail_page,
                        "_shards": {"total": 1, "successful": 1, "failed": 0},
                        "hits": {
                            "total": {"value": len(self.records), "relation": "eq"},
                            "hits": rows,
                        },
                    },
                ).encode(),
            )
        return httpx.Response(404, json={"error": "unsupported mock operation"})
