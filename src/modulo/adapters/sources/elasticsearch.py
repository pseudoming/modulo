"""Elasticsearch PIT and search_after reader."""

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from modulo.core.domain.data_models import RawRecord, SourceError
from modulo.core.domain.interfaces import MAX_RECORD_BYTES, SourceAdapter
from modulo.shared.serialization import digest, loads, ordered_json


class ElasticsearchSource(SourceAdapter):
    """One bounded PIT + search_after batch; does not tail or mutate the index."""

    name = "es"
    version = "es.pit.v2"

    def __init__(
        self,
        url: str,
        index: str,
        *,
        query: dict[str, Any] | None = None,
        page_size: int = 500,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("ES URL requires http(s), without credentials, query, or fragment")
        if not index or not 1 <= page_size <= 10_000:
            raise ValueError("index is required; page_size must be 1..10000")
        if (username is None) != (password is None) or username == "" or password == "":
            raise ValueError("ES username and password must both be provided and nonempty")
        if api_key is not None and username is not None:
            raise ValueError("ES accepts either username/password or API key, not both")
        if api_key == "":
            raise ValueError("ES API key must be nonempty")
        self.url, self.index = url.rstrip("/"), index
        self.query, self.page_size = query or {"match_all": {}}, page_size
        self.api_key, self.transport = api_key, transport
        self.auth = httpx.BasicAuth(username, password) if username is not None else None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "es",
            "version": self.version,
            "url": self.url,
            "index": self.index,
            "query_json": json.dumps(self.query, sort_keys=True),
            "page_size": self.page_size,
            "snapshot": "point_in_time",
            "mock": self.transport is not None,
        }

    def read(self) -> Iterator[RawRecord]:
        yield from self._read()

    def read_from(self, checkpoints):
        yield from self._read(checkpoints)

    def _read(self, checkpoints=None) -> Iterator[RawRecord]:
        resume_key = "es:" + digest([self.url, self.index, self.query])
        saved = checkpoints.get(resume_key, {}) if checkpoints is not None else {}
        headers = {"Authorization": f"ApiKey {self.api_key}"} if self.api_key else {}
        with httpx.Client(
            base_url=self.url + "/",
            headers=headers,
            auth=self.auth,
            timeout=30,
            transport=self.transport,
            follow_redirects=False,
        ) as client:
            pit_id = saved.get("pit_id")
            failed = complete = False
            resume_allowed = bool(pit_id)
            seen = getattr(self, "seen", {})

            def request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
                with client.stream(method, path, **kwargs) as response:
                    if response.status_code == 404 and resume_allowed and path == "_search":
                        raise ExpiredPIT()
                    if not response.is_success:
                        raise SourceError(
                            f"ES {method} request failed: HTTP {response.status_code}"
                        )
                    content = bytearray()
                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        content.extend(chunk)
                        if len(content) > getattr(self, "max_page_bytes", 64 * 1024 * 1024):
                            raise SourceError("ES page exceeds memory budget; reduce page-size")
                data = loads(bytes(content))
                if not isinstance(data, dict):
                    raise SourceError("ES response is not an object")
                return data

            def open_pit():
                opened = request(
                    "POST",
                    quote(self.index, safe=",*-") + "/_pit",
                    params={"keep_alive": "2m", "allow_partial_search_results": "false"},
                )
                value = opened.get("id")
                if not isinstance(value, str) or not value:
                    raise SourceError("ES did not return PIT id")
                seen.clear()
                return value

            try:
                if not pit_id:
                    pit_id = open_pit()
                after = saved.get("after")
                observed = saved.get("observed", 0)
                total = saved.get("total")
                while True:
                    body = {
                        "pit": {"id": pit_id, "keep_alive": "2m"},
                        "size": self.page_size,
                        "sort": ["_shard_doc"],
                        "query": self.query,
                        "track_total_hits": True,
                    }
                    if after is not None:
                        body["search_after"] = after
                    try:
                        data = request(
                            "POST",
                            "_search",
                            params={"allow_partial_search_results": "false"},
                            json=body,
                        )
                    except ExpiredPIT:
                        resume_allowed = False
                        checkpoints.pop(resume_key, None)
                        pit_id, after, observed, total = open_pit(), None, 0, None
                        continue
                    resume_allowed = False
                    pit_id = data.get("pit_id", pit_id)
                    if (
                        not isinstance(pit_id, str)
                        or not pit_id
                        or data.get("timed_out") is not False
                        or data.get("terminated_early")
                        or data.get("_shards", {}).get("failed") != 0
                    ):
                        raise SourceError(
                            "ES incomplete page: timeout, shard failure, or invalid PIT"
                        )
                    hits = data.get("hits", {})
                    count = hits.get("total", {})
                    if (
                        count.get("relation") != "eq"
                        or type(count.get("value")) is not int
                        or count["value"] < 0
                    ):
                        raise SourceError("ES total hit count is not exact")
                    if total is not None and total != count["value"]:
                        raise SourceError("ES PIT total changed")
                    total = count["value"]
                    rows = hits.get("hits")
                    if not isinstance(rows, list):
                        raise SourceError("ES hits must be an array")
                    if not rows:
                        if observed != total:
                            raise SourceError("ES pages ended before the declared total")
                        complete = True
                        if checkpoints is not None:
                            checkpoints.pop(resume_key, None)
                            seen.clear()
                        break
                    for hit in rows:
                        index, record_id = hit.get("_index"), hit.get("_id")
                        if not isinstance(index, str) or not isinstance(record_id, str):
                            raise SourceError("ES hit missing identity")
                        key = (index, record_id)
                        if key in seen:
                            raise SourceError("ES returned a duplicate hit in the PIT")
                        observed += 1
                        if observed > total:
                            raise SourceError("ES returned more hits than declared")
                        if "_source" not in hit:
                            raise SourceError("ES _source is unavailable")
                        # Prompt structures retain field order and numeric types.
                        # Canonical artifact JSON would reorder tool schemas and
                        # turn decimal arguments into strings before extraction.
                        payload = ordered_json(hit["_source"]).encode()
                        if len(payload) > getattr(self, "max_record_bytes", MAX_RECORD_BYTES):
                            raise SourceError("ES record exceeds byte limit")
                        sort = hit.get("sort")
                        if checkpoints is not None and (not isinstance(sort, list) or not sort):
                            raise SourceError("ES hit missing resumable sort values")
                        seen[key] = True
                        locator = f"{quote(index, safe='')}/{quote(record_id, safe='')}"
                        if checkpoints is not None:
                            locator = f"es:{digest(self.url)[:16]}:" + locator
                        yield RawRecord(
                            locator,
                            payload,
                            {
                                resume_key: {
                                    "pit_id": pit_id,
                                    "after": sort,
                                    "observed": observed,
                                    "total": total,
                                }
                            }
                            if checkpoints is not None
                            else None,
                        )
                    next_after = rows[-1].get("sort")
                    if not isinstance(next_after, list) or not next_after or next_after == after:
                        raise SourceError("ES search_after did not advance")
                    after = next_after
            except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
                failed = True
                raise SourceError(f"ES read failed ({type(exc).__name__})") from exc
            except BaseException:
                failed = True
                raise
            finally:
                if checkpoints is not None and not complete and isinstance(pit_id, str) and pit_id:
                    confirmed = checkpoints.get(resume_key)
                    if confirmed and confirmed["pit_id"] != pit_id:
                        # A rejected page can rotate the PIT without advancing a
                        # single committed hit. Keep its ID, retain the old cursor.
                        checkpoints[resume_key] = {**confirmed, "pit_id": pit_id}
                if pit_id and (checkpoints is None or complete):
                    try:
                        closed = request("DELETE", "_pit", json={"id": pit_id})
                        if closed.get("succeeded") is not True:
                            raise SourceError("ES PIT cleanup was not acknowledged")
                    except Exception as exc:
                        if not failed:
                            raise SourceError("ES PIT cleanup failed") from exc


class ExpiredPIT(SourceError):
    """A saved snapshot expired; restart the scan and let the journal deduplicate."""
