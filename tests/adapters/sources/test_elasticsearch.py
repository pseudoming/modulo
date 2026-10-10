import json
from decimal import Decimal

import httpx
import pytest

from modulo.adapters.sources import ElasticsearchSource
from modulo.core.domain.data_models import SourceError
from modulo.shared.serialization import loads
from tests.support.mock_es import MockElasticsearch


def test_es_pagination_rotating_pit_and_cleanup():
    mock = MockElasticsearch([{"n": i} for i in range(5)])
    source = ElasticsearchSource(
        "https://mock.invalid", "logs-*", page_size=2, transport=mock.transport()
    )
    records = list(source.read())
    assert [json.loads(row.data)["n"] for row in records] == list(range(5))
    assert mock.searches == 4
    assert mock.closed


def test_es_retains_tool_schema_order_and_exact_numeric_types():
    record = {
        "tools": [{"function": {"name": "f", "parameters": {"z": 2, "a": 1}}}],
        "arguments": {"z": Decimal("1.2300"), "a": 2**80},
    }
    mock = MockElasticsearch([record])
    raw = next(
        ElasticsearchSource("https://mock.invalid", "logs", transport=mock.transport()).read()
    )
    restored = loads(raw.data)
    assert list(restored["tools"][0]["function"]["parameters"]) == ["z", "a"]
    assert list(restored["arguments"]) == ["z", "a"]
    assert restored == record
    assert isinstance(restored["arguments"]["z"], Decimal)


def test_es_timeout_keeps_prefix_but_raises_and_closes():
    mock = MockElasticsearch([{"n": i} for i in range(4)], fail_page=2)
    records = ElasticsearchSource(
        "https://mock.invalid", "logs", page_size=2, transport=mock.transport()
    ).read()
    assert json.loads(next(records).data)["n"] == 0
    assert json.loads(next(records).data)["n"] == 1
    with pytest.raises(SourceError, match="incomplete page"):
        next(records)
    assert mock.closed


@pytest.mark.parametrize("fault", ["shards", "total", "duplicate", "missing_source", "cursor"])
def test_es_rejects_incomplete_or_broken_pages(fault):
    mock = MockElasticsearch([{"n": 0}, {"n": 1}])

    def broken(request):
        response = mock.handle(request)
        if request.url.path == "/_search":
            data = response.json()
            if fault == "shards":
                data["_shards"]["failed"] = 1
            elif fault == "total":
                data["hits"]["total"]["relation"] = "gte"
            elif fault == "duplicate" and data["hits"]["hits"]:
                data["hits"]["hits"][1]["_id"] = "0"
            elif fault == "missing_source" and data["hits"]["hits"]:
                data["hits"]["hits"][0].pop("_source")
            elif fault == "cursor" and data["hits"]["hits"]:
                data["hits"]["hits"][-1].pop("sort")
            return httpx.Response(200, json=data)
        return response

    source = ElasticsearchSource(
        "https://mock.invalid", "logs", transport=httpx.MockTransport(broken)
    )
    with pytest.raises(SourceError):
        list(source.read())
    assert mock.closed


def test_es_errors_do_not_echo_response_data():
    transport = httpx.MockTransport(lambda request: httpx.Response(403, text="private-prompt"))
    with pytest.raises(SourceError) as captured:
        list(ElasticsearchSource("https://mock.invalid", "logs", transport=transport).read())
    assert "private-prompt" not in str(captured.value)


def test_credentials_not_allowed_in_url():
    with pytest.raises(ValueError):
        ElasticsearchSource("https://user:secret@example.invalid", "logs")
