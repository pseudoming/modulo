"""Remote Write wire format, authentication and bounded retry behavior."""

import math

import cramjam
import httpx
import pytest

from modulo.adapters.exports.prometheus.adapter import Metric, encode, send
from modulo.adapters.exports.prometheus.remote_write_pb2 import WriteRequest


def test_wire_uses_snappy_blocks_sorted_labels_and_exact_millisecond_timestamps():
    metrics = [
        Metric("modulo_calls", 123, {"scope": "审核", "model": 'a\\b"c'}),
        Metric("modulo_token_delta", float("nan"), {"scope": "审核"}),
    ]
    data = encode(metrics, 1791849600123)
    decoded = WriteRequest.FromString(bytes(cramjam.snappy.decompress_raw(data)))
    assert [(label.name, label.value) for label in decoded.timeseries[0].labels] == [
        ("__name__", "modulo_calls"),
        ("model", 'a\\b"c'),
        ("scope", "审核"),
    ]
    assert decoded.timeseries[0].samples[0].value == 123
    assert decoded.timeseries[0].samples[0].timestamp == 1791849600123
    assert math.isnan(decoded.timeseries[1].samples[0].value)
    assert all(metadata.type == 2 for metadata in decoded.metadata)


@pytest.mark.parametrize(
    "metrics,match",
    [
        ([Metric("modulo_calls", 2**53 + 1, {"scope": "a"})], "float64"),
        ([Metric("modulo_calls", float("inf"), {"scope": "a"})], "infinite"),
        ([Metric("modulo_calls", 1, {"scope": ""})], "nonempty"),
        ([Metric("modulo_calls", 1, {"scope": "a"})] * 2, "duplicate"),
    ],
)
def test_invalid_payload_is_never_silently_rounded_or_duplicated(metrics, match):
    with pytest.raises(ValueError, match=match):
        encode(metrics, 1)


def client(monkeypatch, handler):
    actual = httpx.Client
    monkeypatch.setattr(
        "modulo.adapters.exports.prometheus.adapter.httpx.Client",
        lambda **kwargs: actual(transport=httpx.MockTransport(handler), **kwargs),
    )
    sleeps = []
    monkeypatch.setattr("modulo.adapters.exports.prometheus.adapter.time.sleep", sleeps.append)
    return sleeps


def test_retries_keep_identical_payload_and_auth_then_stop_on_success(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response([503, 429, 204][len(requests) - 1])

    sleeps = client(monkeypatch, handler)
    send("http://localhost/api/v1/write", b"payload", bearer_token="private")
    assert [r.content for r in requests] == [b"payload"] * 3
    assert sleeps == [0.5, 1]
    for request in requests:
        assert request.headers["Authorization"] == "Bearer private"
        assert request.headers["Content-Encoding"] == "snappy"
        assert request.headers["X-Prometheus-Remote-Write-Version"] == "0.1.0"


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404, 413])
def test_client_errors_and_redirects_do_not_retry_or_leak_response_bodies(monkeypatch, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status, text="reflected private password", headers={"Location": "/else"}
        )

    sleeps = client(monkeypatch, handler)
    with pytest.raises(ValueError) as exc:
        send("http://localhost/write", b"payload", username="writer", password="secret")
    assert f"HTTP {status}" in str(exc.value)
    assert "private" not in str(exc.value) and "secret" not in str(exc.value)
    assert len(requests) == 1 and sleeps == []
    assert requests[0].headers["Authorization"].startswith("Basic ")


def test_transport_failure_stops_after_three_attempts(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("secret URL or header")

    sleeps = client(monkeypatch, handler)
    with pytest.raises(ValueError, match="transport failure") as exc:
        send("http://localhost/write", b"payload")
    assert sleeps == [0.5, 1]
    assert "secret" not in str(exc.value)
