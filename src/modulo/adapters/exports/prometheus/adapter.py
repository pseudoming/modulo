"""Encode snapshot gauges with the official Remote Write wire schema and send them."""

import math
import ssl
import time
from dataclasses import dataclass

import cramjam
import httpx

from modulo import __version__

from .remote_write_pb2 import WriteRequest


@dataclass(frozen=True)
class Metric:
    name: str
    value: int | float
    labels: dict[str, str]


def encode(metrics: list[Metric], timestamp_ms: int) -> bytes:
    """One sample per series; retain integers until the required float64 wire boundary."""
    request = WriteRequest()
    seen = set()
    for metric in metrics:
        labels = tuple(sorted({"__name__": metric.name, **metric.labels}.items()))
        if labels in seen:
            raise ValueError("duplicate Prometheus series")
        seen.add(labels)
        if any(not key or not value or len(value.encode()) > 1024 for key, value in labels):
            raise ValueError("Prometheus labels must be nonempty and at most 1024 UTF-8 bytes")
        if isinstance(metric.value, int) and abs(metric.value) > 2**53:
            raise ValueError("metric integer exceeds exact float64 range; use the result bundle")
        if math.isinf(metric.value):
            raise ValueError("infinite Prometheus metric")
        series = request.timeseries.add()
        for key, value in labels:
            series.labels.add(name=key, value=value)
        series.samples.add(value=metric.value, timestamp=timestamp_ms)
    for name in sorted({metric.name for metric in metrics}):
        request.metadata.add(type=2, metric_family_name=name, help="Modulo result snapshot gauge")
    return bytes(cramjam.snappy.compress_raw(request.SerializeToString()))


def send(
    url: str,
    payload: bytes,
    *,
    username: str | None = None,
    password: str | None = None,
    bearer_token: str | None = None,
    ca_file: str | None = None,
    timeout_seconds: float = 30,
) -> None:
    """Bounded retry of the same samples; never log credentials or reflected response bodies."""
    headers = {
        "Content-Encoding": "snappy",
        "Content-Type": "application/x-protobuf",
        "User-Agent": f"modulo/{__version__}",
        "X-Prometheus-Remote-Write-Version": "0.1.0",
    }
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    context = ssl.create_default_context(cafile=ca_file) if ca_file else True
    with httpx.Client(
        auth=(username, password) if username is not None else None,
        headers=headers,
        verify=context,
        timeout=timeout_seconds,
        follow_redirects=False,
    ) as client:
        for attempt in range(3):
            try:
                # Do not buffer an untrusted error response (it may echo secrets).
                with client.stream("POST", url, content=payload) as response:
                    status = response.status_code
                if 200 <= status < 300:
                    return
                retry = status == 429 or status >= 500
                message = f"Prometheus Remote Write returned HTTP {status}"
                if status == 404:
                    message += "; check the URL and --web.enable-remote-write-receiver"
            except httpx.TransportError as exc:
                retry = True
                message = f"Prometheus Remote Write transport failure ({type(exc).__name__})"
            except httpx.HTTPError as exc:
                raise ValueError("invalid Prometheus HTTP request") from exc
            if not retry or attempt == 2:
                raise ValueError(message + "; export not confirmed, some samples may be stored")
            time.sleep(0.5 * 2**attempt)
