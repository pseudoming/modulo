"""Synthetic image HTTP responses, without making any socket connections."""

import io
import socket

import httpx
from PIL import Image


def png_bytes(size=(640, 480)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, "PNG")
    return buffer.getvalue()


def mock_image_http(monkeypatch, handler=None):
    requests = []

    def send(_transport, request):
        requests.append(request)
        if handler:
            return handler(request)
        return httpx.Response(
            200, headers={"content-type": "image/png"}, stream=httpx.ByteStream(png_bytes())
        )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", send)
    return requests
