import hashlib
import os
import socket
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from modulo.adapters.protocols.shared.messages.chat import message
from modulo.application import image_cache
from modulo.application.image_cache import (
    ImageCache,
    ImageDownloadError,
    ImageDownloadOptions,
    download_image,
)
from modulo.core.domain.data_models import Call
from modulo.shared.serialization import dumps, loads
from tests.support.image_http import mock_image_http, png_bytes

URL = "https://images.example.test/photo.png?signature=not-a-real-secret"


def image_call(**block):
    return Call(
        call_id="image-test",
        protocol="litellm",
        messages=[
            message(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": URL}, **block},
                    ],
                }
            )
        ],
    )


def options(tmp_path, **kwargs):
    return ImageDownloadOptions(enabled=True, cache_dir=tmp_path / "images", **kwargs)


def test_disabled_creates_no_directory_and_does_not_resolve_or_connect(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("disabled downloads must not perform I/O")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    config = options(tmp_path).model_copy(update={"enabled": False})
    call = image_call()
    with ImageCache(config) as cache:
        assert cache.prepare(call) is call
        with pytest.raises(ImageDownloadError, match="image_download_disabled"):
            cache.get(URL)
    assert not config.cache_dir.exists()


@pytest.mark.parametrize("scope", ["call", "message"])
def test_known_input_gaps_do_not_trigger_downloads(tmp_path, monkeypatch, scope):
    requests = mock_image_http(monkeypatch)
    call = image_call()
    if scope == "call":
        call = call.model_copy(update={"gaps": ["missing_request"]})
    else:
        call = call.model_copy(
            update={
                "messages": [
                    call.messages[0].model_copy(
                        update={"gaps": ["attachment_content_order_missing"]}
                    ),
                    call.messages[0],
                ]
            }
        )
    with ImageCache(options(tmp_path)) as cache:
        assert cache.prepare(call) is call
    assert requests == []


def test_download_reuse_and_corruption_recovery_preserve_evidence(tmp_path, monkeypatch):
    requests = mock_image_http(monkeypatch)
    call = image_call()
    with ImageCache(options(tmp_path)) as cache:
        prepared = cache.prepare(call)
        image = prepared.messages[0].content_parts[0]
        assert (image.width, image.height) == (640, 480)
        assert image.sha256 == hashlib.sha256(png_bytes()).hexdigest()
        assert image.download.historical_content == "unverified"
        assert image.download.url_sha256 == hashlib.sha256(URL.encode()).hexdigest()
        assert image.download.byte_size == len(png_bytes())
        datetime.fromisoformat(image.download.downloaded_at)
        assert image.dimensions_source.endswith(":downloaded_image_header")
        assert image.url == URL and call.messages[0].content_parts[0].width is None
        cached = cache.root / image.download.cache_file
        assert cached.read_bytes() == png_bytes()
        assert cache.prepare(call) == prepared
        assert len(requests) == 1
        assert requests[0].url.host == "93.184.216.34"
        assert requests[0].headers["host"] == "images.example.test"
        assert requests[0].extensions["sni_hostname"] == "images.example.test"
        assert "authorization" not in requests[0].headers and "cookie" not in requests[0].headers
        assert not list(cache.root.glob(".pending-*"))
        assert "not-a-real-secret" not in (cached.parent / "metadata.json").read_text()
        cached.write_bytes(b"corrupt")
        assert cache.prepare(call).messages[0].content_parts[0].sha256 == image.sha256
        assert len(requests) == 2


@pytest.mark.parametrize(
    "block",
    [
        {"width": 640, "height": 480},
        {"detail": "high"},
        {"image_grid_thw": [1, 32, 40]},
        {"width": -1},
    ],
)
def test_sufficient_or_unusable_image_evidence_is_not_downloaded(tmp_path, monkeypatch, block):
    requests = mock_image_http(monkeypatch)
    with ImageCache(options(tmp_path)) as cache:
        call = image_call(**block)
        assert cache.prepare(call) == call
    assert requests == []


@pytest.mark.parametrize("width,expected", [(640, []), (1, ["conflicting_image_dimensions"])])
def test_partial_dimensions_are_completed_without_hiding_conflicts(
    tmp_path, monkeypatch, width, expected
):
    mock_image_http(monkeypatch)
    with ImageCache(options(tmp_path)) as cache:
        image = cache.prepare(image_call(width=width)).messages[0].content_parts[0]
        assert image.download is not None
        assert image.gaps == expected


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "/etc/passwd",
        "ftp://example.test/i",
        "https://user:password@example.test/i",
    ],
)
def test_unsupported_url_never_resolves(tmp_path, monkeypatch, url):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid URL must not resolve")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    with pytest.raises(ImageDownloadError, match="invalid_image_url"):
        download_image(url, 100)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
    ],
)
def test_private_and_reserved_destinations_are_blocked_before_http(monkeypatch, address):
    requests = mock_image_http(monkeypatch)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)),
        ],
    )
    with pytest.raises(ImageDownloadError, match="image_url_address_not_public"):
        download_image(URL, 10000)
    assert requests == []


def test_explicit_host_exception_supports_private_dns_without_wildcards(monkeypatch):
    requests = mock_image_http(monkeypatch)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.0.2", 443)),
        ],
    )
    allowed = ImageDownloadOptions(allow_private_hosts=["IMAGES.EXAMPLE.TEST."])
    assert allowed.allow_private_hosts == ["images.example.test"]
    data, _ = download_image(URL, 10000, allowed.allow_private_hosts)
    assert data == png_bytes()
    assert requests[0].url.host == "198.18.0.2"
    assert requests[0].extensions["sni_hostname"] == "images.example.test"
    with pytest.raises(ImageDownloadError, match="image_url_address_not_public"):
        download_image(
            "https://images.example.test.evil.test/i", 10000, allowed.allow_private_hosts
        )
    assert len(requests) == 1


def test_host_exception_is_rechecked_after_redirect(monkeypatch):
    requests = mock_image_http(
        monkeypatch,
        lambda _request: httpx.Response(
            302,
            headers={"location": "https://other.example.test/i"},
            stream=httpx.ByteStream(b""),
        ),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.0.2", 443)),
        ],
    )
    with pytest.raises(ImageDownloadError, match="image_url_address_not_public"):
        download_image(URL, 10000, ["images.example.test"])
    assert len(requests) == 1


def test_redirect_keeps_pinned_tls_identity_and_does_not_forward_cookies(monkeypatch):
    def respond(request):
        if request.headers["host"] == "images.example.test":
            return httpx.Response(
                302,
                headers={
                    "location": "https://cdn.example.test/final.png",
                    "set-cookie": "credential=secret; Domain=.example.test",
                },
                stream=httpx.ByteStream(b""),
            )
        return httpx.Response(200, stream=httpx.ByteStream(png_bytes()))

    requests = mock_image_http(monkeypatch, respond)
    data, final = download_image(URL, 10000)
    assert data == png_bytes() and final == "https://cdn.example.test/final.png"
    assert len(requests) == 2 and "cookie" not in requests[1].headers
    assert requests[1].extensions["sni_hostname"] == "cdn.example.test"


@pytest.mark.parametrize(
    "location,reason",
    [
        ("http://cdn.example.test/i", "image_redirect_downgrade"),
        ("https://internal.example.test/i", "image_url_address_not_public"),
        (URL, "image_redirect_limit"),
    ],
)
def test_unsafe_or_looping_redirects_fail(monkeypatch, location, reason):
    requests = mock_image_http(
        monkeypatch,
        lambda _request: httpx.Response(
            302, headers={"location": location}, stream=httpx.ByteStream(b"")
        ),
    )
    public_resolve = socket.getaddrinfo
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: (
            [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]
            if host == "internal.example.test"
            else public_resolve(host, *args, **kwargs)
        ),
    )
    with pytest.raises(ImageDownloadError, match=reason):
        download_image(URL, 10000)
    assert len(requests) <= 4


@pytest.mark.parametrize(
    "status,headers,data,limit,reason",
    [
        (404, {}, b"", 10000, "image_http_status:404"),
        (200, {"content-type": "text/html"}, b"<html>", 10000, "image_content_type_unsupported"),
        (200, {"content-encoding": "gzip"}, b"", 10000, "image_content_encoding_unsupported"),
        (200, {"content-length": "100000"}, b"", 10000, "image_download_too_large"),
        (200, {}, b"x" * 200, 100, "image_download_too_large"),
        (200, {"content-length": "10"}, b"x", 100, "image_download_incomplete"),
        (200, {}, b"not an image", 100, "invalid_downloaded_image"),
    ],
)
def test_download_failures_have_precise_gaps_and_no_cached_partial_files(
    tmp_path, monkeypatch, status, headers, data, limit, reason
):
    mock_image_http(
        monkeypatch,
        lambda _request: httpx.Response(status, headers=headers, stream=httpx.ByteStream(data)),
    )
    with ImageCache(options(tmp_path, max_file_bytes=limit)) as cache:
        part = cache.prepare(image_call()).messages[0].content_parts[0]
        assert part.gaps == [reason] and part.width is None and part.download is None
        assert list(cache.root.iterdir()) == []


def test_timeout_does_not_expose_url_secrets(tmp_path, monkeypatch):
    def timeout(_request):
        raise httpx.ReadTimeout(URL)

    mock_image_http(monkeypatch, timeout)
    with ImageCache(options(tmp_path)) as cache:
        part = cache.prepare(image_call()).messages[0].content_parts[0]
        assert part.gaps == ["image_download_timeout"]


def test_expiry_and_oldest_first_size_cleanup_leave_unrelated_files(tmp_path, monkeypatch):
    mock_image_http(monkeypatch)
    config = options(tmp_path, max_file_bytes=10000, max_cache_bytes=10000, ttl_seconds=30)
    with ImageCache(config) as cache:
        unrelated = cache.root / "keep.txt"
        unrelated.write_text("user file")
        _, first = cache.get(URL)
        first_file = cache.root / first.cache_file
        os.utime(first_file.parent, (time.time() - 10, time.time() - 10))
        for i in range(5):
            cache.get(URL + f"&image={i}")
        assert not first_file.exists()
        assert sum(f.stat().st_size for f in cache.root.glob("*/*")) <= config.max_cache_bytes
        for directory in cache.root.iterdir():
            if directory.is_dir():
                os.utime(directory, (time.time() - 31, time.time() - 31))
        cache.cleanup()
        assert list(cache.root.iterdir()) == [unrelated]


def test_expired_metadata_is_not_reused_even_when_directory_was_touched(tmp_path, monkeypatch):
    requests = mock_image_http(monkeypatch)
    with ImageCache(options(tmp_path, ttl_seconds=30)) as cache:
        _, info = cache.get(URL)
        metadata = (cache.root / info.cache_file).parent / "metadata.json"
        old = loads(metadata.read_bytes())
        old["downloaded_at"] = (datetime.now(UTC) - timedelta(seconds=31)).isoformat()
        metadata.write_text(dumps(old))
        _, fresh = cache.get(URL)
        assert fresh.downloaded_at != old["downloaded_at"] and len(requests) == 2


def test_cleanup_on_exit_and_symlinks_do_not_delete_targets(tmp_path, monkeypatch):
    mock_image_http(monkeypatch)
    unrelated = tmp_path / "outside"
    unrelated.mkdir()
    (unrelated / "image").write_text("keep")
    with ImageCache(options(tmp_path, ttl_seconds=30)) as cache:
        link = cache.root / ("a" * 64)
        link.symlink_to(unrelated, target_is_directory=True)
        _, info = cache.get(URL)
        directory = (cache.root / info.cache_file).parent
        os.utime(directory, (time.time() - 31, time.time() - 31))
        assert directory.exists()
    assert not directory.exists()
    assert (unrelated / "image").read_text() == "keep"
    assert link.is_symlink()


def test_write_failure_removes_temporary_files_and_keeps_gap(tmp_path, monkeypatch):
    mock_image_http(monkeypatch)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    with ImageCache(options(tmp_path)) as cache:
        monkeypatch.setattr(image_cache.Path, "write_bytes", fail)
        part = cache.prepare(image_call()).messages[0].content_parts[0]
        assert part.gaps == ["image_cache_error"]
        assert list(cache.root.iterdir()) == []
