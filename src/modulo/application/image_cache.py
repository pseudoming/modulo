"""Opt-in URL image retrieval and bounded, expiring local cache for one import."""

import errno
import ipaddress
import os
import re
import socket
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modulo.adapters.protocols.shared.messages.images import image_header
from modulo.core.domain.data_models import ImageDownload
from modulo.shared.serialization import bytes_digest, dumps, loads


class ImageDownloadOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, alias_generator=lambda name: name.replace("_", "-")
    )

    enabled: bool = False
    cache_dir: Path = Field(
        default_factory=lambda: (
            Path(tempfile.gettempdir()) / f"modulo-images-{getattr(os, 'getuid', lambda: 'user')()}"
        )
    )
    max_file_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_cache_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    ttl_seconds: int = Field(default=86400, gt=0)
    allow_private_hosts: list[str] = Field(default_factory=list)

    @field_validator("allow_private_hosts")
    @classmethod
    def hostnames(cls, values):
        hosts = []
        for value in values:
            if (
                not value.rstrip(".")
                or any(char in value for char in "/\\*:@?#[]")
                or any(char.isspace() for char in value)
            ):
                raise ValueError(
                    "allow-private-hosts requires exact hostnames, without URLs or ports"
                )
            hosts.append(value.rstrip(".").encode("idna").decode("ascii").lower())
        return sorted(set(hosts))

    @field_validator("enabled", mode="before")
    @classmethod
    def boolean(cls, value):
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if type(value) is not bool:
            raise ValueError("image-download.enabled must be true or false")
        return value

    @field_validator("max_file_bytes", "max_cache_bytes", "ttl_seconds", mode="before")
    @classmethod
    def integer(cls, value):
        if isinstance(value, str):
            return int(value)
        if type(value) is not int:
            raise ValueError("image download limits must be integers")
        return value

    @model_validator(mode="after")
    def limits(self):
        if self.max_cache_bytes < self.max_file_bytes:
            raise ValueError("max-cache-bytes must be at least max-file-bytes")
        return self


class ImageDownloadError(ValueError):
    """Fixed reason codes only; never include URL query credentials or body text."""


def _target(value, allow_private_hosts):
    try:
        url = httpx.URL(value)
        if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
            raise ImageDownloadError("invalid_image_url")
        addresses = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(
                url.host,
                url.port or (443 if url.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
        if not addresses or (
            url.raw_host.decode("ascii").lower().rstrip(".") not in allow_private_hosts
            and any(
                not address.is_global or address.is_multicast or address.is_reserved
                for address in addresses
            )
        ):
            raise ImageDownloadError("image_url_address_not_public")
        # Pin the checked IP, retaining the original HTTP Host and TLS identity.
        address = min(addresses, key=lambda item: (item.version, str(item)))
        return url, url.copy_with(host=str(address), fragment=None)
    except httpx.InvalidURL as exc:
        raise ImageDownloadError("invalid_image_url") from exc


def download_image(value, max_bytes, allow_private_hosts=()):
    """Bound streamed bytes; validate every redirect without proxy or cookie inheritance."""
    deadline = time.monotonic() + 30
    try:
        for hop in range(4):
            url, target = _target(value, allow_private_hosts)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ImageDownloadError("image_download_timeout")
            # A new connection per hop also prevents TLS sessions being reused
            # between different hostnames that happen to resolve to the same IP.
            with httpx.Client(trust_env=False, follow_redirects=False) as client:
                with client.stream(
                    "GET",
                    target,
                    headers={"Host": url.netloc.decode("ascii"), "Accept-Encoding": "identity"},
                    extensions={"sni_hostname": url.raw_host.decode("ascii")},
                    timeout=min(10, remaining),
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or hop == 3:
                            raise ImageDownloadError("image_redirect_limit_or_missing_location")
                        value = str(url.join(location))
                        if url.scheme == "https" and httpx.URL(value).scheme != "https":
                            raise ImageDownloadError("image_redirect_downgrade")
                        continue
                    if response.status_code != 200:
                        raise ImageDownloadError(f"image_http_status:{response.status_code}")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ImageDownloadError("image_content_encoding_unsupported")
                    content_type = response.headers.get("content-type", "").split(";")[0].lower()
                    if content_type and not (
                        content_type.startswith("image/")
                        or content_type == "application/octet-stream"
                    ):
                        raise ImageDownloadError("image_content_type_unsupported")
                    length = response.headers.get("content-length")
                    if length is not None and (not length.isdecimal() or int(length) > max_bytes):
                        raise ImageDownloadError("image_download_too_large")
                    data = bytearray()
                    for chunk in response.iter_raw():
                        if time.monotonic() > deadline:
                            raise ImageDownloadError("image_download_timeout")
                        if len(data) + len(chunk) > max_bytes:
                            raise ImageDownloadError("image_download_too_large")
                        data.extend(chunk)
                    if length is not None and len(data) != int(length):
                        raise ImageDownloadError("image_download_incomplete")
                    return bytes(data), str(url)
    except httpx.TimeoutException as exc:
        raise ImageDownloadError("image_download_timeout") from exc
    except (httpx.RequestError, OSError) as exc:
        raise ImageDownloadError("image_download_network_error") from exc


class ImageCache:
    def __init__(self, options: ImageDownloadOptions):
        self.options = options
        self.root = options.cache_dir.absolute()

    def __enter__(self):
        if self.options.enabled:
            if self.root.is_symlink():
                raise ValueError("image cache directory must not be a symlink")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.cleanup()
        return self

    def __exit__(self, *_):
        if self.options.enabled:
            self.cleanup()

    @staticmethod
    def _remove(directory):
        # Only our two files, never recursive deletion of a user-supplied path.
        for name in ("image", "metadata.json"):
            (directory / name).unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass

    def cleanup(self):
        now, entries = time.time(), []
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            pending = directory.name.startswith(".pending-")
            if not pending and not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                continue
            try:
                age = now - directory.stat().st_mtime
                if age >= self.options.ttl_seconds:
                    self._remove(directory)
                elif not pending:
                    size = sum(
                        file.lstat().st_size
                        for file in (directory / "image", directory / "metadata.json")
                        if file.exists() or file.is_symlink()
                    )
                    entries.append((directory.stat().st_mtime, directory.name, size, directory))
            except FileNotFoundError:
                continue  # Another importer may have just evicted this entry.
        total = sum(entry[2] for entry in entries)
        for _, _, size, directory in sorted(entries):
            if total <= self.options.max_cache_bytes:
                break
            self._remove(directory)
            total -= size

    def get(self, url):
        if not self.options.enabled:
            raise ImageDownloadError("image_download_disabled")
        key = bytes_digest(url.encode("utf-8"))
        directory = self.root / key
        if not directory.is_symlink():
            try:
                files = (directory / "image", directory / "metadata.json")
                if any(file.is_symlink() or not file.is_file() for file in files):
                    raise ValueError("invalid cache files")
                with files[1].open("rb") as stream:
                    info = ImageDownload.model_validate(loads(stream.read(8193)))
                with files[0].open("rb") as stream:
                    data = stream.read(self.options.max_file_bytes + 1)
                age = time.time() - datetime.fromisoformat(info.downloaded_at).timestamp()
                if (
                    info.url_sha256 == key
                    and info.cache_file == f"{key}/image"
                    and 0 <= age < self.options.ttl_seconds
                    and len(data) == info.byte_size <= self.options.max_file_bytes
                    and bytes_digest(data) == info.sha256
                ):
                    return data, info
            except (OSError, ValueError):
                pass
            self._remove(directory)
        else:
            raise ImageDownloadError("image_cache_entry_is_symlink")

        data, final_url = download_image(
            url, self.options.max_file_bytes, self.options.allow_private_hosts
        )
        try:
            image_header(data)
        except ValueError as exc:
            raise ImageDownloadError("invalid_downloaded_image") from exc
        info = ImageDownload(
            url_sha256=key,
            final_url_sha256=bytes_digest(final_url.encode("utf-8")),
            sha256=bytes_digest(data),
            byte_size=len(data),
            downloaded_at=datetime.now(UTC).isoformat(),
            cache_file=f"{key}/image",
        )
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=self.root) as temporary:
            staging = Path(temporary)
            (staging / "image").write_bytes(data)
            (staging / "metadata.json").write_text(dumps(info.model_dump()), encoding="utf-8")
            try:
                staging.rename(directory)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                # Another import populated this URL first; use its verified bytes.
                return self.get(url)
        self.cleanup()
        return data, info

    def prepare(self, call):
        if (
            not self.options.enabled
            or call.gaps
            or not call.messages
            or any(message.gaps for message in call.messages)
        ):
            return call
        messages = []
        for message in call.messages:
            parts = []
            for part in message.content_parts:
                if (
                    part.type != "image"
                    or not part.url
                    or (part.width is not None and part.height is not None)
                    or part.sha256
                    or "image_grid_thw" in part.options
                    or set(part.gaps) - {"incomplete_image_dimensions"}
                    or part.options.get("detail") not in (None, "auto")
                ):
                    parts.append(part)
                    continue
                try:
                    data, info = self.get(part.url)
                    header = image_header(data)
                    if any(getattr(part, k) not in (None, header[k]) for k in ("width", "height")):
                        header["gaps"].append("conflicting_image_dimensions")
                    part = part.model_copy(
                        update={
                            **header,
                            "download": info,
                            "dimensions_source": f"{part.url_source}:downloaded_image_header",
                        }
                    )
                except ImageDownloadError as exc:
                    part = part.model_copy(update={"gaps": [*part.gaps, str(exc)]})
                except (OSError, ValueError):
                    part = part.model_copy(update={"gaps": [*part.gaps, "image_cache_error"]})
                parts.append(part)
            messages.append(message.model_copy(update={"content_parts": parts}))
        return call.model_copy(update={"messages": messages})
