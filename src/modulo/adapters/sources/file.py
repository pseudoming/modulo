"""Read one log file or a fixed, ordered set of matching files without altering bytes."""

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from modulo.core.domain.data_models import RawRecord, SourceError
from modulo.core.domain.interfaces import MAX_RECORD_BYTES, SourceAdapter


class FileSource(SourceAdapter):
    name = "file"
    version = "file.lines.v2"
    ordered_records = True

    def __init__(
        self,
        path: Path,
        *,
        pattern: str | None = None,
        max_record_bytes: int = MAX_RECORD_BYTES,
        allow_empty: bool = False,
    ) -> None:
        self.path = path.resolve()
        self.pattern = pattern
        self.max_record_bytes = max_record_bytes
        if pattern is None:
            if self.path.is_dir():
                raise ValueError("--pattern is required when --path is a log directory")
            paths = [self.path]
        else:
            if not self.path.is_dir():
                raise ValueError("--pattern requires a log directory")
            if not pattern.strip() or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError("--pattern must be a nonempty pattern within the log directory")
            paths = sorted(path for path in self.path.glob(pattern) if path.is_file())
            if not paths and not allow_empty:
                raise ValueError("no log files match --pattern")
        self.snapshots = {path: _snapshot(path) for path in paths}
        self.files = [(path, snapshot["sha256"]) for path, snapshot in self.snapshots.items()]

    def read_from(self, checkpoints):
        """Resume complete lines; verify the consumed prefix before accepting appends."""
        for path, expected in self.files:
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = f"{stat.st_dev}:{stat.st_ino}"
                snapshot = self.snapshots[path]
                if identity != snapshot["identity"]:
                    raise SourceError("source file was replaced after selection")
                size = snapshot["size_bytes"]
                if stat.st_size < size:
                    raise SourceError("source file was truncated after selection")
                key = "file:" + identity
                saved = checkpoints.get(key, {})
                offset = saved.get("offset", 0)
                if size < offset:
                    raise SourceError("source was truncated before its checkpoint")
                hasher = hashlib.sha256()
                left = offset
                while left:
                    block = stream.read(min(left, 1024 * 1024))
                    if not block:
                        raise SourceError("source ended before its checkpoint")
                    hasher.update(block)
                    left -= len(block)
                if saved and hasher.hexdigest() != saved["sha256"]:
                    raise SourceError("source changed before its checkpoint")
                line_no = saved.get("line", 0)
                while stream.tell() < size:
                    line = stream.readline(min(self.max_record_bytes + 1, size - stream.tell()))
                    if not line:
                        raise SourceError("source ended before its selected size")
                    line_no += 1
                    if len(line) > self.max_record_bytes:
                        raise SourceError(f"record exceeds byte limit at line:{line_no}")
                    if not line.endswith(b"\n"):
                        raise SourceError("waiting_for_complete_log_line")
                    hasher.update(line)
                    if line.strip():
                        yield RawRecord(
                            f"file:{path.as_posix()}@{identity}:line:{line_no}",
                            line,
                            {
                                key: {
                                    "offset": stream.tell(),
                                    "line": line_no,
                                    "sha256": hasher.hexdigest(),
                                }
                            },
                        )
                if hasher.hexdigest() != expected:
                    raise SourceError(f"source file changed during import: {path.name}")

    def describe(self) -> dict[str, Any]:
        description = {
            "kind": "file",
            "version": self.version,
            "path": str(self.path),
            "format": "line_records",
        }
        if self.pattern is None:
            description["sha256"] = self.files[0][1]
            description["size_bytes"] = self.snapshots[self.path]["size_bytes"]
        else:
            description.update(
                pattern=self.pattern,
                order="relative_path_ascending",
                files=[
                    {
                        "path": path.relative_to(self.path).as_posix(),
                        "sha256": expected,
                        "size_bytes": self.snapshots[path]["size_bytes"],
                    }
                    for path, expected in self.files
                ],
            )
        return description

    def read(self) -> Iterator[RawRecord]:
        for path, expected in self.files:
            prefix = f"file:{path.relative_to(self.path).as_posix()}:" if self.pattern else ""
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                line_no = 0
                size = self.snapshots[path]["size_bytes"]
                while stream.tell() < size:
                    line = stream.readline(min(self.max_record_bytes + 1, size - stream.tell()))
                    if not line:
                        raise SourceError("source ended before its selected size")
                    line_no += 1
                    locator = f"{prefix}line:{line_no}"
                    if len(line) > self.max_record_bytes:
                        raise SourceError(f"record exceeds byte limit at {locator}")
                    hasher.update(line)
                    if line.strip():
                        yield RawRecord(locator, line)
            if hasher.hexdigest() != expected:
                raise SourceError(f"source file changed during import: {path.name}")


def _snapshot(path):
    """Hash a fixed prefix so a live file can keep receiving new lines."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        left = stat.st_size
        while left:
            block = stream.read(min(left, 1024 * 1024))
            if not block:
                raise OSError("source file changed while selecting its snapshot")
            hasher.update(block)
            left -= len(block)
    return {
        "sha256": hasher.hexdigest(),
        "size_bytes": stat.st_size,
        "identity": f"{stat.st_dev}:{stat.st_ino}",
    }
