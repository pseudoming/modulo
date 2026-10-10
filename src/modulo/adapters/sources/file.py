"""Read one log file or a fixed, ordered set of matching files without altering bytes."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from modulo.core.domain.data_models import RawRecord, SourceError
from modulo.core.domain.interfaces import MAX_RECORD_BYTES, SourceAdapter
from modulo.shared.file_hash import file_sha256


class FileSource(SourceAdapter):
    name = "file"
    version = "file.lines.v2"
    ordered_records = True

    def __init__(
        self, path: Path, *, pattern: str | None = None, max_record_bytes: int = MAX_RECORD_BYTES
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
            if not paths:
                raise ValueError("no log files match --pattern")
        self.files = [(path, file_sha256(path)) for path in paths]

    def describe(self) -> dict[str, Any]:
        description = {
            "kind": "file",
            "version": self.version,
            "path": str(self.path),
            "format": "line_records",
        }
        if self.pattern is None:
            description["sha256"] = self.files[0][1]
        else:
            description.update(
                pattern=self.pattern,
                order="relative_path_ascending",
                files=[
                    {"path": path.relative_to(self.path).as_posix(), "sha256": expected}
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
                while line := stream.readline(self.max_record_bytes + 1):
                    line_no += 1
                    locator = f"{prefix}line:{line_no}"
                    if len(line) > self.max_record_bytes:
                        raise SourceError(f"record exceeds byte limit at {locator}")
                    hasher.update(line)
                    if line.strip():
                        yield RawRecord(locator, line)
            if hasher.hexdigest() != expected:
                raise SourceError(f"source file changed during import: {path.name}")
