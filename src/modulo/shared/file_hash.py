"""Compute a file's SHA-256 without loading the whole file into memory."""

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
