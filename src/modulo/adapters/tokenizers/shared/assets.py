"""Verify local, explicitly locked reference assets before opening a backend."""

from pathlib import Path
from typing import Any

from modulo.core.domain.interfaces import AssetError
from modulo.shared.serialization import bytes_digest


def verified_asset(
    root: Path, asset: str, lock: dict[str, Any], family: str, repos: tuple[str, ...]
) -> tuple[Path, dict[str, Any]]:
    if asset not in lock.get("assets", {}):
        raise AssetError("asset_not_locked")
    entry = lock["assets"][asset]
    if entry["family"] != family or entry["repo"] not in repos:
        raise AssetError("asset_family_mismatch")
    directory = root / asset
    if directory.is_dir() and {p.name for p in directory.iterdir()} != set(entry["files"]):
        raise AssetError("asset_directory_contents_mismatch")
    for name, expected in entry["files"].items():
        path = directory / name
        if not path.is_file():
            raise AssetError("asset_missing")
        if bytes_digest(path.read_bytes()) != expected:
            raise AssetError("asset_hash_mismatch")
    return directory, entry
