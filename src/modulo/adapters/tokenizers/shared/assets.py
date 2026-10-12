"""Verify local, explicitly locked reference assets before opening a backend."""

from pathlib import Path
from typing import Any

from modulo.core.domain.interfaces import AssetError
from modulo.shared.serialization import bytes_digest


def verified_asset(
    root: Path, asset: str, lock: dict[str, Any], repo: str, required_files: tuple[str, ...] = ()
) -> tuple[Path, dict[str, Any]]:
    if asset not in lock.get("assets", {}):
        raise AssetError("asset_not_locked")
    entry = lock["assets"][asset]
    if entry["repo"] != repo:
        raise AssetError("asset_repository_mismatch")
    if not set(required_files) <= entry["files"].keys():
        raise AssetError("required_asset_not_locked")
    directory = root / asset
    # Loaders open only explicitly named, locked files. Unread documentation,
    # licenses and developer references do not affect a calculation.
    for name, expected in entry["files"].items():
        path = directory / name
        if not path.is_file():
            raise AssetError("asset_missing")
        if bytes_digest(path.read_bytes()) != expected:
            raise AssetError("asset_hash_mismatch")
    return directory, entry
