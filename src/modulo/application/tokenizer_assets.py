"""Read the tokenizer asset catalog and explicitly prepare its locked files."""

import time
from importlib.resources import files
from pathlib import Path
from urllib.request import Request, urlopen

from modulo.shared.serialization import bytes_digest, loads


def asset_lock() -> dict:
    return loads(files("modulo.application").joinpath("assets-lock.json").read_bytes())


def get_backend_name(asset: str, lock: dict) -> str:
    entry = lock.get("assets", {}).get(asset)
    if entry is None:
        raise ValueError(f"unknown tokenizer asset: {asset}")
    backend = entry.get("family")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(f"tokenizer asset has no backend: {asset}")
    return backend


def prepare_assets(root: Path, only: list[str] | None = None) -> list[str]:
    entries = asset_lock()["assets"]
    selected = list(entries) if only is None else list(dict.fromkeys(only))
    unknown = set(selected) - set(entries)
    if unknown:
        raise ValueError(f"unknown tokenizer assets: {', '.join(sorted(unknown))}")
    verified = []
    for asset in selected:
        entry = entries[asset]
        directory = root / asset
        directory.mkdir(parents=True, exist_ok=True)
        for name, expected in entry["files"].items():
            path = directory / name
            if path.is_file() and bytes_digest(path.read_bytes()) == expected:
                verified.append(f"{asset}/{name}")
                continue
            url = (
                f"https://huggingface.co/api/resolve-cache/models/{entry['repo']}/"
                f"{entry['revision']}/{name}"
            )
            for attempt in range(3):
                try:
                    with urlopen(
                        Request(url, headers={"User-Agent": "modulo/0.1"}), timeout=45
                    ) as response:
                        data = response.read(32 * 1024 * 1024 + 1)
                    if bytes_digest(data) != expected:
                        raise ValueError(f"asset digest mismatch: {asset}/{name}")
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            temporary = path.with_suffix(path.suffix + ".download")
            try:
                temporary.write_bytes(data)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            verified.append(f"{asset}/{name}")
    return verified
