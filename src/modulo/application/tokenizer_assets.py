"""Read the tokenizer asset catalog and explicitly prepare its locked files."""

import time
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from modulo.shared.serialization import bytes_digest, loads


def load_tokenizer_assets() -> dict:
    """Load the built-in catalog of fixed tokenizer files and calculation parameters."""
    packaged = files("modulo").joinpath("static/tokenizer-assets.json")
    if packaged.is_file():
        return loads(packaged.read_bytes())
    # Editable installs use the repository's single source of truth. Never look
    # relative to the working directory, which may contain customer material.
    root = Path(__file__).resolve().parents[3]
    return loads((root / "static/tokenizer-assets.json").read_bytes())


def prepare_assets(
    root: Path,
    only: list[str] | None = None,
    *,
    source: str = "global",
    catalog: dict | None = None,
) -> list[str]:
    if source not in {"global", "cn"}:
        raise ValueError("asset source must be global or cn")
    entries = (load_tokenizer_assets() if catalog is None else catalog)["assets"]
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
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and bytes_digest(path.read_bytes()) == expected:
                verified.append(f"{asset}/{name}")
                continue
            url = download_url(entry, name, source)
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


def download_url(entry: dict, filename: str, source: str) -> str:
    if source == "cn":
        mirror = entry.get("modelscope")
        if not mirror or not mirror.get("repo") or not mirror.get("revision"):
            raise ValueError(f"no locked ModelScope source for {entry['repo']}")
        query = urlencode({"Revision": mirror["revision"], "FilePath": filename})
        return f"https://modelscope.cn/api/v1/models/{quote(mirror['repo'], safe='/')}/repo?{query}"
    return (
        f"https://huggingface.co/api/resolve-cache/models/{quote(entry['repo'], safe='/')}/"
        f"{quote(entry['revision'], safe='')}/{quote(filename, safe='/')}"
    )
