"""Read independent reference metadata; explicitly prepare test-only source files."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def reference_assets() -> dict:
    catalog = json.loads((ROOT / "static/tokenizer-assets.json").read_text())
    references = json.loads((ROOT / "tests/fixtures/tokenizers/reference-assets.json").read_text())[
        "assets"
    ]
    for name, entry in catalog["assets"].items():
        reference = references[name]
        files = {**entry["files"], **reference["files"]}
        entry.update(reference)
        entry["files"] = files
    return catalog


def verify_reference_files(directory: Path, entry: dict) -> None:
    for name, expected in entry["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"reference digest mismatch: {directory.name}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", help="Prepare only these model references")
    args = parser.parse_args()
    # Reuse download/checksum handling, never the implementation under test for
    # expected answers. Normal modulo prepare-assets does not read this catalog.
    from modulo.application.tokenizer_assets import prepare_assets

    prepared = prepare_assets(ROOT / "assets/tokenizers", args.assets, catalog=reference_assets())
    print(f"Verified {len(prepared)} runtime and test reference files.")


if __name__ == "__main__":
    main()
