"""Collect Python, system, dependency versions and source hashes for this run."""

import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

from modulo.shared.file_hash import file_sha256


def collect_runtime_info(plugin_modules: list[str]) -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    code = {
        str(path.relative_to(package)): file_sha256(path) for path in sorted(package.rglob("*.py"))
    }
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    plugins = {}
    for name in plugin_modules:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"plugin has no inspectable local source: {name}")
        if spec.submodule_search_locations:
            paths = [
                path
                for location in spec.submodule_search_locations
                for path in sorted(Path(location).rglob("*.py"))
            ]
        else:
            paths = [Path(spec.origin)]
        plugins[name] = {str(path): file_sha256(path) for path in paths}
    return {
        "schema_version": "execution.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "code": code,
        "plugins": plugins,
    }
