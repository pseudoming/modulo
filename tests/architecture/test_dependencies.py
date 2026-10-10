"""Executable dependency rules for the current modular monolith."""

import ast
import subprocess
import sys
from importlib.util import resolve_name

import pytest

from tests.support.paths import ROOT

PACKAGE = ROOT / "src/modulo"
ALLOWED = {
    "shared": {"shared"},
    "core": {"core", "shared"},
    "adapters": {"core", "adapters", "shared"},
    "storage": {"core", "storage", "shared"},
    "application": {"core", "adapters", "storage", "application", "shared"},
    "cli": {"core", "application", "storage", "cli", "shared"},
}
CORE_ALLOWED = {
    "domain": {"domain"},
    "processing": {"domain", "processing"},
}


def imports(path):
    relative = path.relative_to(PACKAGE).with_suffix("")
    module = "modulo." + ".".join(relative.parts)
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = resolve_name("." * node.level + name, package)
            yield name


@pytest.mark.parametrize("layer", ALLOWED)
def test_dependencies_point_toward_contracts_and_never_back_to_entry_points(layer):
    violations = []
    for path in (PACKAGE / layer).rglob("*.py"):
        for dependency in imports(path):
            if dependency.split(".")[0] == "tests":
                violations.append((str(path.relative_to(PACKAGE)), dependency))
            if not dependency.startswith("modulo."):
                if layer == "shared" and dependency.split(".")[0] not in sys.stdlib_module_names:
                    violations.append((str(path.relative_to(PACKAGE)), dependency))
                continue
            target = dependency.split(".")[1]
            if target not in ALLOWED[layer]:
                violations.append((str(path.relative_to(PACKAGE)), dependency))
            if (
                layer == "core"
                and target == "shared"
                and dependency != "modulo.shared.serialization"
            ):
                violations.append((str(path.relative_to(PACKAGE)), dependency))
            if layer == "core" and dependency.startswith("modulo.core."):
                group = path.relative_to(PACKAGE / "core").parts[0]
                if group in CORE_ALLOWED and dependency.split(".")[2] not in CORE_ALLOWED[group]:
                    violations.append((str(path.relative_to(PACKAGE)), dependency))
            if layer == "adapters" and target == "adapters":
                family = path.relative_to(PACKAGE).parts[1]
                if len(dependency.split(".")) > 2 and dependency.split(".")[2] != family:
                    violations.append((str(path.relative_to(PACKAGE)), dependency))
    assert not violations


def test_core_has_no_runtime_io_or_plugin_discovery_dependencies():
    allowed = (
        sys.stdlib_module_names - {"os", "sys", "pathlib", "socket", "subprocess", "importlib"}
    ) | {
        "pydantic",
        "modulo",
    }
    for path in [
        *(PACKAGE / "core").rglob("*.py"),
        PACKAGE / "shared/__init__.py",
        PACKAGE / "shared/serialization.py",
    ]:
        assert {name.split(".")[0] for name in imports(path)} <= allowed
        tree = ast.parse(path.read_text())
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"open", "eval", "exec", "__import__"}
        ], path


def test_core_import_does_not_load_concrete_runtime_packages():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import modulo.core.main; import sys; "
            "assert not any(name in sys.modules for name in "
            "('modulo.adapters', 'modulo.application', 'modulo.storage', 'modulo.cli', "
            "'modulo.shared.file_hash', 'duckdb', 'httpx', 'transformers', 'yaml'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert completed.stderr == ""
