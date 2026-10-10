import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from modulo.application.registry import Registry, builtins
from modulo.cli import main
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle


def package(tmp_path, monkeypatch, name, files):
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    for filename, source in files.items():
        target = root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dedent(source))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


PROTOCOL = """
from modulo.core.domain.data_models import Parsed
from modulo.core.domain.interfaces import ProtocolAdapter

class ExampleProtocol(ProtocolAdapter):
    name = "example"
    version = "example.v1"

    def parse(self, record):
        return Parsed([], "example_non_llm")
"""


def test_three_plugin_namespaces_and_no_silent_replacement():
    registry = builtins()
    assert registry.describe() == {
        "sources": ["es", "file"],
        "protocols": ["dify", "langchain", "litellm", "newapi", "oneapi"],
        "tokenizers": [
            "deepseek",
            "glm",
            "hunyuan",
            "ling",
            "longcat",
            "mimo",
            "minimax",
            "moonshot",
            "qwen",
            "ring",
            "stepfun",
        ],
    }
    with pytest.raises(ValueError, match="already registered"):
        registry.register("protocols", "dify", object())
    with pytest.raises(ValueError, match="unknown plugin"):
        registry.get("tokenizers", "arbitrary-model-name")


def test_explicit_extension_module(tmp_path, monkeypatch):
    (tmp_path / "test_extension.py").write_text(
        "def register(registry):\n    registry.register('protocols', 'extension', object())\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = builtins()
    assert "extension" not in registry.describe()["protocols"]
    registry.load("test_extension")
    assert "extension" in registry.describe()["protocols"]


def test_discovery_walks_selected_package_and_skips_helpers(tmp_path, monkeypatch):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_nested",
        {
            "nested/__init__.py": "",
            "nested/adapter.py": PROTOCOL + "\nAlias = ExampleProtocol\n",
            "reexport.py": "from .nested.adapter import ExampleProtocol",
            "_private.py": "raise AssertionError('private modules must not be scanned')",
            "helpers.py": """
                from modulo.core.domain.interfaces import ProtocolAdapter

                class Incomplete(ProtocolAdapter):
                    name = "incomplete"
                    version = "test.v1"

                class Helper(ProtocolAdapter):
                    def parse(self, record):
                        raise NotImplementedError
            """,
        },
    )
    registry = builtins()
    registry.load(name)
    assert registry.describe()["protocols"] == [
        "dify",
        "example",
        "langchain",
        "litellm",
        "newapi",
        "oneapi",
    ]
    assert registry.get("protocols", "example").parse({}).ignored == "example_non_llm"


def test_new_builtin_module_requires_no_central_registration(tmp_path, monkeypatch):
    import modulo.adapters.protocols

    registry = builtins()  # Packages and their existing modules are already imported.
    (tmp_path / "added_adapter.py").write_text(PROTOCOL)
    monkeypatch.setattr(
        modulo.adapters.protocols, "__path__", [*modulo.adapters.protocols.__path__, str(tmp_path)]
    )
    discovered = builtins()
    assert "example" in discovered.describe()["protocols"]
    assert "example" not in registry.describe()["protocols"]


def test_explicit_plugins_do_not_leak_to_new_registries(tmp_path, monkeypatch):
    name = package(tmp_path, monkeypatch, "discovery_isolated", {"adapter.py": PROTOCOL})
    first = builtins()
    assert name not in sys.modules
    first.load(name)
    first.get("protocols", "example").local_state = "first registry"
    second = builtins()
    assert "example" not in second.describe()["protocols"]
    second.load(name)  # Discovery must work even when Python has cached the module.
    assert not hasattr(second.get("protocols", "example"), "local_state")


def test_discovery_keeps_sources_and_tokenizers_as_unopened_factories(tmp_path, monkeypatch):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_factories",
        {
            "source.py": """
                from modulo.core.domain.interfaces import SourceAdapter

                class ExampleSource(SourceAdapter):
                    name = "factory-test"
                    version = "test.v1"

                    def __init__(self, *args):
                        raise AssertionError("discovery must not open a source")

                    def describe(self):
                        return {}

                    def read(self):
                        return iter(())
            """,
            "tokenizer.py": """
                from modulo.core.domain.interfaces import TokenizerBackend

                class ExampleTokenizer(TokenizerBackend):
                    name = "factory-test"
                    version = "test.v1"

                    def __init__(self, *args):
                        raise AssertionError("discovery must not load assets")

                    def count(self, call, binding):
                        return {}
            """,
        },
    )
    registry = Registry()
    registry.load(name)
    assert registry.get("sources", "factory-test").__name__ == "ExampleSource"
    assert registry.get("tokenizers", "factory-test").__name__ == "ExampleTokenizer"


def test_duplicate_names_fail_without_partial_registration(tmp_path, monkeypatch):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_duplicate",
        {"first.py": PROTOCOL, "second.py": PROTOCOL},
    )
    registry = builtins()
    before = registry.describe()
    with pytest.raises(ValueError, match="already registered: protocols/example"):
        registry.load(name)
    assert registry.describe() == before


@pytest.mark.parametrize("declaration", ['name = ""', 'version = ""', "version = None"])
def test_invalid_declarations_are_not_silently_ignored(tmp_path, monkeypatch, declaration):
    field = declaration.split(" = ")[0]
    old = 'name = "example"' if field == "name" else 'version = "example.v1"'
    name = package(
        tmp_path,
        monkeypatch,
        "invalid_" + tmp_path.name,
        {"adapter.py": PROTOCOL.replace(old, declaration)},
    )
    registry = Registry()
    with pytest.raises(ValueError, match="requires name and version"):
        registry.load(name)
    assert registry.describe()["protocols"] == []


def test_import_errors_do_not_silently_skip_adapters(tmp_path, monkeypatch):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_import_failure",
        {"a_good.py": PROTOCOL, "z_broken.py": "import missing_modulo_plugin_dependency"},
    )
    registry = Registry()
    with pytest.raises(ValueError, match="cannot import plugin module: .*z_broken"):
        registry.load(name)
    assert registry.describe()["protocols"] == []


def test_invalid_protocol_constructor_does_not_partially_register(tmp_path, monkeypatch):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_constructor",
        {
            "a_good.py": PROTOCOL,
            "z_invalid.py": PROTOCOL.replace('name = "example"', 'name = "requires-config"')
            + "\n    def __init__(self, required):\n        self.required = required\n",
        },
    )
    registry = Registry()
    with pytest.raises(ValueError, match="no-argument constructor"):
        registry.load(name)
    assert registry.describe()["protocols"] == []


def test_legacy_hook_owns_registration_without_double_discovery(tmp_path, monkeypatch):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_legacy_hook",
        {
            "__init__.py": PROTOCOL + "\ndef register(registry):\n"
            "    registry.register('protocols', 'example', ExampleProtocol())\n"
        },
    )
    registry = builtins()
    registry.load(name)
    assert registry.get("protocols", "example").version == "example.v1"
    with pytest.raises(ValueError, match="already registered"):
        registry.load(name)


def test_standard_startup_does_not_load_mock_or_transformers():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from modulo.cli import main; import sys; main(['plugins']); "
            "assert 'tests.support.mock_es' not in sys.modules; "
            "assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(completed.stdout)["sources"] == ["es", "file"]


def test_declared_extension_adapters_work_through_unified_cli(tmp_path, monkeypatch, capsys):
    name = package(
        tmp_path,
        monkeypatch,
        "discovery_cli",
        {
            "source.py": """
                from pathlib import Path
                from modulo.core.domain.data_models import RawRecord
                from modulo.core.domain.interfaces import SourceAdapter

                class SyntheticSource(SourceAdapter):
                    name = "synthetic"
                    version = "synthetic.v1"

                    def __init__(self, path):
                        self.path = Path(path)

                    def describe(self):
                        return {"kind": self.name, "path": str(self.path)}

                    def read(self):
                        yield RawRecord("row:1", self.path.read_bytes())
            """,
            "protocol.py": """
                from modulo.core.domain.data_models import Call, Message, Parsed, Usage
                from modulo.core.domain.interfaces import ProtocolAdapter

                class SyntheticProtocol(ProtocolAdapter):
                    name = "synthetic"
                    version = "synthetic.v1"

                    def parse(self, record):
                        return Parsed([Call(
                            call_id=record["id"], protocol=self.name, model="synthetic-model",
                            messages=[Message(role="user", content=record["text"])],
                            usage=Usage(input_tokens=1),
                        )])
            """,
            "tokenizer.py": """
                from modulo.core.domain.interfaces import TokenizerBackend

                class SyntheticTokenizer(TokenizerBackend):
                    name = "synthetic"
                    version = "synthetic.v1"

                    def __init__(self, *_args):
                        pass

                    def count(self, call, binding):
                        # A declared one-token test vector, not a real tokenizer.
                        assert call.messages[0].content == "Hello"
                        return {"input_tokens": 1}
            """,
        },
    )
    path = tmp_path / "input.json"
    path.write_text('{"id":"synthetic-1","text":"Hello"}')
    config = tmp_path / "source.json"
    config.write_text(json.dumps({"path": str(path)}))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "modelnames": ["synthetic-model"],
                        "asset": "synthetic",
                    },
                ]
            }
        )
    )
    # The extension's asset declaration is independent of user model aliases.
    monkeypatch.setattr(
        "modulo.cli.commands.asset_lock",
        lambda: {"assets": {"synthetic": {"family": "synthetic"}}},
    )
    assert (
        main(
            [
                "--plugin",
                name,
                "import",
                "--source",
                "synthetic",
                "--adapter",
                "synthetic",
                "--source-config",
                str(config),
                "--bindings",
                str(bindings),
                "--output",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["summary"]["verdicts"] == {"CONSISTENT": 1}
    bundle = Path(imported["run"])
    assert details(bundle)[0]["measurement"]["input_delta"] == 0
    verify_bundle(bundle)
    execution = json.loads((bundle / "execution.json").read_text())
    assert {Path(path).name for path in execution["plugins"][name]} == {
        "__init__.py",
        "source.py",
        "protocol.py",
        "tokenizer.py",
    }
    assert "synthetic" not in builtins().describe()["protocols"]
