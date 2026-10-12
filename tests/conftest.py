import socket
from contextlib import ExitStack

import pytest

from modulo.application.factory import make_calculator
from modulo.application.registry import builtins
from modulo.application.tokenizer_assets import load_tokenizer_assets
from modulo.cli.configuration import read_config
from modulo.core.domain.bindings import Bindings
from tests.support.mock_es import MockElasticsearch
from tests.support.paths import ROOT
from tests.tools.references.assets import reference_assets


@pytest.fixture
def mock_es():
    with ExitStack() as patches:

        def install(mock: MockElasticsearch):
            patches.enter_context(mock.patch_source())

        yield install


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("tests must not open external network connections")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def pytest_addoption(parser):
    parser.addoption(
        "--require-assets",
        action="store_true",
        help="Fail instead of skipping when the official tokenizer assets are absent",
    )


@pytest.fixture
def official_assets(request):
    root = ROOT / "assets/tokenizers"
    for asset, entry in reference_assets()["assets"].items():
        for filename in entry["files"]:
            if not (root / asset / filename).is_file():
                if request.config.getoption("--require-assets"):
                    pytest.fail(
                        "prepare test references with tests/tools/references/assets.py "
                        "before running acceptance tests"
                    )
                pytest.skip("official tokenizer assets are not prepared")
    return root


@pytest.fixture
def calculator(official_assets):
    bindings = Bindings.model_validate(
        {"models": read_config(ROOT / "config/modulo.yaml")["models"]}
    )
    return make_calculator(builtins(), official_assets, bindings, load_tokenizer_assets())


@pytest.fixture
def dify_record():
    return {
        "id": "call-1",
        "node_type": "llm",
        "status": "succeeded",
        "created_at": 1791417600,
        "process_data": {
            "model_mode": "chat",
            "model_name": "test-model",
            "model_provider": "test",
            "prompts": [{"role": "user", "text": "Hello", "files": []}],
        },
        "outputs": {"text": "hi", "usage": {"prompt_tokens": 13, "completion_tokens": 1}},
    }


@pytest.fixture
def unit_registry():
    class FixedReference:
        version = "fixed-test-vector.v1"

        def __init__(self, *_args):
            pass

        def count(self, call, binding):
            return {"input_tokens": 13, "visible_output_tokens": 1}

    registry = builtins()
    registry.register("tokenizers", "test-only", FixedReference)
    return registry


@pytest.fixture
def unit_calculator(tmp_path, unit_registry):
    bindings = Bindings.model_validate(
        {"models": [{"aliases": ["test-model"], "name": "test-only"}]}
    )
    return make_calculator(
        unit_registry, tmp_path, bindings, {"assets": {"test-only": {"family": "test-only"}}}
    )
