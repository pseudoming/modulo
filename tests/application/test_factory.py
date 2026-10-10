import pytest

from modulo.application.factory import make_calculator
from modulo.application.registry import Registry, builtins
from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Message, Usage


def test_distinct_assets_can_use_the_same_backend_without_sharing_loaded_state(tmp_path):
    class SharedTokenizer:
        version = "synthetic.v1"

        def __init__(self, root, asset, lock):
            self.asset = asset

        def count(self, call, binding):
            return {"input_tokens": {"first-reference": 13, "second-reference": 17}[self.asset]}

    registry = Registry()
    registry.register("tokenizers", "shared-code", SharedTokenizer)
    bindings = Bindings(
        models=[
            Binding(modelnames=["model-a", "alias-a"], asset="first-reference"),
            Binding(modelnames=["model-b"], asset="second-reference"),
        ]
    )
    lock = {
        "assets": {
            "first-reference": {"family": "shared-code"},
            "second-reference": {"family": "shared-code"},
        }
    }
    calculator = make_calculator(registry, tmp_path, bindings, lock)
    for model, expected in [("model-a", 13), ("model-b", 17), ("alias-a", 13)]:
        result = calculator.calculate(
            Call(
                call_id=model,
                protocol="synthetic",
                model=model,
                messages=[Message(role="user", content="Hello")],
                usage=Usage(input_tokens=expected),
            )
        )
        assert result.backend == "shared-code"
        assert result.input_tokens == expected and result.verdict == "CONSISTENT"


@pytest.mark.parametrize(
    "entry, error",
    [
        (None, "unknown tokenizer asset"),
        ({}, "asset has no backend"),
        ({"family": " "}, "asset has no backend"),
        ({"family": "not-registered"}, "unknown plugin"),
    ],
)
def test_asset_backend_must_be_explicitly_declared_and_registered(tmp_path, entry, error):
    bindings = Bindings(models=[Binding(modelnames=["qwen"], asset="qwen-next")])
    lock = {"assets": {} if entry is None else {"qwen-next": entry}}
    with pytest.raises(ValueError, match=error):
        make_calculator(builtins(), tmp_path, bindings, lock)
