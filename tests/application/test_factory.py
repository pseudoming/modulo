import pytest

from modulo.application.factory import make_calculator
from modulo.application.registry import Registry, builtins
from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Message, Usage
from modulo.core.domain.interfaces import TokenizerBackend


def test_named_models_share_family_code_but_load_separately_and_reuse_aliases(tmp_path):
    loaded = []

    class SharedTokenizer(TokenizerBackend):
        family = "shared-code"
        version = "synthetic.v1"

        def __init__(self, root, lock):
            loaded.append(self.name)

        def count(self, call, binding):
            return {"input_tokens": self.tokens}

    class FirstTokenizer(SharedTokenizer):
        name = "first-reference"
        tokens = 13

    class SecondTokenizer(FirstTokenizer):
        name = "second-reference"
        tokens = 17

    registry = Registry()
    for implementation in (FirstTokenizer, SecondTokenizer):
        registry.register("tokenizers", implementation.name, implementation)
    bindings = Bindings(
        models=[
            Binding(aliases=["model-a", "alias-a"], name="first-reference"),
            Binding(aliases=["model-b"], name="second-reference"),
        ]
    )
    lock = {
        "assets": {
            "first-reference": {"family": "shared-code"},
            "second-reference": {"family": "shared-code"},
        }
    }
    calculator = make_calculator(registry, tmp_path, bindings, lock)
    for model, name, expected in [
        ("model-a", "first-reference", 13),
        ("model-b", "second-reference", 17),
        ("alias-a", "first-reference", 13),
    ]:
        result = calculator.calculate(
            Call(
                call_id=model,
                protocol="synthetic",
                model=model,
                messages=[Message(role="user", content="Hello")],
                usage=Usage(input_tokens=expected),
            )
        )
        assert result.backend == name
        assert result.input_tokens == expected and result.input_verdict == "CONSISTENT"
    assert loaded == ["first-reference", "second-reference"]


@pytest.mark.parametrize(
    "name, lock, error",
    [
        ("qwen-3.8-2.4t", {"assets": {}}, "unknown tokenizer asset"),
        ("qwen-next", {"assets": {"qwen-next": {"family": "qwen"}}}, "unknown plugin"),
    ],
)
def test_model_requires_its_own_registered_name_and_locked_asset(tmp_path, name, lock, error):
    bindings = Bindings(models=[Binding(aliases=["qwen"], name=name)])
    with pytest.raises(ValueError, match=error):
        make_calculator(builtins(), tmp_path, bindings, lock)
