import pytest

from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Message, Usage


@pytest.mark.assets
def test_unknown_model_no_guessed_fallback(calculator):
    call = Call(
        call_id="unknown",
        protocol="synthetic",
        model="qwen-turbo",
        messages=[Message(role="user", content="Hi")],
        usage=Usage(input_tokens=1),
    )
    value = calculator.calculate(call)
    assert value.reason == "model_binding_missing"
    assert value.input_tokens is None
    assert value.verdict == "UNVERIFIABLE"


@pytest.mark.parametrize("model", ["test-model", "test-alias"])
def test_same_model_can_use_different_explicit_platform_backends(
    unit_calculator, unit_registry, model
):
    class AnotherReference:
        version = "other-test-vector.v1"

        def __init__(self, *_args):
            pass

        def count(self, call, binding):
            return {"input_tokens": 17, "visible_output_tokens": 1}

    unit_registry.register("tokenizers", "other-test", AnotherReference)
    unit_calculator.lock["assets"]["other-test"] = {"family": "other-test"}
    unit_calculator.bindings = Bindings(
        models=[
            Binding(
                modelnames=["test-model", "test-alias"], asset="test-only", platform="volcengine"
            ),
            Binding(
                modelnames=["test-model", "test-alias"], asset="other-test", platform="huaweicloud"
            ),
        ]
    )
    call = Call(
        call_id="same-name",
        protocol="test",
        model=model,
        messages=[Message(role="user", content="Hello")],
        usage=Usage(input_tokens=13),
    )
    first = unit_calculator.calculate(call.model_copy(update={"platform": "volcengine"}))
    second = unit_calculator.calculate(call.model_copy(update={"platform": "huaweicloud"}))
    assert (first.input_tokens, first.input_delta) == (13, 0)
    assert (second.input_tokens, second.input_delta) == (17, -4)


def test_global_binding_applies_to_known_and_unknown_platforms(unit_calculator):
    unit_calculator.bindings = Bindings.model_validate(
        {"models": [{"modelnames": ["test-model"], "asset": "test-only"}]}
    )
    call = Call(
        call_id="one",
        protocol="test",
        model="test-model",
        messages=[Message(role="user", content="Hello")],
        usage=Usage(input_tokens=13),
    )
    assert unit_calculator.calculate(call).input_tokens == 13
    assert (
        unit_calculator.calculate(
            call.model_copy(
                update={"platform": "volcengine", "platform_source": "literal:volcengine"}
            )
        ).input_tokens
        == 13
    )
