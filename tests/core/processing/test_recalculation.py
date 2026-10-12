import pytest

from modulo.core.domain.bindings import Binding, Bindings
from modulo.core.domain.data_models import Call, Measurement, Message, Usage


@pytest.mark.parametrize(
    "input_reported, output_reported, input_verdict, output_verdict, verdict",
    [
        (10, 20, "CONSISTENT", "CONSISTENT", "CONSISTENT"),
        (10, 21, "CONSISTENT", "DEVIATION", "DEVIATION"),
        (10, None, "CONSISTENT", "UNVERIFIABLE", "UNVERIFIABLE"),
        (11, 20, "DEVIATION", "CONSISTENT", "DEVIATION"),
        (11, 19, "DEVIATION", "DEVIATION", "DEVIATION"),
        (11, None, "DEVIATION", "UNVERIFIABLE", "DEVIATION"),
        (None, 20, "UNVERIFIABLE", "CONSISTENT", "UNVERIFIABLE"),
        (None, 21, "UNVERIFIABLE", "DEVIATION", "DEVIATION"),
        (None, None, "UNVERIFIABLE", "UNVERIFIABLE", "UNVERIFIABLE"),
    ],
)
def test_call_verdict_combines_comparisons_without_cancelling_differences(
    unit_calculator, input_reported, output_reported, input_verdict, output_verdict, verdict
):
    class Reference:
        version = "test"

        def count(self, call, binding):
            return {"input_tokens": 10}

        def count_output(self, call):
            return {"output_execution": "completed", "output_tokens": 20}

    unit_calculator.backends["test-only"] = Reference()
    measured = unit_calculator.calculate(
        Call(
            call_id="one",
            protocol="synthetic",
            model="test-model",
            messages=[Message(role="user", content="hello")],
            output_text="answer",
            usage=Usage(input_tokens=input_reported, output_tokens=output_reported),
        )
    )
    assert measured.input_verdict == input_verdict
    assert measured.output_verdict == output_verdict
    assert measured.verdict == verdict
    assert Measurement.model_validate_json(measured.model_dump_json()) == measured
    if input_reported == 11 and output_reported == 19:
        assert measured.input_delta + measured.output_delta == 0
        assert measured.verdict == "DEVIATION"


@pytest.mark.parametrize("failed_side", ["input", "output"])
@pytest.mark.parametrize("other_deviates", [False, True])
def test_failed_component_cannot_pass_but_keeps_other_difference(
    unit_calculator, failed_side, other_deviates
):
    class Reference:
        version = "test"

        def count(self, call, binding):
            if failed_side == "input":
                raise ValueError("unavailable")
            return {"input_tokens": 10}

        def count_output(self, call):
            if failed_side == "output":
                raise ValueError("unavailable")
            return {"output_execution": "completed", "output_tokens": 20}

    unit_calculator.backends["test-only"] = Reference()
    measured = unit_calculator.calculate(
        Call(
            call_id="one",
            protocol="synthetic",
            model="test-model",
            messages=[Message(role="user", content="hello")],
            usage=Usage(
                input_tokens=10 + int(other_deviates), output_tokens=20 + int(other_deviates)
            ),
        )
    )
    assert getattr(measured, f"{failed_side}_verdict") == "UNVERIFIABLE"
    assert measured.verdict == ("DEVIATION" if other_deviates else "UNVERIFIABLE")


@pytest.mark.assets
def test_unknown_model_no_guessed_fallback(calculator):
    call = Call(
        call_id="unknown",
        protocol="synthetic",
        model="qwen-turbo",
        messages=[Message(role="user", content="Hi")],
        usage=Usage(input_tokens=1, output_tokens=0),
    )
    value = calculator.calculate(call)
    assert value.reason == "model_binding_missing"
    assert value.input_tokens is None
    assert value.verdict == "UNVERIFIABLE"
    assert value.output_verdict == "UNVERIFIABLE"
    assert value.reported_output_tokens == 0 and value.output_tokens is None
    assert value.output_reason == "model_binding_missing"


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
            Binding(aliases=["test-model", "test-alias"], name="test-only", platform="volcengine"),
            Binding(
                aliases=["test-model", "test-alias"], name="other-test", platform="huaweicloud"
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
        {"models": [{"aliases": ["test-model"], "name": "test-only"}]}
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


def test_candidate_failure_retains_other_counts_and_blocks_total(unit_calculator):
    class FailingReference:
        version = "test"

        def count_output(self, call):
            if call.output_text == "fail":
                raise ValueError("private content must not enter the diagnostic")
            return {"output_execution": "completed", "output_tokens": 7}

    unit_calculator.backends["test-only"] = FailingReference()
    call = Call(
        call_id="failure",
        protocol="synthetic",
        model="test-model",
        output_candidates=[
            {"index": 0, "output_text": "fail"},
            {"index": 1, "output_text": "available"},
        ],
        usage=Usage(output_tokens=14),
    )
    count = unit_calculator.calculate(call)
    assert count.output_candidates[0].execution == "error"
    assert count.output_candidates[1].tokens == 7
    assert count.output_reason == "candidate[0]:tokenizer_error:ValueError"
    assert count.output_execution == "error" and count.output_tokens is None
    assert count.output_delta is None and count.output_verdict == "UNVERIFIABLE"


@pytest.mark.parametrize("reported", [0, 10, None])
def test_cache_hit_counts_both_bodies_without_treating_usage_as_fresh_inference(
    unit_calculator, reported
):
    class Reference:
        version = "test"

        def count(self, call, binding):
            return {"input_tokens": 10}

        def count_output(self, call):
            return {"output_tokens": 20, "output_execution": "completed"}

    unit_calculator.backends["test-only"] = Reference()
    call = Call(
        call_id="cached",
        protocol="litellm",
        model="test-model",
        cache_hit=True,
        messages=[Message(role="user", content="q")],
        output_text="a",
        usage=Usage(input_tokens=reported, output_tokens=reported),
    )
    result = unit_calculator.calculate(call)
    assert result.input_tokens == 10 and result.output_tokens == 20
    assert result.execution == result.output_execution == "completed"
    assert result.reported_input_tokens == result.reported_output_tokens == reported
    assert result.input_delta is None and result.output_delta is None
    assert result.verdict == "UNVERIFIABLE"
    assert result.reason == result.output_reason == "cache_hit_usage_not_current_inference"
