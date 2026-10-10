import json

import pytest

from modulo.application.registry import builtins
from tests.support.paths import ROOT


def example(adapter):
    record = json.loads(
        (ROOT / f"examples/{adapter}/reasoning-tools.jsonl").read_text().splitlines()[0]
    )
    if adapter == "dify":
        return record, record["outputs"], record["process_data"]["model_parameters"]
    if adapter == "langchain":
        return (
            record,
            record["outputs"]["generations"][0][0]["message"]["data"]["additional_kwargs"],
            record["extra"]["invocation_params"],
        )
    return record, record["response"]["choices"][0]["message"], record["model_parameters"]


@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm"])
@pytest.mark.parametrize("alias", ["reasoning_content", "reasoning", "thinking"])
def test_reasoning_aliases_keep_exact_output_and_do_not_replace_answer(adapter, alias):
    record, response, _ = example(adapter)
    original = response.pop("reasoning_content")
    response[alias] = original
    call = builtins().get("protocols", adapter).parse(record).calls[0]
    assert call.output_reasoning == original
    assert call.output_reasoning_source.endswith(alias)
    assert call.output_text == "结果是 5。\n"
    assert not call.gaps and not call.output_gaps


@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm"])
def test_conflicting_output_reasoning_does_not_destroy_valid_input(adapter):
    record, response, _ = example(adapter)
    response["thinking"] = "contradictory"
    call = builtins().get("protocols", adapter).parse(record).calls[0]
    assert call.output_reasoning is None and call.output_gaps == ["conflicting_reasoning_fields"]
    assert call.messages and not call.gaps


@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm"])
def test_nested_mode_conflict_is_not_overwritten(adapter):
    record, _, options = example(adapter)
    options.update(enable_thinking=False, extra_body={"enable_thinking": True})
    call = builtins().get("protocols", adapter).parse(record).calls[0]
    assert "conflicting_request_option:enable_thinking" in call.gaps
    assert len(call.render_option_sources["enable_thinking"]) == 2
    assert call.output_reasoning is not None
