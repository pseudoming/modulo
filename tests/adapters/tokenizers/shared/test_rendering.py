import json

import pytest

from modulo.application.registry import builtins
from tests.support.paths import ROOT


@pytest.mark.parametrize("adapter", ["dify", "langchain", "litellm"])
@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": False},
        {"thinking_budget": 0},
        {"reasoning_effort": "low"},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        {"model_kwargs": {"preserve_thinking": False}},
    ],
)
def test_logged_render_override_is_not_silently_replaced(adapter, options):
    path = ROOT / f"examples/{adapter}/basic.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    if adapter == "dify":
        record.get("data", record)["process_data"]["model_parameters"] = options
    elif adapter == "langchain":
        record.setdefault("extra", {})["invocation_params"] = options
    else:
        record["model_parameters"] = options
    calls = builtins().get("protocols", adapter).parse(record).calls
    assert calls
    expected = options
    while any(k in expected for k in ("extra_body", "chat_template_kwargs", "model_kwargs")):
        expected = next(
            expected[k]
            for k in ("extra_body", "chat_template_kwargs", "model_kwargs")
            if k in expected
        )
    for call in calls:
        assert not call.gaps
        assert call.render_options == expected
        assert set(call.render_option_sources) == set(expected)
