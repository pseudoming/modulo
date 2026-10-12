import json

import pytest

from modulo.adapters.protocols import DifyProtocol
from modulo.core.domain.data_models import DataError


@pytest.mark.parametrize("value", [None, 0, 2**70])
def test_missing_zero_and_large_counters_remain_distinct(dify_record, value):
    dify_record["outputs"]["usage"]["prompt_tokens"] = value
    call = DifyProtocol().parse(dify_record).calls[0]
    assert call.usage.input_tokens == value
    assert call.usage.total_tokens is None


@pytest.mark.parametrize("value", [True, "13", -1, 1.5])
def test_invalid_counters_are_not_coerced(dify_record, value):
    dify_record["outputs"]["usage"]["prompt_tokens"] = value
    with pytest.raises(DataError):
        DifyProtocol().parse(dify_record)


def test_text_bytes_and_empty_content_not_normalized(dify_record):
    content = "  A\r\n中e\u0301🙂\t "
    dify_record["process_data"]["prompts"][0]["text"] = content
    call = DifyProtocol().parse(dify_record).calls[0]
    assert call.messages[0].content == content
    dify_record["process_data"]["prompts"][0]["text"] = ""
    assert DifyProtocol().parse(dify_record).calls[0].messages[0].content == ""


def test_dify_envelope_with_separate_attachment_keeps_text_and_order_gap(dify_record):
    dify_record["process_data"]["prompts"][0]["files"] = [{"type": "image"}]
    call = DifyProtocol().parse({"event": "node_finished", "data": dify_record}).calls[0]
    assert call.messages[0].content == "Hello"
    assert call.messages[0].gaps == ["attachment_content_order_missing"]


def test_missing_output_is_not_zero_tokens(dify_record):
    dify_record["outputs"] = None
    call = DifyProtocol().parse(dify_record).calls[0]
    assert call.output_text is None
    assert call.usage.output_tokens is None


@pytest.mark.parametrize("which", ["process_data", "outputs"])
def test_dify_marks_only_the_truncated_component(dify_record, which):
    dify_record[f"{which}_truncated"] = True
    call = DifyProtocol().parse(dify_record).calls[0]
    assert call.usage.input_tokens == 13
    if which == "process_data":
        assert "truncated_process_data" in call.gaps
        assert not call.output_gaps
    else:
        assert not call.gaps
        assert "truncated_output" in call.output_gaps


def test_dify_complete_model_prompt_is_independent_of_workflow_input_preview(dify_record):
    dify_record["inputs_truncated"] = True
    dify_record["process_data"] = json.dumps(dify_record["process_data"])
    dify_record["outputs"] = json.dumps(dify_record["outputs"])
    call = DifyProtocol().parse(dify_record).calls[0]
    assert not call.gaps and not call.output_gaps
    assert call.messages[0].content == "Hello"


@pytest.mark.parametrize(
    "flag", ["inputs_truncated", "process_data_truncated", "outputs_truncated"]
)
@pytest.mark.parametrize("value", ["false", 0, None])
def test_dify_rejects_invalid_truncation_flags(dify_record, flag, value):
    dify_record[flag] = value
    with pytest.raises(DataError, match=f"invalid_{flag}"):
        DifyProtocol().parse(dify_record)
