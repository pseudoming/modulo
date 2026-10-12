"""Batch totals cannot be copied to individual inputs or cover missing members."""

from collections import Counter
from copy import deepcopy

import pytest

from modulo.core.domain.data_models import Call, InputBatch, Measurement, Usage
from modulo.core.processing.aggregation import summarize


def batch_rows():
    result = []
    for index in range(2):
        call = Call(
            call_id=f"run/input/{index}",
            protocol="langchain",
            model="m",
            input_batch=InputBatch(
                run_id="run", index=index, size=2, usage=Usage(input_tokens=30, output_tokens=9)
            ),
        )
        result.append(
            {
                "call_key": f"key-{index}",
                "call": call.model_dump(),
                "identity_conflict": False,
                "observed_dimensions": [{"platform": None, "model": "m"}],
                "measurement": Measurement(
                    execution="completed",
                    input_tokens=10 + index * 10,
                    output_execution="completed",
                    output_tokens=4 + index,
                ).model_dump(),
            }
        )
    return result


def summary(rows):
    return summarize(rows, Counter(), [], {}, batch_read_complete=True).summary


def test_batch_totals_have_their_own_conclusion_and_are_not_multiplied():
    result = summary(batch_rows())
    batch = result["input_batches"][0]
    assert batch["input_tokens"] == 30 and batch["output_tokens"] == 9
    assert batch["input_delta"] == batch["output_delta"] == 0
    assert batch["verdict"] == "CONSISTENT"
    assert result["input_verdicts"] == {"UNVERIFIABLE": 2}
    assert result["by_model"]["m"]["reported_input_sum"] is None
    assert result["by_model"]["m"]["recalculated_input_sum"] == 30


@pytest.mark.parametrize(
    "defect", ["missing_member", "conflict", "duplicate_index", "usage_conflict", "partial_output"]
)
def test_partial_or_conflicting_batch_is_not_reported_as_complete(defect):
    rows = batch_rows()
    if defect == "missing_member":
        rows.pop()
    elif defect == "conflict":
        rows[1]["identity_conflict"] = True
    elif defect == "duplicate_index":
        rows[1]["call"]["input_batch"]["index"] = 0
    elif defect == "usage_conflict":
        rows[1]["call"]["input_batch"]["usage"]["input_tokens"] = 99
    else:
        rows[1]["measurement"]["output_execution"] = "partial"
    batch = summary(rows)["input_batches"][0]
    assert batch["verdict"] == "UNVERIFIABLE"
    assert batch["output_delta"] is None
    if defect != "partial_output":
        assert batch["input_delta"] is None


def test_batch_differences_keep_their_sign_and_do_not_cancel_between_sides():
    rows = batch_rows()
    for row in rows:
        row["call"]["input_batch"]["usage"] = {
            **row["call"]["input_batch"]["usage"],
            "input_tokens": 31,
            "output_tokens": 8,
        }
    batch = summary(deepcopy(rows))["input_batches"][0]
    assert batch["input_delta"] == 1 and batch["output_delta"] == -1
    assert batch["verdict"] == "DEVIATION"


def test_grouped_coverage_and_token_sums_use_the_same_comparable_calls():
    rows = batch_rows()
    rows[0]["call"]["usage"] = Usage(input_tokens=12, output_tokens=5).model_dump()
    rows[0]["measurement"] = Measurement(
        execution="completed",
        input_tokens=10,
        reported_input_tokens=12,
        input_delta=2,
        input_verdict="DEVIATION",
        output_execution="partial",
        visible_output_tokens=4,
        output_reason="output_incomplete",
    ).model_dump()
    rows[1]["call"]["cache_hit"] = True
    # Body is recounted, but there is no per-call usage comparison.
    result = summary(rows)["by_platform_model"][0]
    assert result["calls"] == 2 and result["cache_hit_calls"] == 1
    assert result["input_completed"] == 2 and result["output_completed"] == 1
    assert result["recalculated_input_sum"] == 30
    assert result["compared_input_recalculated_sum"] == 10
    assert result["compared_input_reported_sum"] == 12
    assert result["signed_input_delta_sum"] == 2
    assert result["compared_output_reported_sum"] is None
    assert result["compared_output_recalculated_sum"] is None
    assert result["input_verdicts"] == {"DEVIATION": 1, "UNVERIFIABLE": 1}
    assert result["output_execution"] == {"partial": 1, "completed": 1}
