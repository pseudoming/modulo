import copy
import json

import pytest

from modulo.adapters.protocols import DifyProtocol
from modulo.adapters.sources import ElasticsearchSource, FileSource
from modulo.application.image_cache import ImageDownloadOptions
from modulo.application.importer import run_batch
from modulo.shared.serialization import dumps, loads
from modulo.storage.query_index import details
from modulo.storage.result_bundle import verify_bundle
from tests.support.image_http import mock_image_http
from tests.support.mock_es import MockElasticsearch


def run_records(tmp_path, records, calculator, **kwargs):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(dumps(row) + "\n" if isinstance(row, dict) else row + "\n" for row in records)
    )
    return run_batch(
        FileSource(source),
        DifyProtocol(),
        calculator,
        namespace="test/app",
        output_root=tmp_path / "runs",
        **kwargs,
    )


@pytest.mark.parametrize("scenario", ["conflict", "unbound", "text-model", "filtered"])
def test_excluded_or_ineligible_calls_do_not_fetch_images(
    tmp_path, monkeypatch, unit_calculator, dify_record, scenario
):
    requests = mock_image_http(monkeypatch)
    dify_record["process_data"]["prompts"][0]["text"] = [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.test/image.png"},
        }
    ]
    records, kwargs = [dify_record], {}
    if scenario != "text-model":
        unit_calculator.lock["assets"]["test-only"]["image_processing"] = {"enabled": True}
    if scenario == "conflict":
        second = copy.deepcopy(dify_record)
        second["outputs"]["usage"]["prompt_tokens"] += 1
        records.append(second)
    elif scenario == "unbound":
        dify_record["process_data"]["model_name"] = "not-bound"
    elif scenario == "filtered":
        kwargs["models"] = ["not-selected"]
    path, _ = run_records(
        tmp_path,
        records,
        unit_calculator,
        image_download=ImageDownloadOptions(enabled=True, cache_dir=tmp_path / "images"),
        **kwargs,
    )
    verify_bundle(path)
    assert requests == []


def test_idempotency_distinct_same_content_and_reopen(tmp_path, unit_calculator, dify_record):
    second = copy.deepcopy(dify_record)
    second["id"] = "call-2"
    records = [dify_record, second, dify_record]
    first, reused = run_records(tmp_path, records, unit_calculator)
    assert not reused
    repeated, reused = run_records(tmp_path, records, unit_calculator)
    assert reused and repeated == first
    summary = loads((first / "summary.json").read_bytes())
    assert summary["observed_calls"] == 2
    assert summary["counters"]["duplicate_calls"] == 1
    assert summary["by_model"]["test-model"]["reported_input_sum"] == 26
    (first / "query.duckdb").unlink()
    assert {row["call"]["call_id"] for row in details(first)} == {"call-1", "call-2"}


def test_bad_json_retained_and_partial_run(tmp_path, unit_calculator, dify_record):
    path, _ = run_records(tmp_path, ['{"id":1,"id":2}', dify_record], unit_calculator)
    summary = loads((path / "summary.json").read_bytes())
    assert summary["run_status"] == "PARTIAL"
    assert summary["observed_calls"] == 1
    assert summary["counters"]["invalid_records"] == 1
    assert len((path / "raw.jsonl").read_text().splitlines()) == 2


def test_identity_conflict_excludes_unreliable_usage(tmp_path, unit_calculator, dify_record):
    conflict = copy.deepcopy(dify_record)
    conflict["outputs"]["usage"]["prompt_tokens"] += 3
    path, _ = run_records(tmp_path, [dify_record, conflict], unit_calculator)
    summary = loads((path / "summary.json").read_bytes())
    assert summary["run_status"] == "PARTIAL"
    assert summary["observed_calls"] == 1
    assert summary["by_model"]["test-model"]["reported_input_sum"] is None
    row = details(path)[0]
    assert row["measurement"]["reason"] == "identity_conflict"
    assert len(row["source_refs"]) == 2


def test_missing_usage_keeps_recount_but_never_passes(tmp_path, unit_calculator, dify_record):
    dify_record["outputs"].pop("usage")
    path, _ = run_records(tmp_path, [dify_record], unit_calculator)
    row = details(path)[0]
    assert row["measurement"]["input_tokens"] == 13
    assert row["measurement"]["input_verdict"] == "UNVERIFIABLE"
    assert row["measurement"]["input_delta"] is None


def test_signed_delta_and_no_output_false_comparison(tmp_path, unit_calculator, dify_record):
    dify_record["outputs"]["usage"]["prompt_tokens"] = 10
    dify_record["outputs"]["usage"]["completion_tokens"] = 200
    path, _ = run_records(tmp_path, [dify_record], unit_calculator)
    row = details(path)[0]
    assert row["measurement"]["input_delta"] == -3
    assert row["measurement"]["input_verdict"] == "DEVIATION"
    assert row["measurement"]["output_delta"] is None
    assert row["measurement"]["output_verdict"] == "UNVERIFIABLE"
    assert row["measurement"]["model_identity"] == "UNVERIFIED"


def test_payload_projection_and_budget(tmp_path, unit_calculator, dify_record):
    path, _ = run_records(
        tmp_path,
        [{"message": json.dumps(dify_record)}, {"message": "{}"}],
        unit_calculator,
        payload_field="message",
        max_records=1,
    )
    summary = loads((path / "summary.json").read_bytes())
    assert summary["observed_calls"] == 1
    assert not summary["batch_read_complete"]
    assert summary["run_status"] == "PARTIAL"


def test_budget_closes_es_pit(tmp_path, unit_calculator, dify_record):
    mock = MockElasticsearch([dify_record, dify_record])
    source = ElasticsearchSource("https://mock.invalid", "logs", transport=mock.transport())
    path, _ = run_batch(
        source,
        DifyProtocol(),
        unit_calculator,
        namespace="test",
        output_root=tmp_path / "runs",
        max_records=1,
    )
    assert mock.closed
    assert verify_bundle(path)["status"] == "PARTIAL"


def test_tampered_bundle_is_rejected(tmp_path, unit_calculator, dify_record):
    path, _ = run_records(tmp_path, [dify_record], unit_calculator)
    (path / "results.jsonl").write_text("")
    with pytest.raises(ValueError, match="hash mismatch"):
        details(path)


def test_namespace_keeps_identical_ids_separate(tmp_path, unit_calculator, dify_record):
    path, _ = run_records(tmp_path, [dify_record], unit_calculator)
    path2, _ = run_batch(
        FileSource(tmp_path / "source.jsonl"),
        DifyProtocol(),
        unit_calculator,
        namespace="another-account",
        output_root=tmp_path / "runs",
    )
    assert details(path)[0]["call_key"] != details(path2)[0]["call_key"]


def test_backend_failure_still_persists_raw_and_error(
    tmp_path, unit_calculator, dify_record, monkeypatch
):
    def fail(call, binding_matches=None):
        raise RuntimeError("private text must not escape into diagnostic")

    monkeypatch.setattr(unit_calculator, "calculate", fail)
    path, _ = run_records(tmp_path, [dify_record], unit_calculator)
    assert verify_bundle(path)["status"] == "PARTIAL"
    row = details(path)[0]
    assert row["measurement"]["execution"] == "error"
    assert row["measurement"]["reason"] == "backend_error:RuntimeError"
    assert row["measurement"]["input_verdict"] == "UNVERIFIABLE"
    assert "private text" not in (path / "diagnostics.jsonl").read_text()


def test_manifest_plan_cannot_be_edited_without_changing_identity(
    tmp_path, unit_calculator, dify_record
):
    path, _ = run_records(tmp_path, [dify_record], unit_calculator)
    manifest = loads((path / "manifest.json").read_bytes())
    manifest["plan"]["namespace"] = "changed"
    (path / "manifest.json").write_text(dumps(manifest))
    with pytest.raises(ValueError, match="run id"):
        verify_bundle(path)


def test_model_selection_is_exact_and_keeps_exclusions_and_missing_models(
    tmp_path, unit_calculator, dify_record, monkeypatch
):
    records = [dify_record, copy.deepcopy(dify_record)]
    for index, model in enumerate(["Test-Model", "other-model", None]):
        record = copy.deepcopy(dify_record)
        record["id"] = f"other-{index}"
        record["process_data"]["model_name"] = model
        records.append(record)
    calculate = unit_calculator.calculate
    called = []

    def selected_only(call, binding_matches=None):
        called.append(call.call_id)
        return calculate(call, binding_matches)

    monkeypatch.setattr(unit_calculator, "calculate", selected_only)
    path, _ = run_records(tmp_path, records, unit_calculator, models=["test-model", "not-present"])
    summary = loads((path / "summary.json").read_bytes())
    assert called == ["call-1"]
    assert summary["selection"] == {
        "schema_version": "selection.v2",
        "models": ["not-present", "test-model"],
        "platforms": None,
        "parsed_calls": 4,
        "selected_calls": 1,
        "excluded_calls": 2,
        "unresolved_calls": 1,
        "matched_models": ["test-model"],
        "unmatched_models": ["not-present"],
        "matched_platforms": [],
        "unmatched_platforms": [],
    }
    assert summary["counters"]["duplicate_calls"] == 1
    assert summary["run_status"] == "COMPLETE" and summary["batch_read_complete"]
    assert len((path / "raw.jsonl").read_text().splitlines()) == 5
    diagnostics = [loads(line) for line in (path / "diagnostics.jsonl").read_text().splitlines()]
    assert {row["reason"] for row in diagnostics} == {
        "model_not_selected",
        "model_selection_unresolved",
    }
    assert all(row["source_refs"] for row in diagnostics)


@pytest.mark.parametrize("reverse", [False, True])
def test_filter_never_hides_conflicting_model_variant(
    tmp_path, unit_calculator, dify_record, reverse
):
    conflict = copy.deepcopy(dify_record)
    conflict["process_data"]["model_name"] = "another-model"
    records = [dify_record, conflict]
    if reverse:
        records.reverse()
    path, _ = run_records(tmp_path, records, unit_calculator, models=["test-model"])
    assert verify_bundle(path)["status"] == "PARTIAL"
    row = details(path)[0]
    assert row["identity_conflict"]
    assert row["measurement"]["reason"] == "identity_conflict"
    assert row["measurement"]["input_tokens"] is None
    assert len(row["source_refs"]) == 2


def test_model_filter_is_frozen_and_order_does_not_change_identity(
    tmp_path, unit_calculator, dify_record
):
    path, _ = run_records(tmp_path, [dify_record], unit_calculator, models=["test-model", "absent"])
    repeated, reused = run_records(
        tmp_path, [dify_record], unit_calculator, models=["absent", "test-model", "test-model"]
    )
    assert repeated == path and reused
    changed, reused = run_records(tmp_path, [dify_record], unit_calculator, models=["test-model"])
    assert changed != path and not reused
    assert verify_bundle(path)["plan"]["selection"]["models"] == ["absent", "test-model"]


@pytest.mark.parametrize("budget", [1, 2])
def test_filter_does_not_hide_parse_errors_or_extend_read_budget(
    tmp_path, unit_calculator, dify_record, budget
):
    path, _ = run_records(
        tmp_path,
        [dify_record, "not JSON"],
        unit_calculator,
        models=["not-present"],
        max_records=budget,
    )
    summary = loads((path / "summary.json").read_bytes())
    assert summary["run_status"] == "FAILED"
    assert summary["observed_calls"] == 0
    assert summary["counters"]["raw_records"] == budget
    assert summary["batch_read_complete"] == (budget == 2)


@pytest.mark.parametrize("models", [[], [""], [" "], "test-model"])
def test_empty_or_invalid_model_list_does_not_mean_all(
    tmp_path, unit_calculator, dify_record, models
):
    with pytest.raises(ValueError, match="nonempty list"):
        run_records(tmp_path, [dify_record], unit_calculator, models=models)
