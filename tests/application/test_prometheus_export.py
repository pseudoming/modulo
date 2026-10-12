"""Export verified results, retaining missing data, scope and per-side denominators."""

import json
import math
from collections import Counter
from datetime import datetime

import pytest

from modulo.application.prometheus_export import PrometheusOptions, export_prometheus
from modulo.core.domain.bindings import Bindings
from modulo.core.domain.data_models import Call, Measurement, Usage
from modulo.core.domain.sampling import SamplingOptions
from modulo.core.processing.aggregation import summarize
from modulo.core.processing.sampling import sample_calls
from modulo.storage.result_bundle import BundleWriter


def row(number, *, reported=12, counted=10, platform="供应商 A", model="model", cache=False):
    call = Call(
        call_id=str(number),
        protocol="litellm",
        platform=platform,
        model=model,
        usage=Usage(input_tokens=reported),
        cache_hit=cache,
    )
    delta = (
        reported - counted if reported is not None and counted is not None and not cache else None
    )
    measured = Measurement(
        execution="completed" if counted is not None else "not_run",
        input_tokens=counted,
        reported_input_tokens=reported,
        input_delta=delta,
        input_verdict="UNVERIFIABLE"
        if delta is None
        else "CONSISTENT"
        if delta == 0
        else "DEVIATION",
        reason="cache_hit_usage_not_current_inference" if cache else None,
        output_execution="partial",
        visible_output_tokens=3,
        output_reason="output_incomplete",
    )
    return {
        "call_key": str(number),
        "call": call.model_dump(),
        "measurement": measured.model_dump(),
        "identity_conflict": False,
        "observed_dimensions": [{"platform": platform, "model": model}],
    }


def bundle(tmp_path, rows, *, sampling=None, complete=True, missing_time=False):
    population = len(rows)
    diagnostics = []
    report = None
    if sampling is not None:
        for item in rows:
            item["call"]["started_at"] = "2026-10-13T02:01:00Z"
            item["source_refs"] = []
        if missing_time:
            rows[-1]["call"]["started_at"] = None
        selected, report = sample_calls(rows, sampling, diagnostics)
        rows = list(selected)
    result = summarize(
        rows,
        Counter(raw_records=population),
        diagnostics,
        {
            "parsed_calls": population,
            "selected_calls": population,
            "excluded_calls": 0,
            "unresolved_calls": 0,
        },
        batch_read_complete=complete,
        sampling=report,
    )
    with BundleWriter(tmp_path / "results") as writer:
        directory, _ = writer.publish(
            {},
            result,
            execution={},
            bindings=Bindings(),
            asset_lock={},
            rebuild_index=lambda directory: None,
        )
    return directory


def capture(monkeypatch):
    sent = []

    def encode(metrics, timestamp):
        sent.append((metrics, timestamp))
        return b"encoded"

    monkeypatch.setattr("modulo.application.prometheus_export.encode", encode)
    monkeypatch.setattr("modulo.application.prometheus_export.send", lambda *a, **k: None)
    return sent


def value(metrics, name, **labels):
    matches = [
        item.value
        for item in metrics
        if item.name == "modulo_" + name
        and all(item.labels.get(key) == val for key, val in labels.items())
    ]
    assert len(matches) == 1
    return matches[0]


def test_snapshot_keeps_comparable_cohort_and_independent_directions(tmp_path, monkeypatch):
    directory = bundle(
        tmp_path,
        [
            row(0),
            row(1, reported=None, counted=90),
            row(2, reported=100, counted=None),
            row(3, reported=2, counted=5),
            row(4, cache=True),
        ],
    )
    sent = capture(monkeypatch)
    options = PrometheusOptions(url="http://localhost:9090/api/v1/write", scope="gateway")
    result = export_prometheus(directory, options)
    metrics, timestamp = sent[0]
    assert result["export"] == "sent" and result["sample_timestamp_ms"] == timestamp
    assert value(metrics, "calls") == 5
    assert value(metrics, "recalculated_calls", direction="input") == 4
    assert value(metrics, "recalculated_calls", direction="output") == 0
    assert value(metrics, "compared_calls", direction="input") == 2
    assert value(metrics, "compared_tokens", direction="input", kind="reported") == 14
    assert value(metrics, "compared_tokens", direction="input", kind="recalculated") == 15
    for kind, expected in {"signed": -1, "absolute": 5, "positive": 2, "negative": -3}.items():
        assert value(metrics, "token_delta", direction="input", kind=kind) == expected
    assert math.isnan(value(metrics, "token_delta", direction="output", kind="signed"))
    assert value(metrics, "cache_hit_calls") == 1
    assert value(metrics, "verdict_calls", direction="overall", verdict="DEVIATION") == 2
    assert value(metrics, "reason_calls", direction="input", reason="cache_hit") == 1
    assert all(item.labels["scope"] == "gateway" for item in metrics)
    assert not any(key in item.labels for item in metrics for key in ("run_id", "call_id", "path"))
    created = datetime.fromisoformat(result["result_created_at"]).timestamp()
    assert value(metrics, "result_timestamp_seconds") == created
    export_prometheus(directory, options)
    assert value(sent[1][0], "calls") == 5  # snapshot, not an export counter
    assert value(sent[1][0], "result_timestamp_seconds") == created


def test_empty_result_has_zero_denominators_but_no_fake_model_or_token_metrics(
    tmp_path, monkeypatch
):
    sent = capture(monkeypatch)
    export_prometheus(
        bundle(tmp_path, []), PrometheusOptions(url="http://localhost/api/v1/write", scope="empty")
    )
    metrics, _ = sent[0]
    assert value(metrics, "selection_calls", selection="selected") == 0
    assert not any(item.name in {"modulo_calls", "modulo_compared_tokens"} for item in metrics)


def test_tampered_result_fails_before_network(tmp_path, monkeypatch):
    directory = bundle(tmp_path, [row(0)])
    (directory / "results.jsonl").write_text("changed")
    sent = capture(monkeypatch)
    with pytest.raises(ValueError, match="hash mismatch"):
        export_prometheus(directory, PrometheusOptions(url="http://localhost/write", scope="a"))
    assert sent == []


def test_unknown_dimensions_do_not_collide_with_empty_values_or_literal_names(
    tmp_path, monkeypatch
):
    directory = bundle(
        tmp_path,
        [
            row(0, model=None, platform=None),
            row(1, model="", platform=""),
            row(2, model="(unknown)", platform="(unknown)"),
        ],
    )
    sent = capture(monkeypatch)
    export_prometheus(directory, PrometheusOptions(url="http://localhost/write", scope="unknown"))
    calls = [metric for metric in sent[0][0] if metric.name == "modulo_calls"]
    assert len(calls) == 2
    assert [(metric.labels.get("model"), metric.value) for metric in calls] == [
        (None, 2),
        ("(unknown)", 1),
    ]


def test_series_limit_is_checked_before_network_and_results_remain_unchanged(tmp_path, monkeypatch):
    directory = bundle(tmp_path, [row(0)])
    original = (directory / "manifest.json").read_bytes()
    sent = capture(monkeypatch)
    with pytest.raises(ValueError, match="max-series"):
        export_prometheus(
            directory, PrometheusOptions(url="http://localhost/write", scope="a", max_series=10)
        )
    assert sent == [] and (directory / "manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    "values",
    [
        {"url": "file:///tmp/write"},
        {"url": "http://name:secret@host/write"},
        {"url": "http://host/write?token=secret"},
        {"username": "writer"},
        {"username": ""},
        {"bearer-token-env": ""},
        {"password-env": "PASSWORD"},
        {"username": "writer", "password-env": "PASSWORD", "bearer-token-env": "TOKEN"},
        {"timeout-seconds": "nan"},
        {"max-series": 0},
        {"scope": ""},
    ],
)
def test_export_options_reject_invalid_destinations_and_auth(values):
    with pytest.raises(ValueError) as exc:
        PrometheusOptions.model_validate({"url": "http://localhost/write", "scope": "a", **values})
    assert "secret" not in str(exc.value)


def test_credentials_are_explicit_and_never_in_export_output(tmp_path, monkeypatch):
    directory = bundle(tmp_path, [])
    sent = []
    monkeypatch.setattr("modulo.application.prometheus_export.send", lambda *a, **k: sent.append(k))
    options = PrometheusOptions(
        url="http://localhost/write", scope="a", bearer_token_env="TEST_TOKEN"
    )
    monkeypatch.delenv("TEST_TOKEN", raising=False)
    with pytest.raises(ValueError, match="environment variable is missing"):
        export_prometheus(directory, options)
    assert sent == []
    monkeypatch.setenv("TEST_TOKEN", "private-token")
    result = export_prometheus(directory, options)
    assert sent[0]["bearer_token"] == "private-token"
    assert "private-token" not in json.dumps(result)


def window(count=3):
    return SamplingOptions(
        mode="window",
        window_minutes=5,
        count=count,
        seed="BUYER-SECRET-SEED",
        before="2026-10-13T02:05:00Z",
    )


def test_time_sample_auto_exports_only_sample_metrics_without_population(tmp_path, monkeypatch):
    directory = bundle(tmp_path, [row(i) for i in range(12)], sampling=window())
    sent = capture(monkeypatch)
    result = export_prometheus(
        directory, PrometheusOptions(url="http://localhost/write", scope="a")
    )
    assert result["visibility"] == "sample-only"
    metrics, _ = sent[0]
    assert value(metrics, "calls") == 3
    assert value(metrics, "sample_only") == value(metrics, "sampled_result") == 1
    assert value(metrics, "sampling_quota") == 3
    assert value(metrics, "sampling_window_seconds") == 300
    forbidden = {
        "modulo_source_records",
        "modulo_selection_calls",
        "modulo_sampling_population",
        "modulo_run_status",
        "modulo_batch_read_complete",
    }
    assert not {item.name for item in metrics} & forbidden
    assert "BUYER-SECRET-SEED" not in repr(metrics)
    assert not any("ratio" in item.name or "eligible" in repr(item.labels) for item in metrics)


@pytest.mark.parametrize("count,complete", [(0, True), (2, True), (10, False)])
def test_insufficient_or_incomplete_sample_sends_no_http_request(
    tmp_path, monkeypatch, count, complete
):
    directory = bundle(
        tmp_path, [row(i) for i in range(count)], sampling=window(), complete=complete
    )
    sent = capture(monkeypatch)
    result = export_prometheus(
        directory, PrometheusOptions(url="http://localhost/write", scope="a")
    )
    assert result == {"export": "withheld", "reason": "sample_not_publishable", "scope": "a"}
    assert sent == []


def test_buyer_can_explicitly_export_population_to_internal_prometheus(tmp_path, monkeypatch):
    directory = bundle(tmp_path, [row(i) for i in range(12)], sampling=window())
    sent = capture(monkeypatch)
    export_prometheus(
        directory, PrometheusOptions(url="http://localhost/write", scope="a", visibility="full")
    )
    metrics, _ = sent[0]
    assert value(metrics, "source_records", kind="raw") == 12
    assert value(metrics, "selection_calls", selection="selected") == 12
    assert value(metrics, "sampling_population", kind="eligible_calls") == 12
    assert value(metrics, "sample_only") == 0


def test_unknown_time_keeps_window_local_even_if_quota_is_filled(tmp_path, monkeypatch):
    directory = bundle(tmp_path, [row(i) for i in range(12)], sampling=window(), missing_time=True)
    sent = capture(monkeypatch)
    result = export_prometheus(
        directory, PrometheusOptions(url="http://localhost/write", scope="a")
    )
    assert result["export"] == "withheld" and sent == []


def test_fixed_count_can_be_explicitly_restricted(tmp_path, monkeypatch):
    directory = bundle(
        tmp_path, [row(i) for i in range(12)], sampling=SamplingOptions(mode="count", count=3)
    )
    sent = capture(monkeypatch)
    result = export_prometheus(
        directory,
        PrometheusOptions(url="http://localhost/write", scope="a", visibility="sample-only"),
    )
    assert result["export"] == "sent"
    assert not any(item.name == "modulo_sampling_population" for item in sent[0][0])


@pytest.mark.parametrize("sampling", [None, SamplingOptions(mode="ratio", ratio=".1")])
def test_sample_only_refuses_full_or_proportional_samples(tmp_path, monkeypatch, sampling):
    directory = bundle(tmp_path, [row(i) for i in range(10)], sampling=sampling)
    sent = capture(monkeypatch)
    with pytest.raises(ValueError, match="fixed count or window"):
        export_prometheus(
            directory,
            PrometheusOptions(url="http://localhost/write", scope="a", visibility="sample-only"),
        )
    assert sent == []
