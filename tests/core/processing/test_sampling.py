"""Sampling invariants: reproducibility, atomic calls, time boundaries and disk parity."""

from contextlib import nullcontext
from datetime import UTC, datetime

import pytest

from modulo.core.domain.sampling import SamplingOptions
from modulo.core.processing.sampling import sample_calls
from modulo.storage.working_set import WorkingSet


def row(number, *, platform="p", model="m", started="2026-10-13T10:01:00+08:00", batch=None):
    return {
        "call_key": str(number),
        "call": {
            "call_id": str(number),
            "protocol": "test",
            "started_at": started,
            "input_batch": batch,
        },
        "observed_dimensions": [{"platform": platform, "model": model}],
        "identity_conflict": False,
        "source_refs": [{"locator": f"line:{number}"}],
    }


def sample(rows, *, working=None, **options):
    diagnostics = []
    selected, report = sample_calls(rows, SamplingOptions(**options), diagnostics, working=working)
    return list(selected), report, diagnostics


def keys(rows):
    return {item["call_key"] for item in rows}


def test_all_is_identity_and_does_not_read_the_iterator_early():
    def rows():
        raise AssertionError("must remain lazy")
        yield

    stream = rows()
    selected, report = sample_calls(stream, SamplingOptions(), [])
    assert selected is stream and report is None


@pytest.mark.parametrize("disk", [False, True])
def test_count_is_reproducible_in_each_platform_model_stratum(tmp_path, disk):
    rows = [row(i, platform=f"p{i // 10}", model=f"m{i % 2}") for i in range(40)]
    with WorkingSet(tmp_path / "work") if disk else nullcontext(None) as working:
        chosen, report, excluded = sample(
            rows, mode="count", count=2, seed="buyer", working=working
        )
        expected, _, _ = sample(reversed(rows), mode="count", count=2, seed="buyer")
        assert keys(chosen) == keys(expected)
        assert report["eligible_units"] == report["eligible_calls"] == 40
        assert report["sampled_units"] == report["sampled_calls"] == 16
        assert len(excluded) == 24
        assert all(item["reason"] == "not_sampled" and item["source_refs"] for item in excluded)
        assert all(group["sampled_calls"] == 2 for group in report["strata"])
    different, _, _ = sample(rows, mode="count", count=2, seed="different")
    assert keys(chosen) != keys(different)


def test_ratio_decisions_survive_reordering_and_growth():
    rows = [row(i) for i in range(100)]
    chosen, report, _ = sample(rows[:50], mode="ratio", ratio="0.2", seed="buyer")
    grown, _, _ = sample(reversed(rows), mode="ratio", ratio="0.2", seed="buyer")
    assert 0 < len(chosen) < 50
    assert keys(chosen) == keys(grown) & keys(rows[:50])
    assert report["eligible_units"] == 50
    assert report["sampled_units"] == len(chosen)
    all_rows, _, _ = sample(rows, mode="ratio", ratio="1")
    assert keys(all_rows) == keys(rows)


def test_shared_usage_run_is_never_split_even_if_it_exceeds_the_quota_in_calls():
    rows = [row(i, batch={"run_id": "shared", "size": 3, "index": i}) for i in range(3)]
    chosen, report, _ = sample(rows, mode="count", count=1)
    assert keys(chosen) == {"0", "1", "2"}
    assert len({item["sampling_unit"] for item in chosen}) == 1
    assert report["sampled_units"] == report["eligible_units"] == 1
    assert report["sampled_calls"] == 3


def test_window_is_start_inclusive_end_exclusive_and_never_guesses_missing_times():
    rows = [
        row(0, started="2026-10-13T01:59:59.999999Z"),
        row(1, started="2026-10-13T10:00:00+08:00"),
        row(2, started="2026-10-13T02:04:59.999999Z"),
        row(3, started="2026-10-13T10:05:00+08:00"),
        row(4, started=None),
        row(5, started="2026-10-13T10:01:00"),
        row(6, started="invalid"),
    ]
    chosen, report, excluded = sample(
        rows, mode="window", window_minutes=5, count=100, before="2026-10-13T02:05:00Z"
    )
    assert keys(chosen) == {"1", "2"}
    assert report["outside_window_calls"] == 2
    assert report["time_unresolved_calls"] == 3
    assert report["eligible_calls"] == 2
    assert sum(item["level"] == "warning" for item in excluded) == 3


def test_window_quota_is_global_and_shared_run_times_must_agree():
    rows = [row(i, platform=f"p{i}") for i in range(10)]
    rows.extend(
        [
            row(10, started="2026-10-13T10:00:00+08:00", batch={"run_id": "conflicting"}),
            row(11, started="2026-10-13T10:01:00+08:00", batch={"run_id": "conflicting"}),
        ]
    )
    chosen, report, _ = sample(
        rows, mode="window", window_minutes=5, count=3, before="2026-10-13T02:05:00Z"
    )
    assert len(chosen) == 3
    assert len(report["strata"]) == 1
    assert report["time_unresolved_calls"] == 2
    assert report["eligible_calls"] == 10


def test_clock_is_frozen_at_previous_complete_utc_window():
    options = SamplingOptions(mode="window", window_minutes=5, count=10)
    frozen = options.resolve(datetime.fromisoformat("2026-10-13T10:07:32+08:00"))
    start, end = frozen.bounds()
    assert start == datetime(2026, 10, 13, 2, 0, tzinfo=UTC)
    assert end == datetime(2026, 10, 13, 2, 5, tzinfo=UTC)
    assert options.before is None
    historical = options.model_copy(update={"before": end})
    assert historical.resolve(datetime(2026, 10, 20, tzinfo=UTC)).before == end
    with pytest.raises(ValueError, match="freeze"):
        options.bounds()


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "ratio"},
        {"mode": "ratio", "ratio": 0},
        {"mode": "ratio", "ratio": "NaN"},
        {"mode": "ratio", "ratio": "1.01"},
        {"mode": "count"},
        {"mode": "count", "count": True},
        {"mode": "count", "count": 1.1},
        {"mode": "count", "count": 100_001},
        {"mode": "count", "count": 1, "ratio": ".1"},
        {"mode": "all", "count": 1},
        {"mode": "window", "count": 1},
        {"mode": "window", "count": 1, "window-minutes": 0},
        {"mode": "window", "count": 1, "window-minutes": 5, "before": "2026-10-13T10:05:00"},
        {"mode": "window", "count": 1, "window-minutes": 5, "before": 1000},
        {"mode": "all", "before": "2026-10-13T10:05:00Z"},
        {"mode": "unknown"},
    ],
)
def test_invalid_sampling_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        SamplingOptions.model_validate(values)
