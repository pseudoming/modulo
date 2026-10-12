"""Select reproducible call samples after assembly, preserving shared-usage Runs."""

import heapq
from datetime import UTC, datetime

from modulo.shared.serialization import digest

from ..domain.interfaces import MemoryEvents
from ..domain.sampling import SamplingOptions


def sample_calls(rows, options: SamplingOptions, diagnostics, *, working=None):
    if options.mode == "all":
        return rows, None

    # These collections use the existing disk working set in production. Only
    # one group's count-sized priority heap needs to be held in memory.
    units = working.mapping("sampling_units") if working is not None else {}
    members = working.sequence("sampling_rows") if working is not None else []
    candidates = working.groups("sampling_candidates") if working is not None else MemoryEvents()
    chosen = working.mapping("sampling_chosen") if working is not None else {}
    report = {
        **options.model_dump(mode="json", by_alias=True),
        "method": "sha256-threshold" if options.mode == "ratio" else "sha256-bottom-k",
        "unit": "call_or_shared_usage_run",
        "matched_calls": 0,
        "eligible_calls": 0,
        "eligible_units": 0,
        "sampled_calls": 0,
        "sampled_units": 0,
        "outside_window_calls": 0,
        "time_unresolved_calls": 0,
        "strata": working.sequence("sampling_strata") if working is not None else [],
    }
    bounds = options.bounds()
    if bounds:
        report["window_start"], report["window_end"] = (t.isoformat() for t in bounds)

    for row in rows:
        report["matched_calls"] += 1
        call = row["call"]
        batch = call.get("input_batch")
        key = digest(["run", call["protocol"], batch["run_id"]]) if batch else row["call_key"]
        dimensions = row["observed_dimensions"]
        platforms, models = (
            {item[field] for item in dimensions} for field in ("platform", "model")
        )
        stratum = [
            next(iter(platforms)) if len(platforms) == 1 else None,
            next(iter(models)) if len(models) == 1 else None,
        ]
        started = _time(call.get("started_at")) if bounds else None
        if bounds and row["identity_conflict"]:
            started = None
        unit = units.get(key)
        if unit is None:
            unit = {"stratum": stratum, "started": started, "calls": 0}
        else:
            if unit["stratum"] != stratum:
                unit["stratum"] = [None, None]
            if unit["started"] != started:
                unit["started"] = None
        unit["calls"] += 1
        units[key] = unit
        members.append({**row, "sampling_unit": key})

    for key, unit in units.items():
        reason = None
        if bounds:
            if unit["started"] is None:
                reason = "sampling_time_unresolved"
                report["time_unresolved_calls"] += unit["calls"]
            elif not bounds[0] <= datetime.fromisoformat(unit["started"]) < bounds[1]:
                reason = "outside_sampling_window"
                report["outside_window_calls"] += unit["calls"]
        unit["reason"] = reason
        units[key] = unit
        if reason:
            continue
        report["eligible_units"] += 1
        report["eligible_calls"] += unit["calls"]
        # Time quotas apply to the entire selected window. Computational quotas
        # are per platform/model. A shared-usage Run is always one sampling unit.
        stratum = () if bounds else tuple(unit["stratum"])
        candidates.append(
            stratum,
            {"key": key, "priority": digest([options.seed, key]), "calls": unit["calls"]},
        )

    threshold = None
    if options.mode == "ratio":
        numerator, denominator = options.ratio.as_integer_ratio()
        threshold = numerator * 2**256 // denominator
    for stratum, entries in candidates.items():
        stats = {
            "platform": stratum[0] if stratum else None,
            "model": stratum[1] if stratum else None,
            "eligible_units": 0,
            "eligible_calls": 0,
            "sampled_units": 0,
            "sampled_calls": 0,
        }

        def counted_entries(entries=entries, stats=stats):
            for entry in entries:
                stats["eligible_units"] += 1
                stats["eligible_calls"] += entry["calls"]
                yield entry

        selected = (
            (entry for entry in counted_entries() if int(entry["priority"], 16) < threshold)
            if threshold is not None
            else heapq.nsmallest(
                options.count,
                counted_entries(),
                key=lambda entry: (entry["priority"], entry["key"]),
            )
        )
        for entry in selected:
            chosen[entry["key"]] = True
            stats["sampled_units"] += 1
            stats["sampled_calls"] += entry["calls"]
        report["sampled_units"] += stats["sampled_units"]
        report["sampled_calls"] += stats["sampled_calls"]
        report["strata"].append(stats)

    def selected_rows():
        for row in members:
            key = row["sampling_unit"]
            if key in chosen:
                yield row
                continue
            reason = units[key]["reason"] or "not_sampled"
            diagnostics.append(
                {
                    "level": "warning" if reason == "sampling_time_unresolved" else "info",
                    "reason": reason,
                    "call_key": row["call_key"],
                    "sampling_unit": key,
                    "source_refs": row["source_refs"],
                }
            )

    return selected_rows(), report


def _time(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC).isoformat() if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
