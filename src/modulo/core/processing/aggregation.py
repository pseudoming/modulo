"""Aggregate observed measurements without turning missing values into zero."""

from collections import Counter

from modulo.shared.serialization import dumps

from ..domain.data_models import BatchResult, Measurement


def summarize(
    rows: list[dict],
    counters: Counter,
    diagnostics: list[dict],
    selection: dict,
    *,
    batch_read_complete: bool,
    working=None,
) -> BatchResult:
    errors = any(item["level"] == "error" for item in diagnostics)
    status = ("PARTIAL" if rows else "FAILED") if errors else "COMPLETE"
    models = working.mapping("model_totals") if working is not None else {}
    platform_models = working.mapping("platform_totals") if working is not None else {}
    verdicts: Counter = Counter()
    input_verdicts: Counter = Counter()
    reasons: Counter = Counter()
    statuses: Counter = Counter()
    output_statuses: Counter = Counter()
    output_reasons: Counter = Counter()
    output_verdicts: Counter = Counter()
    for row in rows:
        call, measured = row["call"], row["measurement"]
        statuses[call["status"]] += 1
        verdicts[measured["verdict"]] += 1
        input_verdicts[measured["input_verdict"]] += 1
        output_statuses[measured["output_execution"]] += 1
        output_verdicts[measured["output_verdict"]] += 1
        if measured["output_reason"]:
            output_reasons[measured["output_reason"]] += 1
        if measured["reason"]:
            reasons[measured["reason"]] += 1
        model = call["model"] or "(unknown)"

        def empty_group():
            return {
                "calls": 0,
                "cache_hit_calls": 0,
                "identity_conflicts": 0,
                "reported_input_sum": 0,
                "input_present": 0,
                "reported_output_sum": 0,
                "output_present": 0,
                "recalculated_input_sum": 0,
                "input_recalculated": 0,
                "image_input_recalculated": 0,
                "image_input_token_sum": 0,
                "compared_calls": 0,
                "signed_input_delta_sum": 0,
                "absolute_input_delta_sum": 0,
                "recalculated_output_sum": 0,
                "output_recalculated": 0,
                "output_compared_calls": 0,
                "signed_output_delta_sum": 0,
                "absolute_output_delta_sum": 0,
                "visible_output_recalculated": 0,
                "visible_output_token_sum": 0,
                "reasoning_output_recalculated": 0,
                "reasoning_output_token_sum": 0,
                "tool_output_recalculated": 0,
                "tool_output_token_sum": 0,
            }

        dimensions = row["observed_dimensions"]
        platforms_seen = {item["platform"] for item in dimensions}
        models_seen = {item["model"] for item in dimensions}
        platform_key = call["platform"] if len(platforms_seen) == 1 else None
        model_key = call["model"] if len(models_seen) == 1 else None
        groups = ((models, model), (platform_models, (platform_key, model_key)))
        for collection, group_key in groups:
            group = collection.get(group_key, empty_group())
            group["calls"] += 1
            if row["identity_conflict"]:
                group["identity_conflicts"] += 1
                collection[group_key] = group
                continue
            group["cache_hit_calls"] += call.get("cache_hit") is True
            for direction in ("input", "output"):
                value = call["usage"][direction + "_tokens"]
                if value is not None:
                    group[f"reported_{direction}_sum"] += value
                    group[f"{direction}_present"] += 1
            for direction in ("input", "output"):
                if measured[f"{direction}_tokens"] is not None:
                    group[f"recalculated_{direction}_sum"] += measured[f"{direction}_tokens"]
                    group[f"{direction}_recalculated"] += 1
                delta = measured[f"{direction}_delta"]
                if delta is not None:
                    group[
                        "compared_calls" if direction == "input" else "output_compared_calls"
                    ] += 1
                    group[f"signed_{direction}_delta_sum"] += delta
                    group[f"absolute_{direction}_delta_sum"] += abs(delta)
            for component in ("visible", "reasoning", "tool"):
                value = measured[f"{component}_output_tokens"]
                if value is not None:
                    group[f"{component}_output_token_sum"] += value
                    group[f"{component}_output_recalculated"] += 1
            if measured.get("image_input_tokens") is not None:
                group["image_input_token_sum"] += measured["image_input_tokens"]
                group["image_input_recalculated"] += 1
            collection[group_key] = group
    # All-missing sums remain null. Counts make partial sums explicit.
    for collection in (models, platform_models):
        for key in collection:
            group = collection[key]
            for direction in ("input", "output"):
                if group[f"{direction}_present"] == 0:
                    group[f"reported_{direction}_sum"] = None
                if not group[f"{direction}_recalculated"]:
                    group[f"recalculated_{direction}_sum"] = None
                if not group["compared_calls" if direction == "input" else "output_compared_calls"]:
                    group[f"signed_{direction}_delta_sum"] = None
                    group[f"absolute_{direction}_delta_sum"] = None
            for component in ("visible", "reasoning", "tool"):
                if not group[f"{component}_output_recalculated"]:
                    group[f"{component}_output_token_sum"] = None
            if not group["image_input_recalculated"]:
                group["image_input_token_sum"] = None
            collection[key] = group
    selection["selected_calls"] = len(rows)
    platform_rows = working.sequence("platform_summary") if working is not None else []
    for (platform, model), values in (
        sorted(platform_models.items(), key=lambda item: dumps(item[0]))
        if isinstance(platform_models, dict)
        else platform_models.items()
    ):
        platform_rows.append({"platform": platform, "model": model, **values})
    warnings = working.sequence("warnings") if working is not None else []
    warnings.extend(item for item in diagnostics if item["level"] == "warning")
    summary = {
        "schema_version": "summary.v3",
        "run_status": status,
        "observed_calls": len(rows),
        "counters": dict(counters),
        "call_status": dict(statuses),
        "verdicts": dict(verdicts),
        "input_verdicts": dict(input_verdicts),
        "unverifiable_reasons": dict(reasons),
        "output_execution": dict(output_statuses),
        "output_verdicts": dict(output_verdicts),
        "output_reasons": dict(output_reasons),
        "by_model": models,
        "by_model_scope": "all_platforms_combined",
        "by_platform_model": platform_rows,
        "diagnostic_count": len(diagnostics),
        "warnings": warnings,
        "input_batches": _input_batches(rows, working),
        "model_identity": "UNVERIFIED",
        "comparison_scope": "local_reference_input_output",
        "input_scope": "local_reference_input",
        "source_completeness": "unknown_before_batch_boundary",
        "output_scope": "local_reference_output",
        "selection": selection,
        "batch_read_complete": batch_read_complete,
    }
    return BatchResult(rows, diagnostics, summary)


def _input_batches(rows, working=None):
    """Compare Run totals once, separately from per-input conclusions and sums."""
    from ..domain.interfaces import MemoryEvents

    batches = working.groups("input_batches") if working is not None else MemoryEvents()
    for row in rows:
        batch = row["call"].get("input_batch")
        if batch:
            batches.append(
                (row["call"]["protocol"], batch["run_id"]),
                {
                    "call_key": row["call_key"],
                    "identity_conflict": row["identity_conflict"],
                    "call": {"input_batch": batch},
                    "measurement": row["measurement"],
                },
            )
    results = working.sequence("batch_totals") if working is not None else []
    for (protocol, run_id), members in batches.items():
        members = list(members)
        first = members[0]["call"]["input_batch"]
        size = first["size"]
        consistent = all(
            row["call"]["input_batch"]["size"] == size
            and row["call"]["input_batch"]["usage"] == first["usage"]
            and not row["identity_conflict"]
            for row in members
        )
        complete = consistent and sorted(
            row["call"]["input_batch"]["index"] for row in members
        ) == list(range(size))
        result = {
            "protocol": protocol,
            "run_id": run_id,
            "size": size,
            "selected_inputs": len(members),
            "complete": complete,
            "call_keys": [row["call_key"] for row in members],
            "usage": first["usage"] if consistent else None,
        }
        for direction in ("input", "output"):
            values = [row["measurement"][direction + "_tokens"] for row in members]
            execution_key = "execution" if direction == "input" else "output_execution"
            counted = (
                sum(values)
                if complete
                and all(v is not None for v in values)
                and all(row["measurement"][execution_key] == "completed" for row in members)
                else None
            )
            reported = first["usage"][direction + "_tokens"] if consistent else None
            delta = reported - counted if reported is not None and counted is not None else None
            result.update(
                {
                    direction + "_tokens": counted,
                    direction + "_delta": delta,
                    direction + "_verdict": "UNVERIFIABLE"
                    if delta is None
                    else ("CONSISTENT" if delta == 0 else "DEVIATION"),
                    direction + "_reason": (
                        "input_batch_incomplete_or_conflicting"
                        if not complete
                        else "input_batch_recalculation_incomplete"
                        if counted is None
                        else "reported_batch_usage_missing"
                        if reported is None
                        else None
                    ),
                }
            )
        result["verdict"] = Measurement(
            input_verdict=result["input_verdict"], output_verdict=result["output_verdict"]
        ).verdict
        results.append(result)
    return results
