"""Aggregate observed measurements without turning missing values into zero."""

from collections import Counter

from modulo.shared.serialization import dumps

from ..domain.data_models import BatchResult


def summarize(
    rows: list[dict],
    counters: Counter,
    diagnostics: list[dict],
    selection: dict,
    *,
    batch_read_complete: bool,
) -> BatchResult:
    errors = any(item["level"] == "error" for item in diagnostics)
    status = ("PARTIAL" if rows else "FAILED") if errors else "COMPLETE"
    models: dict[str, dict] = {}
    platform_models: dict[tuple[str | None, str | None], dict] = {}
    verdicts: Counter = Counter()
    reasons: Counter = Counter()
    statuses: Counter = Counter()
    output_statuses: Counter = Counter()
    output_reasons: Counter = Counter()
    for row in rows:
        call, measured = row["call"], row["measurement"]
        statuses[call["status"]] += 1
        verdicts[measured["verdict"]] += 1
        output_statuses[measured["output_execution"]] += 1
        if measured["output_reason"]:
            output_reasons[measured["output_reason"]] += 1
        if measured["reason"]:
            reasons[measured["reason"]] += 1
        model = call["model"] or "(unknown)"

        def empty_group():
            return {
                "calls": 0,
                "identity_conflicts": 0,
                "reported_input_sum": 0,
                "input_present": 0,
                "reported_output_sum": 0,
                "output_present": 0,
                "recalculated_input_sum": 0,
                "input_recalculated": 0,
                "compared_calls": 0,
                "signed_input_delta_sum": 0,
                "absolute_input_delta_sum": 0,
                "visible_output_recalculated": 0,
                "visible_output_token_sum": 0,
                "reasoning_output_recalculated": 0,
                "reasoning_output_token_sum": 0,
            }

        dimensions = row["observed_dimensions"]
        platforms_seen = {item["platform"] for item in dimensions}
        models_seen = {item["model"] for item in dimensions}
        platform_key = call["platform"] if len(platforms_seen) == 1 else None
        model_key = call["model"] if len(models_seen) == 1 else None
        groups = (
            models.setdefault(model, empty_group()),
            platform_models.setdefault((platform_key, model_key), empty_group()),
        )
        for group in groups:
            group["calls"] += 1
            if row["identity_conflict"]:
                group["identity_conflicts"] += 1
                continue
            for direction in ("input", "output"):
                value = call["usage"][direction + "_tokens"]
                if value is not None:
                    group[f"reported_{direction}_sum"] += value
                    group[f"{direction}_present"] += 1
            if measured["input_tokens"] is not None:
                group["recalculated_input_sum"] += measured["input_tokens"]
                group["input_recalculated"] += 1
            for component in ("visible", "reasoning"):
                value = measured[f"{component}_output_tokens"]
                if value is not None:
                    group[f"{component}_output_token_sum"] += value
                    group[f"{component}_output_recalculated"] += 1
            delta = measured["input_delta"]
            if delta is not None:
                group["compared_calls"] += 1
                group["signed_input_delta_sum"] += delta
                group["absolute_input_delta_sum"] += abs(delta)
    # All-missing sums remain null. Counts make partial sums explicit.
    for group in [*models.values(), *platform_models.values()]:
        for direction in ("input", "output"):
            if group[f"{direction}_present"] == 0:
                group[f"reported_{direction}_sum"] = None
        if not group["input_recalculated"]:
            group["recalculated_input_sum"] = None
        for component in ("visible", "reasoning"):
            if not group[f"{component}_output_recalculated"]:
                group[f"{component}_output_token_sum"] = None
        if not group["compared_calls"]:
            group["signed_input_delta_sum"] = None
            group["absolute_input_delta_sum"] = None
    selection["selected_calls"] = len(rows)
    summary = {
        "schema_version": "summary.v3",
        "run_status": status,
        "observed_calls": len(rows),
        "counters": dict(counters),
        "call_status": dict(statuses),
        "verdicts": dict(verdicts),
        "unverifiable_reasons": dict(reasons),
        "output_execution": dict(output_statuses),
        "output_reasons": dict(output_reasons),
        "by_model": models,
        "by_model_scope": "all_platforms_combined",
        "by_platform_model": [
            {"platform": platform, "model": model, **values}
            for (platform, model), values in sorted(
                platform_models.items(), key=lambda item: dumps(item[0])
            )
        ],
        "diagnostic_count": len(diagnostics),
        "model_identity": "UNVERIFIED",
        "comparison_scope": "local_reference_input",
        "source_completeness": "unknown_before_batch_boundary",
        "output_scope": "text_components_reencoding_only",
        "selection": selection,
        "batch_read_complete": batch_read_complete,
    }
    return BatchResult(rows, diagnostics, summary)
