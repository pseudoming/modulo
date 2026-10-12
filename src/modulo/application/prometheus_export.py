"""Export existing result snapshots; never read upstream logs or run a tokenizer."""

import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modulo.adapters.exports.prometheus.adapter import Metric, encode, send
from modulo.core.processing.aggregation import add_to_group, empty_group
from modulo.shared.serialization import loads
from modulo.storage.result_bundle import verify_bundle

VERDICTS = ("CONSISTENT", "DEVIATION", "UNVERIFIABLE")
REASONS = (
    "cache_hit",
    "batch_usage",
    "missing_usage",
    "conflict",
    "error",
    "unsupported",
    "missing_content",
    "other",
)


class PrometheusOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=lambda name: name.replace("_", "-"),
        populate_by_name=True,
        hide_input_in_errors=True,
    )

    url: str
    scope: str = Field(min_length=1, max_length=128)
    username: str | None = Field(default=None, min_length=1)
    password_env: str | None = Field(default=None, min_length=1)
    bearer_token_env: str | None = Field(default=None, min_length=1)
    ca_file: Path | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=300, allow_inf_nan=False)
    max_series: int = Field(default=10000, ge=1, le=100000)
    visibility: Literal["auto", "full", "sample-only"] = "auto"

    @model_validator(mode="after")
    def validate_destination(self):
        try:
            url = urlsplit(self.url)
            valid = (
                url.scheme in {"http", "https"}
                and url.hostname
                and url.port != 0
                and not (url.username or url.password or url.query or url.fragment)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(
                "url must be an HTTP(S) endpoint without credentials, query or fragment"
            )
        if bool(self.username) != bool(self.password_env):
            raise ValueError("username and password-env must be configured together")
        if self.bearer_token_env and self.username:
            raise ValueError("Basic auth and bearer-token-env are mutually exclusive")
        return self


def export_prometheus(directory: Path, options: PrometheusOptions) -> dict:
    password = _secret(options.password_env)
    token = _secret(options.bearer_token_env)
    manifest = verify_bundle(directory)
    summary = loads((directory / "summary.json").read_bytes())
    sampling = summary.get("sampling")
    mode = (sampling or {}).get("mode", "all")
    visibility = options.visibility
    if visibility == "auto":
        visibility = "sample-only" if mode == "window" else "full"
    if visibility == "sample-only":
        if mode not in {"count", "window"}:
            raise ValueError("sample-only requires a fixed count or window sample")
        # An underfilled quota would reveal the entire small population. Keep
        # it local. No diagnostic counts, denominator or status reach the receiver.
        strata = sampling["strata"]
        if (
            not sampling["frame_complete"]
            or sampling["time_unresolved_calls"]
            or not strata
            or any(group["sampled_units"] != sampling["count"] for group in strata)
        ):
            return {
                "export": "withheld",
                "reason": "sample_not_publishable",
                "scope": options.scope,
            }
    timestamp_ms = time.time_ns() // 1_000_000
    metrics = snapshot_metrics(
        directory,
        manifest,
        summary,
        options.scope,
        timestamp_ms,
        options.max_series,
        sample_only=visibility == "sample-only",
    )
    payload = encode(metrics, timestamp_ms)
    send(
        options.url,
        payload,
        username=options.username,
        password=password,
        bearer_token=token,
        ca_file=str(options.ca_file) if options.ca_file else None,
        timeout_seconds=options.timeout_seconds,
    )
    return {
        "run_id": manifest["run_id"],
        "scope": options.scope,
        "series": len(metrics),
        "sample_timestamp_ms": timestamp_ms,
        "result_created_at": manifest["created_at"],
        "run_status": manifest["status"],
        "export": "sent",
        "visibility": visibility,
    }


def snapshot_metrics(
    directory, manifest, summary, scope, timestamp_ms, max_series, *, sample_only=False
):
    """Reuse core accumulators, including for bundles saved before grouped coverage existed."""
    metrics = []

    def add(name, value, **labels):
        if len(metrics) >= max_series:
            raise ValueError(
                "Prometheus export exceeds max-series; narrow input selection or raise the limit"
            )
        metrics.append(Metric("modulo_" + name, value, {"scope": scope, **labels}))

    add("snapshot_timestamp_seconds", timestamp_ms / 1000)
    sampling = summary.get("sampling")
    add("sample_only", int(sample_only))
    add("sampled_result", int(sampling is not None))
    created = datetime.fromisoformat(manifest["created_at"])
    if created.tzinfo is None:
        raise ValueError("result timestamp must include a timezone")
    add("result_timestamp_seconds", created.timestamp())
    if not sample_only:
        add("batch_read_complete", int(summary["batch_read_complete"]))
        for status in ("COMPLETE", "PARTIAL", "FAILED"):
            add("run_status", int(summary["run_status"] == status), status=status)
        for kind in ("raw", "invalid", "ignored"):
            add("source_records", summary["counters"].get(kind + "_records", 0), kind=kind)
        for selection in ("parsed", "selected", "excluded", "unresolved"):
            add("selection_calls", summary["selection"][selection + "_calls"], selection=selection)
        if sampling:
            for kind in (
                "eligible_calls",
                "eligible_units",
                "sampled_calls",
                "sampled_units",
                "outside_window_calls",
                "time_unresolved_calls",
            ):
                add("sampling_population", sampling[kind], kind=kind)
    if sampling:
        # Only the quota and window duration are public, never the true ratio,
        # population, seed, candidate IDs or outside-window counts.
        add("sample_units", sampling["sampled_units"])
        add("sampling_quota", sampling["count"] if sampling["count"] is not None else float("nan"))
        add(
            "sampling_window_seconds",
            sampling["window-minutes"] * 60
            if sampling["window-minutes"] is not None
            else float("nan"),
        )

    groups = {}
    # Read one saved call at a time. This shares aggregation with core rather
    # than maintaining a second interpretation of counts, conflicts or deltas.
    with (directory / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = loads(line)
            if "input_verdict" not in row["measurement"]:
                raise ValueError("result predates separate input/output verdicts; re-import logs")
            dimensions = row["observed_dimensions"]
            platforms = {item["platform"] for item in dimensions}
            models = {item["model"] for item in dimensions}
            key = (
                row["call"]["protocol"],
                (row["call"]["platform"] or None) if len(platforms) == 1 else None,
                (row["call"]["model"] or None) if len(models) == 1 else None,
            )
            if key not in groups:
                # Each group needs more than 40 series. Bound memory before
                # reading an unbounded number of distinct log model names.
                if len(groups) >= max_series // 40:
                    raise ValueError("Prometheus export exceeds max-series; too many model groups")
                groups[key] = empty_group()
            add_to_group(groups[key], row)
    for (adapter, platform, model), group in groups.items():
        labels = {"adapter": adapter}
        # Omit unknown dimensions; a real literal '(unknown)' remains distinct.
        if platform:
            labels["platform"] = platform
        if model:
            labels["model"] = model
        add("group_snapshot_timestamp_seconds", timestamp_ms / 1000, **labels)
        for name in ("calls", "cache_hit_calls", "identity_conflicts"):
            add(name, group[name], **labels)
        for direction in ("overall", "input", "output"):
            counts = group["verdicts" if direction == "overall" else direction + "_verdicts"]
            for verdict in VERDICTS:
                add(
                    "verdict_calls",
                    counts.get(verdict, 0),
                    **labels,
                    direction=direction,
                    verdict=verdict,
                )
        for direction in ("input", "output"):
            sides = {**labels, "direction": direction}
            compared = group["compared_calls" if direction == "input" else "output_compared_calls"]
            add("recalculated_calls", group[direction + "_completed"], **sides)
            add("compared_calls", compared, **sides)
            for status in ("completed", "partial", "not_run", "error"):
                add(
                    "execution_calls",
                    group[direction + "_execution"].get(status, 0),
                    **sides,
                    status=status,
                )
            for kind in ("reported", "recalculated"):
                add(
                    "compared_tokens",
                    group[f"compared_{direction}_{kind}_sum"] if compared else float("nan"),
                    **sides,
                    kind=kind,
                )
            signed = group[f"signed_{direction}_delta_sum"]
            absolute = group[f"absolute_{direction}_delta_sum"]
            for kind, value in (
                ("signed", signed),
                ("absolute", absolute),
                ("positive", (absolute + signed) // 2),
                ("negative", (signed - absolute) // 2),
            ):
                add("token_delta", value if compared else float("nan"), **sides, kind=kind)
            reasons = Counter()
            for reason, count in group[f"{direction}_reasons"].items():
                reasons[_reason_category(reason)] += count
            for reason in REASONS:
                add("reason_calls", reasons[reason], **sides, reason=reason)
    # Run-level usage is kept separate from per-input calls, without IDs as labels.
    batches = summary.get("input_batches", [])
    add("run_usage_groups", len(batches))
    for direction in ("overall", "input", "output"):
        counts = Counter(
            batch["verdict" if direction == "overall" else direction + "_verdict"]
            for batch in batches
        )
        for verdict in VERDICTS:
            add("run_usage_verdict_groups", counts[verdict], direction=direction, verdict=verdict)
    return metrics


def _secret(name):
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required credential environment variable is missing: {name}")
    if "\n" in value or "\r" in value:
        raise ValueError("credential must not contain line breaks")
    return value


def _reason_category(reason):
    if reason.startswith("cache_hit"):
        return "cache_hit"
    if "only_at_batch_level" in reason:
        return "batch_usage"
    if reason in {"reported_input_missing", "reported_output_missing"}:
        return "missing_usage"
    if "conflict" in reason or "ambiguous" in reason:
        return "conflict"
    if "error" in reason:
        return "error"
    if "unsupported" in reason or "binding" in reason or "unknown_model" in reason:
        return "unsupported"
    if "missing" in reason or "incomplete" in reason or "truncated" in reason:
        return "missing_content"
    return "other"
