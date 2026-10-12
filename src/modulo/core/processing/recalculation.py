"""Recalculate token counts and determine signed differences and result states."""

from collections.abc import Callable, Iterable
from typing import Any

from modulo.shared.serialization import digest

from ..domain.bindings import BindingConflict, Bindings
from ..domain.data_models import Call, CandidateMeasurement, Measurement, OutputContent
from ..domain.interfaces import (
    AssetError,
    BackendLoader,
    Calculator,
    TokenizerBackend,
    UnsupportedInput,
)


def _count_output(backend, call):
    count = getattr(backend, "count_output", None)
    if count is None:
        return {"output_reason": "output_counting_not_supported"}
    if not call.output_candidates:
        return count(call)
    details, counts, result = [], [], {}
    for candidate in call.output_candidates:
        single = call.model_copy(
            update={
                **{key: getattr(candidate, key) for key in OutputContent.model_fields},
                "output_candidates": [],
            }
        )
        try:
            values = count(single)
        except Exception as exc:
            values = {
                "output_execution": "error",
                "output_reason": f"tokenizer_error:{type(exc).__name__}",
            }
        counts.append(values)
        for key in (
            "asset_revision",
            "asset_hash",
            "template_hash",
            "output_render_options",
            "output_render_option_sources",
        ):
            if key in values:
                result[key] = values[key]
        details.append(
            CandidateMeasurement(
                index=candidate.index,
                execution=values.get("output_execution", "not_run"),
                tokens=values.get("output_tokens"),
                body_tokens=values.get("output_body_tokens"),
                visible_tokens=values.get("visible_output_tokens"),
                reasoning_tokens=values.get("reasoning_output_tokens"),
                tool_tokens=values.get("tool_output_tokens"),
                reason=values.get("output_reason"),
                reasoning_source=values.get("reasoning_output_source"),
                rendered_sha256=values.get("rendered_output_sha256"),
                ids_sha256=values.get("output_ids_sha256"),
                ending=values.get("output_ending"),
                termination=values.get("output_termination"),
                render_options=values.get("output_render_options", {}),
                render_option_sources=values.get("output_render_option_sources", {}),
            )
        )
    complete = not call.output_gaps and all(item.execution == "completed" for item in details)
    result.update(
        output_candidates=details,
        output_execution="completed"
        if complete
        else "error"
        if any(item.execution == "error" for item in details)
        else "partial",
        output_reason=None
        if complete
        else ",".join(
            [
                *call.output_gaps,
                *(
                    f"candidate[{item.index}]:{item.reason or item.execution}"
                    for item in details
                    if item.execution != "completed"
                ),
            ]
        ),
    )
    # Candidates are separate generated sequences. Sum their independently
    # encoded counts, never concatenate their text or repeat request usage.
    for key in (
        "output_tokens",
        "output_body_tokens",
        "visible_output_tokens",
        "reasoning_output_tokens",
        "tool_output_tokens",
    ):
        values = [item.get(key) for item in counts]
        if not call.output_gaps and all(value is not None for value in values):
            if key != "output_tokens" or complete:
                result[key] = sum(values)
    return result


def calculate_calls(
    selected: Iterable[dict],
    calculator: Calculator,
    diagnostics: list[dict],
    prepare_call: Callable[[Call, list[int]], Call] | None = None,
    rows=None,
    cache=None,
) -> list[dict]:
    """Calculate selected calls without losing conflicts or component failures."""
    rows = [] if rows is None else rows
    for row in selected:
        key = row["call_key"]
        call = Call.model_validate(row["call"])
        try:
            if row["identity_conflict"]:
                measured = Measurement(
                    reason="identity_conflict", output_reason="identity_conflict"
                )
            elif row["binding_error"]:
                measured = Measurement(
                    execution="error",
                    reason=row["binding_error"],
                    output_reason=row["binding_error"],
                    reported_input_tokens=call.usage.input_tokens,
                    reported_output_tokens=call.usage.output_tokens,
                )
            else:
                if prepare_call and row["binding_matches"]:
                    call = prepare_call(call, row["binding_matches"])
                    row["call"] = call.model_dump()
                cache_key = None
                if cache is not None:
                    cache_key = digest([call.model_dump(), row["binding_matches"]])
                if cache_key is not None and cache_key in cache:
                    measured = Measurement.model_validate(cache[cache_key])
                else:
                    measured = calculator.calculate(call, row["binding_matches"])
                    if (
                        cache_key is not None
                        and measured.execution != "error"
                        and measured.output_execution != "error"
                    ):
                        cache[cache_key] = measured.model_dump()
        except Exception as exc:
            measured = Measurement(execution="error", reason=f"backend_error:{type(exc).__name__}")
        if measured.execution == "error":
            diagnostics.append({"level": "error", "reason": measured.reason, "call_key": key})
        if measured.output_execution == "error":
            diagnostics.append(
                {
                    "level": "error",
                    "reason": measured.output_reason,
                    "component": "output",
                    "call_key": key,
                }
            )
        row["measurement"] = measured.model_dump()
        rows.append(row)
    return rows


class Recalculator:
    def __init__(
        self, bindings: Bindings, load_backend: BackendLoader, lock: dict[str, Any]
    ) -> None:
        self.bindings, self.load_backend, self.lock = bindings, load_backend, lock
        self.backends: dict[str, TokenizerBackend | str] = {}

    def calculate(self, call: Call, binding_matches: list[int] | None = None) -> Measurement:
        base: dict[str, Any] = {
            "reported_input_tokens": call.usage.input_tokens,
            "reported_output_tokens": call.usage.output_tokens,
        }
        input_gap = ",".join(call.gaps) or ("missing_messages" if call.messages is None else None)
        matches = self.bindings.match_call(call) if binding_matches is None else binding_matches
        try:
            binding = self.bindings.select(matches)
        except BindingConflict as exc:
            return Measurement(**base, execution="error", reason=str(exc), output_reason=str(exc))
        if binding is None:
            if call.platform is None and call.platform_source != "none":
                return Measurement(
                    **base, reason="platform_unresolved", output_reason="platform_unresolved"
                )
            missing_reason = (
                "platform_model_binding_missing"
                if call.platform is not None
                else "model_binding_missing"
            )
            return Measurement(
                **base,
                applicability="unsupported",
                reason=input_gap or missing_reason,
                output_reason=missing_reason,
            )
        base.update(backend=binding.name, applicability="supported")
        key = binding.reference_key
        if key not in self.backends:
            try:
                self.backends[key] = self.load_backend(binding)
            except AssetError as exc:
                self.backends[key] = str(exc)
        backend = self.backends[key]
        if isinstance(backend, str):
            return Measurement(**base, reason=backend, output_reason=backend)
        base["backend_version"] = backend.version
        try:
            output = _count_output(backend, call)
            if output:
                base.update(output)
                if base.get("output_execution") == "completed":
                    reported = call.usage.output_tokens
                    if call.cache_hit:
                        base["output_reason"] = "cache_hit_usage_not_current_inference"
                    elif reported is None:
                        base["output_reason"] = (
                            "reported_output_only_at_batch_level"
                            if call.input_batch and call.input_batch.usage.output_tokens is not None
                            else "reported_output_missing"
                        )
                    else:
                        delta = reported - base["output_tokens"]
                        base.update(
                            output_delta=delta,
                            output_verdict="CONSISTENT" if delta == 0 else "DEVIATION",
                        )
            else:
                base["output_reason"] = "output_counting_not_supported"
        except Exception as exc:
            base.update(
                output_execution="error", output_reason=f"tokenizer_error:{type(exc).__name__}"
            )
        if input_gap:
            return Measurement(**base, reason=input_gap)
        try:
            values = backend.count(call, binding)
        except UnsupportedInput as exc:
            return Measurement(
                **{**base, **exc.evidence, "applicability": "unsupported"}, reason=str(exc)
            )
        except Exception as exc:
            # Keep type, never exception text potentially containing request content.
            return Measurement(
                **base, execution="error", reason=f"tokenizer_error:{type(exc).__name__}"
            )
        reported = call.usage.input_tokens
        delta = None if reported is None or call.cache_hit else reported - values["input_tokens"]
        verdict = "UNVERIFIABLE" if delta is None else "CONSISTENT" if delta == 0 else "DEVIATION"
        reason = None
        if call.cache_hit:
            reason = "cache_hit_usage_not_current_inference"
        elif reported is None:
            reason = (
                "reported_input_only_at_batch_level"
                if call.input_batch and call.input_batch.usage.input_tokens is not None
                else "reported_input_missing"
            )
        return Measurement(
            **{**base, **values},
            execution="completed",
            input_delta=delta,
            input_verdict=verdict,
            reason=reason,
        )
