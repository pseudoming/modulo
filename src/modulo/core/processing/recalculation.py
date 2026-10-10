"""Recalculate token counts and determine signed differences and result states."""

from collections.abc import Iterable
from typing import Any

from ..domain.bindings import BindingConflict, Bindings
from ..domain.data_models import Call, Measurement
from ..domain.interfaces import (
    AssetError,
    BackendLoader,
    Calculator,
    TokenizerBackend,
    UnsupportedInput,
)


def calculate_calls(
    selected: Iterable[dict], calculator: Calculator, diagnostics: list[dict]
) -> list[dict]:
    """Calculate selected calls without losing conflicts or component failures."""
    rows = []
    for row in selected:
        key = row["call_key"]
        call = Call.model_validate(row["call"])
        try:
            if row["identity_conflict"]:
                measured = Measurement(reason="identity_conflict")
            elif row["binding_error"]:
                measured = Measurement(
                    execution="error",
                    reason=row["binding_error"],
                    reported_input_tokens=call.usage.input_tokens,
                )
            else:
                measured = calculator.calculate(call, row["binding_matches"])
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
        base: dict[str, Any] = {"reported_input_tokens": call.usage.input_tokens}
        input_gap = ",".join(call.gaps) or ("missing_messages" if call.messages is None else None)
        matches = self.bindings.match_call(call) if binding_matches is None else binding_matches
        try:
            binding = self.bindings.select(matches)
        except BindingConflict as exc:
            return Measurement(**base, execution="error", reason=str(exc))
        if binding is None:
            if call.platform is None and call.platform_source != "none":
                return Measurement(**base, reason="platform_unresolved")
            missing_reason = (
                "platform_model_binding_missing"
                if call.platform is not None
                else "model_binding_missing"
            )
            return Measurement(
                **base, applicability="unsupported", reason=input_gap or missing_reason
            )
        base.update(backend=self.lock["assets"][binding.asset]["family"], applicability="supported")
        key = binding.asset
        if key not in self.backends:
            try:
                self.backends[key] = self.load_backend(binding)
            except AssetError as exc:
                self.backends[key] = str(exc)
        backend = self.backends[key]
        if isinstance(backend, str):
            return Measurement(**base, reason=backend)
        base["backend_version"] = backend.version
        try:
            count_output = getattr(backend, "count_output", None)
            if count_output is not None:
                base.update(count_output(call))
        except Exception as exc:
            base.update(
                output_execution="error", output_reason=f"tokenizer_error:{type(exc).__name__}"
            )
        if input_gap:
            return Measurement(**base, reason=input_gap)
        try:
            values = backend.count(call, binding)
        except UnsupportedInput as exc:
            return Measurement(**{**base, "applicability": "unsupported"}, reason=str(exc))
        except Exception as exc:
            # Keep type, never exception text potentially containing request content.
            return Measurement(
                **base, execution="error", reason=f"tokenizer_error:{type(exc).__name__}"
            )
        reported = call.usage.input_tokens
        delta = None if reported is None else reported - values["input_tokens"]
        verdict = "UNVERIFIABLE" if delta is None else "CONSISTENT" if delta == 0 else "DEVIATION"
        return Measurement(
            **{**base, **values},
            execution="completed",
            input_delta=delta,
            verdict=verdict,
            reason="reported_input_missing" if reported is None else None,
        )
