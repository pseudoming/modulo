"""LangChain serialized Run parser with per-child message and metadata handling."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from modulo.adapters.protocols.shared.messages.candidates import expected_candidates, group_outputs
from modulo.adapters.protocols.shared.messages.chat import message, output_fields, request_options
from modulo.core.domain.data_models import (
    Call,
    DataError,
    InputBatch,
    Message,
    Parsed,
    ProtocolContext,
    Usage,
)
from modulo.core.domain.interfaces import ProtocolAdapter

from ..shared.fields import identifier, mapping, text_or_none, timestamp, usage_from


def _terminal_message(body):
    """A Run's terminal generation may still serialize as AIMessageChunk."""
    if body.get("chunk_position") not in (None, "last"):
        raise DataError("invalid_langchain_chunk_position")
    chunks = body.get("tool_call_chunks")
    additional = mapping(body.get("additional_kwargs") or {}, "additional_kwargs")
    if chunks and not additional.get("tool_calls"):
        if not isinstance(chunks, list):
            raise DataError("incomplete_langchain_tool_chunks")
        calls = []
        for chunk in chunks:
            if not isinstance(chunk, dict) or any(
                not isinstance(chunk.get(k), str) for k in ("id", "name", "args")
            ):
                raise DataError("incomplete_langchain_tool_chunks")
            calls.append(
                {
                    "id": chunk["id"],
                    "type": "function",
                    "function": {"name": chunk["name"], "arguments": chunk["args"]},
                }
            )
        body = {**body, "additional_kwargs": {**additional, "tool_calls": calls}}
    # The framework has already aggregated these chunks at on_llm_end. Keep
    # the complete calls (including original argument strings), not a second
    # copy of its chunk-index bookkeeping. Raw chunks remain in the archive.
    return {k: v for k, v in body.items() if k not in {"chunk_position", "tool_call_chunks"}}


def _parse_messages(
    value: Any, *, path="inputs.messages."
) -> tuple[list[Message] | None, list[str]]:
    if not isinstance(value, list) or not value:
        return None, ["missing_messages"]
    messages = []
    for index, entry in enumerate(value):
        entry = mapping(entry, "message")
        body_path = f"{path}{index}."
        if entry.get("type") == "constructor":
            body = mapping(entry.get("kwargs"), "message.kwargs")
            body_path += "kwargs."
            ids = entry.get("id", [])
            role = ids[-1] if isinstance(ids, list) and ids else None
        else:
            body = mapping(entry.get("data", entry), "message.data")
            if "data" in entry:
                body_path += "data."
            role = entry.get("type", body.get("role"))
        if role in {"ChatMessage", "chat"}:
            role = body.get("role")
        role = {
            "HumanMessage": "user",
            "human": "user",
            "AIMessage": "assistant",
            "ai": "assistant",
            "SystemMessage": "system",
            "system": "system",
            "ToolMessage": "tool",
            "tool": "tool",
        }.get(role, role)
        try:
            if role == "tool":
                # These belong to the framework's tool execution, not the
                # message sent to the model. The original Run remains archived.
                if body.get("status") not in (None, "success", "error"):
                    raise DataError("invalid_langchain_tool_status")
                body = {k: v for k, v in body.items() if k not in {"status", "artifact"}}
            messages.append(message(body, role=role, path=body_path))
        except DataError as exc:
            return None, [str(exc)]
    return messages, []


def _generations(generations, reported, group_index):
    if not generations:
        return {"output_gaps": ["missing_output"]}, reported, []
    if not isinstance(generations, list):
        return {"output_gaps": ["invalid_langchain_generations"]}, reported, []
    candidates, usages, usage_gaps = [], [], []
    for index, generation in enumerate(generations):
        try:
            generation = mapping(generation, "generation")
            info = mapping(generation.get("generation_info") or {}, "generation_info")
            raw = mapping(generation.get("message") or {}, "generation.message")
            body = mapping(raw.get("kwargs", raw.get("data", raw)), "generation.message.body")
            body = _terminal_message(body) if body else {"content": generation.get("text")}
            if body.get("usage_metadata") is not None:
                try:
                    usages.append(
                        usage_from(body["usage_metadata"], "input_tokens", "output_tokens")
                    )
                except DataError:
                    usage_gaps.append("invalid_langchain_usage")
            fields = output_fields(
                body,
                path=f"outputs.generations.{group_index}.{index}.message.",
                termination=info,
                termination_path=f"outputs.generations.{group_index}.{index}.generation_info.",
            )
        except DataError as exc:
            fields = {"output_gaps": [str(exc)]}
        candidates.append({"index": index, **fields})
    response = group_outputs(candidates)
    # LangChain's OpenAI integration copies request usage onto each generation.
    # Equal copies are one report, never candidate subtotals. Conflicting scopes
    # cannot be resolved by summing or arbitrarily choosing the first message.
    values = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        seen = {
            getattr(item, key) for item in [reported, *usages] if getattr(item, key) is not None
        }
        if len(seen) > 1:
            usage_gaps.append("conflicting_langchain_usage")
        else:
            values[key] = next(iter(seen), None)
    if usage_gaps:
        response["output_gaps"] = sorted(set([*response.get("output_gaps", []), *usage_gaps]))
        return response, Usage(), sorted(set(usage_gaps))
    return response, Usage(**values), []


class LangChainProtocol(ProtocolAdapter):
    """RunCollectorCallbackHandler/LangSmith Run JSON, not arbitrary console logs."""

    name = "langchain"
    version = "langchain.run.v5"

    def parse(self, record: dict[str, Any]) -> Parsed:
        return self._parse(record, "")

    def _parse(self, record: dict[str, Any], path: str) -> Parsed:
        if "run_type" not in record:
            raise DataError("missing run_type; expected serialized LangChain Run")
        calls: list[Call] = []
        contexts: list[ProtocolContext] = []
        children = record.get("child_runs") or []
        if not isinstance(children, list):
            raise DataError("child_runs must be an array")
        for index, child in enumerate(children):
            parsed = self._parse(mapping(child, "child_run"), f"{path}child_runs.{index}.")
            calls.extend(parsed.calls)
            contexts.extend(parsed.contexts)
        if record["run_type"] not in {"llm", "chat_model"}:
            return Parsed(calls, None if calls else "non_llm_run", contexts)
        inputs = mapping(record.get("inputs") or {}, "inputs")
        raw_messages = inputs.get("messages")
        nested = isinstance(raw_messages, list) and any(
            isinstance(item, list) for item in raw_messages
        )
        groups = raw_messages if nested else [raw_messages]
        output = mapping(record.get("outputs") or {}, "outputs")
        generations = output.get("generations") or []
        if not isinstance(generations, list):
            raise DataError("invalid_langchain_generations")
        # Retain unmatched outputs as unavailable-input rows; never silently drop
        # an extra group or move a later answer onto an earlier input.
        size = max(len(groups), len(generations), 1)
        mismatch = bool(generations) and len(groups) != len(generations)
        llm_output = mapping(output.get("llm_output") or {}, "llm_output")
        total_usage = usage_from(
            llm_output.get("token_usage"), "prompt_tokens", "completion_tokens"
        )
        extra = mapping(record.get("extra") or {}, "extra")
        metadata = mapping(extra.get("metadata") or {}, "metadata")
        params = mapping(extra.get("invocation_params") or {}, "invocation_params")
        options = request_options(("extra.invocation_params", params))
        option_gaps = options.pop("gaps")
        start, end = timestamp(record.get("start_time")), timestamp(record.get("end_time"))
        duration = None
        if start and end:
            delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
            micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
            if micros < 0:
                raise DataError("end_time precedes start_time")
            duration = str(Decimal(micros) / 1000)
        status = "failed" if record.get("error") else "success" if end else "unknown"
        run_id = identifier(record.get("id"), "id")
        for index in range(size):
            messages_path = f"{path}inputs.messages." + (f"{index}." if nested else "")
            try:
                messages, gaps = _parse_messages(
                    groups[index] if index < len(groups) else None, path=messages_path
                )
            except DataError as exc:
                messages, gaps = None, [str(exc)]
            response, reported, usage_gaps = _generations(
                generations[index] if index < len(generations) else [],
                total_usage if size == 1 else Usage(),
                index,
            )
            gaps.extend([*option_gaps, *usage_gaps])
            if mismatch:
                response["output_gaps"] = [
                    *response.get("output_gaps", []),
                    "langchain_input_output_group_mismatch",
                ]
            response = expected_candidates(response, ("extra.invocation_params", params))
            calls.append(
                Call(
                    call_id=run_id if size == 1 else f"{run_id}/input/{index}",
                    protocol=self.name,
                    model=text_or_none(
                        metadata.get(
                            "ls_model_name", params.get("model_name", params.get("model"))
                        ),
                        "model",
                    ),
                    provider=text_or_none(metadata.get("ls_provider"), "provider"),
                    status=status,
                    started_at=start,
                    duration_ms=duration,
                    messages=messages,
                    input_batch=InputBatch(run_id=run_id, index=index, size=size, usage=total_usage)
                    if size > 1
                    else None,
                    **response,
                    **options,
                    usage=reported,
                    gaps=sorted(set(gaps)),
                )
            )
            contexts.append(
                ProtocolContext(metadata=metadata, metadata_path=f"{path}extra.metadata")
            )
        return Parsed(calls, contexts=contexts)
