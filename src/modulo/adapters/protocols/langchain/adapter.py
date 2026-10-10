"""LangChain serialized Run parser with per-child message and metadata handling."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from modulo.core.domain.data_models import Call, DataError, Message, Parsed, ProtocolContext
from modulo.core.domain.interfaces import ProtocolAdapter

from ..shared.chat import message, output_fields, request_options
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


def _parse_messages(value: Any) -> tuple[list[Message] | None, list[str]]:
    if not isinstance(value, list) or not value:
        return None, ["missing_messages"]
    messages = []
    for entry in value:
        entry = mapping(entry, "message")
        if entry.get("type") == "constructor":
            body = mapping(entry.get("kwargs"), "message.kwargs")
            ids = entry.get("id", [])
            role = ids[-1] if isinstance(ids, list) and ids else None
        else:
            body = mapping(entry.get("data", entry), "message.data")
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
            messages.append(message(body, role=role, path="inputs.messages."))
        except DataError as exc:
            return None, [str(exc)]
    return messages, []


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
        batch = inputs.get("messages")
        gaps: list[str] = []
        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
            batch = batch[0]
        elif isinstance(batch, list) and any(isinstance(item, list) for item in batch):
            batch = None
            gaps.append("unsupported_batch")
        messages, message_gaps = _parse_messages(batch)
        gaps.extend(message_gaps)
        extra = mapping(record.get("extra") or {}, "extra")
        metadata = mapping(extra.get("metadata") or {}, "metadata")
        params = mapping(extra.get("invocation_params") or {}, "invocation_params")
        options, option_sources, tools, option_gaps = request_options(
            ("extra.invocation_params", params)
        )
        gaps.extend(option_gaps)
        output = mapping(record.get("outputs") or {}, "outputs")
        generations = output.get("generations") or []
        message_data: dict[str, Any] = {}
        visible_text = None
        output_gaps = []
        if generations:
            if (
                isinstance(generations, list)
                and len(generations) == 1
                and isinstance(generations[0], list)
                and len(generations[0]) == 1
            ):
                generation = mapping(generations[0][0], "generation")
                visible_text = text_or_none(generation.get("text"), "generation.text")
                message = mapping(generation.get("message") or {}, "generation.message")
                message_data = mapping(
                    message.get("kwargs", message.get("data", message)), "generation.message.body"
                )
                try:
                    message_data = _terminal_message(message_data)
                except DataError as exc:
                    output_gaps.append(str(exc))
            else:
                output_gaps.append("unsupported_multiple_generations")
        response = output_fields(
            message_data if message_data else {"content": visible_text}, path="outputs.generations."
        )
        if output_gaps:
            response = {"output_gaps": output_gaps}
        usage = message_data.get("usage_metadata")
        if usage is not None:
            reported = usage_from(usage, "input_tokens", "output_tokens")
        else:
            llm_output = mapping(output.get("llm_output") or {}, "llm_output")
            reported = usage_from(
                llm_output.get("token_usage"), "prompt_tokens", "completion_tokens"
            )
        start, end = timestamp(record.get("start_time")), timestamp(record.get("end_time"))
        duration = None
        if start and end:
            delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
            micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
            if micros < 0:
                raise DataError("end_time precedes start_time")
            duration = str(Decimal(micros) / 1000)
        status = "failed" if record.get("error") else "success" if end else "unknown"
        calls.append(
            Call(
                call_id=identifier(record.get("id"), "id"),
                protocol=self.name,
                model=text_or_none(
                    metadata.get("ls_model_name", params.get("model_name", params.get("model"))),
                    "model",
                ),
                provider=text_or_none(metadata.get("ls_provider"), "provider"),
                status=status,
                started_at=start,
                duration_ms=duration,
                messages=messages,
                **response,
                tools_json=tools,
                render_options=options,
                render_option_sources=option_sources,
                usage=reported,
                gaps=sorted(set(gaps)),
            )
        )
        contexts.append(ProtocolContext(metadata=metadata, metadata_path=f"{path}extra.metadata"))
        return Parsed(calls, contexts=contexts)
