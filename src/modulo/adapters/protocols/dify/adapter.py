"""Dify node-execution parser, including node_finished envelopes."""

from decimal import Decimal, InvalidOperation
from typing import Any

from modulo.core.domain.data_models import Call, DataError, Message, Parsed, ProtocolContext
from modulo.core.domain.interfaces import ProtocolAdapter

from ..shared.chat import message, output_fields, request_options
from ..shared.fields import identifier, json_object, mapping, text_or_none, timestamp, usage_from
from .agent import SCHEMA, DifySession, parse_invocation, parse_thought


def _parse_messages(value: Any) -> tuple[list[Message] | None, list[str]]:
    if not isinstance(value, list) or not value:
        return None, ["missing_messages"]
    messages = []
    for entry in value:
        entry = mapping(entry, "message")
        try:
            messages.append(message(entry, content_key="text", path="process_data.prompts."))
        except DataError as exc:
            return None, [str(exc)]
    return messages, []


class DifyProtocol(ProtocolAdapter):
    name = "dify"
    version = "dify.node-execution.v6"

    def new_session(self):
        return DifySession(self)

    def parse(self, record: dict[str, Any]) -> Parsed:
        if record.get("schema_version") == SCHEMA:
            return parse_invocation(record)
        if {"message_id", "position", "message_token", "answer_token"} <= record.keys():
            return parse_thought(record)
        if "event" in record:
            if record["event"] != "node_finished":
                return Parsed([], "non_node_finished_event")
            node = mapping(record.get("data"), "data")
        else:
            node = record
        if "node_type" not in node:
            raise DataError("missing node_type; expected Dify node execution")
        if node["node_type"] != "llm":
            return Parsed([], "non_llm_node")
        data = json_object(node.get("process_data"), "process_data")
        output = json_object(node.get("outputs"), "outputs")
        for field in ("inputs_truncated", "process_data_truncated", "outputs_truncated"):
            if field in node and type(node[field]) is not bool:
                raise DataError(f"invalid_{field}")
        messages, gaps = _parse_messages(data.get("prompts"))
        if node.get("process_data_truncated"):
            gaps.append("truncated_process_data")
        if data.get("model_mode") != "chat":
            gaps.append("unsupported_or_missing_model_mode")
        options, option_sources, tools, option_gaps = request_options(
            ("process_data", data), ("process_data.model_config", data.get("model_config"))
        )
        gaps.extend(option_gaps)
        response = output_fields(output, content_key="text", path="outputs.")
        if node.get("outputs_truncated"):
            response["output_gaps"] = [*response.get("output_gaps", []), "truncated_output"]
        elapsed = node.get("elapsed_time")
        duration = None
        if elapsed is not None:
            try:
                number = Decimal(str(elapsed))
                if not number.is_finite() or number < 0:
                    raise InvalidOperation
                duration = str(number * 1000)
            except (InvalidOperation, ValueError) as exc:
                raise DataError("invalid elapsed_time") from exc
        reported = usage_from(
            output.get("usage", data.get("usage")), "prompt_tokens", "completion_tokens"
        )
        status = {"succeeded": "success", "failed": "failed"}.get(node.get("status"), "unknown")
        return Parsed(
            [
                Call(
                    call_id=identifier(node.get("id"), "id"),
                    protocol=self.name,
                    model=text_or_none(data.get("model_name"), "model_name"),
                    provider=text_or_none(data.get("model_provider"), "model_provider"),
                    status=status,
                    started_at=timestamp(node.get("created_at")),
                    duration_ms=duration,
                    messages=messages,
                    **response,
                    tools_json=tools,
                    render_options=options,
                    render_option_sources=option_sources,
                    usage=reported,
                    gaps=sorted(set(gaps)),
                )
            ],
            contexts=[
                ProtocolContext(
                    metadata=node.get("metadata"),
                    metadata_path="data.metadata" if "event" in record else "metadata",
                )
            ],
        )
