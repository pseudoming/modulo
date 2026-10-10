"""Reconstruct one OpenAI-compatible completion, without adding usage snapshots."""

from dataclasses import dataclass, field

from modulo.core.domain.data_models import DataError, Usage
from modulo.shared.serialization import loads

from .chat import output_fields
from .fields import usage_from


@dataclass
class StreamResult:
    output: dict = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    response_id: str | None = None
    model: str | None = None
    done: bool = False
    finish_reason: str | None = None


def assemble_sse(lines: list[str]) -> StreamResult:
    result = StreamResult()
    message = {"role": "assistant"}
    tools = {}
    gaps = []
    pending = []

    def event(data):
        if result.done:
            gaps.append("data_after_stream_done")
            return
        if data == "[DONE]":
            result.done = True
            return
        try:
            payload = loads(data)
        except ValueError:
            gaps.append("invalid_stream_json")
            return
        if not isinstance(payload, dict):
            gaps.append("invalid_stream_event")
            return
        if payload.get("error"):
            gaps.append("stream_error")
            return
        for key in ("id", "model"):
            value = payload.get(key)
            prior = result.response_id if key == "id" else result.model
            if value is not None:
                if not isinstance(value, str) or (prior is not None and prior != value):
                    gaps.append(f"conflicting_stream_{key}")
                elif key == "id":
                    result.response_id = value
                else:
                    result.model = value
        # OpenAI usage is a snapshot, usually in a final choices=[] event.
        if payload.get("usage") is not None:
            try:
                result.usage = usage_from(payload["usage"], "prompt_tokens", "completion_tokens")
            except DataError:
                gaps.append("invalid_stream_usage")
                result.usage = Usage()
        choices = payload.get("choices")
        if choices == []:
            return
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            gaps.append("unsupported_stream_choices")
            return
        choice = choices[0]
        if type(choice.get("index")) is not int or choice["index"] != 0:
            gaps.append("unsupported_stream_choices")
            return
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            gaps.append("invalid_stream_delta")
            return
        if result.finish_reason and any(v is not None and v != "" for v in delta.values()):
            gaps.append("delta_after_finish")
        if choice.get("finish_reason") is not None:
            if not isinstance(choice["finish_reason"], str):
                gaps.append("invalid_finish_reason")
            elif result.finish_reason and result.finish_reason != choice["finish_reason"]:
                gaps.append("conflicting_finish_reason")
            else:
                result.finish_reason = choice["finish_reason"]
        if delta.get("role") not in (None, "assistant"):
            gaps.append("unsupported_stream_role")
        if set(delta) - {
            "role",
            "content",
            "reasoning_content",
            "reasoning",
            "thinking",
            "tool_calls",
        }:
            gaps.append("unsupported_stream_delta_fields")
        thought = [
            delta[k]
            for k in ("reasoning_content", "reasoning", "thinking")
            if delta.get(k) is not None
        ]
        if thought and (
            any(not isinstance(v, str) for v in thought) or any(v != thought[0] for v in thought)
        ):
            gaps.append("conflicting_stream_reasoning")
        elif thought:
            message["reasoning_content"] = message.get("reasoning_content", "") + thought[0]
        if delta.get("content") is not None:
            if not isinstance(delta["content"], str):
                gaps.append("unsupported_stream_content")
            else:
                message["content"] = message.get("content", "") + delta["content"]
        calls = delta.get("tool_calls") or []
        if not isinstance(calls, list):
            gaps.append("invalid_stream_tools")
            return
        for call in calls:
            if (
                not isinstance(call, dict)
                or type(call.get("index")) is not int
                or call["index"] < 0
            ):
                gaps.append("invalid_stream_tool_index")
                continue
            if set(call) - {"index", "id", "type", "function"}:
                gaps.append("unsupported_stream_tool_fields")
            target = tools.setdefault(
                call["index"], {"type": "function", "function": {"name": "", "arguments": ""}}
            )
            for key in ("id", "type"):
                if call.get(key) is not None:
                    if not isinstance(call[key], str) or (
                        key in target and target[key] != call[key]
                    ):
                        gaps.append(f"conflicting_stream_tool_{key}")
                    else:
                        target[key] = call[key]
            fn = call.get("function") or {}
            if not isinstance(fn, dict) or set(fn) - {"name", "arguments"}:
                gaps.append("unsupported_stream_tool_function")
                continue
            for key in ("name", "arguments"):
                if fn.get(key) is not None:
                    if not isinstance(fn[key], str):
                        gaps.append("invalid_stream_tool_fragment")
                    else:
                        target["function"][key] += fn[key]

    for line in lines:
        line = line.rstrip("\r\n")
        if line == "":
            if pending:
                event("\n".join(pending))
                pending = []
        elif line.startswith("data:"):
            data = line[5:]
            pending.append(data[1:] if data.startswith(" ") else data)
        elif line == "[DONE]":
            if pending:
                event("\n".join(pending))
                pending = []
            event(line)
        elif not line.startswith((":", "event:", "id:", "retry:")):
            gaps.append("unsupported_sse_line")
    if pending:
        event("\n".join(pending))
    if not result.done:
        gaps.append("stream_done_missing")
    if not result.finish_reason:
        gaps.append("stream_finish_missing")
    if tools:
        if sorted(tools) != list(range(len(tools))):
            gaps.append("stream_tool_index_gap")
        for tool in tools.values():
            if not tool.get("id") or not tool["function"]["name"]:
                gaps.append("incomplete_stream_tool")
            try:
                if not isinstance(loads(tool["function"]["arguments"]), dict):
                    gaps.append("invalid_stream_tool_arguments")
            except ValueError:
                gaps.append("invalid_stream_tool_arguments")
        message["tool_calls"] = [tools[k] for k in sorted(tools)]
    result.output = output_fields(message, path="response_sse.assembled.")
    if gaps:
        result.output["output_gaps"] = sorted(set([*result.output.get("output_gaps", []), *gaps]))
    if any(
        reason in gaps
        for reason in (
            "invalid_stream_usage",
            "conflicting_stream_id",
            "conflicting_stream_model",
            "data_after_stream_done",
            "stream_error",
        )
    ):
        result.usage = Usage()
    return result
