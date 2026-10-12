"""Reconstruct each completion choice without adding request usage snapshots."""

from dataclasses import dataclass, field

from modulo.adapters.protocols.shared.fields import usage_from
from modulo.adapters.protocols.shared.messages.candidates import group_outputs
from modulo.adapters.protocols.shared.messages.chat import output_fields
from modulo.adapters.protocols.shared.messages.content import content_blocks
from modulo.core.domain.data_models import DataError, Usage
from modulo.shared.serialization import loads, ordered_json

from .common import StreamResult


@dataclass
class _Choice:
    message: dict = field(default_factory=lambda: {"role": "assistant"})
    tools: dict = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    stopping: dict = field(default_factory=dict)
    parts: list = field(default_factory=list)

    def record_content(self, kind, value):
        if isinstance(value, str):
            if value == "" and kind == "text" and self.parts:
                return  # Empty transport deltas do not interrupt a tool/reasoning block.
            if self.parts and self.parts[-1].get("scalar") == kind:
                self.parts[-1]["text"] += value
            else:
                self.parts.append({"type": kind, "text": value, "scalar": kind})
        else:
            self.parts.extend(content_blocks(value, "delta", reasoning=kind == "reasoning").ordered)

    def add(self, choice):
        message, tools, gaps = self.message, self.tools, self.gaps
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            gaps.append("invalid_stream_delta")
            return
        if self.finish_reason and any(v is not None and v != "" for v in delta.values()):
            gaps.append("delta_after_finish")
        if choice.get("finish_reason") is not None:
            if not isinstance(choice["finish_reason"], str):
                gaps.append("invalid_finish_reason")
            elif self.finish_reason and self.finish_reason != choice["finish_reason"]:
                gaps.append("conflicting_finish_reason")
            else:
                self.finish_reason = choice["finish_reason"]
        for key in ("stop_reason", "stop_sequence"):
            if key in choice and (choice.get("finish_reason") or choice[key] is not None):
                if key in self.stopping and self.stopping[key] != choice[key]:
                    gaps.append("conflicting_output_stop_reason")
                self.stopping[key] = choice[key]
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
            any(not isinstance(v, (str, list, dict)) for v in thought)
            or any(v != thought[0] for v in thought)
        ):
            gaps.append("conflicting_stream_reasoning")
        elif thought:
            _append(message, "reasoning_content", thought[0])
            self.record_content("reasoning", thought[0])
        if delta.get("content") is not None:
            if not isinstance(delta["content"], (str, list)):
                gaps.append("unsupported_stream_content")
            else:
                _append(message, "content", delta["content"])
                self.record_content("text", delta["content"])
        calls = delta.get("tool_calls") or []
        if not isinstance(calls, list):
            gaps.append("invalid_stream_tools")
            return
        if calls:
            if not tools:
                self.parts.append({"type": "stream_tools"})
            elif self.parts and self.parts[-1]["type"] != "stream_tools":
                gaps.append("interleaved_stream_tool_fragments")
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

    def output(self, index):
        message, tools, gaps = self.message, self.tools, self.gaps
        if not self.finish_reason:
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
        output = output_fields(
            {**message, **self.stopping},
            path=f"response_sse.choices.{index}.",
            finish_reason=self.finish_reason,
        )
        parts = []
        structured_tools = any(part["type"] == "tool_calls" for part in self.parts)
        for part in self.parts:
            if part["type"] == "stream_tools":
                if tools and not structured_tools:
                    parts.append({"type": "tool_calls", "tool_calls": message["tool_calls"]})
            else:
                parts.append({k: v for k, v in part.items() if k != "scalar"})
        if parts:
            output["output_parts_json"] = ordered_json(parts)
        if gaps:
            output["output_gaps"] = sorted(set([*output.get("output_gaps", []), *gaps]))
        return {"index": index, **output}


def _append(message, key, value):
    prior = message.get(key)
    if prior is None:
        message[key] = value
    elif isinstance(prior, str) and isinstance(value, str):
        message[key] = prior + value
    else:
        # Keep block order when a producer mixes text and structured fragments.
        message[key] = (prior if isinstance(prior, list) else [prior]) + (
            value if isinstance(value, list) else [value]
        )


def assemble_sse(lines: list[str], *, require_done=True) -> StreamResult:
    result = StreamResult()
    states, gaps, pending = {}, [], []

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
        # Usage is request-level. A final choices=[] event may carry the total.
        if payload.get("usage") is not None:
            try:
                result.usage = usage_from(payload["usage"], "prompt_tokens", "completion_tokens")
            except DataError:
                gaps.append("invalid_stream_usage")
                result.usage = Usage()
        choices = payload.get("choices")
        if not isinstance(choices, list):
            gaps.append("unsupported_stream_choices")
            return
        seen = set()
        for choice in choices:
            if (
                not isinstance(choice, dict)
                or type(choice.get("index")) is not int
                or choice["index"] < 0
            ):
                gaps.append("invalid_stream_choice_index")
                continue
            index = choice["index"]
            state = states.setdefault(index, _Choice())
            if index in seen:
                state.gaps.append("duplicate_stream_choice_index")
                continue
            seen.add(index)
            state.add(choice)

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
    if not require_done and states and all(state.finish_reason for state in states.values()):
        # SDK callbacks expose decoded chunks, not the HTTP [DONE] sentinel.
        # Each candidate still needs its own native finish_reason.
        result.done = True
    if not result.done:
        gaps.append("stream_done_missing")
    result.output = group_outputs([state.output(index) for index, state in states.items()], gaps)
    if len(states) == 1:
        result.finish_reason = next(iter(states.values())).finish_reason
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
