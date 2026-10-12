"""Read Anthropic Messages bodies; all contents remain offline log data."""

from modulo.core.domain.data_models import DataError, Usage
from modulo.shared.serialization import loads, ordered_json

from ..fields import usage_from
from .common import options_fields, parse_message, parse_output, remap_sources, with_gaps


def request_fields(body, path, *, extra_roots=()):
    fields = options_fields(body, path, dialect="anthropic", extra_roots=extra_roots)
    messages, gaps = [], []
    if body.get("system") is not None:
        try:
            messages.append(
                parse_message(
                    {"role": "system", "content": body["system"]},
                    path,
                    original={"system": body["system"]},
                )
            )
            # The system prompt is top-level, not messages[0].content.
            messages[-1] = remap_sources(messages[-1], {path + ".content": path + ".system"})
        except DataError as exc:
            gaps.append(str(exc))
    values = body.get("messages")
    if not isinstance(values, list) or not values:
        gaps.append("missing_messages")
        values = []
    for index, item in enumerate(values):
        source = f"{path}.messages.{index}"
        try:
            if not isinstance(item, dict) or item.get("role") not in {
                "user",
                "assistant",
                "system",
            }:
                raise DataError("invalid_anthropic_message")
            if set(item) - {"role", "content"}:
                gaps.append("unsupported_anthropic_message_fields")
            content = item.get("content")
            if not isinstance(content, list) or not any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content
            ):
                messages.append(parse_message(item, source, original=item))
                continue
            if item["role"] != "user":
                raise DataError("tool_result_on_non_user")
            start, pending = 0, []

            def flush(pending, source, item, start):
                if not pending:
                    return
                parsed = parse_message({"role": "user", "content": pending}, source, original=item)
                # Restore source indexes after separating tool results from the
                # enclosing user message. Text/image order is unchanged.
                messages.append(
                    remap_sources(
                        parsed,
                        {
                            f"{source}.content.{relative}": f"{source}.content.{start + relative}"
                            for relative in range(len(pending))
                        },
                    )
                )

            for position, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    if not pending:
                        start = position
                    pending.append(block)
                    continue
                flush(pending, source, item, start)
                pending = []
                if set(block) - {"type", "tool_use_id", "content", "is_error", "cache_control"}:
                    gaps.append("unsupported_tool_result_fields")
                messages.append(
                    parse_message(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": block.get("content"),
                        },
                        f"{source}.content.{position}",
                        original=block,
                    )
                )
            flush(pending, source, item, start)
        except DataError as exc:
            gaps.append(str(exc))
    if body.get("container") or body.get("context_management") or body.get("mcp_servers"):
        gaps.append("unresolved_anthropic_context")
    fields["messages"] = messages or None
    return with_gaps(fields, gaps)


def usage_fields(value):
    usage = usage_from(value, "input_tokens", "output_tokens")
    if not isinstance(value, dict):
        return usage
    cache_keys = ("cache_creation_input_tokens", "cache_read_input_tokens")
    present = [value[key] for key in cache_keys if key in value]
    if any(v is not None and (type(v) is not int or v < 0) for v in present):
        raise DataError("invalid_anthropic_cache_usage")
    if present:
        total_input = (
            usage.input_tokens + sum(present)
            if usage.input_tokens is not None and None not in present
            else None
        )
        usage = usage.model_copy(update={"input_tokens": total_input})
    return usage


def response_fields(body, path):
    if body.get("type") == "error" or body.get("error"):
        return {"output_gaps": ["gateway_error_response"]}, Usage(), body.get("model")
    gaps = []
    if body.get("type") != "message" or body.get("role") != "assistant":
        gaps.append("invalid_anthropic_response")
    fields = parse_output(
        {
            "role": "assistant",
            "content": body.get("content"),
            **({"stop_sequence": body["stop_sequence"]} if "stop_sequence" in body else {}),
        },
        path,
        body.get("stop_reason"),
        original=body,
        gaps=gaps,
    )
    return fields, usage_fields(body.get("usage")), body.get("model")


class Stream:
    """Assemble one message by content-block index; usage updates are totals."""

    def __init__(self):
        self.body, self.blocks, self.arguments, self.closed = {}, {}, {}, set()
        self.gaps, self.done = [], False
        self.delta_seen = False

    def add(self, event):
        kind = event.get("type")
        if kind == "ping":
            return
        if self.done:
            self.gaps.append("data_after_stream_done")
            return
        if kind == "message_start":
            if self.body:
                self.gaps.append("duplicate_message_start")
                return
            self.body = dict(event.get("message") or {})
            if self.body.get("content"):
                self.gaps.append("unexpected_initial_stream_content")
        elif kind in {"content_block_start", "content_block_delta", "content_block_stop"}:
            index = event.get("index")
            if type(index) is not int or index < 0:
                self.gaps.append("invalid_stream_block_index")
                return
            if not self.body:
                self.gaps.append("stream_start_missing")
            if kind == "content_block_start":
                block = event.get("content_block")
                if index in self.blocks or not isinstance(block, dict):
                    self.gaps.append("invalid_stream_block_start")
                else:
                    self.blocks[index] = dict(block)
                return
            if index not in self.blocks or index in self.closed:
                self.gaps.append("invalid_stream_block_order")
                return
            block = self.blocks[index]
            if kind == "content_block_stop":
                self.closed.add(index)
                if index in self.arguments:
                    # Anthropic's input is an object, including when streamed.
                    # Keep its raw JSON separately, without changing its type
                    # to the Responses/Chat arguments string contract.
                    value = self.arguments[index]
                    try:
                        parsed = loads(value)
                        if not isinstance(parsed, dict):
                            raise ValueError()
                        block["input"] = parsed
                    except ValueError:
                        self.gaps.append("invalid_stream_tool_arguments")
                return
            delta = event.get("delta") or {}
            mapping = {
                "text_delta": ("text", "text"),
                "thinking_delta": ("thinking", "thinking"),
                "signature_delta": ("thinking", "signature"),
                "input_json_delta": ("tool_use", "partial_json"),
            }
            target = mapping.get(delta.get("type"))
            if delta.get("type") == "citations_delta" and block.get("type") == "text":
                block.setdefault("citations", []).append(delta.get("citation"))
            elif (
                target and block.get("type") == target[0] and isinstance(delta.get(target[1]), str)
            ):
                if target[1] == "partial_json":
                    if block.get("input") and index not in self.arguments:
                        self.gaps.append("conflicting_stream_tool_input")
                    self.arguments[index] = self.arguments.get(index, "") + delta[target[1]]
                else:
                    block[target[1]] = block.get(target[1], "") + delta[target[1]]
            else:
                self.gaps.append("unsupported_anthropic_stream_delta")
        elif kind == "message_delta":
            if not self.body:
                self.gaps.append("stream_start_missing")
            delta = event.get("delta") or {}
            if not isinstance(delta, dict):
                self.gaps.append("invalid_message_delta")
                return
            for key, value in delta.items():
                if key in {"stop_reason", "stop_sequence"}:
                    if self.body.get(key) is not None and self.body[key] != value:
                        self.gaps.append("conflicting_finish_reason")
                    self.body[key] = value
                elif value is not None:
                    self.gaps.append("unsupported_message_delta")
            usage = event.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    self.gaps.append("invalid_stream_usage")
                else:
                    prior = self.body.get("usage") or {}
                    if any(
                        key in prior
                        and type(value) is int
                        and type(prior[key]) is int
                        and value < prior[key]
                        for key, value in usage.items()
                    ):
                        self.gaps.append("conflicting_stream_usage")
                    self.body["usage"] = {**prior, **usage}
            self.delta_seen = True
        elif kind == "message_stop":
            self.done = True
        elif kind == "error":
            self.gaps.append("stream_error")
        else:
            self.gaps.append("unsupported_anthropic_stream_event")

    def finish(self):
        if not self.body:
            self.gaps.append("stream_start_missing")
        if not self.done:
            self.gaps.append("stream_done_missing")
        if not self.delta_seen or not self.body.get("stop_reason"):
            self.gaps.append("stream_finish_missing")
        if set(self.blocks) != self.closed:
            self.gaps.append("stream_block_stop_missing")
        if sorted(self.blocks) != list(range(len(self.blocks))):
            self.gaps.append("stream_block_index_gap")
        self.body["content"] = [self.blocks[i] for i in sorted(self.blocks)]
        fields, usage, model = response_fields(self.body, "response_sse.anthropic")
        if self.arguments:
            fields["output_structured_content_json"] = ordered_json(
                {
                    "message": self.body,
                    "tool_input_json": {str(k): v for k, v in self.arguments.items()},
                }
            )
        if any(
            gap in self.gaps
            for gap in (
                "invalid_stream_usage",
                "conflicting_stream_usage",
                "stream_error",
                "duplicate_message_start",
            )
        ):
            usage = Usage()
        return with_gaps(fields, self.gaps, "output_gaps"), usage, model
