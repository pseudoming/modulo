"""Read Responses API items as messages, not as parallel completion choices."""

from copy import deepcopy

from modulo.core.domain.data_models import DataError, Message, Usage
from modulo.shared.serialization import ordered_json

from ..fields import usage_from
from .common import options_fields, parse_message, parse_output, remap_sources, with_gaps


def _assistant(items, path, *, allow_incomplete=False):
    blocks, origins, gaps, message_count = [], [], [], 0
    for item, source in items:
        kind = item.get("type", "message")
        if item.get("status") not in (None, "completed") and not (
            allow_incomplete and item.get("status") == "incomplete"
        ):
            gaps.append("incomplete_response_item")
        if kind == "message":
            if item.get("role") != "assistant":
                gaps.append("invalid_response_output_role")
            message_count += 1
            content = item.get("content")
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
                origins.append(source + ".content")
            elif isinstance(content, list):
                blocks.extend(content)
                origins.extend(f"{source}.content.{i}" for i in range(len(content)))
            else:
                gaps.append("missing_response_item_content")
        elif kind == "reasoning":
            blocks.append(item)
            origins.append(source)
        elif kind == "function_call":
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                gaps.append("missing_function_call_id")
            blocks.append(
                {
                    "type": "tool_call",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "args": item.get("arguments"),
                }
            )
            origins.append(source)
        else:
            blocks.append(item)
            origins.append(source)
    fields = parse_output(
        {"role": "assistant", "content": blocks},
        path,
        "max_output_tokens" if allow_incomplete else None,
        original=[item for item, _ in items],
        gaps=gaps,
    )
    if message_count > 1:
        groups, pending = [], []
        for item, source in items:
            has_message = any(v.get("type", "message") == "message" for v, _ in pending)
            if has_message and item.get("type", "message") in {"message", "reasoning"}:
                groups.append(pending)
                pending = []
            pending.append((item, source))
        if pending:
            groups.append(pending)
        fields["output_messages"] = [
            _message(_assistant(group, path, allow_incomplete=allow_incomplete), group)
            for group in groups
        ]
    # Body blocks are assembled into one assistant turn; keep original paths
    # for reasoning rather than pointing at the temporary flattened array.
    source = fields.get("output_reasoning_source")
    if source:
        for index in reversed(range(len(origins))):
            source = source.replace(f"{path}.content.{index}", origins[index])
        fields["output_reasoning_source"] = source
    return fields


def _message(output, items):
    return Message(
        role="assistant",
        content=output.get("output_text"),
        reasoning_content=output.get("output_reasoning"),
        reasoning_source=output.get("output_reasoning_source"),
        structured_content_json=ordered_json([item for item, _ in items]),
        assistant_parts_json=output.get("output_parts_json"),
        tool_calls_json=output.get("output_tool_calls_json"),
        gaps=output.get("output_gaps", []),
    )


def request_fields(body, path, *, extra_roots=()):
    fields = options_fields(body, path, dialect="responses", extra_roots=extra_roots)
    messages, gaps, pending = [], [], []
    if body.get("previous_response_id") or body.get("conversation") or body.get("prompt"):
        gaps.append("unresolved_responses_context")
    if body.get("truncation") not in (None, "disabled"):
        gaps.append("responses_truncation_unknown")
    instructions = body.get("instructions")
    if isinstance(instructions, str):
        messages.append(
            parse_message(
                {"role": "system", "content": instructions},
                path,
                original={"instructions": instructions},
            )
        )
    elif instructions is not None:
        gaps.append("unsupported_response_instructions")
    values = body.get("input")
    if isinstance(values, str):
        messages.append(
            parse_message({"role": "user", "content": values}, path, original={"input": values})
        )
        values = []
    elif not isinstance(values, list):
        gaps.append("missing_messages")
        values = []

    def flush():
        if not pending:
            return
        output = _assistant(pending, path + ".input")
        messages.append(_message(output, pending))
        pending.clear()

    for index, item in enumerate(values):
        source = f"{path}.input.{index}"
        try:
            if not isinstance(item, dict):
                raise DataError("invalid_response_input_item")
            kind = item.get("type", "message")
            if kind in {"reasoning", "function_call"} or (
                kind == "message" and item.get("role") == "assistant"
            ):
                if kind in {"message", "reasoning"} and any(
                    v.get("type", "message") == "message" for v, _ in pending
                ):
                    flush()
                pending.append((item, source))
                continue
            flush()
            if kind == "message":
                if set(item) - {"type", "id", "status", "role", "content", "phase"}:
                    gaps.append("unsupported_response_message_fields")
                messages.append(
                    parse_message(
                        {"role": item.get("role"), "content": item.get("content")},
                        source,
                        original=item,
                    )
                )
            elif kind == "function_call_output":
                messages.append(
                    parse_message(
                        {
                            "role": "tool",
                            "tool_call_id": item.get("call_id"),
                            "content": item.get("output"),
                        },
                        source,
                        original=item,
                    )
                )
                messages[-1] = remap_sources(
                    messages[-1], {source + ".content": source + ".output"}
                )
            elif kind == "item_reference":
                gaps.append("unresolved_response_item_reference")
            else:
                gaps.append("unsupported_response_input_item:" + str(kind))
        except DataError as exc:
            gaps.append(str(exc))
    flush()
    if not messages:
        gaps.append("missing_messages")
    fields["messages"] = messages or None
    return with_gaps(fields, gaps)


def response_fields(body, path):
    gaps = []
    values = body.get("output")
    if not isinstance(values, list):
        values = []
        gaps.append("missing_response_output")
    items = []
    for index, item in enumerate(values):
        if isinstance(item, dict):
            items.append((item, f"{path}.output.{index}"))
        else:
            gaps.append("invalid_response_output_item")
    if not items:
        gaps.append("missing_output")
    output = _assistant(
        items,
        path,
        allow_incomplete=body.get("status") == "incomplete"
        and (body.get("incomplete_details") or {}).get("reason") == "max_output_tokens",
    )
    status = body.get("status")
    if body.get("error") or status == "failed":
        gaps.append("gateway_error_response")
    if status == "completed":
        output["output_finish_reason"] = (
            "tool_calls" if output.get("output_tool_calls_json") else "stop"
        )
    elif status == "incomplete":
        output["output_finish_reason"] = (body.get("incomplete_details") or {}).get(
            "reason", "incomplete"
        )
        if output["output_finish_reason"] != "max_output_tokens":
            gaps.append("incomplete_response")
    else:
        gaps.append("response_completion_missing")
    usage = usage_from(body.get("usage"), "input_tokens", "output_tokens")
    details = (body.get("usage") or {}).get("output_tokens_details")
    if (
        isinstance(details, dict)
        and details.get("reasoning_tokens", 0)
        and output.get("output_reasoning") in (None, "")
    ):
        gaps.append("missing_output_reasoning")
    output["output_structured_content_json"] = ordered_json(body)
    return with_gaps(output, gaps, "output_gaps"), usage, body.get("model")


class Stream:
    """Use completed item/response snapshots once, validating observed deltas."""

    def __init__(self):
        self.body, self.items, self.closed = {}, {}, set()
        self.gaps, self.done, self.sequence = [], False, None
        self.fragments = {}

    def add(self, event):
        if self.done:
            self.gaps.append("data_after_stream_done")
            return
        sequence = event.get("sequence_number")
        if sequence is not None:
            if (
                type(sequence) is not int
                or sequence < 0
                or (self.sequence is not None and sequence <= self.sequence)
            ):
                self.gaps.append("invalid_response_event_sequence")
            else:
                self.sequence = sequence
        kind = event.get("type", "")
        if kind in {
            "response.created",
            "response.in_progress",
            "response.queued",
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            body = event.get("response")
            if not isinstance(body, dict):
                self.gaps.append("invalid_response_snapshot")
                return
            for key in ("id", "model"):
                if (
                    self.body.get(key) is not None
                    and body.get(key) is not None
                    and self.body[key] != body[key]
                ):
                    self.gaps.append("conflicting_stream_" + key)
            self.body = deepcopy(body)
            if kind in {"response.completed", "response.incomplete", "response.failed"}:
                expected = kind.removeprefix("response.")
                if body.get("status") != expected:
                    self.gaps.append("conflicting_response_status")
                self.done = True
                values = body.get("output")
                if isinstance(values, list):
                    for index in self.items:
                        if index >= len(values):
                            self.gaps.append("missing_response_output_item")
                        else:
                            self._check(index, values[index])
            return
        if kind == "error":
            self.gaps.append("stream_error")
            return
        index = event.get("output_index")
        if type(index) is not int or index < 0:
            self.gaps.append("unsupported_responses_stream_event")
            return
        if kind in {"response.output_item.added", "response.output_item.done"}:
            item = event.get("item")
            if not isinstance(item, dict):
                self.gaps.append("invalid_stream_item")
                return
            if kind.endswith(".added"):
                if index in self.items:
                    self.gaps.append("duplicate_stream_item")
                else:
                    self.items[index] = deepcopy(item)
            else:
                self._check(index, item)
                if index in self.closed:
                    self.gaps.append("duplicate_stream_item_done")
                self.items[index] = deepcopy(item)
                self.closed.add(index)
            return
        item = self.items.get(index)
        if item is None or index in self.closed:
            self.gaps.append("invalid_stream_item_order")
            return
        if event.get("item_id") is not None and event["item_id"] != item.get("id"):
            self.gaps.append("conflicting_stream_item_id")
        if kind in {
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            summary = ".reasoning_summary_part." in kind
            key, position = (
                ("summary", event.get("summary_index"))
                if summary
                else ("content", event.get("content_index"))
            )
            parts = item.setdefault(key, [])
            part = event.get("part")
            if (
                type(position) is not int
                or position < 0
                or position > len(parts)
                or not isinstance(part, dict)
            ):
                self.gaps.append("invalid_stream_content_index")
            elif kind.endswith(".added") and position == len(parts):
                parts.append(deepcopy(part))
            elif kind.endswith(".done") and position < len(parts):
                # Citation annotations may arrive only with the completed
                # part. They are preserved below, but do not change its text.
                prior = {k: v for k, v in parts[position].items() if k != "annotations"}
                final = {k: v for k, v in part.items() if k != "annotations"}
                if prior != final:
                    self.gaps.append("conflicting_stream_content")
                parts[position] = deepcopy(part)
            else:
                self.gaps.append("invalid_stream_content_order")
        elif kind.startswith("response.function_call_arguments."):
            self._text(event, item, "arguments", (index, "arguments"), kind)
        elif kind.startswith(
            (
                "response.output_text.",
                "response.reasoning_text.",
                "response.reasoning_summary_text.",
                "response.refusal.",
            )
        ):
            if kind == "response.output_text.annotation.added":
                return  # Annotations are metadata retained in the final snapshot.
            summary = ".reasoning_summary_text." in kind
            key = "summary" if summary else "content"
            position = event.get("summary_index" if summary else "content_index")
            parts = item.get(key, [])
            if type(position) is not int or not 0 <= position < len(parts):
                self.gaps.append("invalid_stream_content_index")
                return
            target = "refusal" if kind.startswith("response.refusal.") else "text"
            self._text(event, parts[position], target, (index, key, position, target), kind)
        else:
            self.gaps.append("unsupported_responses_stream_event:" + str(kind))

    def _text(self, event, target, key, identity, kind):
        value = event.get("delta") if kind.endswith(".delta") else event.get(key)
        if not isinstance(value, str):
            self.gaps.append("invalid_stream_text")
        elif kind.endswith(".delta"):
            target[key] = target.get(key, "") + value
            self.fragments[identity] = target[key]
        elif kind.endswith(".done"):
            if identity in self.fragments and self.fragments[identity] != value:
                self.gaps.append("conflicting_stream_text")
            target[key] = value
            self.fragments[identity] = value
        else:
            self.gaps.append("unsupported_responses_stream_event")

    def _check(self, index, item):
        prior = self.items.get(index)
        if not isinstance(item, dict):
            self.gaps.append("invalid_stream_item")
            return
        if prior:
            for key in ("id", "type", "call_id", "name", "role"):
                if key in prior and prior[key] != item.get(key):
                    self.gaps.append("conflicting_stream_item_" + key)
            if index in self.closed and prior != item:
                self.gaps.append("conflicting_stream_item_snapshot")
        for identity, text in self.fragments.items():
            if identity[0] != index:
                continue
            value = item
            try:
                for key in identity[1:]:
                    value = value[key]
                if text != value:
                    self.gaps.append("conflicting_stream_text")
            except (KeyError, IndexError, TypeError):
                self.gaps.append("missing_stream_text")

    def finish(self):
        if not self.done:
            self.gaps.append("stream_done_missing")
            self.body["output"] = [self.items[i] for i in sorted(self.items)]
        if sorted(self.items) != list(range(len(self.items))):
            self.gaps.append("stream_item_index_gap")
        fields, usage, model = response_fields(self.body, "response_sse.responses")
        if any(
            gap in self.gaps
            for gap in ("conflicting_stream_id", "conflicting_stream_model", "stream_error")
        ):
            usage = Usage()
        return with_gaps(fields, self.gaps, "output_gaps"), usage, model
