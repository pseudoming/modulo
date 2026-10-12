"""Count recorded assistant content and apply its logged termination rule."""

from modulo.core.domain.data_models import Message
from modulo.core.domain.interfaces import UnsupportedInput
from modulo.shared.serialization import bytes_digest, digest, ordered_json

LIMIT_REASONS = {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded"}


def before_marker(profile, parts, marker):
    if profile.output_text(parts).count(marker) != 1:
        raise UnsupportedInput("unfinished_output_boundary_unknown")
    if isinstance(parts, str):
        return parts.split(marker, 1)[0]
    result = []
    for text, special in parts:
        if marker in text:
            result.append((text.split(marker, 1)[0], special))
            break
        result.append((text, special))
    return result


def termination_parts(profile, call, body, normal_ending):
    options = call.generation_options
    if call.generation_gaps:
        raise UnsupportedInput(",".join(call.generation_gaps))
    for key, value in options.items():
        if value is None:
            continue
        if key == "stop":
            valid = isinstance(value, list) and all(isinstance(v, str) and v for v in value)
        elif key == "stop_token_ids":
            valid = isinstance(value, list) and all(type(v) is int and v >= 0 for v in value)
        elif key in {"ignore_eos", "include_stop_str_in_output"}:
            valid = type(value) is bool
        elif key in {"max_tokens", "max_completion_tokens", "max_output_tokens", "min_tokens"}:
            valid = type(value) is int and value >= (0 if key == "min_tokens" else 1)
        else:
            valid = False
        if not valid:
            raise UnsupportedInput("invalid_generation_option:" + key)
    reason = call.output_finish_reason
    if reason is None:
        raise UnsupportedInput("missing_output_finish_reason")
    matched = call.output_stop_reason
    if reason in LIMIT_REASONS:
        if matched is not None:
            raise UnsupportedInput("conflicting_output_stop_reason")
        return normal_ending[:0], None, "token_limit_no_end_marker"
    if reason not in {
        "stop",
        "tool_calls",
        "function_call",
        "end_turn",
        "tool_use",
        "eos",
        "stop_sequence",
    }:
        raise UnsupportedInput("output_finish_reason:" + reason)
    if reason in {"tool_calls", "function_call", "tool_use"} and not (
        call.output_tool_calls_json or any(m.tool_calls_json for m in call.output_messages)
    ):
        raise UnsupportedInput("missing_output_tool_calls")
    if type(matched) is int:
        try:
            tokenizer = profile.tokenizer
            if hasattr(tokenizer, "get_vocab") and matched not in tokenizer.get_vocab().values():
                raise ValueError()
            text = tokenizer.decode([matched])
        except (ValueError, KeyError, OverflowError):
            raise UnsupportedInput("unknown_output_stop_token_id") from None
        return profile.literal_output(text), [matched], "logged_stop_token"
    if isinstance(matched, str):
        if not matched or (options.get("stop") and matched not in options["stop"]):
            raise UnsupportedInput("conflicting_output_stop_sequence")
        if options.get("include_stop_str_in_output", False):
            if not profile.output_text(body).endswith(matched):
                raise UnsupportedInput("included_stop_sequence_missing")
            return normal_ending[:0], None, "logged_stop_sequence_included"
        if matched in profile.output_text(body):
            # A normalized tool/body can already reconstruct this stop marker.
            # Without its original boundary, appending it could count it twice.
            raise UnsupportedInput("output_stop_boundary_ambiguous")
        return profile.literal_output(matched), None, "logged_stop_sequence_excluded"
    if reason == "stop_sequence":
        raise UnsupportedInput("missing_output_stop_sequence")
    if (
        reason == "stop"
        and call.output_stop_reason_source is None
        and (options.get("stop") or options.get("stop_token_ids") or options.get("ignore_eos"))
    ):
        raise UnsupportedInput("output_stop_reason_missing")
    if options.get("ignore_eos"):
        raise UnsupportedInput("output_eos_conflicts_with_ignore_eos")
    return normal_ending, None, "reference_end_marker"


def count_output(self, call, encode):
    if (
        call.output_gaps
        and call.output_text is None
        and call.output_reasoning is None
        and not call.output_tool_calls_json
        and not call.output_messages
    ):
        return {"output_execution": "not_run", "output_reason": ",".join(call.output_gaps)}
    split_gap = None
    try:
        text, reasoning, source = self.split_output(
            call.output_text, call.output_reasoning, call.output_reasoning_source
        )
    except UnsupportedInput as exc:
        # A truncated inline thinking channel still contains countable text.
        # No closing marker or answer is inferred from it.
        split_gap = str(exc)
        text = None
        reasoning = call.output_text[len(self.inline_think_tags[0]) :]
        source = "content:inline_thinking"
    if (
        text is None
        and reasoning is None
        and not call.output_tool_calls_json
        and not call.output_messages
    ):
        return {"output_execution": "not_run", "output_reason": "missing_output"}
    try:
        visible_tokens = None if text is None else len(encode(text))
        reasoning_tokens = None if reasoning is None else len(encode(reasoning))
        if call.output_messages:
            texts = [message.content for message in call.output_messages]
            thoughts = [message.reasoning_content for message in call.output_messages]
            visible_tokens = sum(len(encode(value)) for value in texts if value is not None)
            reasoning_tokens = (
                sum(len(encode(value)) for value in thoughts if value is not None)
                if any(value is not None for value in thoughts)
                else None
            )
    except Exception as exc:
        return {
            "output_execution": "error",
            "output_reason": f"tokenizer_error:{type(exc).__name__}",
        }
    result = {
        "asset_revision": self.entry["revision"],
        "asset_hash": self.asset_hash,
        "template_hash": self.template_hash,
        "visible_output_tokens": visible_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "reasoning_output_source": source,
        "output_execution": "partial",
    }
    try:
        mode_gaps = [
            gap
            for gap in call.gaps
            if gap.startswith("conflicting_request_option:") or gap == "invalid_request_options"
        ]
        if mode_gaps:
            raise UnsupportedInput(",".join(mode_gaps))
        options, sources = self.resolve_options(call)
        result.update(
            output_render_options=options,
            output_render_option_sources=sources,
            output_render_options_json=ordered_json(options),
        )
        thinking = self.thinking_enabled(options)
        # Reuse the same argument validation/projection as input messages.
        responses = self.chat_messages(
            call.model_copy(
                update={
                    "messages": [
                        m.model_copy(update={"content": m.content or ""})
                        for m in call.output_messages
                    ]
                    or [
                        Message(
                            role="assistant",
                            content=text or "",
                            # Empty reasoning here is only a tool-rendering scaffold.
                            # Missing real reasoning still blocks the body/total below.
                            reasoning_content=reasoning if reasoning is not None else "",
                            tool_calls_json=call.output_tool_calls_json,
                            assistant_parts_json=call.output_parts_json,
                        )
                    ]
                }
            )
        )
        response = responses if call.output_messages else responses[0]
        body, ending = self.render_output(response, options)
        tool_counts = []
        for item in responses:
            if not item.get("tool_calls"):
                continue
            if self.uses_content_blocks and item.get("content_blocks"):
                for part in item["content_blocks"]:
                    if part["type"] != "tool_calls":
                        continue
                    fragment, _ = self.render_output(
                        {
                            **item,
                            "content": "",
                            "reasoning_content": "",
                            "content_blocks": [part],
                            "tool_calls": part["tool_calls"],
                        },
                        options,
                    )
                    tool_counts.append(len(self.encode_output(fragment)))
                continue
            rendered, _ = self.render_output(item, options)
            plain, _ = self.render_output({**item, "tool_calls": []}, options)
            if rendered[: len(plain)] == plain:
                tool_counts.append(len(self.encode_output(rendered[len(plain) :])))
            elif not self.uses_content_blocks:
                raise UnsupportedInput("output_tool_boundary_mismatch")
        if tool_counts:
            result["tool_output_tokens"] = sum(tool_counts)
        if call.output_gaps:
            raise UnsupportedInput(",".join(call.output_gaps))
        if thinking and (
            reasoning is None or any(m.reasoning_content is None for m in call.output_messages)
        ):
            raise UnsupportedInput("missing_output_reasoning")
        if not thinking and reasoning:
            raise UnsupportedInput("output_reasoning_conflicts_with_mode")
        early_end = call.output_finish_reason in LIMIT_REASONS or (
            call.output_stop_reason is not None
        )
        if split_gap and not early_end:
            raise UnsupportedInput(split_gap)
        if (
            text is None
            and not call.output_tool_calls_json
            and not early_end
            and not call.output_messages
        ):
            raise UnsupportedInput("missing_output_text")
        last_blocks = responses[-1].get("content_blocks", [])
        if early_end and (
            not responses[-1].get("tool_calls")
            or (
                self.uses_content_blocks
                and last_blocks
                and last_blocks[-1]["type"] in {"text", "reasoning"}
            )
        ):
            # A token limit or custom stop does not generate closing channel tags.
            last = responses[-1]
            last_text = call.output_messages[-1].content if call.output_messages else text
            key = "reasoning_content" if last_text is None else "content"
            value = last.get(key)
            if value is None:
                raise UnsupportedInput("missing_output_text")
            marker = "__MODULO_UNFINISHED_CHANNEL_8d01__"
            replacement = {**last, key: marker}
            if self.uses_content_blocks and last.get("content_blocks"):
                parts = last["content_blocks"]
                if parts[-1]["type"] not in {"text", "reasoning"}:
                    raise UnsupportedInput("unfinished_output_boundary_unknown")
                value = parts[-1]["text"]
                replacement["content_blocks"] = [*parts[:-1], {**parts[-1], "text": marker}]
            scaffold, _ = self.render_output([*responses[:-1], replacement], options)
            body = before_marker(self, scaffold, marker) + self.literal_output(value)
        result["output_body_tokens"] = len(self.encode_output(body))
        ending, extra_ids, termination = termination_parts(self, call, body, ending)
        parts = body + ending
        if (
            isinstance(parts, list)
            and termination == "logged_stop_sequence_excluded"
            and body
            and ending
        ):
            if body[-1][1] == ending[0][1]:
                # The removed stop string continues the same text segment.
                parts = [*body[:-1], (body[-1][0] + ending[0][0], body[-1][1]), *ending[1:]]
        if self.custom_template and isinstance(parts, list):
            parts = [(self.output_text(parts), True)]
        ids = (
            self.encode_output(parts) if extra_ids is None else self.encode_output(body) + extra_ids
        )
        result.update(
            output_execution="completed",
            output_reason=None,
            output_tokens=len(ids),
            output_ids_sha256=digest(ids),
            rendered_output_sha256=bytes_digest(self.output_text(parts).encode()),
            output_ending=self.output_text(ending),
            output_termination=termination,
        )
    except UnsupportedInput as exc:
        result["output_reason"] = str(exc)
    except Exception as exc:
        result.update(
            output_execution="error", output_reason=f"tokenizer_error:{type(exc).__name__}"
        )
    return result
