"""Extract logged content blocks without treating metadata as model text."""

from dataclasses import dataclass, field

from modulo.adapters.protocols.shared.messages.images import image_part

REDACTED = {"redacted-by-litellm", "redacted by litellm. 'litellm.turn_off_message_logging=True'"}


@dataclass
class Content:
    text: str | None = None
    reasoning: str | None = None
    reasoning_sources: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    parts: list[dict] = field(default_factory=list)
    ordered: list[dict] = field(default_factory=list)


def content_blocks(value, path, *, reasoning=False):
    """Keep component text even when another block cannot be reconstructed.

    Preserve the block sequence. Whether a template can represent that sequence
    is checked by the tokenizer, not by a protocol parser.
    """
    result = Content()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        result.gaps.append("invalid_content_blocks")
        return result
    texts, thoughts = [], []
    for index, block in enumerate(value):
        source = f"{path}.{index}"
        if isinstance(block, str):
            block = {"type": "text", "text": block}
        if not isinstance(block, dict):
            result.gaps.append("invalid_content_block")
            continue
        kind = block.get("type", "reasoning" if reasoning else None)
        if not isinstance(kind, str):
            result.gaps.append("invalid_content_block_type")
            continue
        if kind in {"redacted_thinking", "redacted_reasoning"}:
            result.gaps.append("redacted_reasoning")
            continue
        if kind in {"summary_text", "reasoning_summary"}:
            result.gaps.append("reasoning_summary_only")
            continue
        thought = reasoning or kind in {"thinking", "reasoning", "reasoning_text"}
        if kind in {
            "text",
            "input_text",
            "output_text",
            "refusal",
            "thinking",
            "reasoning",
            "reasoning_text",
        }:
            keys = (
                ("thinking", "reasoning", "text")
                if thought
                else ("refusal",)
                if kind == "refusal"
                else ("text", "data")
            )
            values = [(k, block[k]) for k in keys if block.get(k) is not None]
            if not values and thought and isinstance(block.get("content"), list):
                if not block["content"] and (
                    block.get("summary") or block.get("encrypted_content")
                ):
                    result.gaps.append(
                        "reasoning_summary_only" if block.get("summary") else "redacted_reasoning"
                    )
                    continue
                nested = content_blocks(block["content"], source + ".content", reasoning=True)
                result.gaps.extend(nested.gaps)
                if nested.reasoning is not None:
                    thoughts.append(nested.reasoning)
                    result.reasoning_sources.extend(nested.reasoning_sources)
                result.ordered.extend(nested.ordered)
            elif not values:
                result.gaps.append(
                    "reasoning_summary_only"
                    if thought and "summary" in block
                    else "redacted_reasoning"
                    if thought and "encrypted_content" in block
                    else "missing_content_block_text"
                )
                continue
            elif any(not isinstance(v, str) for _, v in values):
                result.gaps.append("invalid_content_block_text")
                continue
            elif any(v != values[0][1] for _, v in values):
                result.gaps.append("conflicting_content_block_text")
                continue
            elif values[0][1] in REDACTED:
                result.gaps.append("redacted_reasoning" if thought else "redacted_messages")
                continue
            elif thought:
                thoughts.append(values[0][1])
                result.reasoning_sources.append(source + "." + values[0][0])
                result.ordered.append({"type": "reasoning", "text": values[0][1]})
            else:
                texts.append(values[0][1])
                result.parts.append({"type": "text", "text": values[0][1]})
                result.ordered.append({"type": "text", "text": values[0][1]})
        elif not reasoning and kind in {"image", "image_url", "input_image"}:
            result.parts.append(image_part(block, source))
            result.ordered.append(result.parts[-1])
        elif not reasoning and kind in {"tool_use", "tool_call"}:
            arguments = block.get("input") if kind == "tool_use" else block.get("args")
            if (
                not isinstance(block.get("name"), str)
                or not isinstance(arguments, (dict, str))
                or not isinstance(block.get("id"), str)
            ):
                result.gaps.append("invalid_tool_content_block")
                continue
            result.calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": arguments},
                }
            )
            result.ordered.append({"type": "tool_calls", "tool_calls": [result.calls[-1]]})
        else:
            result.gaps.append("unsupported_content_block:" + str(kind))
            continue
    if texts:
        result.text = "".join(texts)
    elif not result.gaps and not reasoning:
        result.text = ""  # Explicitly empty/textless assistant content.
    if thoughts:
        result.reasoning = "".join(thoughts)
    elif reasoning and not value:
        result.reasoning = ""
    return result
