"""Synthetic wire bodies using documented Responses and Messages field names.

Only fixture construction lives here. Expected token counts come from the
existing frozen reference vectors, never from these helpers or the parser.
"""

import json
from copy import deepcopy


def request_body(dialect, messages, *, tools=None, options=None, model="synthetic-model"):
    messages = deepcopy(messages)
    result = {"model": model, **(options or {})}
    if messages and messages[0]["role"] == "system":
        result["instructions" if dialect == "responses" else "system"] = messages.pop(0)["content"]
    items = []
    for message in messages:
        role, text = message["role"], message["content"]
        if dialect == "responses":
            if message.get("reasoning_content") is not None:
                items.append(
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": message["reasoning_content"]}
                        ],
                        "summary": [],
                    }
                )
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": text,
                    }
                )
                continue
            items.append({"type": "message", "role": role, "content": text})
            for call in message.get("tool_calls", []):
                fn = call["function"]
                arguments = fn["arguments"]
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": fn["name"],
                        "arguments": arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    }
                )
        else:
            if role == "tool":
                items.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message["tool_call_id"],
                                "content": text,
                            }
                        ],
                    }
                )
                continue
            blocks = []
            if message.get("reasoning_content") is not None:
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": message["reasoning_content"],
                        "signature": "synthetic-signature",
                    }
                )
            blocks.extend(text if isinstance(text, list) else [{"type": "text", "text": text}])
            for call in message.get("tool_calls", []):
                fn = call["function"]
                arguments = fn["arguments"]
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": fn["name"],
                        "input": json.loads(arguments) if isinstance(arguments, str) else arguments,
                    }
                )
            items.append({"role": role, "content": blocks})
    result["input" if dialect == "responses" else "messages"] = items
    if tools is not None:
        result["tools"] = [
            {"type": "function", **t["function"]}
            if dialect == "responses"
            else {"input_schema" if k == "parameters" else k: v for k, v in t["function"].items()}
            for t in tools
        ]
    return result


def response_body(
    dialect,
    text,
    reasoning=None,
    *,
    calls=(),
    input_tokens=None,
    output_tokens=None,
    model="synthetic-model",
):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    blocks = []
    if dialect == "responses":
        if reasoning is not None:
            blocks.append(
                {
                    "type": "reasoning",
                    "id": "thought-1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": reasoning}],
                }
            )
        if text is not None:
            blocks.append(
                {
                    "type": "message",
                    "id": "message-1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            )
        for call in calls:
            fn = call["function"]
            blocks.append(
                {
                    "type": "function_call",
                    "id": "item-" + call["id"],
                    "call_id": call["id"],
                    "name": fn["name"],
                    "arguments": fn["arguments"],
                    "status": "completed",
                }
            )
        return {
            "object": "response",
            "id": "response-1",
            "status": "completed",
            "model": model,
            "output": blocks,
            "usage": usage,
        }
    if reasoning is not None:
        blocks.append(
            {"type": "thinking", "thinking": reasoning, "signature": "synthetic-signature"}
        )
    if text is not None:
        blocks.append({"type": "text", "text": text})
    for call in calls:
        fn = call["function"]
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": fn["name"],
                "input": json.loads(fn["arguments"]),
            }
        )
    return {
        "type": "message",
        "id": "message-1",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": "tool_use" if calls else "end_turn",
        "usage": usage,
    }


def stream_events(dialect, body):
    if dialect == "responses":
        initial = {**body, "status": "in_progress", "output": [], "usage": None}
        events = [{"type": "response.created", "response": initial}]
        for index, item in enumerate(body["output"]):
            initial_item = {**item, "status": "in_progress"} if "status" in item else dict(item)
            if item["type"] == "function_call":
                initial_item["arguments"] = ""
            else:
                initial_item["content"] = []
            events.append(
                {"type": "response.output_item.added", "output_index": index, "item": initial_item}
            )
            common = {"output_index": index, "item_id": item["id"]}
            if item["type"] == "function_call":
                value = item["arguments"]
                for fragment in (value[:2], value[2:]):
                    events.append(
                        {
                            "type": "response.function_call_arguments.delta",
                            **common,
                            "delta": fragment,
                        }
                    )
                events.append(
                    {"type": "response.function_call_arguments.done", **common, "arguments": value}
                )
            else:
                for position, part in enumerate(item["content"]):
                    content = {**common, "content_index": position}
                    events.append(
                        {
                            "type": "response.content_part.added",
                            **content,
                            "part": {**part, "text": ""},
                        }
                    )
                    prefix = (
                        "response.reasoning_text"
                        if item["type"] == "reasoning"
                        else "response.output_text"
                    )
                    for fragment in (part["text"][:2], part["text"][2:]):
                        events.append({"type": prefix + ".delta", **content, "delta": fragment})
                    events.append({"type": prefix + ".done", **content, "text": part["text"]})
                    events.append({"type": "response.content_part.done", **content, "part": part})
            events.append(
                {"type": "response.output_item.done", "output_index": index, "item": item}
            )
        events.append({"type": "response." + body["status"], "response": body})
        return [{**event, "sequence_number": i} for i, event in enumerate(events)]
    events = [
        {
            "type": "message_start",
            "message": {
                **body,
                "content": [],
                "stop_reason": None,
                "usage": {**body["usage"], "output_tokens": 1},
            },
        }
    ]
    for index, block in enumerate(body["content"]):
        key, kind = {
            "text": ("text", "text_delta"),
            "thinking": ("thinking", "thinking_delta"),
            "tool_use": ("input", "input_json_delta"),
        }[block["type"]]
        empty = {**block, key: {} if key == "input" else ""}
        if key == "thinking":
            empty["signature"] = ""
        events.append({"type": "content_block_start", "index": index, "content_block": empty})
        value = json.dumps(block[key], ensure_ascii=False) if key == "input" else block[key]
        for fragment in (value[:2], value[2:]):
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": kind, "partial_json" if key == "input" else key: fragment},
                }
            )
        if key == "thinking":
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": block["signature"]},
                }
            )
        events.append({"type": "content_block_stop", "index": index})
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": body["stop_reason"],
                    **({"stop_sequence": body["stop_sequence"]} if "stop_sequence" in body else {}),
                },
                "usage": {"output_tokens": body["usage"]["output_tokens"]},
            },
            {"type": "message_stop"},
        ]
    )
    return events


def sse(events):
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    )
