"""Pinned DeepSeek V4 / V4.1 references; upstream Python is never executed."""

import ast
import math
from copy import deepcopy

from modulo.core.domain.interfaces import UnsupportedInput
from modulo.shared.serialization import ordered_json

from .shared.huggingface import HuggingFaceTextTokenizer
from .shared.images import image_slots


class DeepSeekV4Tokenizer(HuggingFaceTextTokenizer):
    name = "deepseek-v4-pro-0813"
    family = "deepseek"
    repo = "deepseek-ai/DeepSeek-V4-Pro-0813"
    option_rules = {
        "thinking_mode": ("thinking", "chat"),
        "drop_thinking": bool,
        "reasoning_effort": ("low", "high", "max"),
    }
    roles = {"system", "developer", "user", "assistant", "tool"}
    thinking_parameter = "thinking_mode"
    thinking_values = {
        "enabled": "thinking",
        "disabled": "chat",
        "True": "thinking",
        "False": "chat",
    }

    def compile_template(self, template):
        self.tokenizer.chat_template = None
        return _V4Template(template)

    def translate_option(self, key, value):
        if key == "enable_thinking":
            key = "thinking"
        return super().translate_option(key, value)


class DeepSeekV41Tokenizer(DeepSeekV4Tokenizer):
    name = "deepseek-v4.1"
    repo = "deepseek-ai/DeepSeek-V4.1-Flash"
    option_rules = {
        **DeepSeekV4Tokenizer.option_rules,
        "reasoning_effort": lambda v: (
            (type(v) is int and 1 <= v <= 100)
            or (isinstance(v, str) and v in {"low", "medium", "high", "max"})
        ),
    }
    roles = {"system", "user", "assistant", "tool"}
    image_placeholder = "<｜deepseek_image｜>"

    def compile_template(self, template):
        # V4.1 supplies a Jinja template instead of V4's Python encoder format.
        return HuggingFaceTextTokenizer.compile_template(self, template)

    def image_plan(self, width, height, parameters, grid):
        if grid is not None:
            raise UnsupportedInput("unsupported_image_grid_thw")
        patch = parameters["vision_patch_size"]
        downsample = parameters["vision_downsample_ratio"]
        limit = parameters["vision_max_n_token"]
        ratio_limit = parameters["vision_max_wh_ratio"]
        if ratio_limit is not None and width > height * ratio_limit:
            width = height * ratio_limit
        if width * height < parameters["vision_min_pixels"]:
            scale = math.sqrt(parameters["vision_min_pixels"] / (width * height))
            width, height = int(width * scale), int(height * scale)
        resized_w, resized_h = math.ceil(width / patch) * patch, math.ceil(height / patch) * patch

        def layout(w, h):
            return math.ceil(w // patch / downsample), math.ceil(h // patch / downsample)

        columns, rows = layout(resized_w, resized_h)
        if rows * (columns + 1) + 2 > limit:
            ratio, cell = height / width, patch * downsample
            max_w = math.sqrt((limit - 2) / ratio + 0.25) - 0.5
            max_h = max_w * ratio
            if max_w < 1:
                resized_w, resized_h = cell, (limit - 2) // 2 * cell
            elif max_h < 1:
                resized_w, resized_h = (limit - 3) * cell, cell
            else:
                scale = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
                resized_w, resized_h = (
                    math.floor(side * scale / patch) * patch for side in (width, height)
                )
            columns, rows = layout(resized_w, resized_h)
        if min(rows, columns) <= 0 or rows * (columns + 1) + 2 > limit:
            raise UnsupportedInput("invalid_image_processor_grid")
        return {
            "visual_tokens": rows * columns,
            "replacement": image_slots(self.image_placeholder, rows * (columns + 1) + 2),
            "layout": {"rows": rows, "columns": columns, "width": resized_w, "height": resized_h},
        }


class _V4Template:
    """V4's recorded-message subset, checked against its official encoder.

    The fixed source supplies only literal prompt strings. Control flow stays
    here, with no imports or execution of downloaded Python.
    """

    def __init__(self, source):
        self.constants = {}
        for node in ast.parse(source).body:
            targets = (
                node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            )
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {
                    "REASONING_EFFORT_PROMPTS",
                    "TOOLS_TEMPLATE",
                    "response_format_template",
                }:
                    self.constants[target.id] = ast.literal_eval(node.value)

    def render(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        thinking_mode="thinking",
        drop_thinking=True,
        reasoning_effort="low",
        response_format=None,
        response_schema=None,
        **_,
    ):
        messages = deepcopy(messages)
        schema = response_schema or response_format
        if tools or schema:
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            if tools:
                messages[0]["content"] += "\n\n" + self.constants["TOOLS_TEMPLATE"].format(
                    dsml_token="｜DSML｜",
                    thinking_start_token="<think>",
                    thinking_end_token="</think>",
                    tool_schemas="\n".join(ordered_json(t["function"]) for t in tools),
                )
                drop_thinking = False
            if schema:
                messages[0]["content"] += "\n\n" + self.constants[
                    "response_format_template"
                ].format(schema=ordered_json(schema))
        # V4 merges adjacent user/tool messages, retaining text positions and
        # sorting tool results by the preceding assistant's call IDs.
        merged, order = [], {}
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                order = {t.get("id"): i for i, t in enumerate(message["tool_calls"])}
            if role in {"user", "tool"}:
                block = (role, message["content"], order.get(message.get("tool_call_id"), 0))
                if not merged or merged[-1]["role"] != "user":
                    merged.append({"role": "user", "blocks": []})
                merged[-1]["blocks"].append(block)
            else:
                merged.append(message)
        thinking = thinking_mode == "thinking"
        last_user = max(
            (i for i, m in enumerate(merged) if m["role"] in {"user", "developer"}), default=-1
        )
        if thinking and drop_thinking:
            merged = [m for i, m in enumerate(merged) if m["role"] != "developer" or i >= last_user]
            last_user = max(
                (i for i, m in enumerate(merged) if m["role"] in {"user", "developer"}), default=-1
            )
        parts = ["<｜begin▁of▁sentence｜>"]
        if thinking:
            parts.append(self.constants["REASONING_EFFORT_PROMPTS"][reasoning_effort])
        for i, message in enumerate(merged):
            role = message["role"]
            if role == "system":
                parts.append(message["content"])
            elif role in {"user", "developer"}:
                parts.append("<｜User｜>")
                if role == "developer":
                    parts.append(message["content"])
                else:
                    blocks = message["blocks"]
                    results = iter(
                        sorted((b for b in blocks if b[0] == "tool"), key=lambda b: b[2])
                    )
                    parts.append(
                        "\n\n".join(
                            "<tool_result>" + next(results)[1] + "</tool_result>"
                            if b[0] == "tool"
                            else b[1]
                            for b in blocks
                        )
                    )
            elif role == "assistant":
                if thinking and (not drop_thinking or i > last_user):
                    parts.append(message.get("reasoning_content", "") + "</think>")
                parts.append(message["content"])
                calls = []
                for tool in message.get("tool_calls", []):
                    fn = tool["function"]
                    args = "\n".join(
                        f'<｜DSML｜parameter name="{key}" '
                        f'string="{str(isinstance(value, str)).lower()}">'
                        + (value if isinstance(value, str) else ordered_json(value))
                        + "</｜DSML｜parameter>"
                        for key, value in fn["arguments"].items()
                    )
                    calls.append(f'<｜DSML｜invoke name="{fn["name"]}">\n{args}\n</｜DSML｜invoke>')
                if calls:
                    parts.append(
                        "\n\n<｜DSML｜tool_calls>\n" + "\n".join(calls) + "\n</｜DSML｜tool_calls>"
                    )
                parts.append("<｜end▁of▁sentence｜>")
            if role in {"user", "developer"} and (
                (i == len(merged) - 1 and add_generation_prompt)
                or (i + 1 < len(merged) and merged[i + 1]["role"] == "assistant")
            ):
                parts.append(
                    "<｜Assistant｜>"
                    + (
                        "<think>"
                        if thinking and (not drop_thinking or i >= last_user)
                        else "</think>"
                    )
                )
        return "".join(parts)
