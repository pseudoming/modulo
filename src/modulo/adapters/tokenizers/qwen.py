"""Pinned Qwen references, with per-model thinking and image profiles."""

from .shared.huggingface import HuggingFaceTextTokenizer
from .shared.images import grid_plan, image_slots


class QwenTokenizer(HuggingFaceTextTokenizer):
    family = "qwen"
    output_separator = "\n"
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    image_grid_supported = True

    def image_plan(self, width, height, parameters, grid):
        grid, tokens = grid_plan(width, height, parameters, grid)
        return {
            "visual_tokens": tokens,
            "replacement": "<|vision_start|>"
            + image_slots("<|image_pad|>", tokens)
            + "<|vision_end|>",
            "layout": {"image_grid_thw": grid},
        }


class Qwen38Tokenizer(QwenTokenizer):
    name = "qwen-3.8-2.4t"
    repo = "Qwen/Qwen3.8-2.4T-A95B"
    option_rules = {
        "reasoning_effort": ("xhigh", "medium", "low"),
        "preserve_thinking": bool,
        "enable_thinking": (True,),
    }


class Qwen35Tokenizer(QwenTokenizer):
    option_rules = {"enable_thinking": bool, "add_vision_id": bool}


class Qwen35_122BTokenizer(Qwen35Tokenizer):
    name = "qwen3.5-122b"
    repo = "Qwen/Qwen3.5-122B-A10B"


class Qwen35_397BTokenizer(Qwen35Tokenizer):
    name = "qwen3.5-397b"
    repo = "Qwen/Qwen3.5-397B-A17B"
