"""MiMo's fixed text and image reference."""

from .shared.huggingface import HuggingFaceTextTokenizer
from .shared.images import grid_plan, image_slots


class MiMoV26Tokenizer(HuggingFaceTextTokenizer):
    name = "mimo-v2.6"
    family = "mimo"
    repo = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    image_grid_supported = True
    option_rules = {"enable_thinking": bool}
    # MiMo's template distinguishes a raw string from an argument object.
    parse_tool_arguments = False

    def image_plan(self, width, height, parameters, grid):
        grid, tokens = grid_plan(width, height, parameters, grid)
        return {
            "visual_tokens": tokens,
            "replacement": "<|vision_start|>"
            + image_slots("<|image_pad|>", tokens)
            + "<|vision_end|>",
            "layout": {"image_grid_thw": grid},
        }
