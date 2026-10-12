"""MiniMax's fixed text and image reference."""

from .shared.huggingface import HuggingFaceTextTokenizer
from .shared.images import grid_plan, image_slots


class MiniMaxTokenizer(HuggingFaceTextTokenizer):
    family = "minimax"
    output_separator = "\n"
    thinking_parameter = "thinking_mode"


class MiniMaxM27Tokenizer(MiniMaxTokenizer):
    name = "minimax-m2.7"
    repo = "MiniMaxAI/MiniMax-M2.7"
    option_rules = {"thinking_mode": ("enabled",), "model_identity": str}
    empty_thinking_prefix = "<think>\n"


class MiniMaxM3Tokenizer(MiniMaxTokenizer):
    name = "minimax-m3"
    repo = "MiniMaxAI/MiniMax-M3"
    empty_thinking_prefix = "<mm:think>"
    image_placeholder = "]<]image[>["
    image_grid_supported = True
    option_rules = {"thinking_mode": ("enabled", "disabled", "adaptive"), "model_identity": str}
    roles = {"system", "developer", "root", "user", "assistant", "tool"}
    inline_think_tags = ("<mm:think>", "</mm:think>")

    def image_plan(self, width, height, parameters, grid):
        grid, tokens = grid_plan(width, height, parameters, grid, minimax=True)
        return {
            "visual_tokens": tokens,
            "replacement": "]<]start of image[>["
            + image_slots(self.image_placeholder, tokens)
            + "]<]end of image[>[",
            "layout": {"image_grid_thw": grid},
        }
