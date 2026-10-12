"""GLM's explicitly pinned text reference."""

import math

from modulo.core.domain.interfaces import UnsupportedInput

from .shared.huggingface import HuggingFaceTextTokenizer
from .shared.images import grid_tokens, image_slots


class GLMTokenizer(HuggingFaceTextTokenizer):
    family = "glm"
    # The official template explicitly maps other effort strings to max.
    option_rules = {"reasoning_effort": str, "clear_thinking": bool}

    def render_output(self, response, options):
        if self.custom_template:
            return super().render_output(response, options)
        # GLM's chat template omits termination. The locked tokenizer supplies
        # endoftext; tool turns hand over to the template's observation role.
        last = response[-1] if isinstance(response, list) else response
        ending = "<|observation|>" if last.get("tool_calls") else str(self.tokenizer.eos_token)
        return self.assistant_output(response, options), ending


class GLM52Tokenizer(GLMTokenizer):
    name = "glm-5.2"
    repo = "zai-org/GLM-5.2"
    option_rules = {**GLMTokenizer.option_rules, "enable_thinking": bool}
    thinking_parameter = "enable_thinking"
    thinking_values = {"enabled": True, "disabled": False}


class GLM53Tokenizer(GLMTokenizer):
    name = "glm-5.3"
    repo = "zai-org/GLM-5.3"


class GLM53FlashTokenizer(GLM53Tokenizer):
    name = "glm-5.3-flash"
    repo = "zai-org/GLM-5.3-Flash"
    image_placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    image_grid_supported = True

    def image_plan(self, width, height, parameters, grid):
        factor = (
            parameters["patch_size"] * parameters["merge_size"] * parameters["patch_expand_factor"]
        )
        computed = None
        if width is not None and height is not None:

            def align(value):
                return math.ceil(value / factor) * factor

            h, w = align(height), align(width)
            minimum = parameters["min_image_tokens"] * factor**2
            maximum = parameters["max_image_tokens"] * factor**2
            if h * w < minimum:
                scale = math.sqrt(minimum / (height * width))
                h, w = align(math.ceil(height * scale)), align(math.ceil(width * scale))
            if h * w > maximum:
                low, high = 1, height
                h = w = factor
                while low <= high:
                    candidate = (low + high) // 2
                    ch, cw = align(candidate), align(max(1, math.floor(width * candidate / height)))
                    if ch * cw <= maximum:
                        h, w = ch, cw
                        low = candidate + 1
                    else:
                        high = candidate - 1
            computed = [1, h // parameters["patch_size"], w // parameters["patch_size"]]
        if grid is not None and computed is not None and grid != computed:
            raise UnsupportedInput("conflicting_image_grid_thw")
        grid = computed if grid is None else grid
        tokens = grid_tokens(grid, parameters["merge_size"])
        return {
            "visual_tokens": tokens,
            "replacement": "<|begin_of_image|>"
            + image_slots("<|image|>", tokens)
            + "<|end_of_image|>",
            "layout": {"image_grid_thw": grid},
        }
