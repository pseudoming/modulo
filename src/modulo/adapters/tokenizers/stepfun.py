"""StepFun's fixed text and image reference."""

import math

from modulo.core.domain.interfaces import UnsupportedInput

from .shared.huggingface import HuggingFaceTextTokenizer


class Step37Tokenizer(HuggingFaceTextTokenizer):
    name = "step-3.7"
    family = "stepfun"
    output_separator = "\n"
    repo = "stepfun-ai/Step-3.7-Flash"
    image_placeholder = "<im_patch>"
    # This pinned template renders the provided effort string verbatim.
    option_rules = {"reasoning_effort": str}

    def supports_name(self, message, index):
        return message.role == "system" and index > 0 and message.name == "observation"

    def image_plan(self, width, height, parameters, grid):
        if grid is not None:
            raise UnsupportedInput("unsupported_image_grid_thw")
        if min(width, height) < 32 and max(width, height) / min(width, height) > 4:
            width = height = max(width, height)
        if max(width, height) > 3024:
            scale = 3024 / max(width, height)
            width, height = int(width * scale), int(height * scale)
        longest, shortest = max(width, height), min(width, height)
        if shortest <= 0:
            raise UnsupportedInput("invalid_image_processor_grid")
        window = (
            (shortest if longest / shortest > 1.5 else 0)
            if longest <= 728
            else (min(shortest, 504) if longest / shortest > 4 else 504)
        )
        columns = rows = 0
        if window:

            def crop(side):
                ratio = side / window
                return side if ratio < 1 else window * (int(ratio) + (ratio - side // window > 0.2))

            width, height = crop(width), crop(height)
            columns, rows = (max(1, math.ceil(side / window)) for side in (width, height))
        patch = "<patch_start>" + "<im_patch>" * 81 + "<patch_end>"
        patches = "<patch_newline>".join([patch * columns] * rows)
        return {
            "visual_tokens": rows * columns * 81 + 169,
            "replacement": patches + "<im_start>" + "<im_patch>" * 169 + "<im_end>",
            "layout": {
                "patch_rows": rows,
                "patch_columns": columns,
                "width": width,
                "height": height,
            },
        }
