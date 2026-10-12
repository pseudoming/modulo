"""Ring's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class Ring26Tokenizer(HuggingFaceTextTokenizer):
    name = "ring-2.6"
    family = "ring"
    output_end = "<|role_end|>"
    output_separator = "\n\n"
    repo = "inclusionAI/Ring-2.6-1T"
    option_rules = {"reasoning_effort": ("high", "xhigh")}
