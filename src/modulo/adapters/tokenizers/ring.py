"""Ring's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class RingTokenizer(HuggingFaceTextTokenizer):
    name = "ring"
    family = "ring"
    allowed_repo = "inclusionAI/Ring-2.6-1T"
    option_rules = {"reasoning_effort": ("high", "xhigh")}
