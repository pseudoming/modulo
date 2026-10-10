"""Qwen text reference: one explicitly qualified local asset combination."""

from .shared.huggingface import HuggingFaceTextTokenizer


class QwenTokenizer(HuggingFaceTextTokenizer):
    name = "qwen"
    family = "qwen"
    allowed_repo = "Qwen/Qwen3.8-2.4T-A95B"
    option_rules = {
        "reasoning_effort": ("xhigh", "medium", "low"),
        "preserve_thinking": bool,
        "enable_thinking": (True,),
    }
