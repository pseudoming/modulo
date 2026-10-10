"""Tencent Hunyuan's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class HunyuanTokenizer(HuggingFaceTextTokenizer):
    name = "hunyuan"
    family = "hunyuan"
    allowed_repo = "tencent/Hy4-preview"
    option_rules = {"reasoning_effort": ("high", "no_think"), "preserved_thinking": bool}
    inline_think_tags = ("<think:opensource>", "</think:opensource>")
