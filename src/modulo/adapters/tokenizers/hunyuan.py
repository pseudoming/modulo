"""Tencent Hunyuan's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class HunyuanHy4Tokenizer(HuggingFaceTextTokenizer):
    name = "hy4-preview"
    family = "hunyuan"
    repo = "tencent/Hy4-preview"
    option_rules = {"reasoning_effort": ("high", "no_think"), "preserved_thinking": bool}
    inline_think_tags = ("<think:opensource>", "</think:opensource>")
