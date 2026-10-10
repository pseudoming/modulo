"""LongCat's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class LongCatTokenizer(HuggingFaceTextTokenizer):
    name = "longcat"
    family = "longcat"
    allowed_repo = "meituan-longcat/LongCat-2.0"
    option_rules = {
        "enable_thinking": bool,
        "save_history_reasoning_content": bool,
        "thinking_budget": lambda v: type(v) is int and v >= 0,
        "tool_choice": ("auto", "none", "required"),
    }
    inline_think_tags = ("<longcat_think>", "</longcat_think>")
