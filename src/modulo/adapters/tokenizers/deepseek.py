"""DeepSeek text reference: one explicitly qualified local asset combination."""

from .shared.huggingface import HuggingFaceTextTokenizer


class DeepSeekTokenizer(HuggingFaceTextTokenizer):
    name = "deepseek"
    family = "deepseek"
    allowed_repo = "deepseek-ai/DeepSeek-V4.1-Flash"
    option_rules = {
        "thinking_mode": ("thinking", "chat"),
        "drop_thinking": bool,
        "reasoning_effort": lambda v: (
            (type(v) is int and 1 <= v <= 100)
            or (isinstance(v, str) and v in {"low", "medium", "high", "max"})
        ),
    }
    thinking_parameter = "thinking_mode"
    thinking_values = {
        "enabled": "thinking",
        "disabled": "chat",
        "True": "thinking",
        "False": "chat",
    }

    def translate_option(self, key, value):
        if key == "enable_thinking":
            key = "thinking"
        return super().translate_option(key, value)
