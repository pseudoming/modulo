"""Ling's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class Ling30Tokenizer(HuggingFaceTextTokenizer):
    name = "ling-3.0"
    family = "ling"
    repo = "inclusionAI/Ling-3.0-flash"
    option_rules = {"enable_thinking": bool}

    def translate_option(self, key, value):
        if key == "thinking_option" and value in ("on", "off"):
            return "enable_thinking", value == "on"
        return super().translate_option(key, value)
