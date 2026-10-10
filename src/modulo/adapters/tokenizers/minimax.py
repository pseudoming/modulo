"""MiniMax's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class MiniMaxTokenizer(HuggingFaceTextTokenizer):
    name = "minimax"
    family = "minimax"
    allowed_repo = "MiniMaxAI/MiniMax-M3"
    option_rules = {"thinking_mode": ("enabled", "disabled", "adaptive"), "model_identity": str}
    thinking_parameter = "thinking_mode"
    roles = {"system", "developer", "root", "user", "assistant", "tool"}
    inline_think_tags = ("<mm:think>", "</mm:think>")
