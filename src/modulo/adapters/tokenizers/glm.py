"""GLM's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class GLMTokenizer(HuggingFaceTextTokenizer):
    name = "glm"
    family = "glm"
    allowed_repo = "zai-org/GLM-5.3"
    # The official template explicitly maps other effort strings to max.
    option_rules = {"reasoning_effort": str, "clear_thinking": bool}
