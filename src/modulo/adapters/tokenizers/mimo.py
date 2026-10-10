"""MiMo's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class MiMoTokenizer(HuggingFaceTextTokenizer):
    name = "mimo"
    family = "mimo"
    allowed_repo = "XiaomiMiMo/MiMo-V2.6-Pro-RL"
    option_rules = {"enable_thinking": bool}
    # MiMo's template distinguishes a raw string from an argument object.
    parse_tool_arguments = False
