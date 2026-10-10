"""StepFun's explicitly pinned text reference."""

from .shared.huggingface import HuggingFaceTextTokenizer


class StepFunTokenizer(HuggingFaceTextTokenizer):
    name = "stepfun"
    family = "stepfun"
    allowed_repo = "stepfun-ai/Step-3.7-Flash"
    # This pinned template renders the provided effort string verbatim.
    option_rules = {"reasoning_effort": str}

    def supports_name(self, message, index):
        return message.role == "system" and index > 0 and message.name == "observation"
